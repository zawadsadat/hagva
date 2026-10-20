#!/home/newview/miniconda3/envs/vprtutorial/bin/python3
"""
Phase 1 (R2): build extended per-view score CSVs.

For each dataset, reads module_a/scores/<ds>_400_combined_summary.csv and writes
/mnt/sda/HAGVA_R2/phase1/scores/<ds>_400_extended.csv with the original columns
plus:
  w000 w025 w050 w075 w100   : w*sem + (1-w)*geo  (full float precision)
  rand_s0 .. rand_s4         : random permutation rank, numpy default_rng(seed)
  brisque, niqe, clipiqa_neg : no-reference IQA of the synthesized view (pyiqa)
  dino_global                : 1 - cos(DINOv2 ViT-S/14 CLS(source), CLS(synth))
Lower = accepted first for every column.

Resumable: existing columns in the extended CSV are kept unless --force.
  --groups weights,random,iqa,dino,fusion   which column groups to (re)compute
Phase 1b adds group "fusion": zavg rankavg prod maxgs sem_veto50 sem_veto75 sem_veto90 geo_veto75
Phase 2  adds group "geofix" (opt-in): geo_fixed, w000f w025f w050f w075f, and the 8 fusion rules with suffix f,
         all derived from phase1/scores/<ds>_400_geo_fixed.csv (recompute_geo.py); w100f == w100 so it is omitted.
  --device cuda|cpu
Writes /mnt/sda/HAGVA_R2/phase1/scores/provenance.json (merged across runs).
"""
import argparse, csv, hashlib, json, os, re, sys, time
import numpy as np

# DINOv2 torch.hub pin: the cached 'main' snapshot needs Python>=3.10 syntax; this is
# the last commit before 2025-01-01 (2024-10-26, "Fix: x.item() ... meta device").
DINOV2_COMMIT = "85a24602099d397264d5b30461ad7f3bfd726ca1"

REPO_SCORES = "/mnt/sda/hagva-repo/module_a/scores"
GENWARP     = "/mnt/sda/Zawad/genwarp"
OUT_DIR     = "/mnt/sda/HAGVA_R2/phase1/scores"
DATASETS = {  # key: (score-file prefix, genwarp dir prefix)
    "gardenspoint": "garden",
    "sfu":          "sfu",
    "stlucia":      "st",
}
WEIGHTS = {"w000": 0.0, "w025": 0.25, "w050": 0.5, "w075": 0.75, "w100": 1.0}
SEEDS   = [0, 1, 2, 3, 4]
IQA_COLS  = ["brisque", "niqe", "clipiqa_neg"]
DINO_COL  = "dino_global"
FUSION_COLS = ["zavg", "rankavg", "prod", "maxgs", "sem_veto50", "sem_veto75", "sem_veto90", "geo_veto75"]
GEOFIX_W    = ["w000f", "w025f", "w050f", "w075f"]            # w*sem + (1-w)*geo_fixed  (w100f == w100 == sem, omitted)
GEOFIX_COLS = ["geo_fixed"] + GEOFIX_W + [c + "f" for c in FUSION_COLS]
ALL_NEW   = list(WEIGHTS) + [f"rand_s{s}" for s in SEEDS] + IQA_COLS + [DINO_COL] + FUSION_COLS + GEOFIX_COLS

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def synth_to_stem(synth_file):
    """'001_000_Image127_ele1_r0.10.jpg' -> '000_Image127' (same rule as gate scripts)."""
    stem = os.path.splitext(synth_file)[0]
    m = re.search(r'_ele([\.\d]+)_r([\d.]+)$', stem)
    base = stem[:m.start()]
    return re.sub(r'^\d+_', '', base, count=1)

def read_csv(path):
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        return list(r), r.fieldnames

def write_csv_atomic(path, rows, fieldnames):
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(rows)
    os.replace(tmp, path)

def fmt(x):
    return repr(float(x)) if not isinstance(x, (int, np.integer)) else str(int(x))

# ── column groups ─────────────────────────────────────────────────────────────
def add_weights(rows):
    for r in rows:
        geo, sem = float(r["mean_geo_score"]), float(r["mean_sem_score"])
        for col, w in WEIGHTS.items():
            r[col] = fmt(w * sem + (1.0 - w) * geo)

def add_random(rows):
    n = len(rows)
    for s in SEEDS:
        perm = np.random.default_rng(s).permutation(n)
        for i, r in enumerate(rows):
            r[f"rand_s{s}"] = str(int(perm[i]))

def image_paths(ds, rows):
    """Return {image_stem: (synth_path, source_path)} using <ds>_400_signal1_summary.csv."""
    sig, _ = read_csv(f"{REPO_SCORES}/{ds}_400_signal1_summary.csv")
    gdir = DATASETS[ds]
    out = {}
    for s in sig:
        stem = synth_to_stem(s["synth_file"])
        out[stem] = (f"{GENWARP}/{gdir}_400_out_low/{s['synth_file']}",
                     f"{GENWARP}/{gdir}_400/{s['source_file']}")
    stems = [r["image_stem"] for r in rows]
    missing = [s for s in stems if s not in out]
    assert not missing and len(out) == len(stems) == 400, f"{ds}: mapping mismatch {len(out)} vs {len(stems)}, missing={missing[:3]}"
    for stem in stems:
        for p in out[stem]:
            assert os.path.exists(p), p
    return out

def load_rgb_tensor(path, size=None):
    import torch
    from PIL import Image
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize((size, size), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # 1x3xHxW in [0,1]

def add_iqa(ds, rows, paths, device, prov):
    import torch, pyiqa
    metrics = {}
    for name in ["brisque", "niqe", "clipiqa"]:
        metrics[name] = pyiqa.create_metric(name, device=device)
        info = {"lower_better": bool(metrics[name].lower_better)}
        try:
            from pyiqa.default_model_configs import DEFAULT_CONFIGS
            info["metric_opts"] = json.loads(json.dumps(DEFAULT_CONFIGS[name].get("metric_opts", {}), default=str))
        except Exception as e:
            info["metric_opts_error"] = str(e)
        prov.setdefault("iqa", {})[name] = info
    t0 = time.time()
    with torch.no_grad():
        for i, r in enumerate(rows):
            x = load_rgb_tensor(paths[r["image_stem"]][0]).to(device)
            b = float(metrics["brisque"](x).item())
            q = float(metrics["niqe"](x).item())
            c = float(metrics["clipiqa"](x).item())
            r["brisque"] = fmt(b); r["niqe"] = fmt(q); r["clipiqa_neg"] = fmt(-c)
            if (i + 1) % 100 == 0:
                print(f"  [{ds}] iqa {i+1}/{len(rows)}  {time.time()-t0:.0f}s", flush=True)
    prov["iqa"]["pyiqa_version"] = pyiqa.__version__
    prov["iqa"]["note"] = "clipiqa_neg = -clipiqa so that lower = better for all IQA columns; brisque/niqe already lower=better"
    prov["iqa"]["input"] = "synthesized view, RGB in [0,1], native resolution, no resize"

def add_dino(ds, rows, paths, device, prov):
    import torch, torch.nn.functional as F
    torch.hub.set_dir(os.path.expanduser("~/.cache/torch/hub"))
    model = torch.hub.load(f"facebookresearch/dinov2:{DINOV2_COMMIT}", "dinov2_vits14",
                           skip_validation=True).to(device).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    def cls(path):
        x = load_rgb_tensor(path, size=448).to(device)
        x = (x - mean) / std
        return model(x)  # DINOv2 hub forward() returns the CLS token embedding (identity head)
    t0 = time.time()
    with torch.no_grad():
        for i, r in enumerate(rows):
            sp, so = paths[r["image_stem"]]
            cs = F.cosine_similarity(cls(so), cls(sp)).item()
            r[DINO_COL] = fmt(1.0 - cs)
            if (i + 1) % 100 == 0:
                print(f"  [{ds}] dino {i+1}/{len(rows)}  {time.time()-t0:.0f}s", flush=True)
    ck = os.path.expanduser("~/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth")
    prov["dino_global"] = {
        "model": "facebookresearch/dinov2 dinov2_vits14 (torch.hub)", "hub_commit": DINOV2_COMMIT,
        "weights_file": ck, "weights_sha256": sha256(ck) if os.path.exists(ck) else None,
        "input": "448x448 bicubic resize, ImageNet mean/std norm; CLS token; score = 1 - cosine(source, synth)",
        "source_image": f"{GENWARP}/<ds>_400/<source_file from signal1 csv>",
    }

# ── Phase 1b: non-linear fusion rules (group "fusion") ────────────────────────
FUSION_DEFS = {
    "zavg":       "(z(g) + z(s))/2, z = (x - mean)/std within the dataset (numpy std, ddof=0)",
    "rankavg":    "(rank(g) + rank(s))/2, 0-based ranks over the 400 views, ties -> average rank (scipy rankdata 'average' - 1)",
    "prod":       "g * s",
    "maxgs":      "max(g, s)",
    "sem_veto50": "s if g <= P50(g) else s + 10   (numpy.percentile, linear)",
    "sem_veto75": "s if g <= P75(g) else s + 10",
    "sem_veto90": "s if g <= P90(g) else s + 10",
    "geo_veto75": "g if s <= P75(s) else g + 10   (symmetric control)",
}

def fusion_from(g, s):
    """The 8 Phase-1b fusion rules from geo vector g and sem vector s. Returns (cols dict, percentiles dict)."""
    from scipy.stats import rankdata
    zg = (g - g.mean()) / g.std(); zs = (s - s.mean()) / s.std()
    cols = {
        "zavg":    (zg + zs) / 2.0,
        "rankavg": ((rankdata(g, method="average") - 1) + (rankdata(s, method="average") - 1)) / 2.0,
        "prod":    g * s,
        "maxgs":   np.maximum(g, s),
    }
    pct = {}
    for p in (50, 75, 90):
        pct[f"P{p}(g)"] = float(np.percentile(g, p))
        cols[f"sem_veto{p}"] = np.where(g <= pct[f"P{p}(g)"], s, s + 10.0)
    pct["P75(s)"] = float(np.percentile(s, 75))
    cols["geo_veto75"] = np.where(s <= pct["P75(s)"], g, g + 10.0)
    return cols, pct

def check_fusion(ds, cols, g, s, label=""):
    for k, v in cols.items():
        assert not np.isnan(v).any(), f"{ds}: NaN in {k}{label}"
    assert (cols["maxgs"] >= 0.5 * s + 0.5 * g).all(), f"{ds}: maxgs{label} < w050{label} somewhere"
    unvetoed = {}
    for k in ("sem_veto50", "sem_veto75", "sem_veto90", "geo_veto75"):
        unvetoed[k + label] = int((cols[k] < 10).sum())
        assert unvetoed[k + label] >= 145, f"{ds}: only {unvetoed[k + label]} un-vetoed views in {k}{label} (< 145)"
    return unvetoed

def add_geofix(ds, rows, prov):
    """Phase 2: re-derive the weight and fusion columns with Module A's corrected geometric score.
    Reads /mnt/sda/HAGVA_R2/phase1/scores/<ds>_400_geo_fixed.csv (from phase2/scripts/recompute_geo.py)."""
    path = f"{OUT_DIR}/{ds}_400_geo_fixed.csv"
    assert os.path.exists(path), f"{ds}: {path} missing -> run phase2/scripts/recompute_geo.py first (needs <pool>_400_warp_only + _warp_masks)"
    fixed = {r["image_stem"]: float(r["mean_geo_score_fixed"]) for r in csv.DictReader(open(path, newline=""))}
    assert all(r["image_stem"] in fixed for r in rows), f"{ds}: geo_fixed.csv does not cover all 400 stems"
    g = np.array([fixed[r["image_stem"]] for r in rows])          # 4-decimal, as Module A writes
    s = np.array([float(r["mean_sem_score"]) for r in rows])
    assert not np.isnan(g).any()
    cols = {"geo_fixed": g}
    for col, w in WEIGHTS.items():
        if col != "w100": cols[col + "f"] = w * s + (1.0 - w) * g
    fcols, pct = fusion_from(g, s)
    unvetoed = check_fusion(ds, fcols, g, s, label="f")
    for k, v in fcols.items(): cols[k + "f"] = v
    print(f"[{ds}] geofix asserts OK: un-vetoed counts {unvetoed}; percentiles { {k: round(v, 4) for k, v in pct.items()} }; "
          f"Spearman(geo_old, geo_fixed) = {float(np.corrcoef(np.argsort(np.argsort([float(r['mean_geo_score']) for r in rows])), np.argsort(np.argsort(g)))[0,1]):.4f}")
    for i, r in enumerate(rows):
        for k, v in cols.items(): r[k] = fmt(float(v[i]))
    prov.setdefault("geofix", {})["source"] = path
    prov["geofix"]["definitions"] = {"geo_fixed": "mean_geo_score recomputed by phase2/scripts/recompute_geo.py (Module A formula, regenerated warp/mask resized to synth size)",
                                     "wXXXf": "w*mean_sem_score + (1-w)*geo_fixed", "<rule>f": "Phase-1b fusion rule with g = geo_fixed"}
    prov["geofix"].setdefault("per_dataset", {})[ds] = {"percentiles": pct, "unvetoed_counts": unvetoed}

def add_fusion(ds, rows, prov):
    g = np.array([float(r["mean_geo_score"]) for r in rows])
    s = np.array([float(r["mean_sem_score"]) for r in rows])
    cols, pct = fusion_from(g, s)
    unvetoed = check_fusion(ds, cols, g, s)
    print(f"[{ds}] fusion asserts OK: no NaNs; maxgs >= w050 row-wise; un-vetoed counts (value<10): {unvetoed}; "
          f"percentiles: { {k: round(v, 4) for k, v in pct.items()} }")
    for i, r in enumerate(rows):
        for k, v in cols.items():
            r[k] = fmt(float(v[i]))
    prov.setdefault("fusion", {})["definitions"] = FUSION_DEFS
    prov["fusion"]["inputs"] = "4-decimal mean_geo_score (g) and mean_sem_score (s) from the combined_summary CSV; lower = accepted first"
    prov["fusion"].setdefault("per_dataset", {})[ds] = {"percentiles": pct, "unvetoed_counts": unvetoed,
                                                        "mean_std": {"g": [float(g.mean()), float(g.std())], "s": [float(s.mean()), float(s.std())]}}

def _old_add_fusion_body_removed(ds, rows, prov):
    from scipy.stats import rankdata
    g = np.array([float(r["mean_geo_score"]) for r in rows])
    s = np.array([float(r["mean_sem_score"]) for r in rows])
    zg = (g - g.mean()) / g.std(); zs = (s - s.mean()) / s.std()
    cols = {
        "zavg":    (zg + zs) / 2.0,
        "rankavg": ((rankdata(g, method="average") - 1) + (rankdata(s, method="average") - 1)) / 2.0,
        "prod":    g * s,
        "maxgs":   np.maximum(g, s),
    }
    pct = {}
    for p in (50, 75, 90):
        pct[f"P{p}(g)"] = float(np.percentile(g, p))
        cols[f"sem_veto{p}"] = np.where(g <= pct[f"P{p}(g)"], s, s + 10.0)
    pct["P75(s)"] = float(np.percentile(s, 75))
    cols["geo_veto75"] = np.where(s <= pct["P75(s)"], g, g + 10.0)
    # assertions required by the Phase 1b brief
    for k, v in cols.items():
        assert not np.isnan(v).any(), f"{ds}: NaN in {k}"
    w050 = 0.5 * s + 0.5 * g
    assert (cols["maxgs"] >= w050).all(), f"{ds}: maxgs < w050 somewhere"
    unvetoed = {}
    for k in ("sem_veto50", "sem_veto75", "sem_veto90", "geo_veto75"):
        unvetoed[k] = int((cols[k] < 10).sum())
        assert unvetoed[k] >= 145, f"{ds}: only {unvetoed[k]} un-vetoed views in {k} (< 145)"
    print(f"[{ds}] fusion asserts OK: no NaNs; maxgs >= w050 row-wise; un-vetoed counts (value<10): {unvetoed}; "
          f"percentiles: { {k: round(v, 4) for k, v in pct.items()} }")
    for i, r in enumerate(rows):
        for k, v in cols.items():
            r[k] = fmt(float(v[i]))
    prov.setdefault("fusion", {})["definitions"] = FUSION_DEFS
    prov["fusion"]["inputs"] = "4-decimal mean_geo_score (g) and mean_sem_score (s) from the combined_summary CSV; lower = accepted first"
    prov["fusion"].setdefault("per_dataset", {})[ds] = {"percentiles": pct, "unvetoed_counts": unvetoed,
                                                        "mean_std": {"g": [float(g.mean()), float(g.std())], "s": [float(s.mean()), float(s.std())]}}

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default="weights,random,iqa,dino,fusion")  # geofix is opt-in: --groups geofix
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--datasets", default=",".join(DATASETS))
    a = ap.parse_args()
    groups = set(a.groups.split(","))
    os.makedirs(OUT_DIR, exist_ok=True)
    prov_path = f"{OUT_DIR}/provenance.json"
    prov = json.load(open(prov_path)) if os.path.exists(prov_path) else {}
    import torch
    prov["env"] = {"python": sys.executable, "torch": torch.__version__,
                   "numpy": np.__version__, "cuda": torch.cuda.is_available(),
                   "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    prov.setdefault("runs", []).append({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "groups": sorted(groups),
                                        "device": a.device, "datasets": a.datasets, "argv": sys.argv})
    prov["random"] = {"generator": "numpy.random.default_rng(seed).permutation(400)", "seeds": SEEDS,
                      "semantics": "rank per view in CSV row order; lower rank accepted first"}
    prov["weights"] = {"formula": "w*mean_sem_score + (1-w)*mean_geo_score", "w": WEIGHTS,
                       "note": "computed from the 4-decimal values stored in the combined_summary CSV; "
                               "w050 therefore differs from stored mean_combined_score by <= 5e-5"}

    for ds in a.datasets.split(","):
        src = f"{REPO_SCORES}/{ds}_400_combined_summary.csv"
        out = f"{OUT_DIR}/{ds}_400_extended.csv"
        base_rows, base_fields = read_csv(src)
        if os.path.exists(out) and not a.force:
            rows, fields = read_csv(out)
            assert [r["image_stem"] for r in rows] == [r["image_stem"] for r in base_rows]
        else:
            rows, fields = base_rows, list(base_fields)
        have = set(fields)
        def need(cols): return a.force or not all(c in have for c in cols)
        prov.setdefault("inputs", {})[ds] = {"combined_csv": src, "sha256": sha256(src),
                                              "signal1_csv": f"{REPO_SCORES}/{ds}_400_signal1_summary.csv"}
        if "weights" in groups and need(WEIGHTS): add_weights(rows); print(f"[{ds}] weights done")
        if "random"  in groups and need([f"rand_s{s}" for s in SEEDS]): add_random(rows); print(f"[{ds}] random done")
        if "fusion"  in groups and need(FUSION_COLS): add_fusion(ds, rows, prov); print(f"[{ds}] fusion done")
        if "geofix"  in groups and need(GEOFIX_COLS): add_geofix(ds, rows, prov); print(f"[{ds}] geofix done")
        if ("iqa" in groups and need(IQA_COLS)) or ("dino" in groups and need([DINO_COL])):
            paths = image_paths(ds, rows)
            if "iqa"  in groups and need(IQA_COLS):  add_iqa(ds, rows, paths, a.device, prov);  print(f"[{ds}] iqa done")
            if "dino" in groups and need([DINO_COL]): add_dino(ds, rows, paths, a.device, prov); print(f"[{ds}] dino done")
        fields = list(base_fields) + [c for c in ALL_NEW if c in rows[0]]
        write_csv_atomic(out, rows, fields)
        print(f"[{ds}] wrote {out}  cols={fields}")
    json.dump(prov, open(prov_path, "w"), indent=2)
    print(f"provenance -> {prov_path}")

if __name__ == "__main__":
    main()
