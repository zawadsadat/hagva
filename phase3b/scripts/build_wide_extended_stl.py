#!/mnt/sda/HAGVA_R2/phase1/iqa_venv/bin/python3
"""Build phase3b/scores/stlucia_wide_extended.csv (3b-STL copy of build_wide_extended.py) from <ds>_wide_scores.csv using the Phase 1 helper functions
(phase1/scripts/build_scores.py: WEIGHTS/fmt, add_random, add_iqa). Columns: image_stem, mean_sem_score, geo_fixed, holefrac,
mean_combined_fixed, w100 (= sem), w000f w025f w050f w075f (w*sem + (1-w)*geo_fixed), rand_s0..s4, brisque, niqe, clipiqa_neg.
IQA runs on CPU (pyiqa venv). Run with the venv python."""
import csv, json, os, sys, importlib.util, time
sys.argv += [] ; import numpy as np
spec = importlib.util.spec_from_file_location("bs", "/mnt/sda/HAGVA_R2/phase1/scripts/build_scores.py"); bs = importlib.util.module_from_spec(spec); spec.loader.exec_module(bs)
P3 = "/mnt/sda/HAGVA_R2/phase3b"; GW = "/mnt/sda/Zawad/genwarp"; HOME = "/home/newview/genwarp"
POOLS = {"stlucia": ("st", 640)}   # StLucia only (3b-STL); garden/sfu extended CSVs are never rewritten
prov = {"built": time.strftime("%Y-%m-%d %H:%M:%S"), "helper": "phase1/scripts/build_scores.py (WEIGHTS, add_random, add_iqa)", "per_dataset": {}}
for ds, (pool, _) in POOLS.items():
    rows = list(csv.DictReader(open(f"{P3}/scores/{ds}_wide_scores.csv", newline="")))
    assert len(rows) == 400
    for r in rows:
        g, s = float(r["geo_fixed"]), float(r["mean_sem_score"])
        r["w100"] = bs.fmt(s)
        for col, w in bs.WEIGHTS.items():
            if col != "w100": r[col + "f"] = bs.fmt(w * s + (1.0 - w) * g)
    bs.add_random(rows)                                           # numpy default_rng(seed).permutation(400), seeds 0-4, CSV row order
    paths = {r["image_stem"]: (f"{HOME}/{pool}_400_out_wide/{r['synth_file']}", f"{GW}/{pool}_400/{r['source_file']}") for r in rows}
    for st, (sp, so) in paths.items(): assert os.path.exists(sp) and os.path.exists(so), (sp, so)
    p = {}; bs.add_iqa(ds, rows, paths, "cpu", p); prov["per_dataset"][ds] = p.get("iqa", {})
    fields = ["image_stem", "synth_file", "source_file", "ele", "r", "holefrac", "mean_sem_score", "geo_fixed", "mean_combined_fixed",
              "w100", "w000f", "w025f", "w050f", "w075f"] + [f"rand_s{i}" for i in range(5)] + ["brisque", "niqe", "clipiqa_neg"]
    out = f"{P3}/scores/{ds}_wide_extended.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    print(f"[{ds}] wrote {out} cols={fields}", flush=True)
json.dump(prov, open(f"{P3}/scores/provenance_wide_stlucia.json", "w"), indent=2); print("provenance ->", f"{P3}/scores/provenance_wide_stlucia.json")
