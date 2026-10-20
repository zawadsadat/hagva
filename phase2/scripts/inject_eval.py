#!/home/newview/miniconda3/envs/vprtutorial/bin/python3
"""
Phase 2 single VPR evaluation: inject one condition's 50 views for one draw and run demo.py once.
Injection and GT mechanisms copied from phase1/scripts/r2_gate_{gp,sfu,stl}.py (which are the Phase 1
gate scripts): GardensPoint = Image{200+k}.jpg + extra_query_gt loader patch; SFU = z_hagva_tmp_* +
copy_from loader patch with GT.npz restored from GT_original.npz; StLucia = zzz_inject_NNNN.jpg +
explicit GT.npz column extension in the staging dir.  Called by run_phase2.sh; do not run by hand
while any other gate/VPR process is alive (serial rule).

  inject_eval.py --ds gp|sfu|stl --seed 0 --condition C0|C1|C2|C2b|C4|C5|C6|C7 --descriptor NetVLAD|PatchNetVLAD
                 [--draws-dir phase2/draws] [--views-root phase2/views] [--out-dir DIR] [--save-s] [--dry-run]
Prints a final line:  RESULT ds=<ds> seed=<seed> condition=<C> descriptor=<D> n_injected=<n> auc=<x.xxx>
"""
import argparse, csv, os, re, shutil, subprocess, sys
from glob import glob
import numpy as np

PY = "/home/newview/miniconda3/envs/vprtutorial/bin/python3"
VPR_DIR = "/mnt/sda/Zawad/genwarp/VPR-GIT"
LOADER = f"{VPR_DIR}/datasets/load_dataset.py"
GW = "/mnt/sda/Zawad/genwarp"
P2 = "/mnt/sda/HAGVA_R2/phase2"

CFG = {
    "gp": dict(dataset="GardensPoint", pool="garden", stage="/tmp/hagva_p2_gp_stage",
               images_sub="images/GardensPoint", q_sub="night_right", ref_sub="day_right",
               real_q=f"{VPR_DIR}/images/GardensPoint/night_right", real_ref=f"{VPR_DIR}/images/GardensPoint/day_right",
               marker="# HAGVA_GT_PATCH"),
    "sfu": dict(dataset="SFU", pool="sfu", stage="/tmp/hagva_p2_sfu_stage",
                images_sub="images/SFU", q_sub="jan", ref_sub="dry",
                real_q=f"{VPR_DIR}/images/SFU/jan", real_ref=f"{VPR_DIR}/images/SFU/dry",
                real_gt_dir=f"{GW}/VPR-Query-NEW/images/SFU", marker="# HAGVA_SFU400_GT_PATCH",
                anchor="        # Resize hard GT to match current number of queries (expand or truncate)"),
    "stl": dict(dataset="StLucia", pool="st", stage="/tmp/hagva_p2_stl_stage",
                images_sub="images/StLucia_small", q_sub="180809_1545", ref_sub="100909_0845",
                real_q=f"{VPR_DIR}/images/StLucia_small/180809_1545", real_ref=f"{VPR_DIR}/images/StLucia_small/100909_0845",
                real_gt=f"{VPR_DIR}/images/StLucia_small/GT.npz"),
}
VARIANT = {"C4": "warp_black", "C5": "warp_telea", "C6": "genwarp_black", "C7": "genwarp_telea"}
CONDITIONS = {"gp": ["C0", "C1", "C2", "C2b", "C4", "C5", "C6", "C7"], "sfu": ["C0", "C1", "C2", "C4", "C5", "C6", "C7"], "stl": ["C0", "C1", "C2", "C4", "C5", "C6", "C7"]}

def source_path(ds, cond, row, views_root):
    c = CFG[ds]
    if cond == "C1":  return f"{GW}/{c['pool']}_400_out_low/{row['synth_file']}"
    if cond == "C2":  return f"{GW}/{c['pool']}_400/{row['stem']}.jpg"
    if cond == "C2b":
        assert ds == "gp", "C2b (day_left) is GardensPoint-only"
        return f"{VPR_DIR}/images/GardensPoint/day_left/{row['source_image']}"
    if cond in VARIANT: return f"{views_root}/{ds}/{VARIANT[cond]}/{row['stem']}.jpg"
    raise ValueError(cond)

# ── loader patching (verbatim logic from r2_gate_gp.py / r2_gate_sfu.py) ─────────────────────────
def _strip_marker_block(content, marker):
    if marker in content:
        content = re.sub(rf'\n        {marker}.*?{marker}\n', '', content, flags=re.DOTALL)
    return content

def patch_loader_gp(gt_additions, marker):
    content = _strip_marker_block(open(LOADER).read(), marker)
    patch = f"\n        {marker}\n" + "".join(f"        extra_query_gt[{q}] = {r}\n" for q, r in gt_additions.items()) + f"        {marker}\n"
    anchor = "        q_basenames = [os.path.basename(f) for f in fns_q]"
    assert anchor in content, "GP loader anchor not found"
    open(LOADER, "w").write(content.replace(anchor, patch + anchor))

def patch_loader_sfu(copy_from, marker, anchor):
    content = _strip_marker_block(open(LOADER).read(), marker)
    patch = f"\n        {marker}\n" + "".join(f"        copy_from['{d}'] = '{s}'\n" for d, s in copy_from.items()) + f"        {marker}\n"
    assert anchor in content, "SFU loader anchor not found"
    open(LOADER, "w").write(content.replace(anchor, patch + anchor))

def unpatch_loader(marker):
    content = open(LOADER).read()
    new = _strip_marker_block(content, marker)
    if new != content: open(LOADER, "w").write(new)

def loader_clean():
    t = open(LOADER).read()
    return ("extra_query_gt[" not in t.replace("extra_query_gt = {", "")) and ("copy_from['z_hagva_tmp_" not in t)

# ── staging ─────────────────────────────────────────────────────────────────────────────────────
def setup_staging(ds):
    c = CFG[ds]; stage_q = f"{c['stage']}/{c['images_sub']}/{c['q_sub']}"; stage_ref = f"{c['stage']}/{c['images_sub']}/{c['ref_sub']}"
    os.makedirs(stage_q, exist_ok=True)
    if not os.path.exists(stage_ref): os.symlink(os.path.abspath(c["real_ref"]), stage_ref)
    files = sorted(glob(os.path.join(c["real_q"], "*.jpg")))
    if ds == "sfu": files = files[:385]            # r2_gate_sfu.py: only the original 385 queries
    for img in files:
        dst = os.path.join(stage_q, os.path.basename(img))
        if not os.path.lexists(dst): os.symlink(os.path.abspath(img), dst)
    if ds == "sfu": restore_gt_sfu()
    if ds == "stl": write_gt_stl([])
    return stage_q

def restore_gt_sfu():
    c = CFG["sfu"]; gt_dir = f"{c['stage']}/{c['images_sub']}"
    gt = os.path.join(gt_dir, "GT.npz")
    if os.path.exists(gt): os.remove(gt)
    shutil.copy2(os.path.join(c["real_gt_dir"], "GT_original.npz"), gt)
    o = os.path.join(gt_dir, "GT_original.npz")
    if os.path.exists(o): os.remove(o)

_GT_STL = None
def write_gt_stl(extra_cols):
    global _GT_STL
    c = CFG["stl"]
    if _GT_STL is None:
        g = np.load(c["real_gt"]); _GT_STL = (g["GThard"].astype(bool), g["GTsoft"].astype(bool))
    hard, soft = _GT_STL
    if extra_cols:
        hard = np.hstack([hard, hard[:, extra_cols]]); soft = np.hstack([soft, soft[:, extra_cols]])
    np.savez(os.path.join(c["stage"], c["images_sub"], "GT.npz"), GThard=hard, GTsoft=soft)

# ── main ────────────────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True, choices=CFG); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--condition", required=True); ap.add_argument("--descriptor", required=True, choices=["NetVLAD", "PatchNetVLAD", "CosPlace", "EigenPlaces", "AlexNet", "SAD", "HDC-DELF"])
    ap.add_argument("--draws-dir", default=f"{P2}/draws"); ap.add_argument("--views-root", default=f"{P2}/views")
    ap.add_argument("--out-dir", default=None); ap.add_argument("--save-s", action="store_true"); ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stems-csv", default=None, help="use this stem list (draw-CSV schema) instead of draws/<ds>_seed<k>.csv; any number of rows")
    ap.add_argument("--audit", action="store_true", help="stage + inject + GT exactly as a run, then run hagva-repo/tools/gt_audit.py on the staged set instead of demo.py (CPU only)")
    a = ap.parse_args()
    ds, cond, D = a.ds, a.condition, a.descriptor
    assert cond in CONDITIONS[ds], f"{cond} not valid for {ds} (valid: {CONDITIONS[ds]})"
    c = CFG[ds]
    out_dir = a.out_dir or f"{P2}/runs/{ds}/seed{a.seed if cond != 'C0' else 'base'}/{cond}_{D}"
    rows = [] if cond == "C0" else list(csv.DictReader(open(a.stems_csv or f"{a.draws_dir}/{ds}_seed{a.seed}.csv", newline="")))
    items = []
    for k, r in enumerate(rows):
        src = source_path(ds, cond, r, a.views_root)
        if ds == "gp":  name, gt = f"Image{200+k:03d}.jpg", int(r["ref_idx"])
        elif ds == "sfu": name, gt = f"z_hagva_tmp_{k:03d}_{r['stem']}.jpg", r["source_image"]
        else:             name, gt = f"zzz_inject_{k:04d}.jpg", os.path.splitext(r["source_image"])[0]
        items.append((src, name, gt))
    missing = [s for s, _, _ in items if not os.path.exists(s)]
    if missing: sys.exit(f"MISSING {len(missing)} injection files, e.g. {missing[:3]} -> see NEEDED_FROM_GENWARP.md")
    if a.dry_run:
        print(f"DRY RUN {ds} seed{a.seed} {cond} {D}: {len(items)} files, e.g. {items[:2]}"); return
    assert loader_clean(), "loader dirty before run (hard rule 2)"
    stage_q = setup_staging(ds)
    if ds == "stl": stem_to_col = {os.path.splitext(f)[0]: i for i, f in enumerate(sorted(f for f in os.listdir(c["real_q"]) if f.endswith(".jpg")))}
    injected = []; auc = None; os.makedirs(out_dir, exist_ok=True)
    try:
        if ds == "sfu": restore_gt_sfu()
        for src, name, gt in items:
            dst = os.path.join(stage_q, name); shutil.copy2(src, dst); injected.append(dst)
        if items:
            if ds == "gp":   patch_loader_gp({200 + k: gt for k, (_, _, gt) in enumerate(items)}, c["marker"])
            elif ds == "sfu": patch_loader_sfu({name: gt for _, name, gt in items}, c["marker"], c["anchor"])
            else:             write_gt_stl([stem_to_col[gt] for _, _, gt in items])
        env = {**os.environ, "MPLBACKEND": "Agg"}
        if a.save_s: env["HAGVA_SAVE_S"] = "1"
        if a.audit:
            cls = {"gp": "GardensPointDataset", "sfu": "SFUDataset", "stl": "StLuciaDataset"}[ds]
            res = subprocess.run([PY, "/mnt/sda/hagva-repo/tools/gt_audit.py", VPR_DIR, cls, f"{c['stage']}/{c['images_sub']}"],
                                 capture_output=True, text=True, cwd=VPR_DIR, env=env)
            print("GT AUDIT of the staged, injected set:\n" + res.stdout.strip() + (("\n[stderr] " + res.stderr[-600:]) if res.returncode else ""))
            print(f"staged query files: {len([f for f in os.listdir(stage_q) if f.endswith('.jpg')])} (injected {len(injected)})")
            return
        res = subprocess.run([PY, "demo.py", "--descriptor", D, "--dataset", c["dataset"],
                              "--images_dir", f"{c['stage']}/{c['images_sub']}", "--output_dir", out_dir],
                             capture_output=True, text=True, cwd=VPR_DIR, env=env)
        open(os.path.join(out_dir, "stdout.log"), "w").write(res.stdout); open(os.path.join(out_dir, "stderr.log"), "w").write(res.stderr[-20000:])
        for line in res.stdout.splitlines():
            if "AUC (area under curve)" in line:
                try: auc = float(line.strip().split(":")[-1].strip())
                except ValueError: pass
                break
        if auc is None: print(f"  [rc={res.returncode}] [stderr] {res.stderr[-400:]}")
    finally:
        for dst in injected:
            try: os.remove(dst)
            except OSError: pass
        if ds == "gp": unpatch_loader(c["marker"])
        elif ds == "sfu": unpatch_loader(c["marker"]); restore_gt_sfu()
        else: write_gt_stl([])
    assert loader_clean(), "loader dirty after run (hard rule 2)"
    print(f"RESULT ds={ds} seed={a.seed if cond != 'C0' else '-'} condition={cond} descriptor={D} n_injected={len(items)} auc={'ERROR' if auc is None else f'{auc:.3f}'}")
    sys.exit(0 if auc is not None else 1)

if __name__ == "__main__":
    main()
