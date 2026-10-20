#!/home/newview/miniconda3/envs/vprtutorial/bin/python3
"""
Mask audit: Module A hole masks vs Exp-4-style (pose-matched) hole masks. CPU only.

  audit_masks.py --set gp50   (36 shared views: modA/gardenspoint vs home/garden_50_warp_masks_v2)
  audit_masks.py --set gp400 | sfu400 | stl400  (400 views: modA/<ds>_400 vs /home/newview/genwarp/<pool>_400_warp_masks)
  [--out phase2/mask_audit.csv]

Mask convention in both sources: PNG mode L, 255 = visible warped pixel, 0 = hole.
hole = value < 128. Columns: stem, holefrac_moduleA, holefrac_exp4, pixel_agreement, iou_hole,
plus diagnostics: frac_hole_A_only, frac_hole_E_only, recall_E_by_A (share of exp4 holes that
modA also marks), bottom10_holefrac_A/E (hole fraction in the bottom 10% of rows),
best_shift_dy/dx (vertical/horizontal shift of modA that maximises agreement, |shift|<=16),
agreement_at_best_shift, pct_hall_csv_A (module A CSV column), halluc_pct_log_E (exp4 log).
"""
import argparse, csv, os, re, sys
import numpy as np
from PIL import Image

MODA = "/mnt/sda/Zawad/genwarp/hagva_module_a"
SETS = {
    "gp50": dict(
        modA_dir=f"{MODA}/gardenspoint/signal1_geometric",
        modA_csv=f"{MODA}/gardenspoint/signal1_geometric/signal1_summary.csv",
        exp4_dir="/home/newview/genwarp/garden_50_warp_masks_v2",
        exp4_log="/home/newview/genwarp/garden_50_warp_masks_v2/mask_regen_log.csv",
        # modA mask name uses the SOURCE stem (Image001_mask.png); exp4 uses the synth stem
        modA_name=lambda row: os.path.splitext(row["source_file"])[0] + "_mask.png",
        exp4_name=lambda row: os.path.splitext(row["synth_file"])[0] + "_mask.png",
    ),
}
def _regen_set(moda_name, pool):
    return dict(
        modA_dir=f"{MODA}/{moda_name}_400/signal1_geometric",
        modA_csv=f"{MODA}/{moda_name}_400/signal1_geometric/signal1_summary.csv",
        exp4_dir=f"/home/newview/genwarp/{pool}_400_warp_masks",      # regenerated masks (Zawad, home checkout)
        exp4_log=f"/home/newview/genwarp/{pool}_400_warp_masks/regen_log.csv",
        modA_name=lambda row: os.path.splitext(row["source_file"])[0] + "_mask.png",   # pool stem, e.g. 000_b008
        exp4_name=lambda row: os.path.splitext(row["synth_file"])[0] + "_mask.png",    # synth stem
    )
SETS["gp400"]  = _regen_set("gardenspoint", "garden")
SETS["sfu400"] = _regen_set("sfu", "sfu")
SETS["stl400"] = _regen_set("stlucia", "st")

def load_hole(path):
    a = np.asarray(Image.open(path).convert("L"))
    return a < 128

def best_shift(A, E, maxs=16, step=2):
    """shift modA hole mask A by (dy,dx) and find max agreement with E."""
    best = (0, 0, (A == E).mean())
    H, W = A.shape
    for dy in range(-maxs, maxs + 1, step):
        for dx in range(-maxs, maxs + 1, step):
            As = np.roll(np.roll(A, dy, axis=0), dx, axis=1)
            ys = slice(max(dy, 0), H + min(dy, 0)); xs = slice(max(dx, 0), W + min(dx, 0))
            ag = (As[ys, xs] == E[ys, xs]).mean()
            if ag > best[2]: best = (dy, dx, ag)
    return best

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=SETS, default="gp50")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-shift", action="store_true")
    a = ap.parse_args()
    S = SETS[a.set]
    out = a.out or f"/mnt/sda/HAGVA_R2/phase2/mask_audit{'' if a.set=='gp50' else '_400' if a.set=='gp400' else '_' + a.set}.csv"
    rows = list(csv.DictReader(open(S["modA_csv"], newline="")))
    log = {}
    if S["exp4_log"] and os.path.exists(S["exp4_log"]):
        for r in csv.DictReader(open(S["exp4_log"], newline="")):
            log[r.get("synth_file") or r.get("out_file")] = r
    # fall back to any csv in exp4_dir for the 400 set
    elif a.set == "gp400":
        for f in os.listdir(S["exp4_dir"]) if os.path.isdir(S["exp4_dir"]) else []:
            if f.endswith(".csv"):
                for r in csv.DictReader(open(os.path.join(S["exp4_dir"], f), newline="")):
                    k = r.get("synth_file") or r.get("out_file") or r.get("filename")
                    if k: log[k] = r
    res = []; missing = []
    for r in rows:
        pa = os.path.join(S["modA_dir"], S["modA_name"](r)); pe = os.path.join(S["exp4_dir"], S["exp4_name"](r))
        if not (os.path.exists(pa) and os.path.exists(pe)):
            missing.append((r["synth_file"], os.path.exists(pa), os.path.exists(pe))); continue
        A = load_hole(pa); E = load_hole(pe)
        if A.shape != E.shape:
            missing.append((r["synth_file"], f"shape {A.shape} vs {E.shape}", "")); continue
        H = A.shape[0]; b = slice(int(0.9 * H), H)
        inter = (A & E).sum(); union = (A | E).sum()
        d = dict(stem=os.path.splitext(r["synth_file"])[0],
                 source_stem=os.path.splitext(r["source_file"])[0],
                 holefrac_moduleA=round(A.mean(), 5), holefrac_exp4=round(E.mean(), 5),
                 pixel_agreement=round((A == E).mean(), 5),
                 iou_hole=round(inter / union, 5) if union else 1.0,
                 frac_hole_A_only=round((A & ~E).mean(), 5), frac_hole_E_only=round((E & ~A).mean(), 5),
                 recall_E_by_A=round(inter / E.sum(), 4) if E.sum() else "",
                 bottom10_holefrac_A=round(A[b].mean(), 4), bottom10_holefrac_E=round(E[b].mean(), 4),
                 pct_hall_csv_A=r.get("pct_hallucinated", ""),
                 halluc_pct_log_E=(log.get(r["synth_file"], {}) or {}).get("halluc_pct", ""))
        # root-cause test: Module A cropped GenWarp's 512x512 output to (W,H) top-left instead of resizing.
        # Re-create that transform from the exp4 mask: resize to 512x512 (NEAREST), paste top-left, pad holes.
        E512 = np.asarray(Image.fromarray((~E).astype(np.uint8) * 255).resize((512, 512), Image.NEAREST)) < 128
        T = np.ones_like(A); hh, ww = min(H, 512), min(A.shape[1], 512); T[:hh, :ww] = E512[:hh, :ww]
        d["agreement_after_512crop_transform"] = round((A == T).mean(), 5)
        d["holefrac_A_cols_ge512"] = round(A[:, 512:].mean(), 4) if A.shape[1] > 512 else ""
        if not a.no_shift:
            dy, dx, ag = best_shift(A, E)
            d.update(best_shift_dy=dy, best_shift_dx=dx, agreement_at_best_shift=round(ag, 5))
        res.append(d)
    if not res:
        print("no comparable pairs; missing:", missing[:5]); sys.exit(1)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(res[0].keys())); w.writeheader(); w.writerows(res)
    M = lambda k: np.mean([float(x[k]) for x in res if x[k] != ""])
    print(f"wrote {out}: {len(res)} pairs, {len(missing)} missing/unmatched")
    if missing: print("  missing e.g.:", missing[:3])
    print(f"  holefrac modA {M('holefrac_moduleA'):.4f} | exp4 {M('holefrac_exp4'):.4f} | agreement {M('pixel_agreement'):.4f} | IoU(hole) {M('iou_hole'):.4f}")
    print(f"  hole A-only {M('frac_hole_A_only'):.4f} | hole E-only {M('frac_hole_E_only'):.4f} | recall of exp4 holes by modA {M('recall_E_by_A'):.3f}")
    print(f"  bottom-10%-rows hole frac: modA {M('bottom10_holefrac_A'):.3f} vs exp4 {M('bottom10_holefrac_E'):.3f}")
    if not a.no_shift:
        import collections
        print(f"  best shift (dy,dx) histogram: {collections.Counter((x['best_shift_dy'], x['best_shift_dx']) for x in res).most_common(5)}; agreement at best shift {M('agreement_at_best_shift'):.4f}")
    print(f"  agreement after 512-crop transform of exp4 mask: {M('agreement_after_512crop_transform'):.4f}; modA hole frac in cols>=512: {M('holefrac_A_cols_ge512') if res[0]['holefrac_A_cols_ge512']!='' else float('nan'):.3f}")
    if any(x["pct_hall_csv_A"] != "" for x in res):
        print(f"  modA CSV pct_hallucinated mean {M('pct_hall_csv_A'):.1f}% vs modA mask hole {100*M('holefrac_moduleA'):.1f}%")

if __name__ == "__main__":
    main()
