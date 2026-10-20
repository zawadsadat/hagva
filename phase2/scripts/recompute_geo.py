#!/home/newview/miniconda3/envs/vprtutorial/bin/python3
"""
Recompute Module A Signal 1 (geometric score) with correctly handled warp/mask inputs.  CPU only.

Reproduces cell 2 of /mnt/sda/hagva-repo/module_a/VPR.ipynb exactly:
    synth_np, warped_np in [0,1] float32 (H,W,3); mask_np in [0,1] (H,W)
    visible   = mask_np >= 0.5
    score_map = ones(H,W);  diff = |synth - warped|.mean(axis=2)
    p99       = percentile(diff[visible], 99);  score_map[visible] = clip(diff/p99, 0, 1)[visible]
    mean_geo_score = score_map.mean()  (written with 4 decimals, as Module A does)
    pct_hall       = (score_map > 0.5).mean()*100
but reads the warp-only render and the mask from the regenerated folders
    /home/newview/genwarp/<pool>_400_warp_only/<synth stem>.jpg
    /home/newview/genwarp/<pool>_400_warp_masks/<synth stem>_mask.png
and resizes them to the synthesized view's size (BILINEAR for the image, NEAREST for the mask —
the resize Module A intended) instead of Module A's top-left crop of GenWarp's 512x512 output.

Output: /mnt/sda/HAGVA_R2/phase1/scores/<ds>_400_geo_fixed.csv with
    image_stem, mean_geo_score_old, mean_geo_score_fixed, deadband_frac_old,
    [holefrac_old, holefrac_fixed, pct_hall_old, pct_hall_fixed, warp_size, mask_size]
deadband_frac_old = fraction of all pixels that lie in Module A's dead band (columns >= 512 or
rows >= 512 of the synthesized size) AND are hole in Module A's stored mask.

Guard: refuses to run for a dataset unless BOTH regenerated folders hold 400 files (poll read-only).
    --validate-old : instead recompute the OLD score from Module A's own stored
                     <stem>_warped.jpg / <stem>_mask.png (cropped outputs) and compare with the CSV,
                     to check that the formula reproduces Module A (JPEG re-read of the warped image
                     is the only deviation). Writes <ds>_400_geo_validate_old.csv.
Usage: recompute_geo.py [--ds gp sfu stl] [--validate-old] [--force]
"""
import argparse, csv, os, re, sys
import numpy as np
from PIL import Image

GW = "/mnt/sda/Zawad/genwarp"
REGEN = "/home/newview/genwarp"          # regenerated warp-only + masks land here (Zawad, 2026-10-10)
MODA = f"{GW}/hagva_module_a"
SCORES_IN = "/mnt/sda/hagva-repo/module_a/scores"
OUT_DIR = "/mnt/sda/HAGVA_R2/phase1/scores"
DS = {"gp": ("gardenspoint", "garden"), "sfu": ("sfu", "sfu"), "stl": ("stlucia", "st")}

def geo_score(synth_np, warped_np, mask_np):
    """verbatim Module A compute_geometric_score"""
    H, W = mask_np.shape
    score_map = np.ones((H, W), dtype=np.float32)
    visible = mask_np >= 0.5
    if visible.any():
        diff = np.abs(synth_np - warped_np).mean(axis=2)
        visible_diff = diff[visible]
        p99 = np.percentile(visible_diff, 99)
        diff_norm = np.clip(diff / p99, 0.0, 1.0) if p99 > 0 else np.zeros_like(diff)
        score_map[visible] = diff_norm[visible]
    return score_map

def to_np(pil):  # Module A: np.array(pil, dtype=float32)/255.
    return np.array(pil, dtype=np.float32) / 255.

def count(d, suffix):
    return sum(1 for f in os.listdir(d) if f.endswith(suffix)) if os.path.isdir(d) else 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", nargs="+", default=["gp", "sfu", "stl"], choices=DS)
    ap.add_argument("--validate-old", action="store_true")
    ap.add_argument("--force", action="store_true", help="skip the 400-file guard (not recommended)")
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    for ds in a.ds:
        name, pool = DS[ds]
        sig = list(csv.DictReader(open(f"{SCORES_IN}/{name}_400_signal1_summary.csv", newline="")))
        old = {r["image_stem"]: r for r in csv.DictReader(open(f"{SCORES_IN}/{name}_400_combined_summary.csv", newline=""))}
        synth_dir = f"{GW}/{pool}_400_out_low"
        moda_dir = f"{MODA}/{name}_400/signal1_geometric"
        warp_dir = f"{REGEN}/{pool}_400_warp_only"; mask_dir = f"{REGEN}/{pool}_400_warp_masks"
        if a.validate_old:
            warp_dir, mask_dir = moda_dir, moda_dir
            out = f"{OUT_DIR}/{name}_400_geo_validate_old.csv"
        else:
            nw, nm = count(warp_dir, ".jpg"), count(mask_dir, "_mask.png")
            if (nw < 400 or nm < 400) and not a.force:
                print(f"[{ds}] NOT READY: {warp_dir} has {nw}/400 .jpg, {mask_dir} has {nm}/400 _mask.png -> skipping (read-only poll)")
                continue
            out = f"{OUT_DIR}/{name}_400_geo_fixed.csv"
        rows = []; maxdev = 0.0
        for i, r in enumerate(sig):
            stem = os.path.splitext(r["source_file"])[0]            # '000_Image127' = image_stem
            sstem = os.path.splitext(r["synth_file"])[0]            # '001_000_Image127_ele1_r0.10'
            synth = Image.open(f"{synth_dir}/{r['synth_file']}").convert("RGB"); W, H = synth.size
            if a.validate_old:
                wp, mp = f"{moda_dir}/{stem}_warped.jpg", f"{moda_dir}/{stem}_mask.png"
            else:
                wp, mp = f"{warp_dir}/{sstem}.jpg", f"{mask_dir}/{sstem}_mask.png"
            if not (os.path.exists(wp) and os.path.exists(mp)):
                sys.exit(f"[{ds}] missing {wp if not os.path.exists(wp) else mp} -> add to NEEDED_FROM_GENWARP.md")
            warped = Image.open(wp).convert("RGB"); mask = Image.open(mp).convert("L")
            wsize, msize = warped.size, mask.size
            if warped.size != synth.size: warped = warped.resize(synth.size, Image.BILINEAR)
            if mask.size != synth.size:   mask = mask.resize(synth.size, Image.NEAREST)
            sm = geo_score(to_np(synth), to_np(warped), to_np(mask))
            fixed = float(sm.mean()); pct = float((sm > 0.5).mean() * 100)
            # dead band of the OLD (Module A) mask
            om = np.array(Image.open(f"{moda_dir}/{stem}_mask.png").convert("L"), dtype=np.float32) / 255.
            band = np.zeros_like(om, dtype=bool); band[:, 512:] = True; band[512:, :] = True
            dead = float(((om < 0.5) & band).mean())
            oldv = float(old[stem]["mean_geo_score"])
            rows.append(dict(image_stem=stem, mean_geo_score_old=f"{oldv:.4f}", mean_geo_score_fixed=f"{fixed:.4f}",
                             deadband_frac_old=round(dead, 4), holefrac_old=round(float((om < 0.5).mean()), 4),
                             holefrac_fixed=round(float((to_np(mask) < 0.5).mean()), 4),
                             pct_hall_old=r["pct_hallucinated"], pct_hall_fixed=f"{pct:.2f}",
                             warp_size=f"{wsize[0]}x{wsize[1]}", mask_size=f"{msize[0]}x{msize[1]}"))
            maxdev = max(maxdev, abs(fixed - oldv))
            if (i + 1) % 100 == 0: print(f"  [{ds}] {i+1}/400", flush=True)
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        fx = np.array([float(x["mean_geo_score_fixed"]) for x in rows]); ol = np.array([float(x["mean_geo_score_old"]) for x in rows])
        from scipy.stats import spearmanr
        tag = "validate-old (recomputed from Module A's stored warped/mask)" if a.validate_old else "fixed (regenerated warp/mask)"
        print(f"[{ds}] wrote {out}: n={len(rows)} | {tag}: mean {fx.mean():.4f} vs CSV old {ol.mean():.4f} | "
              f"max|dev| {maxdev:.4f} | Spearman(old,new) {spearmanr(ol, fx).correlation:.4f} | "
              f"deadband_frac_old mean {np.mean([x['deadband_frac_old'] for x in rows]):.4f} | holefrac old {np.mean([x['holefrac_old'] for x in rows]):.4f} new {np.mean([x['holefrac_fixed'] for x in rows]):.4f}")

if __name__ == "__main__":
    main()
