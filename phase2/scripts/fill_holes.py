#!/home/newview/miniconda3/envs/vprtutorial/bin/python3
"""
Phase 2 (a)/(b): build the hole-treated injection variants from (synth, warp-only, mask) triples.
CPU only (numpy + OpenCV). Never writes into GenWarp folders.

  fill_holes.py --ds gp --synth-dir DIR --warp-dir DIR --mask-dir DIR
                [--stems-csv phase2/draws/gp_seed0.csv ...]   # restrict to drawn stems (default: all synth files)
                [--warp-suffix .jpg] [--mask-suffix _mask.png] [--name-by synth|source]
                [--out-root /mnt/sda/HAGVA_R2/phase2/views] [--radius 3] [--thresh 128]
                [--genwarp-fill paste|own] [--dry-run]

Inputs per view (same stem for all three):
   synth : <synth-dir>/<synth_file>                      GenWarp final image (e.g. 001_000_Image127_ele1_r0.10.jpg)
   warp  : <warp-dir>/<name><warp-suffix>                warp-only render (renders['warped']), holes black
   mask  : <mask-dir>/<name><mask-suffix>                PNG L, 255 = visible warped pixel, 0 = disoccluded
   <name> = synth stem ('001_000_Image127_ele1_r0.10', --name-by synth, Zawad's garden_400_warp_* folders)
            or pool/source stem ('000_Image127', --name-by source, module A signal1_geometric folders)
M = mask >= thresh.  Asserts mask.shape == synth.shape == warp.shape (no silent resize).

Outputs (JPEG q95, named <stem>.jpg with stem = pool stem '000_Image127'):
   <out-root>/<ds>/warp_black/     warp with holes = 0                      (C4, original Exp 3)
   <out-root>/<ds>/warp_telea/     cv2.inpaint(warp, ~M, radius, TELEA)     (C5)
   <out-root>/<ds>/genwarp_black/  synth with holes = 0                     (C6, original Exp 4)
   <out-root>/<ds>/genwarp_telea/  synth in M; holes = Telea fill           (C7)
        --genwarp-fill paste (default): holes take EXACTLY the warp_telea pixels, so C5 and C7
              differ only in the visible region (the brief's "replaced by the same Telea fill").
        --genwarp-fill own : holes = cv2.inpaint(synth*M, ~M) (fill derived from GenWarp's own
              visible pixels; seam-free but the fill then also differs between C5 and C7).
   <out-root>/<ds>/manifest.csv : stem, synth_file, holefrac, sizes, variant paths
"""
import argparse, csv, os, re, sys
import numpy as np
import cv2

VARIANTS = ["warp_black", "warp_telea", "genwarp_black", "genwarp_telea"]

def pool_stem(synth_file):
    """'001_000_Image127_ele1_r0.10.jpg' -> '000_Image127' (same rule as the gate scripts)."""
    stem = os.path.splitext(synth_file)[0]
    stem = re.sub(r"_ele[\d.]+_r[\d.]+$", "", stem)
    return re.sub(r"^\d+_", "", stem, count=1)

def imread_rgb(p):
    im = cv2.imread(p, cv2.IMREAD_COLOR)
    if im is None: raise FileNotFoundError(p)
    return im  # BGR; written back with imwrite so channel order is irrelevant

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="gp")
    ap.add_argument("--synth-dir", required=True); ap.add_argument("--warp-dir", required=True); ap.add_argument("--mask-dir", required=True)
    ap.add_argument("--stems-csv", nargs="*", default=[])
    ap.add_argument("--warp-suffix", default=".jpg"); ap.add_argument("--mask-suffix", default="_mask.png")
    ap.add_argument("--name-by", choices=["synth", "source"], default="synth")
    ap.add_argument("--out-root", default="/mnt/sda/HAGVA_R2/phase2/views")
    ap.add_argument("--radius", type=float, default=3.0); ap.add_argument("--thresh", type=int, default=128)
    ap.add_argument("--genwarp-fill", choices=["paste", "own"], default="paste")
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    for d in (a.synth_dir, a.warp_dir, a.mask_dir):
        assert os.path.isdir(d), f"missing input dir {d}"
    assert not any(a.out_root.startswith(p) for p in ("/mnt/sda/Zawad/genwarp", "/home/newview/genwarp", "/mnt/sda/hagva-repo")), "refusing to write into a GenWarp/repo folder"

    synth_files = sorted(f for f in os.listdir(a.synth_dir) if f.lower().endswith((".jpg", ".png")))
    if a.stems_csv:
        want = set()
        for p in a.stems_csv:
            for r in csv.DictReader(open(p, newline="")): want.add(r["stem"])
        synth_files = [f for f in synth_files if pool_stem(f) in want]
        missing = want - {pool_stem(f) for f in synth_files}
        assert not missing, f"{len(missing)} drawn stems have no synth file, e.g. {sorted(missing)[:3]}"
    print(f"{len(synth_files)} views to process -> {a.out_root}/{a.ds}/{{{','.join(VARIANTS)}}}")

    out = {v: os.path.join(a.out_root, a.ds, v) for v in VARIANTS}
    if not a.dry_run:
        for d in out.values(): os.makedirs(d, exist_ok=True)
    man = []
    for i, sf in enumerate(synth_files):
        stem = pool_stem(sf)
        name = os.path.splitext(sf)[0] if a.name_by == "synth" else stem
        ps = os.path.join(a.synth_dir, sf); pw = os.path.join(a.warp_dir, name + a.warp_suffix); pm = os.path.join(a.mask_dir, name + a.mask_suffix)
        for p in (pw, pm):
            if not os.path.exists(p):
                sys.exit(f"MISSING input for {sf}: {p}  -> add to NEEDED_FROM_GENWARP.md; not improvising")
        if a.dry_run:
            man.append(dict(stem=stem, synth_file=sf, warp=pw, mask=pm)); continue
        synth = imread_rgb(ps); warp = imread_rgb(pw)
        mask = cv2.imread(pm, cv2.IMREAD_GRAYSCALE)
        assert mask is not None, pm
        assert mask.shape == synth.shape[:2] == warp.shape[:2], f"size mismatch {sf}: synth {synth.shape[:2]} warp {warp.shape[:2]} mask {mask.shape}"
        M = mask >= a.thresh                                   # True = visible
        hole_u8 = (~M).astype(np.uint8) * 255                  # cv2.inpaint wants non-zero = to be filled
        M3 = M[:, :, None]
        warp_black = np.where(M3, warp, 0).astype(np.uint8)
        warp_telea = cv2.inpaint(warp_black, hole_u8, a.radius, cv2.INPAINT_TELEA) if (~M).any() else warp_black.copy()
        genwarp_black = np.where(M3, synth, 0).astype(np.uint8)
        if a.genwarp_fill == "paste":
            genwarp_telea = np.where(M3, synth, warp_telea).astype(np.uint8)
        else:
            genwarp_telea = cv2.inpaint(genwarp_black, hole_u8, a.radius, cv2.INPAINT_TELEA) if (~M).any() else genwarp_black.copy()
        outs = dict(warp_black=warp_black, warp_telea=warp_telea, genwarp_black=genwarp_black, genwarp_telea=genwarp_telea)
        paths = {}
        for v, im in outs.items():
            paths[v] = os.path.join(out[v], stem + ".jpg")
            ok = cv2.imwrite(paths[v], im, [cv2.IMWRITE_JPEG_QUALITY, a.quality]); assert ok, paths[v]
        # sanity: outside the holes, C5 == C4 pixels and C7 == C6 pixels (before JPEG)
        assert (warp_telea[M] == warp_black[M]).all() and (genwarp_telea[M] == genwarp_black[M]).all()
        man.append(dict(stem=stem, synth_file=sf, holefrac=round(float((~M).mean()), 5), h=mask.shape[0], w=mask.shape[1],
                        warp=pw, mask=pm, **paths))
        if (i + 1) % 50 == 0: print(f"  {i+1}/{len(synth_files)}")
    mpath = os.path.join(a.out_root, a.ds, "manifest.csv" if not a.dry_run else "manifest_dryrun.csv")
    os.makedirs(os.path.dirname(mpath), exist_ok=True)
    with open(mpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(man[0].keys())); w.writeheader(); w.writerows(man)
    print(f"{'DRY RUN: ' if a.dry_run else ''}manifest -> {mpath} ({len(man)} rows)"
          + ("" if a.dry_run else f"; mean hole fraction {np.mean([m['holefrac'] for m in man]):.4f}; genwarp_telea fill = {a.genwarp_fill}"))

if __name__ == "__main__":
    main()
