#!/home/newview/miniconda3/envs/vprtutorial/bin/python3
"""
Texture of the synthesized view (and of its source image) vs the HAGVA signals. CPU only.
Measures per image (grayscale, native resolution, float64):
    lapvar   = variance of cv2.Laplacian(gray, CV_64F)           (sharpness / high-frequency content)
    sobel    = mean of sqrt(Sobel_x^2 + Sobel_y^2), ksize=3       (mean gradient magnitude)
Signals per view: geo_fixed (phase1/scores/<ds>_400_geo_fixed.csv), mean_sem_score (module_a/scores),
    holefrac = regenerated-mask hole fraction (phase2/views/<ds>/manifest.csv).
Outputs phase2/texture_measures_<ds>.csv and phase2/texture_vs_signals.md (Spearman, with p-values).
"""
import csv, os, sys
import numpy as np, cv2
from scipy.stats import spearmanr

P1S = "/mnt/sda/HAGVA_R2/phase1/scores"; P2 = "/mnt/sda/HAGVA_R2/phase2"; GW = "/mnt/sda/Zawad/genwarp"
DS = [("gp", "gardenspoint", "garden"), ("sfu", "sfu", "sfu")]
if len(sys.argv) > 1 and sys.argv[1] == "all": DS.append(("stl", "stlucia", "st"))

def texture(path):
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if g is None: raise FileNotFoundError(path)
    g = g.astype(np.float64)
    lap = cv2.Laplacian(g, cv2.CV_64F).var()
    sx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3); sy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
    return float(lap), float(np.sqrt(sx * sx + sy * sy).mean())

L = ["# Texture of the synthesized view vs the HAGVA signals\n",
     "Per view: variance of the Laplacian (`lapvar`) and mean Sobel gradient magnitude (`sobel`) of the grayscale image at native "
     "resolution, for the synthesized view and for the source image it was synthesized from. Signals: `geo_fixed` (corrected Signal 1), "
     "`mean_sem_score` (Signal 2), `holefrac` (regenerated-mask hole fraction). Spearman ρ (p) over 400 views.\n"]
for ds, name, pool in DS:
    geo = {r["image_stem"]: float(r["mean_geo_score_fixed"]) for r in csv.DictReader(open(f"{P1S}/{name}_400_geo_fixed.csv"))}
    geo_old = {r["image_stem"]: float(r["mean_geo_score"]) for r in csv.DictReader(open(f"/mnt/sda/hagva-repo/module_a/scores/{name}_400_combined_summary.csv"))}
    sem = {r["image_stem"]: float(r["mean_sem_score"]) for r in csv.DictReader(open(f"/mnt/sda/hagva-repo/module_a/scores/{name}_400_combined_summary.csv"))}
    man = {r["stem"]: r for r in csv.DictReader(open(f"{P2}/views/{ds}/manifest.csv"))}
    rows = []
    for stem in sorted(geo):
        sf = man[stem]["synth_file"]
        lv, sb = texture(f"{GW}/{pool}_400_out_low/{sf}")
        lvs, sbs = texture(f"{GW}/{pool}_400/{stem}.jpg")
        rows.append(dict(image_stem=stem, synth_file=sf, lapvar_synth=round(lv, 3), sobel_synth=round(sb, 4), lapvar_src=round(lvs, 3), sobel_src=round(sbs, 4),
                         geo_fixed=geo[stem], geo_old=geo_old[stem], mean_sem_score=sem[stem], holefrac=float(man[stem]["holefrac"])))
    out = f"{P2}/texture_measures_{ds}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    A = {k: np.array([r[k] for r in rows]) for k in rows[0] if k not in ("image_stem", "synth_file")}
    L.append(f"## {name} (n = {len(rows)})\n")
    L.append("| measure | vs geo_fixed | vs mean_sem_score | vs holefrac | vs geo_old (context) |"); L.append("|---|---|---|---|---|")
    for m, lab in [("lapvar_synth", "lapvar, synthesized view"), ("sobel_synth", "sobel, synthesized view"), ("lapvar_src", "lapvar, source image"), ("sobel_src", "sobel, source image")]:
        cells = []
        for sig in ("geo_fixed", "mean_sem_score", "holefrac", "geo_old"):
            r = spearmanr(A[m], A[sig]); cells.append(f"{r.correlation:+.3f} (p={r.pvalue:.1e})")
        L.append(f"| {lab} | " + " | ".join(cells) + " |")
    rs = spearmanr(A["lapvar_synth"], A["lapvar_src"]).correlation; rs2 = spearmanr(A["sobel_synth"], A["sobel_src"]).correlation
    L.append(f"\nSynth-vs-source texture agreement: lapvar ρ = {rs:+.3f}, sobel ρ = {rs2:+.3f}. "
             f"Means: lapvar synth {A['lapvar_synth'].mean():.1f} vs source {A['lapvar_src'].mean():.1f}; sobel synth {A['sobel_synth'].mean():.2f} vs source {A['sobel_src'].mean():.2f}. "
             f"Signal inter-correlations: geo_fixed–sem ρ = {spearmanr(A['geo_fixed'], A['mean_sem_score']).correlation:+.3f}, geo_fixed–holefrac ρ = {spearmanr(A['geo_fixed'], A['holefrac']).correlation:+.3f}, sem–holefrac ρ = {spearmanr(A['mean_sem_score'], A['holefrac']).correlation:+.3f}.\n")
    print(f"[{ds}] wrote {out}; " + "; ".join(f"{m}: geo {spearmanr(A[m], A['geo_fixed']).correlation:+.3f} sem {spearmanr(A[m], A['mean_sem_score']).correlation:+.3f} hole {spearmanr(A[m], A['holefrac']).correlation:+.3f}" for m in ("lapvar_synth", "sobel_synth", "lapvar_src", "sobel_src")))
L.append("Notes: lapvar and sobel are computed on the whole frame including disoccluded regions of the synthesized view; the source image has no holes. "
         "Higher = more high-frequency texture. All correlations are rank-based (Spearman); no causal reading is attached here.\n")
open(f"{P2}/texture_vs_signals.md", "w").write("\n".join(L) + "\n")
print("wrote", f"{P2}/texture_vs_signals.md")
