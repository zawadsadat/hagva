#!/home/newview/miniconda3/envs/vprtutorial/bin/python3
"""Low vs wide pool statistics: hole fraction, semantic score, corrected geo. Writes phase3b/pool_stats.md."""
import csv, collections, numpy as np
from scipy.stats import spearmanr
R = "/mnt/sda/HAGVA_R2"; DS = {"gardenspoint": "gp", "sfu": "sfu"}
def q(v): return f"{v.mean():.3f} ± {v.std(ddof=1):.3f} | min {v.min():.3f} | p25 {np.percentile(v,25):.3f} | median {np.median(v):.3f} | p75 {np.percentile(v,75):.3f} | max {v.max():.3f}"
L = ["# Phase 3b — pool statistics, low (ele 1–5°, r 0.10–0.20) vs wide (ele 8–16°, r 0.30–0.50)\n",
     "Low pool: hole fraction from `phase2/views/<ds>/manifest.csv` (regenerated masks), sem from `module_a/scores/<ds>_400_combined_summary.csv`, geo from `phase1/scores/<ds>_400_geo_fixed.csv`. "
     "Wide pool: `phase3b/scores/<ds>_wide_scores.csv` (score_pool.py, validated on the low pool). 400 views each.\n"]
for ds, k in DS.items():
    man = {r["stem"]: float(r["holefrac"]) for r in csv.DictReader(open(f"{R}/phase2/views/{k}/manifest.csv"))}
    sem_lo = {r["image_stem"]: float(r["mean_sem_score"]) for r in csv.DictReader(open(f"/mnt/sda/hagva-repo/module_a/scores/{ds}_400_combined_summary.csv"))}
    geo_lo = {r["image_stem"]: float(r["mean_geo_score_fixed"]) for r in csv.DictReader(open(f"{R}/phase1/scores/{ds}_400_geo_fixed.csv"))}
    wide = list(csv.DictReader(open(f"{R}/phase3b/scores/{ds}_wide_scores.csv")))
    st = sorted(sem_lo); assert set(st) == set(r["image_stem"] for r in wide)
    lo = {"holefrac": np.array([man[s] for s in st]), "sem": np.array([sem_lo[s] for s in st]), "geo_fixed": np.array([geo_lo[s] for s in st])}
    W = {r["image_stem"]: r for r in wide}
    wi = {"holefrac": np.array([float(W[s]["holefrac"]) for s in st]), "sem": np.array([float(W[s]["mean_sem_score"]) for s in st]), "geo_fixed": np.array([float(W[s]["geo_fixed"]) for s in st])}
    L.append(f"## {ds}\n"); L.append("| quantity | pool | mean ± sd | min | p25 | median | p75 | max |"); L.append("|---|---|---|---|---|---|---|---|")
    for key, lab in [("holefrac", "hole fraction"), ("sem", "mean_sem_score"), ("geo_fixed", "geo_fixed")]:
        for pool, d in [("low", lo), ("wide", wi)]:
            v = d[key]; L.append(f"| {lab} | {pool} | {v.mean():.3f} ± {v.std(ddof=1):.3f} | {v.min():.3f} | {np.percentile(v,25):.3f} | {np.median(v):.3f} | {np.percentile(v,75):.3f} | {v.max():.3f} |")
    L.append("")
    L.append(f"Views with hole fraction > 0.25: low {int((lo['holefrac']>0.25).sum())}/400, wide {int((wi['holefrac']>0.25).sum())}/400; > 0.50: low {int((lo['holefrac']>0.5).sum())}, wide {int((wi['holefrac']>0.5).sum())}. "
             f"Same-view paired Spearman low→wide: sem {spearmanr(lo['sem'], wi['sem']).correlation:+.3f}, geo_fixed {spearmanr(lo['geo_fixed'], wi['geo_fixed']).correlation:+.3f}, holefrac {spearmanr(lo['holefrac'], wi['holefrac']).correlation:+.3f}. "
             f"Within-pool Spearman(sem, holefrac): low {spearmanr(lo['sem'], lo['holefrac']).correlation:+.3f}, wide {spearmanr(wi['sem'], wi['holefrac']).correlation:+.3f}; Spearman(geo_fixed, sem): low {spearmanr(lo['geo_fixed'], lo['sem']).correlation:+.3f}, wide {spearmanr(wi['geo_fixed'], wi['sem']).correlation:+.3f}; "
             f"Spearman(geo_fixed, holefrac): low {spearmanr(lo['geo_fixed'], lo['holefrac']).correlation:+.3f}, wide {spearmanr(wi['geo_fixed'], wi['holefrac']).correlation:+.3f}.\n")
    cells = collections.defaultdict(list)
    for r in wide: cells[(float(r["ele"]), float(r["r"]))].append((float(r["holefrac"]), float(r["mean_sem_score"]), float(r["geo_fixed"])))
    L.append("| wide (ele, r) | n | hole fraction | mean_sem_score | geo_fixed |"); L.append("|---|---|---|---|---|")
    for k2 in sorted(cells):
        a = np.array(cells[k2]); L.append(f"| ({k2[0]:g}°, {k2[1]:.2f}) | {len(a)} | {a[:,0].mean():.3f} | {a[:,1].mean():.3f} | {a[:,2].mean():.3f} |")
    L.append("")
open(f"{R}/phase3b/pool_stats.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))
