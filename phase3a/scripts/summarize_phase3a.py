#!/home/newview/miniconda3/envs/vprtutorial/bin/python3
"""Phase 3a summary: every table side by side PatchNetVLAD (Phase 1/2 results, read-only) vs EigenPlaces (phase3a results).
Writes phase3a/summary_phase3a.md and phase3a/gt_inherit_eigenplaces.csv.  CPU only."""
import csv, os, glob
from collections import defaultdict
import numpy as np
from scipy import stats

R2 = "/mnt/sda/HAGVA_R2"; P1, P2, P3 = f"{R2}/phase1", f"{R2}/phase2", f"{R2}/phase3a"
DS = ["GardensPoint", "SFU", "StLucia"]; DSK = {"GardensPoint": "gp", "SFU": "sfu", "StLucia": "stl"}; NS = [20, 64, 145]
DESCS = ["PatchNetVLAD", "EigenPlaces"]
def f3(x): return "—" if x is None else f"{x:.3f}"

# ── gate results ────────────────────────────────────────────────────────────
G = {}  # (desc, ds, method, n) -> dict(auc, baseline, naive, n_passed)
for r in csv.DictReader(open(f"{P1}/phase1_results.csv")):
    G[("PatchNetVLAD", r["dataset"], r["method"], int(r["n_requested"]))] = dict(auc=float(r["auc"]), base=float(r["baseline"]), naive=float(r["naive"]), np_=int(r["n_passed"]))
for r in csv.DictReader(open(f"{P3}/phase3a_gate_results.csv")):
    G[(r["descriptor"], r["dataset"], r["method"], int(r["n_requested"]))] = dict(auc=float(r["auc"]), base=float(r["baseline"]), naive=float(r["naive"]), np_=int(r["n_passed"]))
def gauc(D, ds, m, n): x = G.get((D, ds, m, n)); return None if x is None else x["auc"]
def gbase(D, ds): x = [v for k, v in G.items() if k[0] == D and k[1] == ds]; return (x[0]["base"], x[0]["naive"]) if x else (None, None)
def rec(D, ds, auc):
    b, nv = gbase(D, ds); return None if auc is None or b is None or b == nv else (auc - nv) / (b - nv)
def rand_stats(D, ds, n):
    v = [gauc(D, ds, f"rand_s{i}", n) for i in range(5)]; v = [x for x in v if x is not None]
    return (np.mean(v), np.std(v, ddof=1), len(v)) if len(v) > 1 else (None, None, len(v))

# ── causal results ──────────────────────────────────────────────────────────
A = defaultdict(dict); base = {}
for path, D in [(f"{P2}/phase2_results.csv", "PatchNetVLAD"), (f"{P3}/phase3a_causal_results.csv", "EigenPlaces")]:
    for r in csv.DictReader(open(path)):
        if r["descriptor"] != D: continue
        if r["condition"] == "C0": base[(D, r["dataset"])] = float(r["auc"])
        else: A[(D, r["dataset"], r["condition"])][int(r["seed"])] = float(r["auc"])
def paired(x, y):
    seeds = sorted(set(x) & set(y)); d = np.array([x[s] - y[s] for s in seeds])
    if len(seeds) < 2: return d, np.nan, np.nan, int((d > 0).sum())
    t = stats.ttest_rel([x[s] for s in seeds], [y[s] for s in seeds]); return d, t.statistic, t.pvalue, int((d > 0).sum())
def cm(D, ds, c): v = list(A.get((D, ds, c), {}).values()); return (np.mean(v), np.std(v, ddof=1)) if len(v) > 1 else (None, None)

# ── GT-inherit for EigenPlaces from S.npy (same logic as summarize_phase2) ──
def gt_inherit(runs_root, D, out_csv):
    rows = []
    for ds in DS:
        k = DSK[ds]
        real_q = {"gp": "/mnt/sda/Zawad/genwarp/VPR-GIT/images/GardensPoint/night_right", "sfu": "/mnt/sda/Zawad/genwarp/VPR-GIT/images/SFU/jan", "stl": "/mnt/sda/Zawad/genwarp/VPR-GIT/images/StLucia_small/180809_1545"}[k]
        qfiles = sorted(f for f in os.listdir(real_q) if f.endswith(".jpg"))
        if k == "sfu": qfiles = qfiles[:385]
        nq = len(qfiles); qidx = {os.path.splitext(f)[0]: i for i, f in enumerate(qfiles)}
        gthard = None if k == "gp" else np.load({"sfu": "/mnt/sda/Zawad/genwarp/VPR-Query-NEW/images/SFU/GT_original.npz", "stl": "/mnt/sda/Zawad/genwarp/VPR-GIT/images/StLucia_small/GT.npz"}[k])["GThard"].astype(bool)
        for seed in range(5):
            draw = list(csv.DictReader(open(f"{P2}/draws/{k}_seed{seed}.csv")))
            for c in ["C1", "C2", "C2b", "C4", "C5", "C6", "C7"]:
                p = f"{runs_root}/{k}/seed{seed}/{c}_{D}/S.npy"
                if not os.path.exists(p): continue
                S = np.load(p)
                if S.shape[1] != nq + len(draw): continue
                top = S.argmax(axis=0)
                for j, r in enumerate(draw):
                    src = os.path.splitext(r["source_image"])[0]; qi = qidx.get(src)
                    if qi is None: continue
                    syn, st = int(top[nq + j]), int(top[qi]); hit = (lambda t: t == qi) if k == "gp" else (lambda t: bool(gthard[t, qi]))
                    rows.append(dict(dataset=ds, seed=seed, condition=c, stem=r["stem"], source_stem=src, src_top1=st, syn_top1=syn, same_top1=int(st == syn), src_hit=int(hit(st)), syn_hit=int(hit(syn))))
    if rows:
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    return rows
gt = {"EigenPlaces": gt_inherit(f"{P3}/runs", "EigenPlaces", f"{P3}/gt_inherit_eigenplaces.csv"),
      "PatchNetVLAD": [dict(r, same_top1=int(r["same_top1"]), src_hit=int(r["src_hit"]), syn_hit=int(r["syn_hit"])) for r in csv.DictReader(open(f"{P2}/gt_inherit.csv"))]}
def gstat(D, ds, c):
    g = [r for r in gt[D] if r["dataset"] == ds and r["condition"] == c]
    if not g: return None
    sh = np.array([r["src_hit"] for r in g]); yh = np.array([r["syn_hit"] for r in g]); same = np.array([r["same_top1"] for r in g])
    return dict(n=len(g), same=same.mean(), src=sh.mean(), syn=yh.mean(), hh=(yh[sh == 1].mean() if (sh == 1).any() else np.nan), hm=(yh[sh == 0].mean() if (sh == 0).any() else np.nan))

# ── write ───────────────────────────────────────────────────────────────────
L = ["# HAGVA Phase 3a — second descriptor (EigenPlaces, ResNet50 2048-d) vs PatchNetVLAD\n",
     "PatchNetVLAD numbers are the Phase 1 / Phase 2 results (read-only); EigenPlaces numbers are from `phase3a_gate_results.csv` (51 rows) and "
     "`phase3a_causal_results.csv` (98 rows), same extended score CSVs, same draws, same filled views. Recovery = (AUC − naive)/(baseline − naive) with each descriptor's own baseline and naive-400.\n",
     "| descriptor | GardensPoint baseline / naive-400 | SFU | StLucia |", "|---|---|---|---|"]
for D in DESCS: L.append(f"| {D} | " + " | ".join(f"{gbase(D, ds)[0]:.3f} / {gbase(D, ds)[1]:.3f}" for ds in DS) + " |")
L.append("\nThe EigenPlaces naive-400 values (0.594 / 0.465 / 0.579) are new numbers; naive damage under EigenPlaces is 0.081 / 0.281 / 0.157 vs 0.047 / 0.078 / 0.146 under PatchNetVLAD.\n")
agree = []  # (claim, ds, dirPNV, dirEIG)
def sgn(x, eps=0.0005): return "≈0" if x is None or abs(x) < eps else ("+" if x > 0 else "−")

# (1) matched-n gate table
L.append("## 1. Matched-n gate table (AUC, recovery in brackets), PatchNetVLAD | EigenPlaces\n")
L.append("| method | n | " + " | ".join(DS) + " |"); L.append("|---|---|" + "---|" * len(DS))
for m, lab in [("w100", "sem-only (w100)"), ("w000f", "fixed geo only (w000f)"), ("w050f", "equal weight, fixed geo (w050f)")]:
    for n in NS:
        cells = []
        for ds in DS:
            p, e = gauc("PatchNetVLAD", ds, m, n), gauc("EigenPlaces", ds, m, n)
            cells.append(f"{f3(p)} [{100*rec('PatchNetVLAD', ds, p):+.0f}%] \\| {f3(e)} [{100*rec('EigenPlaces', ds, e):+.0f}%]" if p is not None and e is not None else "—")
        L.append(f"| {lab} | {n} | " + " | ".join(cells) + " |")
for n in NS:
    cells = []; cells2 = []
    for ds in DS:
        mp, sp, kp = rand_stats("PatchNetVLAD", ds, n); me, se, ke = rand_stats("EigenPlaces", ds, n)
        if mp is None or me is None: cells.append("—"); cells2.append("—"); continue
        cells.append(f"{mp:.3f}±{sp:.3f} [{100*rec('PatchNetVLAD', ds, mp):+.0f}%] \\| {me:.3f}±{se:.3f} [{100*rec('EigenPlaces', ds, me):+.0f}%]")
        wp, we = gauc("PatchNetVLAD", ds, "w100", n), gauc("EigenPlaces", ds, "w100", n)
        cells2.append(f"{wp-mp:+.3f} ({(wp-mp)/sp:+.1f} sd) \\| {we-me:+.3f} ({(we-me)/se:+.1f} sd)")
        if n != 20: agree.append((f"gate: w100 − random mean at n={n}", ds, sgn(wp - mp), sgn(we - me), wp - mp, we - me))
    L.append(f"| random, 5 seeds mean±sd | {n} | " + " | ".join(cells) + " |")
    L.append(f"| sem-only − random mean (in random sd) | {n} | " + " | ".join(cells2) + " |")
for m, lab in [("clipiqa_neg", "CLIP-IQA"), ("brisque", "BRISQUE")]:
    for n in NS:
        cells = []
        for ds in DS:
            p, e = gauc("PatchNetVLAD", ds, m, n), gauc("EigenPlaces", ds, m, n)
            cells.append(f"{f3(p)} [{100*rec('PatchNetVLAD', ds, p):+.0f}%] \\| {f3(e)} [{100*rec('EigenPlaces', ds, e):+.0f}%]" if p is not None and e is not None else "—")
        L.append(f"| {lab} | {n} | " + " | ".join(cells) + " |")
L.append("\nTies (n_passed = n−1) under EigenPlaces: SFU w100 n=145, StLucia w100 n=145, StLucia w000f n=20 — the same cells as under PatchNetVLAD (identical score columns).\n")
for ds in DS:
    for n in NS:
        for m, lab in [("w000f", "w000f − w100"), ("w050f", "w050f − w100")]:
            dp = gauc("PatchNetVLAD", ds, m, n) - gauc("PatchNetVLAD", ds, "w100", n); de = gauc("EigenPlaces", ds, m, n) - gauc("EigenPlaces", ds, "w100", n)
            agree.append((f"gate: {lab} at n={n}", ds, sgn(dp), sgn(de), dp, de))
    mp = rand_stats("PatchNetVLAD", ds, 20)[0]; me = rand_stats("EigenPlaces", ds, 20)[0]
    agree.append(("gate: w100 − random mean at n=20", ds, sgn(gauc("PatchNetVLAD", ds, "w100", 20) - mp), sgn(gauc("EigenPlaces", ds, "w100", 20) - me), gauc("PatchNetVLAD", ds, "w100", 20) - mp, gauc("EigenPlaces", ds, "w100", 20) - me))
    for m, lab in [("clipiqa_neg", "CLIP-IQA − w100 at n=20"), ("brisque", "BRISQUE − w100 at n=20")]:
        dp = gauc("PatchNetVLAD", ds, m, 20) - gauc("PatchNetVLAD", ds, "w100", 20); de = gauc("EigenPlaces", ds, m, 20) - gauc("EigenPlaces", ds, "w100", 20)
        agree.append((f"gate: {lab}", ds, sgn(dp), sgn(de), dp, de))
    agree.append(("naive-400 below baseline", ds, sgn(gbase("PatchNetVLAD", ds)[1] - gbase("PatchNetVLAD", ds)[0]), sgn(gbase("EigenPlaces", ds)[1] - gbase("EigenPlaces", ds)[0]), gbase("PatchNetVLAD", ds)[1] - gbase("PatchNetVLAD", ds)[0], gbase("EigenPlaces", ds)[1] - gbase("EigenPlaces", ds)[0]))

# (2) fixed 5% sem-only
L.append("## 2. Fixed 5 % acceptance (n = 20), semantic-only, vs baseline\n")
L.append("| descriptor | dataset | baseline | naive-400 | w100 @ 5 % | Δ vs baseline | recovery |"); L.append("|---|---|---|---|---|---|---|")
for D in DESCS:
    for ds in DS:
        b, nv = gbase(D, ds); a = gauc(D, ds, "w100", 20)
        L.append(f"| {D} | {ds} | {b:.3f} | {nv:.3f} | {a:.3f} | {a-b:+.3f} | {100*rec(D, ds, a):+.0f}% |")
        agree.append(("gate: w100 @5% − baseline", ds, sgn(a - b, 0.0015), None, a - b, None)) if D == "PatchNetVLAD" else None
for ds in DS:  # fill EIG column for the last claim
    for i, t in enumerate(agree):
        if t[0] == "gate: w100 @5% − baseline" and t[1] == ds and t[3] is None:
            e = gauc("EigenPlaces", ds, "w100", 20) - gbase("EigenPlaces", ds)[0]; agree[i] = (t[0], ds, t[2], sgn(e, 0.0015), t[4], e)
L.append("")

# (3) Phase 2 conditions
L.append("## 3. Causal conditions (mean ± sd over 5 seeds), PatchNetVLAD | EigenPlaces\n")
L.append("| condition | " + " | ".join(DS) + " |"); L.append("|---|" + "---|" * len(DS))
L.append("| C0 baseline | " + " | ".join(f"{base[('PatchNetVLAD', ds)]:.3f} \\| {base[('EigenPlaces', ds)]:.3f}" for ds in DS) + " |")
for c, lab in [("C1", "C1 naive"), ("C2", "C2 source re-injected"), ("C2b", "C2b day_left"), ("C4", "C4 warp_black"), ("C5", "C5 warp_telea"), ("C6", "C6 genwarp_black"), ("C7", "C7 genwarp_telea")]:
    cells = []
    for ds in DS:
        p, e = cm("PatchNetVLAD", ds, c), cm("EigenPlaces", ds, c)
        cells.append("—" if p[0] is None else f"{p[0]:.3f}±{p[1]:.3f} \\| {e[0]:.3f}±{e[1]:.3f}")
    L.append(f"| {lab} | " + " | ".join(cells) + " |")
L.append("\n### 3a. Damage C2 − C1 and direct contrasts: mean, k of 5 > 0, paired t (p), PatchNetVLAD | EigenPlaces\n")
L.append("| contrast | dataset | PatchNetVLAD | EigenPlaces | direction |"); L.append("|---|---|---|---|---|")
for x, y, lab in [("C2", "C1", "C2 − C1 (damage)"), ("C7", "C5", "C7 − C5 (distribution shift, Telea)"), ("C6", "C4", "C6 − C4 (distribution shift, black)"), ("C5", "C1", "C5 − C1 (warp-only Telea vs naive)"), ("C7", "C1", "C7 − C1 (holes filled vs naive)"), ("C4", "C1", "C4 − C1 (warp-only black vs naive)"), ("C6", "C1", "C6 − C1 (holes black vs naive)"), ("C2b", "C2", "C2b − C2 (real viewpoint vs source)")]:
    for ds in DS:
        if x == "C2b" and ds != "GardensPoint": continue
        dp, tp, pp, kp = paired(A.get(("PatchNetVLAD", ds, x), {}), A.get(("PatchNetVLAD", ds, y), {})); de, te, pe, ke = paired(A.get(("EigenPlaces", ds, x), {}), A.get(("EigenPlaces", ds, y), {}))
        if len(dp) < 2 or len(de) < 2: continue
        d = "agrees" if sgn(dp.mean()) == sgn(de.mean()) else "DISAGREES"
        L.append(f"| {lab} | {ds} | {dp.mean():+.4f}, {kp}/5, t={tp:+.2f} (p={pp:.3f}) | {de.mean():+.4f}, {ke}/5, t={te:+.2f} (p={pe:.3f}) | {d} |")
        agree.append((f"causal: {lab}", ds, sgn(dp.mean()), sgn(de.mean()), dp.mean(), de.mean()))
L.append("")

# (4) GT-inherit
L.append("## 4. GT-inheritance: C1 vs C2b vs C2 (C2 = sanity), PatchNetVLAD | EigenPlaces\n")
L.append("| dataset | condition | same top-1 as source | injected hit rate | hit given source hit | hit given source miss |"); L.append("|---|---|---|---|---|---|")
for ds in DS:
    for c in ["C2", "C1", "C2b"]:
        p, e = gstat("PatchNetVLAD", ds, c), gstat("EigenPlaces", ds, c)
        if p is None or e is None: continue
        L.append(f"| {ds} | {c} | {100*p['same']:.0f}% \\| {100*e['same']:.0f}% | {100*p['syn']:.0f}% \\| {100*e['syn']:.0f}% | {100*p['hh']:.0f}% \\| {100*e['hh']:.0f}% | {100*p['hm']:.0f}% \\| {100*e['hm']:.0f}% |")
    p1, e1 = gstat("PatchNetVLAD", ds, "C1"), gstat("EigenPlaces", ds, "C1"); pb, eb = gstat("PatchNetVLAD", ds, "C2b"), gstat("EigenPlaces", ds, "C2b")
    if p1 and e1: agree.append(("gt-inherit: C1 hit rate − source hit rate", ds, sgn(p1["syn"] - p1["src"], 0.005), sgn(e1["syn"] - e1["src"], 0.005), p1["syn"] - p1["src"], e1["syn"] - e1["src"]))
    if pb and eb and p1 and e1: agree.append(("gt-inherit: C1 same-top-1 − C2b same-top-1", ds, sgn(p1["same"] - pb["same"], 0.005), sgn(e1["same"] - eb["same"], 0.005), p1["same"] - pb["same"], e1["same"] - eb["same"]))
L.append(f"\nSource hit rates (real queries) differ by descriptor: " + "; ".join(f"{ds} {100*gstat('PatchNetVLAD', ds, 'C1')['src']:.0f}% \\| {100*gstat('EigenPlaces', ds, 'C1')['src']:.0f}%" for ds in DS) + ".\n")

# (5) claims
L.append("## 5. Direction of every claim across descriptors\n")
L.append("Direction = sign of the stated difference (≈0 if |Δ| < 0.0005 AUC or 0.5 pp). 'agrees' = same sign under both descriptors.\n")
L.append("| claim | dataset | PatchNetVLAD | EigenPlaces | verdict |"); L.append("|---|---|---|---|---|")
na = nd = 0
for claim, ds, sp, se, vp, ve in agree:
    v = "agrees" if sp == se else "DISAGREES"; na += v == "agrees"; nd += v != "agrees"
    L.append(f"| {claim} | {ds} | {sp} ({vp:+.3f}) | {se} ({ve:+.3f}) | {v} |")
L.append(f"\n{na} of {na+nd} claim-dataset pairs agree in direction across descriptors; {nd} disagree.\n")
open(f"{P3}/summary_phase3a.md", "w").write("\n".join(L) + "\n")
print(f"wrote {P3}/summary_phase3a.md; EigenPlaces gt_inherit rows {len(gt['EigenPlaces'])}; claims agree/disagree {na}/{nd}")
