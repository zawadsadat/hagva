#!/home/newview/miniconda3/envs/vprtutorial/bin/python3
"""Phase 3b summary: low (Phase 1/2/3a) vs wide (Phase 3b) pools, both descriptors. Writes phase3b/summary_phase3b.md."""
import csv, os, collections
import numpy as np
from scipy import stats
R = "/mnt/sda/HAGVA_R2"; DS = ["GardensPoint", "SFU", "StLucia"]; NS = [20, 64, 145]; DESC = ["PatchNetVLAD", "EigenPlaces"]
f3 = lambda x: "—" if x is None else f"{x:.3f}"
# ── gate: (pool, desc, ds, method, n) -> (auc, base, naive)
G = {}
for r in csv.DictReader(open(f"{R}/phase1/phase1_results.csv")):
    if r["dataset"] in DS: G[("low", "PatchNetVLAD", r["dataset"], r["method"], int(r["n_requested"]))] = (float(r["auc"]), float(r["baseline"]), float(r["naive"]))
for r in csv.DictReader(open(f"{R}/phase3a/phase3a_gate_results.csv")):
    if r["dataset"] in DS: G[("low", r["descriptor"], r["dataset"], r["method"], int(r["n_requested"]))] = (float(r["auc"]), float(r["baseline"]), float(r["naive"]))
for r in csv.DictReader(open(f"{R}/phase3b/phase3b_gate_results.csv")):
    G[("wide", r["descriptor"], r["dataset"], r["method"], int(r["n_requested"]))] = (float(r["auc"]), float(r["baseline"]), float(r["naive"]))
def auc(p, D, ds, m, n): x = G.get((p, D, ds, m, n)); return None if x is None else x[0]
def bn(p, D, ds): x = [v for k, v in G.items() if k[:3] == (p, D, ds)]; return (x[0][1], x[0][2]) if x else (None, None)
def rnd(p, D, ds, n): v = [auc(p, D, ds, f"rand_s{i}", n) for i in range(5)]; v = [x for x in v if x is not None]; return (np.mean(v), np.std(v, ddof=1), len(v)) if len(v) > 1 else (None, None, len(v))
# ── causal: (pool, desc, ds, cond) -> {seed: auc}
A = collections.defaultdict(dict); C0 = {}
for path, pool, keep in [(f"{R}/phase2/phase2_results.csv", "low", "PatchNetVLAD"), (f"{R}/phase3a/phase3a_causal_results.csv", "low", "EigenPlaces"), (f"{R}/phase3b/phase3b_causal_results.csv", "wide", None)]:
    for r in csv.DictReader(open(path)):
        if keep and r["descriptor"] != keep: continue
        if r["dataset"] not in DS: continue
        if r["condition"] == "C0": C0[(pool, r["descriptor"], r["dataset"])] = float(r["auc"])
        else: A[(pool, r["descriptor"], r["dataset"], r["condition"])][int(r["seed"])] = float(r["auc"])
def paired(x, y):
    s = sorted(set(x) & set(y)); d = np.array([x[i] - y[i] for i in s])
    if len(s) < 2: return d, np.nan, np.nan, int((d > 0).sum())
    t = stats.ttest_rel([x[i] for i in s], [y[i] for i in s]); return d, t.statistic, t.pvalue, int((d > 0).sum())
def ms(p, D, ds, c): v = list(A.get((p, D, ds, c), {}).values()); return (np.mean(v), np.std(v, ddof=1)) if len(v) > 1 else (None, None)

L = ["# HAGVA Phase 3b — wide-viewpoint pools (ele 8–16°, r 0.30–0.50) vs low (ele 1–5°, r 0.10–0.20)\n",
     "Wide pools: 400 views per dataset from the same 400 sources, scored with `score_pool.py` (validated against Module A on the low pool). "
     "Gate runs: `phase3b_gate_results.csv` (186 rows); causal runs: `phase3b_causal_results.csv` (126 rows; C1, C2, C5, C7 × 5 seeds × 3 datasets × 2 descriptors, same draws as Phase 2). "
     "Low-pool numbers are the Phase 1 / 2 (PatchNetVLAD) and Phase 3a (EigenPlaces) results. StLucia wide pool added 2026-10-17 (3b-STL).\n",
     "**Caveat (3b-STL):** StLucia's wide pool has the smallest hole fraction of the three (mean 0.153 vs SFU 0.184 and GardensPoint 0.247; 3/400 views above 0.25 vs 34 (SFU) and 193 (GardensPoint)), "
     "so its wide poses leave more of the image untouched by diffusion; its \"holds at wide\" cells are therefore a weaker test of the gate and of the causal picture than the GardensPoint cells.\n"]
# (1) pool stats
L.append("## 1. Pool statistics, low vs wide\n"); L.append(open(f"{R}/phase3b/pool_stats.md").read().split("\n", 3)[3])
# (2) gate table
L.append("## 2. Matched-n gate table, low | wide (AUC [recovery vs own baseline/naive])\n")
claims = []
for D in DESC:
    L.append(f"### {D}\n")
    L.append("| method | n | " + " | ".join(DS) + " |"); L.append("|---|---|" + "---|" * len(DS))
    L.append("| baseline / naive-400 | — | " + " | ".join(f"{bn('low',D,ds)[0]:.3f} / {bn('low',D,ds)[1]:.3f} \\| {bn('wide',D,ds)[0]:.3f} / {bn('wide',D,ds)[1]:.3f}" for ds in DS) + " |")
    def cell(ds, m, n):
        out = []
        for p in ("low", "wide"):
            a = auc(p, D, ds, m, n); b, nv = bn(p, D, ds)
            out.append("—" if a is None else f"{a:.3f} [{100*(a-nv)/(b-nv):+.0f}%]")
        return " \\| ".join(out)
    for m, lab in [("w100", "sem-only (w100)"), ("w000f", "corrected geo-only (w000f)"), ("w050f", "equal weight, corrected (w050f)"), ("clipiqa_neg", "CLIP-IQA"), ("brisque", "BRISQUE")]:
        for n in NS:
            if m in ("clipiqa_neg", "brisque") and any(auc("low", D, ds, m, n) is None for ds in DS) and n != 20: 
                L.append(f"| {lab} | {n} | " + " | ".join("— \\| " + (f3(auc('wide', D, ds, m, n)) + f" [{100*(auc('wide',D,ds,m,n)-bn('wide',D,ds)[1])/(bn('wide',D,ds)[0]-bn('wide',D,ds)[1]):+.0f}%]" if auc('wide', D, ds, m, n) is not None else "—") for ds in DS) + " |"); continue
            L.append(f"| {lab} | {n} | " + " | ".join(cell(ds, m, n) for ds in DS) + " |")
    for n in NS:
        row = []; row2 = []
        for ds in DS:
            parts = []; parts2 = []
            for p in ("low", "wide"):
                m_, s_, k = rnd(p, D, ds, n); b, nv = bn(p, D, ds); w = auc(p, D, ds, "w100", n)
                if m_ is None or w is None: parts.append("—"); parts2.append("—"); continue
                parts.append(f"{m_:.3f}±{s_:.3f} [{100*(m_-nv)/(b-nv):+.0f}%]"); parts2.append(f"{w-m_:+.3f} ({(w-m_)/s_:+.1f} sd)")
                if p == "wide" and n in (64, 145): claims.append(("gate", D, ds, n, w - m_, (w - m_) / s_))
            row.append(" \\| ".join(parts)); row2.append(" \\| ".join(parts2))
        L.append(f"| random, 5 seeds | {n} | " + " | ".join(row) + " |"); L.append(f"| sem-only − random mean (sd units) | {n} | " + " | ".join(row2) + " |")
    L.append("")
# (3) naive damage
L.append("## 3. Naive-injection damage: naive-400 − baseline, low vs wide\n")
L.append("| descriptor | dataset | baseline | naive low | naive wide | damage low | damage wide | wide/low |"); L.append("|---|---|---|---|---|---|---|---|")
for D in DESC:
    for ds in DS:
        b, nl = bn("low", D, ds); _, nw = bn("wide", D, ds)
        L.append(f"| {D} | {ds} | {b:.3f} | {nl:.3f} | {nw:.3f} | {nl-b:+.3f} | {nw-b:+.3f} | {(nw-b)/(nl-b):.2f}× |")
L.append("")
# (4) causal
L.append("## 4. Causal contrasts (5 seeds, same draws): mean Δ, seeds with Δ > 0, paired t (p) — low | wide\n")
L.append("| descriptor | dataset | contrast | low | wide |"); L.append("|---|---|---|---|---|")
cres = {}
for D in DESC:
    for ds in DS:
        L.append(f"| {D} | {ds} | means C1 / C2 / C5 / C7 | " + " | ".join(" / ".join(f3(ms(p, D, ds, c)[0]) for c in ("C1", "C2", "C5", "C7")) for p in ("low", "wide")) + " |")
        for x, y, lab in [("C2", "C1", "C2 − C1 (damage)"), ("C7", "C5", "C7 − C5 (distribution shift)"), ("C5", "C1", "C5 − C1 (remove diffusion)"), ("C7", "C1", "C7 − C1 (fill holes only)")]:
            cells = []
            for p in ("low", "wide"):
                d, t, pv, k = paired(A.get((p, D, ds, x), {}), A.get((p, D, ds, y), {}))
                cells.append(f"{d.mean():+.4f}, {k}/5, t={t:+.2f} (p={pv:.3f})" if len(d) > 1 else "—")
                if len(d) > 1: cres[(p, D, ds, x, y)] = (d.mean(), k)
            L.append(f"| {D} | {ds} | {lab} | " + " | ".join(cells) + " |")
L.append("")
# (5) pre-registered readings + claims
L.append("## 5. Pre-registered readings and claim-by-claim status\n")
L.append('> The gate "holds at wide viewpoints" iff sem-only beats the random mean at n=64 and n=145 on all datasets under both descriptors (12 cells: 3 datasets × 2 descriptors × n ∈ {64, 145}; the original 3b wording said 8 cells for GardensPoint and SFU).\n')
gate_cells = [(D, ds, n, dv, sd) for (_, D, ds, n, dv, sd) in claims]
ok = sum(1 for c in gate_cells if c[3] > 0)
L.append("| descriptor | dataset | n | sem − random (wide) | sd units | beats random? |"); L.append("|---|---|---|---|---|---|")
for D, ds, n, dv, sd in gate_cells: L.append(f"| {D} | {ds} | {n} | {dv:+.3f} | {sd:+.1f} | {'yes' if dv > 0 else 'no'} |")
NC = len(gate_cells); okGS = sum(1 for c in gate_cells if c[3] > 0 and c[1] != "StLucia"); nGS = sum(1 for c in gate_cells if c[1] != "StLucia")
L.append(f"\n**Gate at wide: {ok}/{NC} cells → {'HOLDS' if ok == NC else 'does not hold (' + str(NC-ok) + ' cell(s) fail)'}.** (GardensPoint + SFU only, the original 8-cell reading: {okGS}/{nGS}.)\n")
L.append('> The causal picture "holds at wide" iff C7 − C5 < 0 in ≥4 of 5 seeds in all 6 dataset×descriptor cells (originally 4, GardensPoint and SFU); if instead C7 − C1 > 0 in ≥4/5 seeds (filling the holes now helps), the holes become a cause at wide poses — report that as the finding, not as a failure.\n')
L.append("| descriptor | dataset | C7 − C5 < 0 seeds (wide) | C7 − C1 > 0 seeds (wide) | C5 − C1 > 0 seeds (wide) |"); L.append("|---|---|---|---|---|")
hold = 0; holes = 0
for D in DESC:
    for ds in DS:
        m75, k75 = cres[("wide", D, ds, "C7", "C5")]; m71, k71 = cres[("wide", D, ds, "C7", "C1")]; m51, k51 = cres[("wide", D, ds, "C5", "C1")]
        hold += (5 - k75) >= 4; holes += k71 >= 4
        L.append(f"| {D} | {ds} | {5-k75}/5 (mean {m75:+.4f}) | {k71}/5 (mean {m71:+.4f}) | {k51}/5 (mean {m51:+.4f}) |")
NCC = len(DS) * len(DESC)
L.append(f"\n**Causal at wide: C7 − C5 < 0 in ≥4/5 seeds in {hold}/{NCC} cells → {'HOLDS' if hold == NCC else 'does not hold in all cells'}; C7 − C1 > 0 in ≥4/5 seeds in {holes}/{NCC} cells → "
         f"{'holes become a cause at wide poses in those cells (reported as the finding)' if holes else 'filling the holes alone does not help at wide poses either'}.**\n")
L.append("### Claim-by-claim\n"); L.append("| claim | descriptor | dataset | low | wide | status |"); L.append("|---|---|---|---|---|---|")
sg = lambda x: "+" if x > 0.0005 else ("−" if x < -0.0005 else "≈0")
for D in DESC:
    for ds in DS:
        for n in (64, 145):
            lo = auc("low", D, ds, "w100", n) - rnd("low", D, ds, n)[0]; wi = auc("wide", D, ds, "w100", n) - rnd("wide", D, ds, n)[0]
            L.append(f"| sem-only beats random mean at n={n} | {D} | {ds} | {sg(lo)} ({lo:+.3f}) | {sg(wi)} ({wi:+.3f}) | {'holds at wide' if (lo > 0) == (wi > 0) else 'breaks at wide'} |")
        for m, lab in [("w000f", "corrected geo-only vs sem-only (n=64)"), ("w050f", "equal weight vs sem-only (n=64)"), ("clipiqa_neg", "CLIP-IQA vs sem-only (n=20)"), ("brisque", "BRISQUE vs sem-only (n=20)")]:
            n = 64 if m in ("w000f", "w050f") else 20
            lo = auc("low", D, ds, m, n) - auc("low", D, ds, "w100", n); wi = auc("wide", D, ds, m, n) - auc("wide", D, ds, "w100", n)
            L.append(f"| {lab} | {D} | {ds} | {sg(lo)} ({lo:+.3f}) | {sg(wi)} ({wi:+.3f}) | {'holds at wide' if sg(lo) == sg(wi) else 'breaks at wide'} |")
        for x, y, lab in [("C2", "C1", "naive damage C2 − C1 > 0"), ("C7", "C5", "C7 − C5 < 0 (visible-region diffusion hurts)"), ("C5", "C1", "C5 − C1 > 0 (warp-only recovers)"), ("C7", "C1", "C7 − C1 ≈ 0 or < 0 (holes not the cause)")]:
            lo = cres[("low", D, ds, x, y)][0]; wi = cres[("wide", D, ds, x, y)][0]
            L.append(f"| {lab} | {D} | {ds} | {sg(lo)} ({lo:+.4f}) | {sg(wi)} ({wi:+.4f}) | {'holds at wide' if sg(lo) == sg(wi) else 'breaks at wide'} |")
L.append("")
open(f"{R}/phase3b/summary_phase3b.md", "w").write("\n".join(L) + "\n"); print("wrote summary_phase3b.md")
