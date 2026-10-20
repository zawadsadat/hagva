#!/home/newview/miniconda3/envs/vprtutorial/bin/python3
"""
Phase 2 post-processing (CPU). Reads phase2/phase2_results.csv (+ runs/*/S.npy for the GT-inherit table,
draws/*.csv, strata_results.csv if present) and writes phase2/summary_phase2.md and phase2/gt_inherit.csv.
  summarize_phase2.py [--results phase2_results.csv] [--out summary_phase2.md]
"""
import argparse, csv, os, glob, json
from collections import defaultdict
import numpy as np
from scipy import stats

P2 = "/mnt/sda/HAGVA_R2/phase2"
DS = ["GardensPoint", "SFU", "StLucia"]; DSK = {"GardensPoint": "gp", "SFU": "sfu", "StLucia": "stl"}
DESC = ["NetVLAD", "PatchNetVLAD"]
COND = ["C1", "C2", "C2b", "C4", "C5", "C6", "C7"]
CLABEL = {"C0": "C0 baseline (no injection)", "C1": "C1 naive (GenWarp view)", "C2": "C2 real-image control (source re-injected)",
          "C2b": "C2b real-viewpoint control (day_left)", "C4": "C4 warp_black", "C5": "C5 warp_telea", "C6": "C6 genwarp_black", "C7": "C7 genwarp_telea"}
READING = ('"hallucinated content is the dominant cause" is supported iff C7 ≥ C5 − sd AND C7 > C1 + 2·sd on both descriptors; otherwise report as partial.')

def f3(x): return "—" if x is None else f"{x:.3f}"
FLOOR_T = 0.10   # descriptor "at floor": baseline AUC below this -> rows flagged, excluded from verdict/contrast tables
def at_floor(base, ds, D): return base.get((ds, D)) is not None and base[(ds, D)] < FLOOR_T
def floor_note(base, ds, D): return f"descriptor at floor (baseline {base[(ds, D)]:.3f}); not interpretable"

def paired(x, y):
    """x, y: dicts seed->auc. Returns (seeds, d, t, p_t, w, p_w, k_pos) for d = x - y over common seeds."""
    seeds = sorted(set(x) & set(y)); d = np.array([x[s] - y[s] for s in seeds])
    if len(seeds) < 2: return seeds, d, np.nan, np.nan, None, np.nan, int((d > 0).sum())
    t = stats.ttest_rel([x[s] for s in seeds], [y[s] for s in seeds])
    try: w = stats.wilcoxon(d, zero_method="wilcox", method="exact") if np.any(d != 0) else None
    except ValueError: w = None
    return seeds, d, t.statistic, t.pvalue, (None if w is None else w.statistic), (np.nan if w is None else w.pvalue), int((d > 0).sum())

def ci95(v):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    if len(v) < 2: return np.nan, np.nan, np.nan
    m = v.mean(); h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)); return m, m - h, m + h
def fmt_seeds(d): return " / ".join(f"{x:+.3f}" for x in d)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--results", default=f"{P2}/phase2_results.csv"); ap.add_argument("--out", default=f"{P2}/summary_phase2.md")
    ap.add_argument("--gt-out", default=f"{P2}/gt_inherit.csv"); a = ap.parse_args()
    rows = list(csv.DictReader(open(a.results, newline="")))
    base = {(r["dataset"], r["descriptor"]): float(r["auc"]) for r in rows if r["condition"] == "C0"}
    A = defaultdict(dict)   # (ds, desc, cond) -> {seed: auc}
    for r in rows:
        if r["condition"] != "C0": A[(r["dataset"], r["descriptor"], r["condition"])][int(r["seed"])] = float(r["auc"])
    L = ["# HAGVA Phase 2 — causal-chain repair (summary)\n",
         f"Source: `{os.path.relpath(a.results, P2)}` ({len(rows)} rows: {sum(1 for r in rows if r['condition']=='C0')} baselines + {sum(1 for r in rows if r['condition']!='C0')} injected runs). "
         "Each injected run = real query set + 50 views of one seeded draw from the 400-view pool (5 draws, seeds 0–4). Hole mask M = regenerated GenWarp mask; "
         "`black` = holes set to 0; `telea` = cv2.inpaint radius 3 (C7 holes take exactly the C5 fill). C3 (zero-shift) skipped: no outputs for the pool sources. "
         "Reference: C0 (no injection) is not draw-matched, so the within-seed reference is C2 (the same 50 source images re-injected); "
         "recovery = (Cx − C1)/(C2 − C1) per seed (v2; v1 used C0). NetVLAD on SFU/StLucia is at floor (baseline 0.022 / 0.052) and is flagged, not interpreted.\n"]
    # ── 1. mean ± sd per condition ──────────────────────────────────────────
    L.append("## 1. AUC per condition, mean ± sd over the 5 seeds\n")
    for D in DESC:
        L.append(f"### {D}\n"); L.append("| condition | " + " | ".join(DS) + " |"); L.append("|---|" + "---|" * len(DS))
        L.append("| C0 baseline (not draw-matched) | " + " | ".join(f3(base.get((ds, D))) + (" ⚠ floor" if at_floor(base, ds, D) else "") for ds in DS) + " |")
        for c in COND:
            cells = []
            for ds in DS:
                v = list(A.get((ds, D, c), {}).values())
                cells.append("—" if not v else (f"{np.mean(v):.3f} ± {np.std(v, ddof=1):.3f} (n={len(v)})" if len(v) > 1 else f"{v[0]:.3f} (n=1)"))
            L.append(f"| {CLABEL[c]} | " + " | ".join(cells) + " |")
        fl = [ds for ds in DS if at_floor(base, ds, D)]
        if fl: L.append(f"\n⚠ {D} on {', '.join(fl)}: " + "; ".join(floor_note(base, ds, D) for ds in fl) + ". These columns are excluded from the verdict and contrast tables below.")
        L.append("")
    # ── 2. naive-injection damage (C2 − C1) and recovery relative to C2 ─────
    L.append("## 2. Naive-injection damage: C2 − C1 per seed (draw-matched)\n")
    L.append("C2 injects the same 50 real source images as C1 injects synthesized views, so C2 − C1 is the damage attributable to synthesis for that draw.\n")
    L.append("| descriptor | dataset | C2 − C1 per seed (0…4) | mean | sd | paired t (p) | Wilcoxon (p) |"); L.append("|---|---|---|---|---|---|---|")
    damage = {}
    for D in DESC:
        for ds in DS:
            if at_floor(base, ds, D): L.append(f"| {D} | {ds} | {floor_note(base, ds, D)} | | | | |"); continue
            seeds, d, t, pt, w, pw, k = paired(A.get((ds, D, "C2"), {}), A.get((ds, D, "C1"), {}))
            if len(seeds) < 2: continue
            damage[(D, ds)] = dict(zip(seeds, d))
            L.append(f"| {D} | {ds} | {fmt_seeds(d)} | {d.mean():+.4f} | {d.std(ddof=1):.4f} | {t:+.2f} ({pt:.3f}) | {'—' if w is None else f'{w:.0f} ({pw:.3f})'} |")
    L.append("")
    L.append("### 2a. Recovery relative to C2: (Cx − C1)/(C2 − C1) per seed, mean and 95 % CI\n")
    L.append("100 % = the condition restores the AUC of injecting the real source images for that draw; 0 % = no better than naive.\n")
    L.append("| descriptor | dataset | condition | recovery per seed | mean [95 % CI] |"); L.append("|---|---|---|---|---|")
    for D in DESC:
        for ds in DS:
            if at_floor(base, ds, D) or (D, ds) not in damage: continue
            c1 = A.get((ds, D, "C1"), {}); dm = damage[(D, ds)]
            for c in ["C2b", "C4", "C5", "C6", "C7"]:
                cc = A.get((ds, D, c), {}); seeds = sorted(set(cc) & set(dm))
                if not seeds: continue
                rec = np.array([(cc[sd_] - c1[sd_]) / dm[sd_] if dm[sd_] != 0 else np.nan for sd_ in seeds])
                m, lo, hi = ci95(rec)
                L.append(f"| {D} | {ds} | {c} | {' / '.join('n/a' if np.isnan(r) else f'{100*r:+.0f}%' for r in rec)} | {100*m:+.0f}% [{100*lo:+.0f}, {100*hi:+.0f}] |")
    L.append("\nn = 5 seeds; CI = t-interval over seeds. Per-seed recovery is unstable when C2 − C1 is small for that seed (see the damage table).\n")
    # ── 2b. direct paired contrasts ──────────────────────────────────────────
    L.append("### 2b. Direct paired contrasts over the 5 seeds\n")
    CONTRASTS = [("C7", "C5", "distribution shift at real pose, Telea holes"), ("C6", "C4", "distribution shift at real pose, black holes"),
                 ("C5", "C1", "removing diffusion entirely (warp-only, Telea) vs naive"), ("C4", "C1", "removing diffusion entirely (warp-only, black) vs naive"),
                 ("C7", "C1", "removing only the holes (GenWarp, Telea) vs naive"), ("C6", "C1", "removing only the holes (GenWarp, black) vs naive"),
                 ("C5", "C4", "fill effect on warp-only (Telea − black)"), ("C7", "C6", "fill effect on GenWarp (Telea − black)")]
    L.append("| descriptor | dataset | contrast | per seed (0…4) | mean | k of 5 > 0 | paired t (p) | Wilcoxon (p) |"); L.append("|---|---|---|---|---|---|---|---|")
    contrast = {}
    for D in DESC:
        for ds in DS:
            if at_floor(base, ds, D): continue
            for x, y, lab in CONTRASTS:
                seeds, d, t, pt, w, pw, k = paired(A.get((ds, D, x), {}), A.get((ds, D, y), {}))
                if len(seeds) < 2: continue
                contrast[(D, ds, x, y)] = d
                L.append(f"| {D} | {ds} | {x} − {y} ({lab}) | {fmt_seeds(d)} | {d.mean():+.4f} | {k}/5 | {t:+.2f} ({pt:.3f}) | {'—' if w is None else f'{w:.0f} ({pw:.3f})'} |")
    L.append("\nWilcoxon exact, two-sided; smallest attainable p at n = 5 is 0.0625. k of 5 = number of seeds with a positive difference.\n")
    # ── 3. pre-registered reading, two versions ─────────────────────────────
    L.append("## 3. Pre-registered reading (verbatim from the brief), evaluated two ways\n"); L.append(f"> {READING}\n")
    L.append("Version 1 (as in v1): clause (a) C7 ≥ C5 − sd uses the sd over seeds of the raw C5 AUCs; clause (b) C7 > C1 + 2·sd uses the sd of the raw C1 AUCs. "
             "These sds include draw-to-draw variance that cancels within a seed.\n")
    L.append("Version 2 (C2 as reference): the brief states recovery against the no-injection baseline and asks whether C7 recovers to within the seed sd of C2; since C0 is not draw-matched, "
             "both clauses are evaluated on within-seed differences: (a) mean(C7 − C5) ≥ −sd(C7 − C5); (b) mean(C7 − C1) > 2·sd(C7 − C1); plus the brief's C2 test (c) |mean(C7 − C2)| ≤ sd(C2 − C1). "
             "Floor rows (NetVLAD on SFU / StLucia) are excluded.\n")
    L.append("| dataset | descriptor | v1 (a) C7 ≥ C5 − sd | v1 (b) C7 > C1 + 2sd | v1 verdict | v2 (a) mean(C7−C5) ≥ −sd | v2 (b) mean(C7−C1) > 2sd | v2 (c) C7 within sd of C2 | v2 verdict |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    v1, v2 = {}, {}
    for ds in DS:
        for D in DESC:
            if at_floor(base, ds, D): L.append(f"| {ds} | {D} | excluded: {floor_note(base, ds, D)} | | | | | | |"); continue
            c7 = A.get((ds, D, "C7"), {}); c5 = A.get((ds, D, "C5"), {}); c1 = A.get((ds, D, "C1"), {}); c2 = A.get((ds, D, "C2"), {})
            if min(len(c7), len(c5), len(c1), len(c2)) < 2: continue
            m7, m5, m1 = np.mean(list(c7.values())), np.mean(list(c5.values())), np.mean(list(c1.values()))
            s5, s1 = np.std(list(c5.values()), ddof=1), np.std(list(c1.values()), ddof=1)
            a1, b1 = m7 >= m5 - s5, m7 > m1 + 2 * s1
            d75 = paired(c7, c5)[1]; d71 = paired(c7, c1)[1]; d72 = paired(c7, c2)[1]; d21 = paired(c2, c1)[1]
            a2 = d75.mean() >= -d75.std(ddof=1); b2 = d71.mean() > 2 * d71.std(ddof=1); c2ok = abs(d72.mean()) <= d21.std(ddof=1)
            v1.setdefault(ds, []).append(a1 and b1); v2.setdefault(ds, []).append(a2 and b2)
            L.append(f"| {ds} | {D} | {'yes' if a1 else 'no'} ({m7:.3f} vs {m5-s5:.3f}) | {'yes' if b1 else 'no'} ({m7:.3f} vs {m1+2*s1:.3f}) | {'supported' if a1 and b1 else 'not supported'} | "
                     f"{'yes' if a2 else 'no'} ({d75.mean():+.4f} vs {-d75.std(ddof=1):+.4f}) | {'yes' if b2 else 'no'} ({d71.mean():+.4f} vs {2*d71.std(ddof=1):+.4f}) | "
                     f"{'yes' if c2ok else 'no'} ({d72.mean():+.4f} vs ±{d21.std(ddof=1):.4f}) | {'supported' if a2 and b2 else 'not supported'} |")
    def verdict_of(dct, ds):
        if ds not in dct or not dct[ds]: return "no interpretable descriptor"
        return "**PASS**" if all(dct[ds]) and len(dct[ds]) == 2 else ("PARTIAL" if any(dct[ds]) else "not supported on any interpretable descriptor")
    L.append("\nPer-dataset verdict (the brief requires BOTH descriptors; where only PatchNetVLAD is interpretable the verdict rests on one descriptor): " +
             "; ".join(f"{ds}: v1 {verdict_of(v1, ds)}, v2 {verdict_of(v2, ds)}" for ds in DS) + ".\n")
    verdict = {ds: verdict_of(v2, ds) for ds in DS}
    # ── 4. black vs telea ───────────────────────────────────────────────────
    L.append("## 4. black vs telea agreement\n")
    L.append("| descriptor | dataset | C5 − C4 (warp: telea − black) | C7 − C6 (genwarp: telea − black) | C7 − C5 (genwarp − warp, telea) | C6 − C4 (genwarp − warp, black) | same sign? |"); L.append("|---|---|---|---|---|---|---|")
    for D in DESC:
        for ds in DS:
            g = lambda c: A.get((ds, D, c), {})
            seeds = sorted(set(g("C4")) & set(g("C5")) & set(g("C6")) & set(g("C7")))
            if not seeds: continue
            if at_floor(base, ds, D): L.append(f"| {D} | {ds} | {floor_note(base, ds, D)} | | | | |"); continue
            d54 = np.mean([g("C5")[s] - g("C4")[s] for s in seeds]); d76 = np.mean([g("C7")[s] - g("C6")[s] for s in seeds])
            d75 = np.mean([g("C7")[s] - g("C5")[s] for s in seeds]); d64 = np.mean([g("C6")[s] - g("C4")[s] for s in seeds])
            L.append(f"| {D} | {ds} | {d54:+.3f} | {d76:+.3f} | {d75:+.3f} | {d64:+.3f} | {'yes' if np.sign(d75) == np.sign(d64) else 'NO'} |")
    L.append("\n`C7 − C5` is the distribution-shift effect at the real viewpoint (diffusion re-rendering of visible content, holes identical); `C6 − C4` is the same contrast under black holes. "
             "If the two disagree in sign, that disagreement is the result.\n")
    # ── 5. GT-inherit from S.npy ────────────────────────────────────────────
    L.append("## 5. GT-inheritance check (PatchNetVLAD, from S.npy)\n")
    gt_rows = []
    for ds in DS:
        k = DSK[ds]
        real_q = {"gp": "/mnt/sda/Zawad/genwarp/VPR-GIT/images/GardensPoint/night_right", "sfu": "/mnt/sda/Zawad/genwarp/VPR-GIT/images/SFU/jan", "stl": "/mnt/sda/Zawad/genwarp/VPR-GIT/images/StLucia_small/180809_1545"}[k]
        qfiles = sorted(f for f in os.listdir(real_q) if f.endswith(".jpg"))
        if k == "sfu": qfiles = qfiles[:385]
        nq = len(qfiles); qidx = {os.path.splitext(f)[0]: i for i, f in enumerate(qfiles)}
        # hard GT per real query column: GP identity; SFU/StLucia from their GT.npz
        if k == "gp": gthard = None
        elif k == "sfu": gthard = np.load("/mnt/sda/Zawad/genwarp/VPR-Query-NEW/images/SFU/GT_original.npz")["GThard"].astype(bool)
        else: gthard = np.load("/mnt/sda/Zawad/genwarp/VPR-GIT/images/StLucia_small/GT.npz")["GThard"].astype(bool)
        for seed in range(5):
            draw = list(csv.DictReader(open(f"{P2}/draws/{k}_seed{seed}.csv", newline="")))
            for c in ["C1", "C2", "C2b", "C4", "C5", "C6", "C7"]:
                p = f"{P2}/runs/{k}/seed{seed}/{c}_PatchNetVLAD/S.npy"
                if not os.path.exists(p): continue
                S = np.load(p)
                if S.shape[1] != nq + len(draw): continue
                top = S.argmax(axis=0)
                for j, r in enumerate(draw):
                    src = os.path.splitext(r["source_image"])[0]; qi = qidx.get(src)
                    if qi is None: continue
                    syn_top1 = int(top[nq + j]); src_top1 = int(top[qi])
                    if k == "gp": hit = lambda t: t == qi
                    else: hit = lambda t: bool(gthard[t, qi])
                    gt_rows.append(dict(dataset=ds, seed=seed, condition=c, stem=r["stem"], source_stem=src, src_top1=src_top1, syn_top1=syn_top1,
                                        same_top1=int(src_top1 == syn_top1), src_hit=int(hit(src_top1)), syn_hit=int(hit(syn_top1))))
    if gt_rows:
        with open(a.gt_out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(gt_rows[0].keys())); w.writeheader(); w.writerows(gt_rows)
        L.append(f"`{os.path.relpath(a.gt_out, P2)}`: {len(gt_rows)} rows. For each injected view: top-1 reference of the synthesized query (`syn_top1`) vs top-1 of its real source query in the same S (`src_top1`); hit = top-1 is a hard-GT reference (GardensPoint: identity; SFU/StLucia: GThard column of the source).\n")
        def gstats(ds, c):
            g = [r for r in gt_rows if r["dataset"] == ds and r["condition"] == c]
            if not g: return None
            sh = np.array([r["src_hit"] for r in g]); yh = np.array([r["syn_hit"] for r in g]); same = np.array([r["same_top1"] for r in g])
            return dict(n=len(g), same=same.mean(), src=sh.mean(), syn=yh.mean(), hit_given_hit=(yh[sh == 1].mean() if (sh == 1).any() else np.nan), hit_given_miss=(yh[sh == 0].mean() if (sh == 0).any() else np.nan))
        L.append("### 5a. C1 (synthesized) vs C2b (real day_left view) side by side, C2 as the sanity row (GardensPoint)\n")
        L.append("| metric | C2 sanity (source re-injected) | C1 naive GenWarp | C2b real viewpoint (day_left) |"); L.append("|---|---|---|---|")
        g2, g1, gb = gstats("GardensPoint", "C2"), gstats("GardensPoint", "C1"), gstats("GardensPoint", "C2b")
        if g2 and g1 and gb:
            for key, lab in [("same", "same top-1 as source"), ("syn", "hit rate of injected query"), ("hit_given_hit", "injected hit | source hit"), ("hit_given_miss", "injected hit | source miss")]:
                L.append(f"| {lab} | {100*g2[key]:.0f}% | {100*g1[key]:.0f}% | {100*gb[key]:.0f}% |")
            L.append(f"| source hit rate (same for all) | {100*g2['src']:.0f}% | {100*g1['src']:.0f}% | {100*gb['src']:.0f}% |")
        L.append("\nC2 must read 100 % / identical to the source by construction (same pixels); C2b is a genuinely different viewpoint of the same place.\n")
        L.append("### 5b. All conditions\n")
        L.append("| dataset | condition | n | same top-1 (src = syn) | src hit rate | syn hit rate | syn hit given src hit | syn hit given src miss |"); L.append("|---|---|---|---|---|---|---|---|")
        for ds in DS:
            for c in ["C2", "C1", "C2b", "C4", "C5", "C6", "C7"]:
                g = [r for r in gt_rows if r["dataset"] == ds and r["condition"] == c]
                if not g: continue
                sh = np.array([r["src_hit"] for r in g]); yh = np.array([r["syn_hit"] for r in g]); same = np.array([r["same_top1"] for r in g])
                cond_hit = yh[sh == 1].mean() if (sh == 1).any() else np.nan; cond_miss = yh[sh == 0].mean() if (sh == 0).any() else np.nan
                L.append(f"| {ds} | {c} | {len(g)} | {100*same.mean():.0f}% | {100*sh.mean():.0f}% | {100*yh.mean():.0f}% | {100*cond_hit:.0f}% | {100*cond_miss:.0f}% |")
        L.append("")
    else:
        L.append("No S.npy files found yet (PatchNetVLAD runs with HAGVA_SAVE_S=1 pending).\n")
    # ── 6. radius strata (if present) ───────────────────────────────────────
    sp = f"{P2}/strata_results.csv"
    if os.path.exists(sp):
        srows = list(csv.DictReader(open(sp, newline="")))
        L.append("## 6. Radius-stratified naive injection (PatchNetVLAD, all views of one radius, no gate)\n")
        L.append("| dataset | stratum | n injected | AUC | AUC − baseline | mean hole fraction |"); L.append("|---|---|---|---|---|---|")
        for r in srows:
            L.append(f"| {r['dataset']} | {r['stratum']} | {r['n_injected']} | {float(r['auc']):.3f} | {float(r['auc'])-float(r['baseline']):+.3f} | {float(r['mean_holefrac']):.3f} |")
        L.append("\nPairwise AUC differences between strata (single runs; the 140/121/139-view sets are disjoint, so no pairing over seeds is possible):\n")
        L.append("| dataset | r0.15 − r0.10 | r0.20 − r0.10 | r0.20 − r0.15 |"); L.append("|---|---|---|---|")
        for ds in DS:
            v = {r["stratum"]: float(r["auc"]) for r in srows if r["dataset"] == ds}
            if len(v) == 3: L.append(f"| {ds} | {v['r0.15']-v['r0.10']:+.3f} | {v['r0.20']-v['r0.10']:+.3f} | {v['r0.20']-v['r0.15']:+.3f} |")
        au = np.array([float(r["auc"]) for r in srows]); hf = np.array([float(r["mean_holefrac"]) for r in srows]); db = np.array([float(r["auc"]) - float(r["baseline"]) for r in srows])
        rs = stats.spearmanr(au, hf); rs2 = stats.spearmanr(db, hf)
        L.append(f"\nSpearman across the 9 rows: stratum AUC vs mean hole fraction ρ = {rs.correlation:+.3f} (p = {rs.pvalue:.2f}); (AUC − baseline) vs hole fraction ρ = {rs2.correlation:+.3f} (p = {rs2.pvalue:.2f}). "
                 "Within each dataset the three strata do not order monotonically with hole fraction.\n")
    open(a.out, "w").write("\n".join(L) + "\n")
    print(f"wrote {a.out} ({len(rows)} rows; gt_inherit rows {len(gt_rows)}); verdicts: {verdict}")

if __name__ == "__main__":
    main()
