#!/home/newview/miniconda3/envs/vprtutorial/bin/python3
"""
Phase 1 post-processing (no VPR runs). Reads phase1_results.csv (+ extended score CSVs for
tie values, checkpoint2_verification.csv for the SFU w050 n=20 footnote) and writes summary.md.

  summarize.py [--results phase1_results.csv] [--out summary.md]
"""
import argparse, csv, json, os, sys
from collections import defaultdict
import numpy as np

P1 = "/mnt/sda/HAGVA_R2/phase1"
DS = ["GardensPoint", "SFU", "StLucia"]
DSKEY = {"GardensPoint": "gardenspoint", "SFU": "sfu", "StLucia": "stlucia"}
NS = [20, 64, 145]
RATE = {20: "5%", 64: "16%", 145: "36%"}          # 20/400, 64/400, 145/400
BASE = {"GardensPoint": 0.802, "SFU": 0.729, "StLucia": 0.594}
NAIVE = {"GardensPoint": 0.755, "SFU": 0.651, "StLucia": 0.448}
WCOLS = ["w000", "w025", "w050", "w075", "w100"]
WVAL = {"w000": 0.0, "w025": 0.25, "w050": 0.5, "w075": 0.75, "w100": 1.0}
RAND = [f"rand_s{i}" for i in range(5)]
IQA = ["brisque", "niqe", "clipiqa_neg", "dino_global"]
FUSION = ["zavg", "rankavg", "prod", "maxgs", "sem_veto50", "sem_veto75", "sem_veto90", "geo_veto75"]
FCOLS = ["w000f", "w025f", "w050f", "w075f"] + [c + "f" for c in FUSION]      # Phase 1c: corrected geometric signal
FLABEL = {"w000f": "w=0.00f (fixed geo only)", "w025f": "w=0.25f", "w050f": "w=0.50f (equal weight, fixed geo)", "w075f": "w=0.75f"}
VETO = {"sem_veto50": ("mean_geo_score", 50), "sem_veto75": ("mean_geo_score", 75), "sem_veto90": ("mean_geo_score", 90),
        "geo_veto75": ("mean_sem_score", 75)}
FUSION_LABEL = {"zavg": "zavg = (z(g)+z(s))/2", "rankavg": "rankavg = (rank g + rank s)/2", "prod": "prod = g·s",
                "maxgs": "maxgs = max(g, s)", "sem_veto50": "sem, geo veto @P50", "sem_veto75": "sem, geo veto @P75",
                "sem_veto90": "sem, geo veto @P90", "geo_veto75": "geo, sem veto @P75 (control)",
                "w050": "w=0.50 (reference)", "w100": "w=1.00 sem-only (reference)"}
CRITERION = ('A fusion rule "beats semantic-only" iff BOTH: (a) AUC ≥ w100 AUC in at least 7 of the 9 (dataset, n) cells, '
             'and (b) mean AUC over the 9 cells ≥ w100 mean (0.7053 from phase1_results.csv).')
LABEL = {"w000": "w=0.00 (geo only)", "w025": "w=0.25", "w050": "w=0.50 (HAGVA, equal weight)",
         "w075": "w=0.75", "w100": "w=1.00 (sem only)", "random": "random (5 seeds, mean ± sd)",
         "brisque": "BRISQUE", "niqe": "NIQE", "clipiqa_neg": "CLIP-IQA (neg.)",
         "dino_global": "DINOv2 global drift"}
ANCHOR = {  # (ds, n): (geo, sem, comb)
    ("GardensPoint", 20): (0.794, 0.803, 0.791), ("GardensPoint", 64): (0.734, 0.807, 0.792), ("GardensPoint", 145): (0.733, 0.812, 0.790),
    ("SFU", 20): (0.727, 0.731, 0.731), ("SFU", 64): (0.713, 0.737, 0.741), ("SFU", 145): (0.695, 0.740, 0.732),
    ("StLucia", 20): (0.589, 0.597, 0.599), ("StLucia", 64): (0.561, 0.589, 0.578), ("StLucia", 145): (0.540, 0.532, 0.546)}
ANCHOR_IDX = {"w000": 0, "w100": 1, "w050": 2}

def rec(auc, ds):
    return (auc - NAIVE[ds]) / (BASE[ds] - NAIVE[ds])

def f3(x):  return "—" if x is None else f"{x:.3f}"
def pct(x): return "—" if x is None else f"{100*x:+.0f}%"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=f"{P1}/phase1_results.csv")
    ap.add_argument("--out", default=f"{P1}/summary.md")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.results, newline="")))
    R = {}      # (ds, method, n) -> row
    for r in rows:
        R[(r["dataset"], r["method"], int(r["n_requested"]))] = r
    def auc(ds, m, n):
        r = R.get((ds, m, n)); return None if r is None else float(r["auc"])

    L = []
    L.append("# HAGVA Phase 1 + 1b + 1c — reviewer-response experiments (matched-n comparison)\n")
    L.append(f"Source: `{os.path.relpath(a.results, P1)}` ({len(rows)} rows). Gate: PatchNetVLAD, VPR-GIT, 400 synthesized views per dataset. "
             "All methods are score columns over the SAME 400 views; a method accepts the n lowest-scoring views "
             "(tau = midpoint between the n-th and (n+1)-th lowest value, strict `<`). "
             "Baseline = no injection; naive = all 400 injected. Recovery = (AUC − naive)/(baseline − naive).\n")
    L.append("| dataset | baseline | naive (400) |\n|---|---|---|")
    for ds in DS: L.append(f"| {ds} | {BASE[ds]:.3f} | {NAIVE[ds]:.3f} |")
    L.append("")

    # ── 0. verification anchors ──────────────────────────────────────────────
    L.append("## 0. Verification anchors\n")
    fails = []
    for (ds, n), cells in ANCHOR.items():
        for m, i in ANCHOR_IDX.items():
            v = auc(ds, m, n)
            if v is None or abs(v - cells[i]) >= 5e-4: fails.append((ds, m, n, v, cells[i]))
    bn = [r for r in rows if float(r["baseline"]) != BASE[r["dataset"]] or float(r["naive"]) != NAIVE[r["dataset"]]]
    r400 = {ds: auc(ds, "rand_s0", 400) for ds in DS}
    L.append(f"- Anchor cells (w000/w050/w100 at n=20/64/145, 3 datasets): {27-len(fails)}/27 match the ablation table to 3 decimals.")
    for ds, m, n, v, exp in fails:
        L.append(f"  - MISMATCH {ds} {m} n={n}: got {f3(v)}, anchor {exp:.3f}" + (" — see footnote [^sfu20]" if (ds, m, n) == ("SFU", "w050", 20) else ""))
    L.append(f"- rand_s0 at n=400 equals naive: " + ", ".join(f"{ds} {f3(r400[ds])} (naive {NAIVE[ds]:.3f})" for ds in DS)
             + (" — OK" if all(r400[ds] is not None and abs(r400[ds]-NAIVE[ds]) < 5e-4 for ds in DS) else " — FAIL"))
    L.append(f"- Baseline/naive printed by the gate scripts matched the table on every row: {'yes' if not bn else 'NO — ' + str(len(bn)) + ' rows differ'}.")
    missing = [(ds, m, n) for ds in DS for m in WCOLS + RAND + IQA for n in NS if (ds, m, n) not in R]
    L.append(f"- Missing cells: {len(missing)}" + (" — " + ", ".join(f"{d}/{m}/n{n}" for d, m, n in missing) if missing else ""))
    L.append("")

    # ── 0b. Phase 1b anchors ─────────────────────────────────────────────────
    frows = [r for r in rows if r["method"] in FUSION]
    if frows:
        L.append("### Phase 1b anchors\n")
        fmiss = [(ds, m, n) for ds in DS for m in FUSION for n in NS if (ds, m, n) not in R]
        L.append(f"- Fusion rows present: {len(frows)}/72" + (f" — missing {fmiss}" if fmiss else "") + ".")
        fties = [r for r in frows if r["n_passed"] != r["n_requested"]]
        L.append(f"- n_passed == n_requested on {len(frows)-len(fties)}/{len(frows)} fusion rows; {len(fties)} tie rows (each shifted by one), listed in the tie report.")
        d90 = {ds: (auc(ds, "sem_veto90", 20), auc(ds, "w100", 20)) for ds in DS}
        L.append("- sem_veto90 vs w100 at n=20: " + ", ".join(f"{ds} {f3(a)} vs {f3(b)} ({'—' if a is None or b is None else f'{a-b:+.3f}'})" for ds, (a, b) in d90.items())
                 + (" — all within 0.01" if all(a is not None and b is not None and abs(a-b) <= 0.01 for a, b in d90.values()) else " — DIFFERENCE > 0.01, see report"))
        md5f = f"{P1}/phase1_results_first130.md5"
        if os.path.exists(md5f):
            import hashlib
            cur = hashlib.md5("".join(open(a.results).readlines()[:130]).encode()).hexdigest()
            saved = open(md5f).read().split()[0]
            L.append(f"- Phase 1 rows (first 130 lines of phase1_results.csv) md5 `{cur}` vs Phase 1 delivery `{saved}`: {'identical' if cur == saved else 'DIFFERENT'}.")
        L.append("")

    # ── 1. main table ────────────────────────────────────────────────────────
    hdr = "| method | " + " | ".join(f"{ds} n={n}" for ds in DS for n in NS) + " |"
    sep = "|---|" + "---|" * (len(DS) * len(NS))
    def rand_cell(ds, n, fn=lambda ds, v: v):
        vs = [auc(ds, m, n) for m in RAND]; vs = [fn(ds, v) for v in vs if v is not None]
        if not vs: return "—"
        return f"{np.mean(vs):.3f} ± {np.std(vs, ddof=1):.3f}" if len(vs) > 1 else f"{vs[0]:.3f}"
    L.append("## 1. AUC at matched acceptance counts\n")
    L.append("Columns = (dataset, n accepted). Bold = best method in that column (random uses its mean). Random: mean ± sample sd over seeds 0–4.\n")
    L.append(hdr); L.append(sep)
    L.append("| *baseline (no injection)* | " + " | ".join(f"*{BASE[ds]:.3f}*" for ds in DS for n in NS) + " |")
    L.append("| *naive (all 400)* | " + " | ".join(f"*{NAIVE[ds]:.3f}*" for ds in DS for n in NS) + " |")
    best = {}
    for ds in DS:
        for n in NS:
            cands = {m: auc(ds, m, n) for m in WCOLS + IQA}
            rv = [auc(ds, m, n) for m in RAND]; rv = [v for v in rv if v is not None]
            if rv: cands["random"] = float(np.mean(rv))
            cands = {k: v for k, v in cands.items() if v is not None}
            best[(ds, n)] = max(cands, key=cands.get) if cands else None
    for m in WCOLS + ["random"] + IQA:
        cells = []
        for ds in DS:
            for n in NS:
                c = rand_cell(ds, n) if m == "random" else f3(auc(ds, m, n))
                if best[(ds, n)] == m: c = f"**{c}**"
                cells.append(c)
        L.append(f"| {LABEL[m]} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("### 1b. Recovery = (AUC − naive)/(baseline − naive)\n")
    L.append(hdr); L.append(sep)
    for m in WCOLS + ["random"] + IQA:
        cells = []
        for ds in DS:
            for n in NS:
                if m == "random":
                    vs = [auc(ds, mm, n) for mm in RAND]; vs = [rec(v, ds) for v in vs if v is not None]
                    cells.append(f"{100*np.mean(vs):+.0f}% ± {100*np.std(vs, ddof=1):.0f}" if len(vs) > 1 else "—")
                else:
                    v = auc(ds, m, n); cells.append(pct(None if v is None else rec(v, ds)))
        L.append(f"| {LABEL[m]} | " + " | ".join(cells) + " |")
    L.append("")

    # ── 2. weight sweep ──────────────────────────────────────────────────────
    L.append("## 2. Fusion-weight sweep (score = w·sem + (1−w)·geo)\n")
    L.append("| w | " + " | ".join(f"{ds} n={n}" for ds in DS for n in NS) + " | " + " | ".join(f"{ds} mean(n)" for ds in DS) + " |")
    L.append("|---|" + "---|" * (len(DS) * len(NS) + len(DS)))
    meanw = {}
    for w in WCOLS:
        cells = [f3(auc(ds, w, n)) for ds in DS for n in NS]
        for ds in DS:
            vs = [auc(ds, w, n) for n in NS]; vs = [v for v in vs if v is not None]
            meanw[(ds, w)] = float(np.mean(vs)) if len(vs) == 3 else None
        L.append(f"| {WVAL[w]:.2f} | " + " | ".join(cells) + " | " + " | ".join(f3(meanw[(ds, w)]) for ds in DS) + " |")
    L.append("")
    L.append("Best w per dataset by mean AUC over n ∈ {20, 64, 145}:\n")
    for ds in DS:
        c = {w: meanw[(ds, w)] for w in WCOLS if meanw[(ds, w)] is not None}
        if c:
            bw = max(c, key=c.get)
            L.append(f"- {ds}: w = {WVAL[bw]:.2f} (mean AUC {c[bw]:.3f}); equal weight w=0.50 gives {f3(c.get('w050'))}.")
    L.append("")

    # ── 3. cross-dataset weight ──────────────────────────────────────────────
    L.append("## 3. Cross-dataset weight selection (held-out, n=20)\n")
    L.append("For each held-out dataset, w is chosen to maximise mean AUC on the other two datasets at n=20; the held-out AUC at that w is reported "
             "next to the held-out dataset's own best w (oracle).\n")
    L.append("| held-out | w chosen on others | AUC on others (mean) | held-out AUC at chosen w | held-out oracle w | held-out oracle AUC | gap |")
    L.append("|---|---|---|---|---|---|---|")
    for ds in DS:
        others = [d for d in DS if d != ds]
        sc = {w: np.mean([auc(d, w, 20) for d in others]) for w in WCOLS if all(auc(d, w, 20) is not None for d in others)}
        own = {w: auc(ds, w, 20) for w in WCOLS if auc(ds, w, 20) is not None}
        if not sc or not own: L.append(f"| {ds} | — | — | — | — | — | — |"); continue
        bw = max(sc, key=sc.get); ow = max(own, key=own.get)
        L.append(f"| {ds} | {WVAL[bw]:.2f} | {sc[bw]:.3f} | {own[bw]:.3f} | {WVAL[ow]:.2f} | {own[ow]:.3f} | {own[bw]-own[ow]:+.3f} |")
    L.append("")

    # ── 4. LODO acceptance rate ──────────────────────────────────────────────
    L.append("## 4. Leave-one-dataset-out acceptance rate\n")
    L.append("Candidate rates 5% / 16% / 36% of the 400 views (n = 20 / 64 / 145). For each held-out dataset, the rate maximising mean recovery "
             "on the other two is applied to the held-out dataset. Primary method: equal-weight HAGVA (w=0.50).\n")
    def lodo_table(m):
        out = ["| held-out | rate chosen on others | mean recovery on others | held-out AUC | held-out recovery | held-out oracle rate | oracle recovery |",
               "|---|---|---|---|---|---|---|"]
        for ds in DS:
            others = [d for d in DS if d != ds]
            sc = {n: np.mean([rec(auc(d, m, n), d) for d in others]) for n in NS if all(auc(d, m, n) is not None for d in others)}
            own = {n: rec(auc(ds, m, n), ds) for n in NS if auc(ds, m, n) is not None}
            if not sc or not own: out.append(f"| {ds} | — | — | — | — | — | — |"); continue
            bn = max(sc, key=sc.get); on = max(own, key=own.get)
            out.append(f"| {ds} | {RATE[bn]} (n={bn}) | {pct(sc[bn])} | {f3(auc(ds, m, bn))} | {pct(own[bn])} | {RATE[on]} (n={on}) | {pct(own[on])} |")
        return out
    L.append("### 4a. w=0.50 (HAGVA)\n"); L += lodo_table("w050"); L.append("")
    L.append("### 4b. Fixed 5% acceptance (n=20) on all three datasets, w=0.50\n")
    L.append("| dataset | AUC | recovery | vs baseline |"); L.append("|---|---|---|---|")
    for ds in DS:
        v = auc(ds, "w050", 20)
        L.append(f"| {ds} | {f3(v)} | {pct(None if v is None else rec(v, ds))} | {'—' if v is None else f'{v-BASE[ds]:+.3f}'} |")
    L.append("")
    L.append("### 4c. LODO-selected rate for every method (held-out recovery)\n")
    L.append("| method | " + " | ".join(f"{ds}: rate → recovery" for ds in DS) + " | mean held-out recovery |")
    L.append("|---|" + "---|" * (len(DS) + 1))
    for m in WCOLS + IQA + RAND:
        cells = []; hs = []
        for ds in DS:
            others = [d for d in DS if d != ds]
            sc = {n: np.mean([rec(auc(d, m, n), d) for d in others]) for n in NS if all(auc(d, m, n) is not None for d in others)}
            if not sc or any(auc(ds, m, n) is None for n in NS): cells.append("—"); continue
            bn = max(sc, key=sc.get); h = rec(auc(ds, m, bn), ds); hs.append(h)
            cells.append(f"{RATE[bn]} → {pct(h)}")
        L.append(f"| {m} | " + " | ".join(cells) + f" | {pct(np.mean(hs)) if len(hs)==3 else '—'} |")
    L.append("")

    # ── 5. tie report ────────────────────────────────────────────────────────
    L.append("## 5. Tie report (rows where n_passed ≠ n_requested)\n")
    L.append("tau is the midpoint between the n-th and (n+1)-th lowest value; when those two values are equal the strict `<` gate admits fewer than n views. "
             "The tied value is the n-th sorted score of that column in the extended CSV.\n")
    L.append("| dataset | method | n_requested | n_passed | tau | tied value | AUC |"); L.append("|---|---|---|---|---|---|---|")
    scores = {}
    for ds in DS:
        p = f"{P1}/scores/{DSKEY[ds]}_400_extended.csv"
        if os.path.exists(p): scores[ds] = list(csv.DictReader(open(p, newline="")))
    nties = 0
    for r in rows:
        if r["n_passed"] != r["n_requested"]:
            nties += 1; n = int(r["n_requested"]); tied = "—"
            if r["dataset"] in scores and r["method"] in scores[r["dataset"]][0]:
                v = np.sort([float(x[r["method"]]) for x in scores[r["dataset"]]])
                tied = f"{v[n-1]!r}" + ("" if n >= len(v) or v[n-1] == v[n] else f" (≠ next {v[n]!r}: not a tie!)")
            L.append(f"| {r['dataset']} | {r['method']} | {n} | {r['n_passed']} | {r['tau']} | {tied} | {float(r['auc']):.3f} |")
    if nties == 0: L.append("| — | — | — | — | — | — | — |")
    L.append("")
    L.append("Note: the original ablation runs labelled \"n=145\" for SFU geo, SFU sem and StLucia sem also admitted 144 views at the same tau "
             "(see `checkpoint1/prior_tmp_results/*/gate_results.csv`); those anchor AUCs are reproduced exactly.\n")

    # ── 6. fusion-rule table (Phase 1b) ──────────────────────────────────────
    if frows:
        cells9 = [(ds, n) for ds in DS for n in NS]
        w100v = {c: auc(c[0], "w100", c[1]) for c in cells9}
        w100_mean = float(np.mean(list(w100v.values()))) if all(v is not None for v in w100v.values()) else None
        L.append("## 6. Alternative fusion rules (Phase 1b)\n")
        L.append("Pre-registered success criterion (fixed before any 1b run):\n")
        L.append(f"> {CRITERION}\n")
        L.append(f"w100 mean over the 9 cells from this file: {w100_mean:.4f}. \"≥\" is evaluated on the 3-decimal AUCs. "
                 "g = mean_geo_score, s = mean_sem_score (4-decimal CSV values); z-scores use the within-dataset mean/std; "
                 "ranks are 0-based with average ties; veto adds +10 to the score when the other signal exceeds the stated percentile.\n")
        L.append("| rule | " + " | ".join(f"{ds} n={n}" for ds, n in cells9) + " | mean (9, 4 dp) | cells ≥ w100 | criterion |")
        L.append("|---|" + "---|" * (len(cells9) + 3))
        verdict = {}
        for m in FUSION + ["w050", "w100"]:
            vals = [auc(ds, m, n) for ds, n in cells9]
            if any(v is None for v in vals) or w100_mean is None:
                L.append(f"| {FUSION_LABEL[m]} | " + " | ".join(f3(v) for v in vals) + " | — | — | incomplete |"); continue
            ge = sum(1 for (c, v) in zip(cells9, vals) if v >= w100v[c] - 1e-9)
            mean_m = float(np.mean(vals))
            ok = (ge >= 7) and (mean_m >= w100_mean - 1e-9)
            verdict[m] = ok
            cells = []
            for c, v in zip(cells9, vals):
                cells.append(f"**{v:.3f}**" if v > w100v[c] + 1e-9 else (f"{v:.3f}" if v >= w100v[c] - 1e-9 else f"<sub>{v:.3f}</sub>"))
            tag = "reference" if m in ("w050", "w100") else ("**PASS**" if ok else "FAIL")
            L.append(f"| {FUSION_LABEL[m]} | " + " | ".join(cells) + f" | {mean_m:.4f} | {ge}/9 | {tag} |")
        L.append("")
        L.append("Bold = strictly above w100 in that cell; plain = equal; small = below. "
                 f"Rules passing the criterion: {', '.join(m for m in FUSION if verdict.get(m)) or 'none'}.\n")

        # ── 7. how often the veto fires inside the semantic top-n ─────────────
        L.append("## 7. Veto activity inside the semantic top-n\n")
        L.append("For each veto column: the number of views that w100 alone (sem-only, midpoint tau, strict `<`) accepts at n, "
                 "but which the veto rejects (value ≥ 10). Zero means the veto column accepts exactly the same set as w100 at that n.\n")
        L.append("| veto column | " + " | ".join(f"{ds} n={n}" for ds, n in cells9) + " |")
        L.append("|---|" + "---|" * len(cells9))
        def topset(ds, col, n):
            v = np.array([float(r[col]) for r in scores[ds]])
            srt = np.sort(v); tau = (srt[n-1] + srt[n]) / 2.0
            return v < tau
        for m in VETO:
            cells = []
            for ds, n in cells9:
                if ds not in scores or m not in scores[ds][0]: cells.append("—"); continue
                top = topset(ds, "mean_sem_score", n)
                vv = np.array([float(r[m]) for r in scores[ds]])
                cells.append(f"{int(((vv >= 10) & top).sum())} of {int(top.sum())}")
            L.append(f"| {m} | " + " | ".join(cells) + " |")
        # supplementary: geo_veto75 against its own base ranking (geo-only)
        cells = []
        for ds, n in cells9:
            if ds not in scores: cells.append("—"); continue
            top = topset(ds, "mean_geo_score", n); vv = np.array([float(r["geo_veto75"]) for r in scores[ds]])
            cells.append(f"{int(((vv >= 10) & top).sum())} of {int(top.sum())}")
        L.append(f"| *geo_veto75 vs geo-only (w000) top-n, for context* | " + " | ".join(cells) + " |")
        L.append("")
        L.append("Un-vetoed view counts per dataset (value < 10), from `provenance.json`: "
                 + "; ".join(f"{ds}: " + ", ".join(f"{k}={v}" for k, v in prov_f["per_dataset"][DSKEY[ds]]["unvetoed_counts"].items()) for ds in DS if (prov_f := json.load(open(f"{P1}/scores/provenance.json")).get("fusion")) and DSKEY[ds] in prov_f["per_dataset"]) + ".\n")

    # ── 8. Phase 1c: fusion with the corrected geometric signal ─────────────────
    frows_c = [r for r in rows if r["method"] in FCOLS]
    if frows_c:
        cells9 = [(ds, n) for ds in DS for n in NS]
        w100v = {c: auc(c[0], "w100", c[1]) for c in cells9}
        w100_mean = float(np.mean(list(w100v.values()))) if all(v is not None for v in w100v.values()) else None
        L.append("## 8. Fusion with the corrected geometric signal (Phase 1c)\n")
        L.append("Module A's Signal 1 cropped GenWarp's 512×512 output instead of resizing it (`phase2/mask_root_cause.md`); `geo_fixed` is the same "
                 "formula recomputed from correctly resized warp-only renders and masks (`phase2/scripts/recompute_geo.py`). Every `f` column is the Phase 1/1b "
                 "column re-derived with `g = geo_fixed`; w100 (semantic only) is unchanged. Δ = AUC(f column) − AUC(old counterpart) in the same cell.\n")
        L.append("Pre-registered success criterion (unchanged from Phase 1b):\n")
        L.append(f"> {CRITERION}\n")
        L.append(f"w100 mean over the 9 cells from this file: {w100_mean:.4f}.\n")
        L.append("| rule | " + " | ".join(f"{ds} n={n}" for ds, n in cells9) + " | mean (9, 4 dp) | mean Δ vs old | cells ≥ w100 | top-n set == w100 | criterion |")
        L.append("|---|" + "---|" * (len(cells9) + 5))
        verdict_c = {}
        def same_set_count(m):
            """cells where the rule's accepted set (midpoint tau, strict <) is identical to w100's (sem) accepted set."""
            k = 0
            for ds, n in cells9:
                if ds not in scores or m not in scores[ds][0]: continue
                v = np.array([float(r[m]) for r in scores[ds]]); sv = np.array([float(r["mean_sem_score"]) for r in scores[ds]])
                tv = np.sort(v); ts = np.sort(sv)
                if set(np.where(v < (tv[n-1]+tv[n])/2)[0]) == set(np.where(sv < (ts[n-1]+ts[n])/2)[0]): k += 1
            return k
        def frow(m, old, label):
            vals = [auc(ds, m, n) for ds, n in cells9]; olds = [auc(ds, old, n) for ds, n in cells9] if old else [None] * 9
            if any(v is None for v in vals) or w100_mean is None:
                L.append(f"| {label} | " + " | ".join(f3(v) for v in vals) + " | — | — | — | — | incomplete |"); return
            ge = sum(1 for c, v in zip(cells9, vals) if v >= w100v[c] - 1e-9); mean_m = float(np.mean(vals))
            ok = (ge >= 7) and (mean_m >= w100_mean - 1e-9); verdict_c[m] = ok
            cells = []
            for c, v, o in zip(cells9, vals, olds):
                core = f"**{v:.3f}**" if v > w100v[c] + 1e-9 else (f"{v:.3f}" if v >= w100v[c] - 1e-9 else f"<sub>{v:.3f}</sub>")
                cells.append(core + (f" ({v-o:+.3f})" if o is not None else ""))
            dm = f"{np.mean([v-o for v, o in zip(vals, olds)]):+.4f}" if all(o is not None for o in olds) else "—"
            same = same_set_count(m)
            tag = "reference" if m in ("w050", "w100") else ("**PASS**" + (" (degenerate: identical to w100)" if same == 9 else "") if ok else "FAIL")
            L.append(f"| {label} | " + " | ".join(cells) + f" | {mean_m:.4f} | {dm} | {ge}/9 | {same}/9 | {tag} |")
        for m in FCOLS:
            old = m[:-1]; frow(m, old, FLABEL[m] if m in FLABEL else FUSION_LABEL[old] + " (f)")
        frow("w050", None, "w=0.50 OLD geo (reference)"); frow("w100", None, "w=1.00 sem-only (reference)")
        L.append("")
        L.append(f"Bold = strictly above w100 in that cell; plain = equal; small = below. \"top-n set == w100\" counts the cells in which the rule accepts exactly the same "
                 f"views as semantic-only (then the AUCs are necessarily equal). Rules passing the criterion with the corrected signal: "
                 f"{', '.join(m + (' (degenerate)' if same_set_count(m) == 9 else '') for m in FCOLS if verdict_c.get(m)) or 'none'}.\n")
        # degeneracy of max(g, s) after the fix: how often is geo_fixed >= sem?
        gs = []
        for ds in DS:
            if ds in scores and "geo_fixed" in scores[ds][0]:
                g = np.array([float(r["geo_fixed"]) for r in scores[ds]]); sv = np.array([float(r["mean_sem_score"]) for r in scores[ds]])
                go = np.array([float(r["mean_geo_score"]) for r in scores[ds]])
                gs.append(f"{ds}: geo_fixed ≥ sem in {int((g >= sv).sum())}/400 views (old geo ≥ sem in {int((go >= sv).sum())}/400)")
        if gs: L.append("maxgs = max(g, s) reduces to the semantic score wherever g < s; " + "; ".join(gs) + ".\n")
        # veto activity for the f vetoes
        L.append("### 8b. Veto activity inside the semantic top-n (f vetoes)\n")
        L.append("Number of views that w100 alone accepts at n but the corrected-geo veto rejects (value ≥ 10).\n")
        L.append("| veto column | " + " | ".join(f"{ds} n={n}" for ds, n in cells9) + " |"); L.append("|---|" + "---|" * len(cells9))
        def topset2(ds, col, n):
            v = np.array([float(r[col]) for r in scores[ds]]); srt = np.sort(v); return v < (srt[n-1] + srt[n]) / 2.0
        for m in ["sem_veto50f", "sem_veto75f", "sem_veto90f", "geo_veto75f"]:
            cells = []
            for ds, n in cells9:
                if ds not in scores or m not in scores[ds][0]: cells.append("—"); continue
                top = topset2(ds, "mean_sem_score", n); vv = np.array([float(r[m]) for r in scores[ds]])
                cells.append(f"{int(((vv >= 10) & top).sum())} of {int(top.sum())}")
            L.append(f"| {m} | " + " | ".join(cells) + " |")
        cells = []
        for ds, n in cells9:
            if ds not in scores or "geo_fixed" not in scores[ds][0]: cells.append("—"); continue
            top = topset2(ds, "geo_fixed", n); vv = np.array([float(r["geo_veto75f"]) for r in scores[ds]])
            cells.append(f"{int(((vv >= 10) & top).sum())} of {int(top.sum())}")
        L.append("| *geo_veto75f vs fixed-geo-only (w000f) top-n, for context* | " + " | ".join(cells) + " |")
        L.append("")
        # ── 9. old vs fixed side by side ────────────────────────────────────────
        L.append("## 9. Old vs corrected geometric signal, side by side\n")
        L.append("| cell | w000 (old geo) | w000f (fixed geo) | Δ | w050 (old) | w050f (fixed) | Δ | w100 (sem) |"); L.append("|---|---|---|---|---|---|---|---|")
        acc = {k: [] for k in ("w000", "w000f", "w050", "w050f", "w100")}
        for ds, n in cells9:
            v = {k: auc(ds, k, n) for k in acc}
            for k in acc: acc[k].append(v[k])
            d0 = "—" if None in (v["w000"], v["w000f"]) else f"{v['w000f']-v['w000']:+.3f}"
            d5 = "—" if None in (v["w050"], v["w050f"]) else f"{v['w050f']-v['w050']:+.3f}"
            L.append(f"| {ds} n={n} | {f3(v['w000'])} | {f3(v['w000f'])} | {d0} | {f3(v['w050'])} | {f3(v['w050f'])} | {d5} | {f3(v['w100'])} |")
        if all(None not in vs for vs in acc.values()):
            m = {k: float(np.mean(vs)) for k, vs in acc.items()}
            L.append(f"| **mean (9)** | {m['w000']:.4f} | {m['w000f']:.4f} | {m['w000f']-m['w000']:+.4f} | {m['w050']:.4f} | {m['w050f']:.4f} | {m['w050f']-m['w050']:+.4f} | {m['w100']:.4f} |")
        L.append("")
        pp = json.load(open(f"{P1}/scores/provenance.json")).get("geofix", {}).get("geo_fixed_recomputation", {}).get("per_dataset", {})
        if pp:
            L.append("Old vs fixed geometric score over the 400 views (from `provenance.json`): " + "; ".join(
                f"{ds}: Spearman {v['spearman_old_fixed']:.3f}, mean {v['old_mean']:.3f} → {v['fixed_mean']:.3f}, hole fraction {v['holefrac_old_mean']:.3f} → {v['holefrac_fixed_mean']:.3f} (dead band {v['deadband_frac_old']:.3f})"
                for ds, v in pp.items()) + ".\n")

    # ── footnote: SFU w050 n=20 ─────────────────────────────────────────────
    ver = f"{P1}/checkpoint2_verification.csv"
    fn = None
    if os.path.exists(ver):
        for v in csv.DictReader(open(ver, newline="")):
            if (v["dataset"], v["method"], v["n_requested"]) == ("SFU", "mean_combined_score", "20"): fn = v
    fr = R.get(("SFU", "w050", 20))
    L.append("## Footnotes\n")
    L.append("[^sfu20]: **SFU, w=0.50, n=20.** The stored `mean_combined_score` column in the original CSV is rounded to 4 decimals and is not "
             "exactly the mean of the stored (also rounded) geo and sem columns; the formula column `w050 = 0.5·sem + 0.5·geo` differs from it by ≤ 5e-5. "
             "Rankings are identical at every anchor cell, but at SFU n=20 the formula value ties at the 20th/21st view, so the midpoint tau admits 19 views. "
             + (f"Formula column (this sweep): {fr['n_passed']} views, AUC {float(fr['auc']):.3f}. " if fr else "")
             + (f"Stored column (verification run, `checkpoint2_verification.csv`): {fn['n_passed']} views, AUC {float(fn['auc']):.3f} — equal to the ablation anchor (0.731). " if fn else "")
             + "Table 1 reports the formula column throughout.")
    L.append("")

    # ── provenance pointer ──────────────────────────────────────────────────
    pp = f"{P1}/scores/provenance.json"
    if os.path.exists(pp):
        prov = json.load(open(pp))
        L.append("## Provenance (see `scores/provenance.json`)\n")
        iqa = prov.get("iqa", {}); dino = prov.get("dino_global", {})
        L.append(f"- pyiqa {iqa.get('pyiqa_version','?')} (BRISQUE, NIQE, CLIP-IQA on the synthesized view at native resolution); clipiqa_neg = −CLIP-IQA.")
        L.append(f"- DINOv2 ViT-S/14 via torch.hub commit `{dino.get('hub_commit','?')}`, 448×448, ImageNet norm, CLS cosine; weights sha256 `{(dino.get('weights_sha256') or '?')[:16]}…`.")
        L.append(f"- Random: numpy default_rng(seed).permutation(400), seeds 0–4. Weights: w·sem + (1−w)·geo on the 4-decimal CSV values.")
        L.append(f"- torch {prov.get('env',{}).get('torch','?')}; gate runs: `/home/newview/miniconda3/envs/vprtutorial/bin/python3`.")
        L.append("")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    open(a.out, "w").write("\n".join(L))
    print(f"wrote {a.out}  ({len(rows)} rows, {len(missing)} missing cells, {nties} tie rows, {len(fails)} anchor mismatches)")

if __name__ == "__main__":
    main()
