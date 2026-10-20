#!/usr/bin/env python3
"""Summarize measured MixVPR outputs with the existing CosPlace table routines."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import warnings
import numpy as np
from scipy import stats
import summarize_cosplace as tables


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project', default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--cosplace-run-id', default='cosplace_20261020')
    args = ap.parse_args()
    root = Path(args.project).resolve()
    output = root/'phase3a/mixvpr'/args.run_id
    assert (output/'completion.json').is_file(), 'Full GPU workload has not completed'
    with (output/'commands.jsonl').open('a') as f:
        f.write(json.dumps({'time':datetime.now(timezone.utc).isoformat(), 'cwd':str(Path.cwd()), 'argv':[sys.executable]+sys.argv})+'\n')
    gates = tables.read(output/'mixvpr_gate_results.csv')
    causal = tables.read(output/'mixvpr_causal_results.csv')
    assert len(gates)==186 and len(causal)==196
    old = []
    for phase, filename, descriptor, pool in [
        ('phase1','phase1_results.csv','PatchNetVLAD','low'),
        ('phase3a','phase3a_gate_results.csv','EigenPlaces','low'),
        ('phase3b','phase3b_gate_results.csv',None,'wide')]:
        old += [{**r,'descriptor':descriptor or r['descriptor'],'pool':pool}
                for r in tables.read(root/phase/filename)]
    oldcausal = []
    for phase, filename, descriptor, pool in [
        ('phase2','phase2_results.csv','PatchNetVLAD','low'),
        ('phase3a','phase3a_causal_results.csv','EigenPlaces','low'),
        ('phase3b','phase3b_causal_results.csv',None,'wide')]:
        oldcausal += [{**r,'pool':pool} for r in tables.read(root/phase/filename)
                      if not descriptor or r['descriptor']==descriptor]
    cos = root/'phase3a/cosplace'/args.cosplace_run_id
    old += tables.read(cos/'cosplace_gate_results.csv')
    oldcausal += tables.read(cos/'cosplace_causal_results.csv')
    tables.DESCRIPTORS = ['PatchNetVLAD','EigenPlaces','CosPlace','MixVPR']
    main_rows = tables.comparison(old+gates)
    contrasts = tables.contrasts(oldcausal+causal)
    mixmain = [r for r in main_rows if r['descriptor']=='MixVPR']
    mixcontrasts = [r for r in contrasts if r['descriptor']=='MixVPR']
    assert len(main_rows)==72 and len(contrasts)==48
    for row in mixcontrasts:
        a,b = row['contrast'].split('-')
        def values(condition):
            records = [r for r in causal if r['pool']==row['pool'] and r['dataset']==row['dataset'] and r['condition']==condition]
            return np.array([float(r['auc_full_precision']) for r in sorted(records,key=lambda r:int(r['seed']))])
        av,bv = values(a),values(b)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore'); tf,pf=stats.ttest_rel(av,bv)
        row.update(mean_delta_full_precision=float((av-bv).mean()), t_full_precision=float(tf), p_full_precision=float(pf))
    for ds in tables.DATASETS:
        for seed in range(5):
            pairs = [r for r in causal if r['dataset']==ds and r['condition']=='C2' and int(r['seed'])==seed]
            assert len(pairs)==2 and abs(float(pairs[0]['auc_full_precision'])-float(pairs[1]['auc_full_precision']))<1e-12
    tables.write(output/'main_comparison.csv',mixmain)
    tables.write(output/'main_comparison_n64.csv',[r for r in mixmain if r['n_requested']==64])
    tables.write(output/'budget_wise_results.csv',[r for r in gates if int(r['n_requested']) in [20,64,145]])
    tables.write(output/'causal_comparison.csv',mixcontrasts)
    tables.write(output/'descriptor_main_comparison.csv',main_rows)
    tables.write(output/'descriptor_causal_comparison.csv',contrasts)
    tables.write(output/'selection_count_exceptions.csv',[r for r in gates if int(r['n_passed'])!=int(r['n_requested'])])
    lines = ['# MixVPR: completed Phase 1–3b experiments','',
        'All MixVPR AUCs were measured on the GPU using the official pretrained ResNet-50/4096 checkpoint. Real views, 2,400 existing generated candidates, DINOv2/IQA scores, five random rankings, five causal draws, and validated query-side GT were reused. No image-generation experiments were run.','',
        '## Main comparison at n=64','',
        '| Regime | Dataset | Real only | Naive 400 | Semantic | Random mean ± SD | Semantic − random |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in mixmain:
        if r['n_requested']==64:
            lines.append(f"| {r['pool']} | {r['dataset']} | {r['baseline']:.3f} | {r['naive']:.3f} | {r['semantic_auc']:.3f} | {r['random_mean']:.4f} ± {r['random_sd']:.4f} | {r['semantic_minus_random']:+.4f} |")
    lines += ['', 'Baseline AUC is measured independently for each dataset and is shared by the two viewpoint regimes because their original real views are identical. All three budgets appear in `main_comparison.csv`. `main_comparison_n64.csv` contains the six main cells. `budget_wise_results.csv` contains all 180 budget/method cells. The 186-row full gate CSV adds six all-400 checks, which equal naive injection.','',
        '## Five-seed causal comparisons','',
        '| Regime | Dataset | Contrast | Mean Δ AUC | Paired t-test p | Full-precision Δ | Full-precision p |',
        '|---|---|---|---:|---:|---:|---:|']
    for r in mixcontrasts:
        lines.append(f"| {r['pool']} | {r['dataset']} | {r['contrast']} | {r['mean_delta_auc']:+.4f} | {r['p_two_sided']:.4g} | {r['mean_delta_full_precision']:+.6f} | {r['p_full_precision']:.4g} |")
    lines += ['', 'The primary statistics use the recorded three-decimal run AUCs for parity with prior phases; unrounded AUCs and paired statistics are also retained. Tests are two-sided, paired by the five original draw seeds, and unadjusted for multiple comparisons. Confidence intervals and seed differences are in `causal_comparison.csv`.','',
        'The full causal CSV contains 196 rows: six real-only C0 cells; C1 (existing synthesis), C2 (source reinjection), C4 (black-hole geometric warp), C5 (Telea geometric warp), C6 (black-hole GenWarp), and C7 (Telea GenWarp) at five seeds in each of six cells; plus five-seed C2b (real day-left viewpoint) for GardensPoint in each regime. Each non-C0 trial injects 50 views. C2b is GardensPoint-only under the established protocol. C3 is not part of this protocol.','',
        '## Comparison across four descriptors','']
    for descriptor in tables.DESCRIPTORS:
        wide = [r for r in main_rows if r['descriptor']==descriptor and r['pool']=='wide' and r['n_requested'] in [64,145]]
        above = sum(r['semantic_minus_baseline']>0 for r in main_rows if r['descriptor']==descriptor and r['n_requested']==64)
        lines.append(f"- {descriptor}: semantic exceeds the five-seed random mean in {sum(r['semantic_minus_random']>0 for r in wide)}/6 wide cells at n=64/145; semantic n=64 exceeds real-only baseline in {above}/6 dataset–regime cells.")
    lines += ['', 'These descriptive counts do not establish significance or universal improvement. `descriptor_main_comparison.csv` and `descriptor_causal_comparison.csv` retain the actual previously recorded PatchNetVLAD, EigenPlaces, and CosPlace values alongside the new measurements. Different descriptors have independent baseline AUCs.', '',
        'MixVPR semantic selection at n=64 exceeds the random mean in all six cells and exceeds its own baseline in GardensPoint narrow/wide and StLucia narrow. It remains below its baseline in SFU narrow/wide and StLucia wide. Relative to the earlier descriptors, MixVPR has the strongest recorded StLucia real-only baseline (0.878), while SFU remains weaker than PatchNetVLAD and EigenPlaces.', '',
        'MixVPR C2−C1 is positive in all six cells; the GardensPoint narrow contrast is not significant at the unadjusted 0.05 level. C7−C5 is positive on GardensPoint in both regimes (recorded Δ=+0.0102/+0.0224, p=0.01156/0.0006853), whereas EigenPlaces and CosPlace have negative means there. PatchNetVLAD has a small positive, nonsignificant GardensPoint wide mean. This does not support a universal diffusion penalty across descriptors and datasets. C7−C5 is negative for MixVPR on SFU and StLucia.', '',
        'Precision sensitivity: StLucia wide C7−C5 has p=0.0560 using the legacy three-decimal AUCs and p=0.0474 using full-precision AUCs. Both are reported rather than treating this borderline contrast as a robust significance claim. All tests remain unadjusted for multiple comparisons.', '',
        '## Methods, counts, and validation','',
        '- Gate score columns: `w100` = DINOv2 semantic; `w000f` = corrected geometry only; `w050f` = equal-weight corrected geometry/semantic fusion; `rand_s0`–`rand_s4` = original random rankings; `brisque` = BRISQUE; `clipiqa_neg` = negative CLIP-IQA. Lower scores rank first. No scores were recomputed.',
        '- Admission uses the existing strict score < midpoint(nth,(n+1)th) rule. Actual admitted counts are in every row. Six inherited threshold-tie exceptions are listed in `selection_count_exceptions.csv`; these policies match CosPlace.',
        '- Narrow: elevation 1–5 degrees and radius 0.10/0.15/0.20; wide: elevation 8/12/16 degrees and radius 0.30/0.40/0.50; azimuth 0. Existing candidate files determine the exact sampled poses.',
        '- MixVPR preprocessing: RGB, PIL bicubic resize to 320 × 320, ToTensor, ImageNet normalization. Batch size 1, eval mode, float32. Cosine similarity uses normalized 4096-dimensional descriptors.',
        '- PR/AUC: unchanged VPR `createPR(matching="multi", n_thresh=100)` followed by `numpy.trapz(P,R)`. Each selection computes its own similarity matrix and PR thresholds.',
        '- Six image checks execute the official upstream `load_image` and `VPRModel.forward` source independently of the adapter. All checkpoint tensors are checked after strict loading. Training-only Lightning and metric-learning dependencies are unnecessary for this inference adapter.',
        '- Three real-only baselines and one 20-view injection were validated against the private VPR demo before full execution. SFU narrow C5 and StLucia wide C7 seed-0 replays validate the other dataset-specific injection paths.',
        '- The framework is copied privately. Only the private demo adds the MixVPR choice and adapter. Only its private loader/GT are staged for injection. HAGVA_REF_GT is explicitly unset, excluding Phase 5/5b.',
        '- Descriptor cache fingerprints include descriptor identity, checkpoint hash, adapter hash, upstream source hashes, preprocessing, ordered input paths and all image SHA-256 hashes. A mismatch raises an error. Each validation similarity matrix has a separate directory and is never reused for another evaluation.',
        '- Original scripts/results, including completed CosPlace files, are hashed before and after the workload. The input hash manifest and verification report document preservation.',
        '- Official source: https://github.com/amaralibey/MixVPR at commit 4043915cef24818003ece1a8112bc8a24e69abe0. Official checkpoint: https://drive.google.com/file/d/1vuz3PvnR7vxnDDLQrdHJaOA04SQrtk5L/view. Checkpoint hash and environment are in `model_provenance.json`.','',
        '## Scripts and commands','',
        'New experiment/analysis scripts: `phase3a/scripts/feature_extractor_mixvpr.py`, `run_mixvpr.py`, `summarize_mixvpr.py`, and `verify_mixvpr.py`. The additional packaging utility is `phase3a/mixvpr/package_results.py`. The runner reuses the established CosPlace implementation; the summarizer imports its table routines without modifying them. Setup logs and commands are included. No previous descriptor scripts/results, manuscript, rebuttal, or GitHub branches were modified.','',
        '```bash',
        f'/home/newview/miniconda3/envs/vprtutorial/bin/python3 -u phase3a/scripts/run_mixvpr.py --run-id {args.run_id} --mode validate',
        f'/home/newview/miniconda3/envs/vprtutorial/bin/python3 -u phase3a/scripts/run_mixvpr.py --run-id {args.run_id} --mode all',
        f'/home/newview/miniconda3/envs/vprtutorial/bin/python3 phase3a/scripts/summarize_mixvpr.py --run-id {args.run_id}',
        f'/home/newview/miniconda3/envs/vprtutorial/bin/python3 phase3a/scripts/verify_mixvpr.py --run-id {args.run_id}',
        '```','']
    (output/'SUMMARY.md').write_text('\n'.join(lines))
    print('Wrote all MixVPR tables and four-descriptor comparisons:',output)
    for row in mixmain:
        if row['n_requested']==64: print(row)


if __name__=='__main__': main()
