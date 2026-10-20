#!/usr/bin/env python3
"""Summarize measured CosPlace runs and compare existing descriptor results."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import warnings

import numpy as np
from scipy import stats


DESCRIPTORS = ['PatchNetVLAD','EigenPlaces','CosPlace']
DATASETS = ['GardensPoint','SFU','StLucia']
BUDGETS = [20,64,145]


def read(path):
    with path.open(newline='') as f: return list(csv.DictReader(f))


def write(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def f3(value): return f'{float(value):.3f}'
def fp(value): return 'undefined' if not np.isfinite(float(value)) else f'{float(value):.4g}'


def comparison(gates):
    result=[]
    for descriptor in DESCRIPTORS:
        for pool in ['low','wide']:
            for dataset in DATASETS:
                for n in BUDGETS:
                    rows=[r for r in gates if r['descriptor']==descriptor and r['pool']==pool and r['dataset']==dataset and int(r['n_requested'])==n]
                    sem=[r for r in rows if r['method']=='w100']; random=sorted([r for r in rows if r['method'].startswith('rand_s')],key=lambda r:r['method'])
                    assert len(sem)==1 and len(random)==5,(descriptor,pool,dataset,n)
                    sem=sem[0]; values=np.array([float(r['auc']) for r in random])
                    result.append({'descriptor':descriptor,'pool':pool,'dataset':dataset,'n_requested':n,
                                   'n_semantic_passed':int(sem['n_passed']),
                                   'n_random_min':min(int(r['n_passed']) for r in random),
                                   'n_random_max':max(int(r['n_passed']) for r in random),
                                   'baseline':float(sem['baseline']),'naive':float(sem['naive']),
                                   'semantic_auc':float(sem['auc']),'random_mean':float(values.mean()),
                                   'random_sd':float(values.std(ddof=1)),
                                   'semantic_minus_random':float(sem['auc'])-float(values.mean()),
                                   'semantic_minus_baseline':float(sem['auc'])-float(sem['baseline'])})
    return result


def contrasts(causal):
    result=[]
    for descriptor in DESCRIPTORS:
        for pool in ['low','wide']:
            for dataset in DATASETS:
                rows=[r for r in causal if r['descriptor']==descriptor and r['pool']==pool and r['dataset']==dataset and r['condition']!='C0']
                for a,b in [('C2','C1'),('C7','C5')]:
                    aa={int(r['seed']):r for r in rows if r['condition']==a}
                    bb={int(r['seed']):r for r in rows if r['condition']==b}
                    assert set(aa)==set(bb)==set(range(5)),(descriptor,pool,dataset,a,b)
                    av=np.array([float(aa[s]['auc']) for s in range(5)])
                    bv=np.array([float(bb[s]['auc']) for s in range(5)])
                    d=av-bv
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore'); t,p=stats.ttest_rel(av,bv)
                    se=d.std(ddof=1)/np.sqrt(5); half=stats.t.ppf(.975,df=4)*se
                    row={'descriptor':descriptor,'pool':pool,'dataset':dataset,'contrast':a+'-'+b,
                         'n_pairs':5,'auc_basis':'recorded_3dp','mean_delta_auc':float(d.mean()),
                         'sd_delta_auc':float(d.std(ddof=1)),'ci95_low':float(d.mean()-half),
                         'ci95_high':float(d.mean()+half),'t_statistic':float(t),'p_two_sided':float(p),
                         'positive_seeds':int((d>0).sum()),'negative_seeds':int((d<0).sum()),
                         **{f'delta_seed{s}':float(d[s]) for s in range(5)}}
                    if descriptor=='CosPlace':
                        af=np.array([float(aa[s]['auc_full_precision']) for s in range(5)])
                        bf=np.array([float(bb[s]['auc_full_precision']) for s in range(5)])
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore'); tf,pf=stats.ttest_rel(af,bf)
                        row.update(mean_delta_full_precision=float((af-bf).mean()),
                                   t_full_precision=float(tf),p_full_precision=float(pf))
                    result.append(row)
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project',default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument('--run-id',required=True);a=ap.parse_args()
    root=Path(a.project).resolve(); output=root/'phase3a/cosplace'/a.run_id
    assert (output/'completion.json').exists(),'Full run has not completed'
    with (output/'commands.jsonl').open('a') as f:
        f.write(json.dumps({'time':datetime.now(timezone.utc).isoformat(),'cwd':str(Path.cwd()),'argv':[sys.executable]+sys.argv})+'\n')
    gates=read(output/'cosplace_gate_results.csv'); causal=read(output/'cosplace_causal_results.csv')
    assert len(gates)==186 and len(causal)==126
    old=[]
    for phase,file,descriptor,pool in [('phase1','phase1_results.csv','PatchNetVLAD','low'),
                                       ('phase3a','phase3a_gate_results.csv','EigenPlaces','low'),
                                       ('phase3b','phase3b_gate_results.csv',None,'wide')]:
        for r in read(root/phase/file): old.append({**r,'descriptor':descriptor or r['descriptor'],'pool':pool})
    oldcausal=[]
    for phase,file,descriptor,pool in [('phase2','phase2_results.csv','PatchNetVLAD','low'),
                                       ('phase3a','phase3a_causal_results.csv','EigenPlaces','low'),
                                       ('phase3b','phase3b_causal_results.csv',None,'wide')]:
        for r in read(root/phase/file):
            if descriptor and r['descriptor']!=descriptor: continue
            oldcausal.append({**r,'pool':pool})
    main_rows=comparison(old+gates); causal_rows=contrasts(oldcausal+causal)
    cosmain=[r for r in main_rows if r['descriptor']=='CosPlace']
    coscausal=[r for r in causal_rows if r['descriptor']=='CosPlace']
    assert len(cosmain)==18 and len(coscausal)==12 and len(main_rows)==54 and len(causal_rows)==36
    for ds in DATASETS:
        for seed in range(5):
            pair=[r for r in causal if r['dataset']==ds and r['condition']=='C2' and int(r['seed'])==seed]
            assert len(pair)==2 and abs(float(pair[0]['auc_full_precision'])-float(pair[1]['auc_full_precision']))<1e-12
    write(output/'main_comparison.csv',cosmain)
    write(output/'budget_wise_results.csv',[r for r in gates if int(r['n_requested']) in BUDGETS])
    write(output/'causal_comparison.csv',coscausal)
    write(output/'descriptor_main_comparison.csv',main_rows)
    write(output/'descriptor_causal_comparison.csv',causal_rows)
    count_exceptions=[r for r in gates if int(r['n_passed'])!=int(r['n_requested'])]
    write(output/'selection_count_exceptions.csv',count_exceptions)
    lines=['# CosPlace: Phase 1–3b extension','',
           'All CosPlace values below were measured using the existing ResNet-50/2048',
           'extractor, original query-side GT handling, existing candidate pools and',
           'unchanged VPR multi-match PR/AUC code. Candidate views and DINO/IQA',
           'ranking scores were reused; new CosPlace descriptors and AUCs were measured.',
           'PatchNetVLAD/EigenPlaces values come from the existing recorded result CSVs.','',
           '## Main comparison at 64 views','',
           '| Pool | Dataset | Baseline | Naive 400 | Semantic 64 | Random mean ± SD | Semantic − random |',
           '|---|---|---:|---:|---:|---:|---:|']
    for r in cosmain:
        if r['n_requested']==64:
            lines.append(f"| {r['pool']} | {r['dataset']} | {r['baseline']:.3f} | {r['naive']:.3f} | {r['semantic_auc']:.3f} | {r['random_mean']:.4f} ± {r['random_sd']:.4f} | {r['semantic_minus_random']:+.4f} |")
    lines+=['','All budgets are in `main_comparison.csv`; individual random seeds, IQA,',
            'corrected geometry and fusion are in `budget_wise_results.csv`. The full',
            'gate table also includes the six all-400 random selections, which equal',
            'naive injection. `n_passed` records strict-cutoff ties; nominal budgets',
            'are not silently converted to a different selection policy.','',
            '## Five-seed paired causal comparisons','',
            '| Pool | Dataset | Contrast | Mean ΔAUC | Paired t | Two-sided p | Full-precision p |',
            '|---|---|---|---:|---:|---:|---:|']
    for r in coscausal:
        lines.append(f"| {r['pool']} | {r['dataset']} | {r['contrast']} | {r['mean_delta_auc']:+.4f} | {fp(r['t_statistic'])} | {fp(r['p_two_sided'])} | {fp(r['p_full_precision'])} |")
    lines+=['','Primary contrasts and p-values use each run’s three-decimal AUC, matching',
            'the previous phase tables. Full-precision CosPlace AUCs and paired tests',
            'are also retained. Tests are two-sided, paired over seeds 0–4, with no',
            'multiple-comparison correction. Five seeds provide limited inferential',
            'precision; a positive C2−C1 means source reinjection outperforms naive',
            'synthesis, while a negative C7−C5 means generated visible content performs',
            'worse than warp-only under the constructed matched fill. C5/C7 holes',
            'match before separate JPEG encoding, not necessarily after decoding.','',
            '## Comparison with previous descriptors','']
    for descriptor in DESCRIPTORS:
        wide=[r for r in main_rows if r['descriptor']==descriptor and r['pool']=='wide' and r['n_requested'] in [64,145]]
        positive=sum(r['semantic_minus_random']>0 for r in wide)
        n64=[r for r in main_rows if r['descriptor']==descriptor and r['n_requested']==64]
        above=sum(r['semantic_minus_baseline']>0 for r in n64)
        lines.append(f'- {descriptor}: semantic ranking exceeds the random mean in {positive}/6 wide comparisons at n=64/145; semantic n=64 exceeds the no-injection baseline in {above}/6 dataset–regime cells.')
    lines+=['','These counts describe measured effects and do not imply significance or',
            'universal improvement. `descriptor_main_comparison.csv` and',
            '`descriptor_causal_comparison.csv` contain the complete comparisons.','',
            '## Protocol and verification','',
            '- Model source: [official CosPlace pretrained models](https://github.com/gmberton/CosPlace#trained-models), ResNet-50 with 2048-dimensional descriptors.',
            '- Pretrained weights are checked tensor-for-tensor after loading and their SHA-256 is recorded in `model_provenance.json`.',
            '- Initial validation compares three real-only baselines and a 20-view injection against the original demo. Similarity matrices, GT and printed AUC must agree.',
            '- Full-run replay checks compare an SFU narrow C5 seed and a StLucia wide C7 seed against the demo.',
            '- All 2,400 injected hard and soft GT columns are checked against their source columns; legitimate zero columns are retained.',
            '- Original scripts/results and the external framework are hashed before and after; changes cause failure.',
            '- Descriptor caching retains the existing RGB/Resize(480)/ImageNet-normalization, batch-size-one, eval-mode extraction and float32 cosine similarities.',
            '- Existing scores, random ranks, five causal draws, three requested budgets and strict midpoint cutoffs are reused without rescoring.',
            '- Commands are recorded in `commands.jsonl`.','',
            '## Scripts and execution','',
            'New scripts: `phase3a/scripts/run_cosplace.py` and',
            '`phase3a/scripts/summarize_cosplace.py`. Existing experimental scripts,',
            'results, datasets, generation code, manuscript and rebuttal are unchanged.',
            'Framework code is copied into this run’s private `framework/` folder;',
            'only that loader copy is temporarily patched for injection.','',
            '```bash',
            f'/home/newview/miniconda3/envs/vprtutorial/bin/python3 -u phase3a/scripts/run_cosplace.py --run-id {a.run_id} --mode validate',
            f'/home/newview/miniconda3/envs/vprtutorial/bin/python3 -u phase3a/scripts/run_cosplace.py --run-id {a.run_id} --mode all',
            f'/home/newview/miniconda3/envs/vprtutorial/bin/python3 phase3a/scripts/summarize_cosplace.py --run-id {a.run_id}',
            '```','']
    (output/'SUMMARY.md').write_text('\n'.join(lines))
    print('Wrote measured main, causal, budget and descriptor-comparison tables to',output)
    for r in cosmain:
        if r['n_requested']==64: print(r['pool'],r['dataset'],'baseline',f3(r['baseline']),'naive',f3(r['naive']),'semantic',f3(r['semantic_auc']),'random',f"{r['random_mean']:.4f}")


if __name__=='__main__': main()
