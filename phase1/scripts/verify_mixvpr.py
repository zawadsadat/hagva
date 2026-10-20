#!/usr/bin/env python3
"""Audit completed MixVPR measurements, original cutoffs, cache provenance and preservation."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy import stats
from run_mixvpr import DATASETS, METHODS, BUDGETS, sha, read_csv, write_json


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project', default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument('--run-id', required=True)
    args = ap.parse_args()
    root = Path(args.project).resolve()
    out = root/'phase3a/mixvpr'/args.run_id
    commands = out/'commands.jsonl'
    with commands.open('a') as f:
        f.write(json.dumps({'time':datetime.now(timezone.utc).isoformat(), 'cwd':str(Path.cwd()), 'argv':[sys.executable]+sys.argv})+'\n')
    assert (out/'completion.json').exists()
    gates = read_csv(out/'mixvpr_gate_results.csv')
    causal = read_csv(out/'mixvpr_causal_results.csv')
    assert len(gates)==186 and len(causal)==196
    key = lambda r:(r['pool'],r['dataset'],r['method'],int(r['n_requested']))
    assert len({key(r) for r in gates})==186
    assert len({(r['pool'],r['dataset'],r['condition'],r['seed']) for r in causal})==196
    assert all(r['descriptor']=='MixVPR' for r in gates+causal)
    old = {key(r):r for r in read_csv(root/'phase3a/cosplace/cosplace_20261020/cosplace_gate_results.csv')}
    for r in gates:
        assert r['n_passed']==old[key(r)]['n_passed']
        assert float(r['tau'])==float(old[key(r)]['tau'])
        assert 0<=float(r['auc_full_precision'])<=1 and float(f"{float(r['auc_full_precision']):.3f}")==float(r['auc'])
        if int(r['n_requested'])==400: assert float(r['auc'])==float(r['naive'])
    # Independent legacy command verifies its actual midpoint/tie implementation.
    cutoff_checks = 0
    legacy_log = []
    for pool in ['low','wide']:
        for _,(dataset,stem,_) in DATASETS.items():
            filename = (f'{stem}_400_extended.csv' if pool=='low' else f'{stem}_wide_extended.csv')
            scores = root/('phase1/scores' if pool=='low' else 'phase3b/scores')/filename
            for method in METHODS:
                budgets = BUDGETS+([400] if method=='rand_s0' else [])
                command = [sys.executable,str(root/'phase3a/scripts/tau_for.py'),str(scores),method]+list(map(str,budgets))
                result = subprocess.run(command,check=True,capture_output=True,text=True)
                legacy_log.append({'argv':command,'stdout':result.stdout})
                with commands.open('a') as f:
                    f.write(json.dumps({'time':datetime.now(timezone.utc).isoformat(), 'cwd':str(Path.cwd()), 'argv':command})+'\n')
                for line in result.stdout.splitlines():
                    n,tau,passed,tie = line.split()
                    r = next(r for r in gates if key(r)==(pool,dataset,method,int(n)))
                    assert float(r['tau'])==float(tau) and int(r['n_passed'])==int(passed)
                    cutoff_checks+=1
    write_json(out/'legacy_cutoff_validation.json',legacy_log)
    assert cutoff_checks==186
    exceptions = [r for r in gates if r['n_passed']!=r['n_requested']]
    assert len(exceptions)==6
    for pool in ['low','wide']:
        for _,(dataset,_,_) in DATASETS.items():
            rows = [r for r in causal if r['pool']==pool and r['dataset']==dataset]
            expected = ['C1','C2','C4','C5','C6','C7']+(['C2b'] if dataset=='GardensPoint' else [])
            for condition in expected:
                subset = [r for r in rows if r['condition']==condition]
                assert {int(r['seed']) for r in subset}==set(range(5))
                assert all(int(r['n_injected'])==50 for r in subset)
            c0 = [r for r in rows if r['condition']=='C0']
            assert len(c0)==1 and int(c0[0]['n_injected'])==0
            assert all(float(r['baseline'])==float(c0[0]['auc']) for r in rows)
    for row in read_csv(out/'causal_comparison.csv'):
        a,b = row['contrast'].split('-')
        def values(cond,field):
            selected = [r for r in causal if r['pool']==row['pool'] and r['dataset']==row['dataset'] and r['condition']==cond]
            return np.array([float(r[field]) for r in sorted(selected,key=lambda r:int(r['seed']))])
        for field,delta_key,p_key in [('auc','mean_delta_auc','p_two_sided'),('auc_full_precision','mean_delta_full_precision','p_full_precision')]:
            av,bv = values(a,field),values(b,field)
            assert np.isclose(float(row[delta_key]),(av-bv).mean(),rtol=0,atol=1e-14)
            assert np.isclose(float(row[p_key]),stats.ttest_rel(av,bv).pvalue,rtol=1e-12,atol=1e-15)
    cache_count = cache_images = 0
    provenance = json.loads((out/'model_provenance.json').read_text())
    for metadata in sorted((out/'descriptor_cache').glob('*.json')):
        meta = json.loads(metadata.read_text())
        assert meta['descriptor']=='MixVPR' and meta['weights']==provenance['weights_sha256']
        assert meta['extractor']==provenance['extractor_sha256']
        assert '320x320' in meta['preprocess'] and 'bicubic' in meta['preprocess']
        digest = hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest()
        with np.load(metadata.with_suffix('.npz')) as data:
            descriptors = data['features']
            assert str(data['digest'])==digest
            assert descriptors.shape==(len(meta['paths']),4096)
            assert np.isfinite(descriptors).all() and np.allclose(np.linalg.norm(descriptors,axis=1),1,atol=1e-5)
        cache_count+=1; cache_images+=len(meta['paths'])
    assert sha(provenance['weights'])==provenance['weights_sha256']
    assert provenance['all_loaded_tensors_equal_checkpoint']
    inputs = json.loads((out/'input_hashes.json').read_text())
    protected = json.loads((out/'preservation_before.json').read_text())
    assert all(sha(path)==digest for path,digest in inputs.items()), 'An experimental input changed'
    assert all(sha(path)==digest for path,digest in protected.items()), 'A preserved previous file changed'
    assert len(read_csv(out/'validation_results.csv'))==4
    assert len(read_csv(out/'causal_replay_validation.csv'))==2
    assert all(r['passed']=='True' for file in ['official_descriptor_validation.csv','validation_results.csv','causal_replay_validation.csv'] for r in read_csv(out/file))
    gt = read_csv(out/'gt_inheritance_audit.csv')
    assert len(gt)==6 and all(r['hard_source_columns_equal']==r['soft_source_columns_equal']=='True' for r in gt)
    for pool,phase in [('low','phase3a'),('wide','phase3b')]:
        folder = root/phase/'mixvpr'/args.run_id
        assert len(read_csv(folder/'gate_results.csv'))==93
        assert len(read_csv(folder/'causal_results.csv'))==98
    report = {'passed':True,'gate_rows':186,'causal_rows':196,'legacy_cutoff_checks':cutoff_checks,
              'threshold_tie_exceptions':len(exceptions),'causal_contrasts_verified':12,
              'cache_groups_verified':cache_count,'cached_descriptor_rows':cache_images,
              'experimental_inputs_verified':len(inputs),'previous_files_preserved':len(protected),
              'initial_demo_oracles':4,'full_run_demo_replays':2,'inherited_gt_columns_verified':2400,
              'checkpoint_sha256':provenance['weights_sha256'],'time':datetime.now(timezone.utc).isoformat()}
    write_json(out/'verification_report.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
