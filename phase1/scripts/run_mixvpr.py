#!/usr/bin/env python3
"""MixVPR extension of Phases 1–3b; original files and results are read-only.

Validation uses official inference source and the VPR demo as independent checks.
Full runs cache batch-size-one MixVPR descriptors, then call the unchanged
framework createPR(matching='multi', n_thresh=100) for every selection.
"""
import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import numpy as np
from PIL import Image
import torch


DATASETS = {
    'gp': ('GardensPoint', 'gardenspoint', 'garden'),
    'sfu': ('SFU', 'sfu', 'sfu'),
    'stl': ('StLucia', 'stlucia', 'st'),
}
METHODS = ['w100', 'w000f', 'w050f'] + [f'rand_s{s}' for s in range(5)] + ['clipiqa_neg', 'brisque']
BUDGETS = [20, 64, 145]


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1048576), b''): h.update(block)
    return h.hexdigest()


def read_csv(path):
    with open(path, newline='') as f: return list(csv.DictReader(f))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with tmp.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    tmp.replace(path)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def stamp(): return datetime.now(timezone.utc).isoformat()


class Experiment:
    def __init__(self, args):
        self.a = args
        self.root = Path(args.project).resolve()
        self.low = self.root / 'phase3a/mixvpr' / args.run_id
        self.wide = self.root / 'phase3b/mixvpr' / args.run_id
        self.low.mkdir(parents=True, exist_ok=True)
        self.frame = self.low / 'framework'
        self.original_frame = Path(args.framework).resolve()
        self.gw, self.renders = Path(args.data_root).resolve(), Path(args.render_root).resolve()
        self.inputs = {}
        self.commands = self.low / 'commands.jsonl'
        with self.commands.open('a') as f:
            f.write(json.dumps({'time': stamp(), 'cwd': os.getcwd(), 'argv': [sys.executable] + sys.argv}) + '\n')
        protected = self.low / 'preservation_before.json'
        if not protected.exists():
            paths = []
            for phase in ['phase1','phase2','phase3a','phase3b']:
                base = self.root / phase
                paths += list(base.glob('*.csv'))
                for folder in ['scripts','scores','raw','draws','strata','views']:
                    for p in (base/folder).rglob('*'):
                        if p.is_file() and p.suffix in ['.py','.sh','.csv','.json'] and '__pycache__' not in p.parts:
                            if p.name not in ['run_mixvpr.py','summarize_mixvpr.py']: paths.append(p)
            for phase in ['phase3a','phase3b']:
                paths += [p for p in (self.root/phase/'cosplace').rglob('*') if p.is_file() and '__pycache__' not in p.parts]
            paths += [self.original_frame / 'demo.py', self.original_frame / 'datasets/load_dataset.py',
                      self.original_frame / 'feature_extraction/feature_extractor_cosplace.py']
            write_json(protected, {str(p): sha(p) for p in paths})
        self.upstream = self.root / 'phase3a/mixvpr/vendor/MixVPR'
        os.environ['HAGVA_MIXVPR_UPSTREAM'] = str(self.upstream)
        os.environ['HAGVA_MIXVPR_CHECKPOINT'] = str(self.root / 'phase3a/mixvpr/assets/resnet50_MixVPR_4096_channels(1024)_rows(4).ckpt')
        self.upstream_hashes = {str(p.relative_to(self.upstream)): sha(p) for p in [
            self.upstream/'models/backbones/resnet.py', self.upstream/'models/aggregators/mixvpr.py',
            self.upstream/'demo.py', self.upstream/'main.py']}
        self.prepare_framework()
        sys.path.insert(0, str(self.frame))
        from evaluation.metrics import createPR
        from feature_extraction.feature_extractor_mixvpr import MixVPRFeatureExtractor
        self.createPR = createPR
        torch.set_num_threads(4)
        self.extractor = MixVPRFeatureExtractor()
        assert self.extractor.device.type == 'cuda', 'CUDA required for this execution; use GPU-enabled execution.'
        self.extractor.model.eval()
        checkpoint = self.extractor.checkpoint
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        actual = self.extractor.model.state_dict()
        assert set(state) == set(actual)
        assert all(torch.equal(state[k], actual[k].detach().cpu()) for k in state), 'Loaded model differs from pretrained checkpoint'
        self.weight_sha = sha(checkpoint)
        self.extractor_sha = sha(self.frame / 'feature_extraction/feature_extractor_mixvpr.py')
        self.metric_sha = sha(self.frame / 'evaluation/metrics.py')
        write_json(self.low / 'model_provenance.json', {
            'descriptor': 'MixVPR', 'backbone': 'ResNet50', 'dimensions': 4096,
            'weights': str(checkpoint), 'weights_sha256': self.weight_sha,
            'weights_url': 'https://drive.google.com/file/d/1vuz3PvnR7vxnDDLQrdHJaOA04SQrtk5L/view',
            'upstream_commit': '4043915cef24818003ece1a8112bc8a24e69abe0',
            'upstream_sources_sha256': self.upstream_hashes,
            'all_loaded_tensors_equal_checkpoint': True,
            'extractor_sha256': self.extractor_sha, 'metrics_sha256': self.metric_sha,
            'torch': torch.__version__, 'numpy': np.__version__, 'python': sys.version,
            'gpu': torch.cuda.get_device_name(0), 'preprocess': 'RGB; Resize((320,320), BICUBIC); ToTensor; ImageNet mean/std',
            'batch_size': 1, 'precision': 'float32; no autocast',
            'metric': "createPR(S, GThard, GTsoft, matching='multi', n_thresh=100); numpy.trapz(P,R)",
            'reporting': 'auc rounds each run to 3 decimals for compatibility; auc_full_precision also retained',
            'time': stamp(),
        })
        spec = importlib.util.spec_from_file_location('_mixvpr_injection', self.root / 'phase3b/scripts/inject_eval_wide.py')
        self.inj = importlib.util.module_from_spec(spec); spec.loader.exec_module(self.inj)
        self.inj.VPR_DIR, self.inj.LOADER, self.inj.GW = str(self.frame), str(self.frame / 'datasets/load_dataset.py'), str(self.gw)
        self.inj._GT_STL = None
        for ds, c in self.inj.CFG.items():
            c['stage'] = str(self.low / 'staging' / ds)
            for key in ['real_q', 'real_ref', 'real_gt']:
                if key == 'real_q': c[key] = str(self.original_frame/c['images_sub']/c['q_sub'])
                elif key == 'real_ref': c[key] = str(self.original_frame/c['images_sub']/c['ref_sub'])
                elif key in c: c[key] = str(self.original_frame/c['images_sub']/'GT.npz')
            if ds == 'sfu': c['real_gt_dir'] = str(self.gw / 'VPR-Query-NEW/images/SFU')
        self.baselines = {}

    def prepare_framework(self):
        if self.frame.exists(): return
        self.frame.mkdir()
        for name in ['datasets', 'feature_extraction', 'evaluation', 'matching']:
            shutil.copytree(self.original_frame / name, self.frame / name,
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        # Copies only; every invocation explicitly unsets the reference-GT override.
        shutil.copy2(self.original_frame / 'demo.py', self.frame / 'demo.py')
        shutil.copy2(self.root/'phase3a/scripts/feature_extractor_mixvpr.py', self.frame/'feature_extraction/feature_extractor_mixvpr.py')
        demo = self.frame/'demo.py'
        source = demo.read_text().replace("'CosPlace', 'EigenPlaces'", "'CosPlace', 'MixVPR', 'EigenPlaces'")
        anchor = "    elif args.descriptor == 'CosPlace':"
        assert source.count(anchor) == 1
        source = source.replace(anchor, "    elif args.descriptor == 'MixVPR':\n        from feature_extraction.feature_extractor_mixvpr import MixVPRFeatureExtractor\n        feature_extractor = MixVPRFeatureExtractor()\n" + anchor)
        demo.write_text(source)
        (self.frame / 'images').symlink_to(self.original_frame / 'images', target_is_directory=True)

    def track(self, path):
        path = Path(path)
        self.inputs[str(path)] = sha(path)

    def features(self, group, paths):
        paths = [Path(p) for p in paths]
        hashes = [sha(p) for p in paths]
        for p, h in zip(paths, hashes): self.inputs[str(p)] = h
        meta = {'paths': list(map(str, paths)), 'sha256': hashes, 'weights': self.weight_sha, 'extractor': self.extractor_sha, 'descriptor': 'MixVPR', 'upstream': self.upstream_hashes, 'preprocess': 'RGB,320x320,bicubic,ImageNet,float32,batch1'}
        digest = hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()
        cache = self.low / 'descriptor_cache' / (group + '.npz')
        cache.parent.mkdir(parents=True, exist_ok=True)
        if cache.exists():
            with np.load(cache) as stored:
                assert str(stored['digest']) == digest, f'Changed descriptor inputs: {group}'
                values = stored['features']
        else:
            print(f'EXTRACT {group}: {len(paths)} images', flush=True)
            batches = []
            for i in range(0, len(paths), 32):
                images = [np.array(Image.open(p).convert('RGB')) for p in paths[i:i+32]]
                batches.append(self.extractor.compute_features(images))
            values = np.vstack(batches)
            np.savez_compressed(cache, features=values, digest=np.array(digest))
            write_json(cache.with_suffix('.json'), meta)
        assert values.shape == (len(paths), 4096) and np.isfinite(values).all()
        assert np.allclose(np.linalg.norm(values, axis=1), 1, atol=1e-5)
        return values

    def pool(self, ds, regime):
        _, key, prefix = DATASETS[ds]
        score = self.root / ('phase1/scores' if regime == 'low' else 'phase3b/scores') / (f'{key}_400_extended.csv' if regime == 'low' else f'{key}_wide_extended.csv')
        selection = self.gw / f'{prefix}_400/selected_images.txt'
        self.track(score); self.track(selection)
        scores = {r['image_stem']: r for r in read_csv(score)}
        source_names = {}
        for line in selection.read_text().splitlines():
            if line.strip():
                index, source = line.strip().split('_', 1); source_names[int(index)] = source
        directory = (self.gw if regime == 'low' else self.renders) / f'{prefix}_400_out_{regime}'
        rows = []
        for p in sorted(directory.glob('*.jpg')):
            m = re.match(r'^\d+_(\d+)_(.+)_ele([\d.]+)_r([\d.]+)\.jpg$', p.name)
            assert m, p
            index, name = int(m[1]), m[2]
            stem = p.stem.split('_', 1)[1].split('_ele')[0]
            assert stem in scores and source_names[index] == name + '.jpg'
            rows.append({**scores[stem], 'stem': stem, 'source_image': source_names[index],
                         'ref_idx': int(re.search(r'Image(\d+)', name)[1]) if ds == 'gp' else '',
                         'synth_file': p.name, 'path': str(p)})
        assert len(rows) == len(scores) == len({r['stem'] for r in rows}) == 400
        for seed in range(5):
            f = self.root / ('phase2' if regime == 'low' else 'phase3b') / 'draws' / f'{ds}_seed{seed}.csv'
            self.track(f)
            by_stem = {r['stem']: r for r in rows}
            for drawn in read_csv(f):
                candidate = by_stem[drawn['stem']]
                assert candidate['synth_file'] == drawn['synth_file'] and candidate['source_image'] == drawn['source_image']
        return rows

    @contextmanager
    def staged(self, ds, rows, paths):
        c, inj = self.inj.CFG[ds], self.inj
        assert inj.loader_clean()
        stage = Path(inj.setup_staging(ds))
        assert len(list(stage.glob('*.jpg'))) == (385 if ds == 'sfu' else 200), 'Contaminated staging queries'
        destinations = []
        original_loader = Path(inj.LOADER).read_bytes()
        try:
            names = []
            for k, (r, p) in enumerate(zip(rows, paths)):
                name = f'Image{200+k:03d}.jpg' if ds == 'gp' else f'z_hagva_tmp_{k:03d}_{r["stem"]}.jpg' if ds == 'sfu' else f'zzz_inject_{k:04d}.jpg'
                dst = stage / name; shutil.copy2(p, dst); destinations.append(dst); names.append(name)
            if ds == 'gp' and rows: inj.patch_loader_gp({200+k: int(r['ref_idx']) for k,r in enumerate(rows)}, c['marker'])
            if ds == 'sfu' and rows: inj.patch_loader_sfu({name:r['source_image'] for name,r in zip(names,rows)}, c['marker'], c['anchor'])
            if ds == 'stl':
                columns = {p.stem:i for i,p in enumerate(sorted(Path(c['real_q']).glob('*.jpg')))}
                inj.write_gt_stl([columns[Path(r['source_image']).stem] for r in rows])
            yield Path(c['stage']) / c['images_sub']
        finally:
            for p in destinations: p.unlink()
            Path(inj.LOADER).write_bytes(original_loader)
            if ds == 'sfu': inj.restore_gt_sfu()
            if ds == 'stl': inj.write_gt_stl([])
            assert inj.loader_clean()

    def loaded(self, ds, location):
        scope = {'__name__': '_mixvpr_loader'}
        path = self.frame / 'datasets/load_dataset.py'
        exec(compile(path.read_text(), str(path), 'exec'), scope)
        cls = scope[{'gp':'GardensPointDataset','sfu':'SFUDataset','stl':'StLuciaDataset'}[ds]]
        return cls(str(location)).load()

    def auc(self, refs, queries, hard, soft):
        refs = refs / np.linalg.norm(refs, axis=1, keepdims=True)
        queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
        similarity = np.matmul(refs, queries.T)
        assert np.isfinite(similarity).all() and similarity.shape == hard.shape == soft.shape
        precision, recall = self.createPR(similarity, hard, soft, matching='multi', n_thresh=100)
        value = float(np.trapz(precision, recall))
        assert np.isfinite(value)
        return {'auc': float(f'{value:.3f}'), 'auc_full_precision': value}, similarity

    def base(self, ds):
        if ds in self.baselines: return self.baselines[ds]
        c = self.inj.CFG[ds]
        with self.staged(ds, [], []) as stage:
            _, _, hard, soft = self.loaded(ds, stage)
        refs = sorted(Path(c['real_ref']).glob('*.jpg'))
        queries = sorted(Path(c['real_q']).glob('*.jpg'))[:385 if ds == 'sfu' else 200]
        ref_d = self.features(ds+'_references', refs)
        query_d = self.features(ds+'_original_queries', queries)
        value, _ = self.auc(ref_d, query_d, hard, soft)
        result = (ref_d, query_d, hard, soft, value)
        self.baselines[ds] = result
        write_csv(self.low/'baseline_results.csv',[
            {'descriptor':'MixVPR','dataset':DATASETS[key][0],
             'n_reference':len(value[0]),'n_original_queries':len(value[1]),**value[4]}
            for key,value in self.baselines.items()])
        print(f'MEASURED BASELINE {ds}: {value}', flush=True)
        return result

    def oracle(self, ds, rows, paths, label, ref_d, query_d, expected_hard, expected_soft):
        with self.staged(ds, rows, paths) as stage:
            _, _, hard, soft = self.loaded(ds, stage)
            assert np.array_equal(hard, expected_hard) and np.array_equal(soft, expected_soft), f'GT mismatch: {label}'
            output = self.low / 'validation' / label
            assert not output.exists(), f'Existing oracle output: {output}'
            output.mkdir(parents=True)
            command = [sys.executable, 'demo.py', '--descriptor','MixVPR','--dataset',DATASETS[ds][0],
                       '--images_dir',str(stage),'--output_dir',str(output)]
            environment = {**os.environ, 'MPLBACKEND':'Agg', 'HAGVA_SAVE_S':'1'}
            environment.pop('HAGVA_REF_GT', None)
            with self.commands.open('a') as f: f.write(json.dumps({'time':stamp(),'cwd':str(self.frame),'argv':command})+'\n')
            res = subprocess.run(command, cwd=self.frame, env=environment, text=True, capture_output=True)
            (output/'stdout.log').write_text(res.stdout); (output/'stderr.log').write_text(res.stderr)
            assert res.returncode == 0, f'Oracle failed: {label}: {res.stderr[-1000:]}'
            match = re.search(r'AUC \(area under curve\): ([\d.]+)',res.stdout)
            assert match, label
            cached, sim = self.auc(ref_d, query_d, hard, soft)
            oracle_s = np.load(output/'S.npy')
            error = float(np.abs(sim-oracle_s).max())
            assert np.allclose(sim,oracle_s,rtol=1e-5,atol=1e-6) and cached['auc'] == float(match[1]), (label,error,cached,match[1])
            return {'dataset':DATASETS[ds][0],'test':label,'n_injected':len(rows),'auc':cached['auc'],
                    'auc_full_precision':cached['auc_full_precision'],'demo_auc':float(match[1]),'S_max_abs_error':error,'passed':True}

    def validate(self):
        path = self.low / 'validation_results.csv'
        if path.exists():
            assert all(r['passed']=='True' for r in read_csv(path)); print('Existing validation passed.',flush=True); return
        self.validate_official()
        records = []
        for ds in DATASETS:
            ref, query, hard, soft, _ = self.base(ds)
            records.append(self.oracle(ds, [], [], ds+'_baseline',ref,query,hard,soft))
        ds = 'gp'; rows = self.pool(ds,'low'); selected = rows[:20]
        ref,query,hard,soft,_ = self.base(ds)
        novel = self.features('gp_validation20',[r['path'] for r in selected])
        with self.staged(ds,selected,[r['path'] for r in selected]) as stage: _,_,h,s = self.loaded(ds,stage)
        records.append(self.oracle(ds, selected,[r['path'] for r in selected],'gp_low_inject20',ref,np.vstack([query,novel]),h,s))
        write_csv(path, records)
        print('VALIDATION PASSED: three independently measured baselines and injected-view descriptor/GT/AUC equivalence.',flush=True)

    def validate_official(self):
        """Execute official preprocessing and forward source without training imports."""
        import ast
        import torchvision.transforms as tvf
        tree = ast.parse((self.upstream/'demo.py').read_text())
        load_image = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'load_image')
        scope = {'Image': Image, 'tvf': tvf}
        exec(compile(ast.Module(body=[load_image], type_ignores=[]), 'official/demo.py', 'exec'), scope)
        tree = ast.parse((self.upstream/'main.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'VPRModel')
        forward = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
        module = ast.Module(body=[forward], type_ignores=[])
        exec(compile(module, 'official/main.py', 'exec'), scope)
        records = []
        for ds in DATASETS:
            c = self.inj.CFG[ds]
            files = [sorted(Path(c['real_q']).glob('*.jpg'))[0], Path(self.pool(ds,'low')[0]['path'])]
            for path in files:
                official = scope['load_image'](str(path))
                adapted = self.extractor.transform(Image.open(path).convert('RGB'))
                assert torch.equal(official, adapted), f'Official preprocessing differs: {path}'
                with torch.no_grad():
                    x = official.unsqueeze(0).to(self.extractor.device)
                    expected = scope['forward'](self.extractor.model, x).cpu().numpy()
                actual = self.extractor.compute_features([np.array(Image.open(path).convert('RGB'))])
                error = float(np.abs(expected-actual).max())
                assert np.allclose(expected, actual, rtol=1e-6, atol=1e-7)
                records.append({'dataset':DATASETS[ds][0], 'image':str(path),
                                'preprocessing_tensor_identical':True, 'descriptor_max_abs_error':error,
                                'dimensions':actual.shape[1], 'finite':bool(np.isfinite(actual).all()),
                                'l2_norm':float(np.linalg.norm(actual)), 'passed':True})
        write_csv(self.low/'official_descriptor_validation.csv', records)
        print('OFFICIAL VALIDATION PASSED: preprocessing and forward descriptors match on six real/synthetic images.', flush=True)

    def run(self):
        assert (self.low/'validation_results.csv').is_file(), 'Run validation before full experiments'
        gates, causal, gt_audits, replays = [], [], [], []
        for ds in DATASETS:
            ref, query, base_h, base_s, baseline = self.base(ds)
            _, _, prefix = DATASETS[ds]
            for regime, phase, out in [('low','phase2',self.low),('wide','phase3b',self.wide)]:
                out.mkdir(parents=True,exist_ok=True)
                rows = self.pool(ds,regime)
                with self.staged(ds,rows,[r['path'] for r in rows]) as stage: _,_,hard,soft = self.loaded(ds,stage)
                nq = len(query)
                assert np.array_equal(hard[:,:nq],base_h) and np.array_equal(soft[:,:nq],base_s)
                real_queries=sorted(Path(self.inj.CFG[ds]['real_q']).glob('*.jpg'))[:nq]
                source_columns={p.name:i for i,p in enumerate(real_queries)}
                source_idx=[source_columns[r['source_image']] for r in rows]
                assert np.array_equal(hard[:,nq:],base_h[:,source_idx]), 'Injected hard GT differs from source'
                assert np.array_equal(soft[:,nq:],base_s[:,source_idx]), 'Injected soft GT differs from source'
                gt_audits.append({'pool':regime,'dataset':DATASETS[ds][0],'n_original_queries':nq,
                                  'n_injected':400,'hard_source_columns_equal':True,'soft_source_columns_equal':True,
                                  'original_empty_columns':int((~base_h.any(axis=0)).sum()),
                                  'injected_empty_columns':int((~hard[:,nq:].any(axis=0)).sum())})
                novel = self.features(ds+'_'+regime+'_synth',[r['path'] for r in rows])
                naive,_ = self.auc(ref,np.vstack([query,novel]),hard,soft)
                for method in METHODS:
                    values = np.array([float(r[method]) for r in rows]); ordered=np.sort(values)
                    for n in BUDGETS + ([400] if method=='rand_s0' else []):
                        tau = float(ordered[-1])+1.0 if n==400 else (float(ordered[n-1])+float(ordered[n]))/2.0
                        chosen=np.flatnonzero(values<tau); cols=np.r_[np.arange(nq),nq+chosen]
                        measured,_=self.auc(ref,np.vstack([query,novel[chosen]]),hard[:,cols],soft[:,cols])
                        gates.append({'pool':regime,'descriptor':'MixVPR','dataset':DATASETS[ds][0],'method':method,
                                      'n_requested':n,'n_passed':len(chosen),'tau':tau,**measured,
                                      'baseline':baseline['auc'],'naive':naive['auc']})
                # C0 is measured real-only AUC, not a borrowed baseline.
                causal.append({'pool':regime,'descriptor':'MixVPR','dataset':DATASETS[ds][0],'seed':'-',
                               'condition':'C0','n_injected':0,**baseline,'baseline':baseline['auc']})
                by_stem={r['stem']:i for i,r in enumerate(rows)}
                draws=[read_csv(self.root/phase/'draws'/f'{ds}_seed{s}.csv') for s in range(5)]
                union=sorted({r['stem'] for draw in draws for r in draw})
                union_pos={st:i for i,st in enumerate(union)}
                variants={}
                for condition,folder in [('C4','warp_black'),('C5','warp_telea'),('C6','genwarp_black'),('C7','genwarp_telea')]:
                    paths=[self.root/phase/'views'/ds/folder/(st+'.jpg') for st in union]
                    variants[condition]=self.features(ds+'_'+regime+'_'+condition,paths)
                sources=[self.gw/f'{prefix}_400'/(st+'.jpg') for st in union]
                source_d=self.features(ds+'_source_control',sources)
                if ds == 'gp':
                    paths = [self.original_frame/'images/GardensPoint/day_left'/rows[by_stem[st]]['source_image'] for st in union]
                    variants['C2b'] = self.features(ds+'_real_viewpoint_control',paths)
                for seed,draw in enumerate(draws):
                    idx=np.array([by_stem[r['stem']] for r in draw]); positions=np.array([union_pos[r['stem']] for r in draw])
                    cols=np.r_[np.arange(nq),nq+idx]
                    for condition,features in [('C1',novel[idx]),('C2',source_d[positions])] + [(c,v[positions]) for c,v in variants.items()]:
                        measured,_=self.auc(ref,np.vstack([query,features]),hard[:,cols],soft[:,cols])
                        causal.append({'pool':regime,'descriptor':'MixVPR','dataset':DATASETS[ds][0],'seed':seed,
                                       'condition':condition,'n_injected':50,**measured,'baseline':baseline['auc']})
                        if seed==0 and ((ds=='sfu' and regime=='low' and condition=='C5') or
                                        (ds=='stl' and regime=='wide' and condition=='C7')):
                            paths=[self.root/phase/'views'/ds/('warp_telea' if condition=='C5' else 'genwarp_telea')/(r['stem']+'.jpg') for r in draw]
                            replays.append(self.oracle(ds,draw,paths,ds+'_'+regime+'_'+condition+'_seed0',
                                                       ref,np.vstack([query,features]),hard[:,cols],soft[:,cols]))
                write_csv(out/'gate_results.csv',[r for r in gates if r['pool']==regime])
                write_csv(out/'causal_results.csv',[r for r in causal if r['pool']==regime])
                print(f'FINISHED {ds} {regime}: baseline={baseline["auc"]:.3f}, naive={naive["auc"]:.3f}',flush=True)
        assert len(gates)==186 and len(causal)==196
        write_csv(self.low/'mixvpr_gate_results.csv',gates)
        write_csv(self.low/'mixvpr_causal_results.csv',causal)
        write_csv(self.low/'gt_inheritance_audit.csv',gt_audits)
        write_csv(self.low/'causal_replay_validation.csv',replays)
        write_json(self.low/'input_hashes.json',self.inputs)
        preservation=json.loads((self.low/'preservation_before.json').read_text())
        assert all(sha(p)==digest for p,digest in preservation.items()), 'Original files changed'
        write_json(self.low/'completion.json',{'time':stamp(),'gates':len(gates),'causal':len(causal),
                                             'preserved_originals':len(preservation),'inputs_hashed':len(self.inputs)})


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project',default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument('--framework',default='/mnt/sda/Zawad/genwarp/VPR-GIT')
    ap.add_argument('--data-root',default='/mnt/sda/Zawad/genwarp')
    ap.add_argument('--render-root',default='/home/newview/genwarp')
    ap.add_argument('--run-id',required=True)
    ap.add_argument('--mode',choices=['validate','all'],default='validate')
    args=ap.parse_args()
    completed=Path(args.project).resolve()/'phase3a/mixvpr'/args.run_id/'completion.json'
    if completed.exists():
        print(f'Run {args.run_id} is complete. Choose a new --run-id for a new evaluation; existing outputs are preserved.')
        return
    e=Experiment(args)
    e.validate()
    if args.mode=='all': e.run()


if __name__=='__main__': main()
