#!/usr/bin/env python3
"""Prepare CosPlace/MixVPR inference dependencies; never run experiments.

Run against a dedicated framework checkout. Requires git and, with
--download-weights, gdown. The source repo and official weights stay separate
from the published code/results. Existing mismatched files are never replaced.
"""
import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess


REPO = Path(__file__).resolve().parents[1]
COMMIT = '4043915cef24818003ece1a8112bc8a24e69abe0'
WEIGHT_NAME = 'resnet50_MixVPR_4096_channels(1024)_rows(4).ckpt'
WEIGHT_SHA = '97528606773e9920e93ca4d211daeabf1c7312480f38b45379ee5551a1b4dd24'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''): h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=REPO,
                        help='project/workspace containing phase1 through phase3b')
    parser.add_argument('--framework', type=Path, required=True,
                        help='dedicated VPR_Tutorial checkout; its demo/loader/adapters will be patched')
    parser.add_argument('--mixvpr', action='store_true', help='obtain the pinned official MixVPR source')
    parser.add_argument('--download-weights', action='store_true', help='download and hash-check MixVPR weights; requires --mixvpr')
    args = parser.parse_args()
    if args.download_weights and not args.mixvpr: parser.error('--download-weights requires --mixvpr')
    project, frame = args.project.expanduser().resolve(), args.framework.expanduser().resolve()
    if not (project/'phase3a/scripts/run_cosplace.py').is_file(): parser.error('project is missing the descriptor runner scripts')
    if not (frame/'demo.py').is_file() or not (frame/'evaluation/metrics.py').is_file():
        parser.error('framework must be an existing VPR_Tutorial checkout')
    # Validate upstream/checkpoint before changing the dedicated framework.
    if args.mixvpr:
        upstream = project/'phase3a/mixvpr/vendor/MixVPR'
        if not upstream.exists():
            upstream.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(['git','clone','https://github.com/amaralibey/MixVPR.git',str(upstream)],check=True)
            subprocess.run(['git','-C',str(upstream),'checkout','--detach',COMMIT],check=True)
        actual = subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip()
        if actual != COMMIT: parser.error(f'existing MixVPR source is at {actual}, expected {COMMIT}; it was not changed')
        dirty = subprocess.check_output(['git','-C',str(upstream),'status','--porcelain','--untracked-files=no'],text=True)
        if dirty: parser.error('existing MixVPR source has modified tracked files; it was not changed')
        checkpoint = project/'phase3a/mixvpr/assets'/WEIGHT_NAME
        if args.download_weights and not checkpoint.exists():
            import gdown
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            temporary = checkpoint.with_suffix('.download')
            if temporary.exists(): parser.error(f'partial download exists: {temporary}; inspect it before retrying')
            result = gdown.download(id='1vuz3PvnR7vxnDDLQrdHJaOA04SQrtk5L',output=str(temporary),quiet=False)
            if result is None or sha(temporary)!=WEIGHT_SHA:
                parser.error(f'checkpoint download/hash failed; inspect {temporary}')
            temporary.replace(checkpoint)
        if checkpoint.exists():
            if sha(checkpoint)!=WEIGHT_SHA: parser.error('existing checkpoint has the wrong SHA-256; it was not changed')
            print('MixVPR checkpoint SHA-256 verified:',WEIGHT_SHA)
        else: print('MixVPR checkpoint is not installed; use --download-weights before running MixVPR.')
    for source, relative in [
        (REPO/'patches/demo.py','demo.py'),
        (REPO/'patches/load_dataset.py','datasets/load_dataset.py'),
        (REPO/'patches/feature_extraction/feature_extractor_cosplace.py','feature_extraction/feature_extractor_cosplace.py'),
        (REPO/'patches/feature_extraction/feature_extractor_eigenplaces.py','feature_extraction/feature_extractor_eigenplaces.py'),
    ]:
        destination = frame/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        if source.resolve()!=destination.resolve(): shutil.copy2(source,destination)
    print('Prepared dedicated framework:',frame)
    print('No VPR evaluation or view generation executed. CosPlace obtains its official weights through torch.hub on first use.')


if __name__=='__main__': main()
