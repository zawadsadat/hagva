# CosPlace and MixVPR reproduction

These extensions reuse the original Phase 1–3b candidate images, six extended
score CSVs, source mappings and five-seed draw lists. They measure new descriptors
and AUCs without regenerating views or rescoring DINOv2/IQA.

## Code to use

| Path | Role |
|---|---|
| `phase3a/scripts/run_cosplace.py` | CosPlace validation and complete evaluation |
| `phase3a/scripts/summarize_cosplace.py` | CosPlace tables and comparison routines reused by MixVPR |
| `phase3a/scripts/run_mixvpr.py` | MixVPR validation and complete evaluation |
| `phase3a/scripts/feature_extractor_mixvpr.py` | Inference adapter; imports official backbone and aggregator source |
| `phase3a/scripts/summarize_mixvpr.py` | MixVPR and four-descriptor tables |
| `phase3a/scripts/verify_mixvpr.py` | Full local-run scientific/cache/preservation audit |
| `tools/setup_descriptor_extensions.py` | Framework adapter setup and pinned MixVPR download |
| `patches/feature_extraction/feature_extractor_cosplace.py` | Evaluated CosPlace extractor for the framework |
| `patches/feature_extraction/feature_extractor_eigenplaces.py` | Evaluated EigenPlaces extractor for the framework |

The existing `phase3b/scripts/inject_eval_wide.py`, framework demo and dataset
loader are dependencies. The runners load the shared injection/staging logic,
copy the framework privately, and preserve the source framework. Use these
Python runners for the new descriptors: legacy gate shell scripts contain
baseline dictionaries for PatchNetVLAD/EigenPlaces only.

## Recorded model settings

| Descriptor | Model | Dimensions | Input |
|---|---|---:|---|
| CosPlace | ResNet-50 | 2048 | RGB, shorter-side Resize(480), ImageNet normalization |
| MixVPR | ResNet-50, layer 4 cropped, MixVPR depth 4, channels 1024, output rows 4 | 4096 | RGB, fixed 320 × 320 PIL bicubic resize, ImageNet normalization |

Both use float32, batch size one and eval mode. Descriptor rows are normalized
for cosine similarity. Every selected subset uses the original framework
`createPR(matching='multi', n_thresh=100)` and `numpy.trapz(precision, recall)`.
No reference-side augmentation or Phase 5/5b evaluation is used.

CosPlace weights: [official ResNet50/2048 checkpoint](https://github.com/gmberton/CosPlace/releases/download/v1.0/ResNet50_2048_cosplace.pth).
SHA-256: `1083431a283b755fd9b49302711fe1d31b061a73d3b743c2971e895224ecd89a`.
The evaluated adapter obtains them through `torch.hub` on first use.

MixVPR source: [official repository](https://github.com/amaralibey/MixVPR),
commit `4043915cef24818003ece1a8112bc8a24e69abe0`.
Weights: [official ResNet50/4096 checkpoint](https://drive.google.com/file/d/1vuz3PvnR7vxnDDLQrdHJaOA04SQrtk5L/view),
filename `resnet50_MixVPR_4096_channels(1024)_rows(4).ckpt`.
SHA-256: `97528606773e9920e93ca4d211daeabf1c7312480f38b45379ee5551a1b4dd24`.
The setup helper checks the source commit and weight checksum. The adapter
imports only inference modules; Lightning and metric-learning are unnecessary.

The recorded GPU environment used Python 3.9.23, PyTorch 2.6.0,
torchvision 0.21.0, NumPy 1.26.4, SciPy 1.13.1 and Pillow 11.3.0 on an
RTX 3080. The framework also needs its plotting/loading dependencies, including
Matplotlib and tqdm. `gdown==5.2.2` was used only for the MixVPR checkpoint
download. Install PyTorch for your CUDA environment using its official guidance;
these recorded versions are not a complete environment lockfile.

## Setup on another machine

Use a full working copy of this repository for these extensions so that the
previous result CSVs required by the summary routines are available. This is a
separate workflow from the older `prepare_phase_workspace.py` input-only helper.
Do not put the work copy inside the published repository.

```bash
cp -a /path/to/hagva /scratch/hagva_run
cd /scratch/hagva_run

# Activate the GPU VPR environment first. gdown is for downloading only.
python -m pip install gdown==5.2.2
python tools/setup_descriptor_extensions.py \
  --project /scratch/hagva_run \
  --framework /scratch/VPR_Tutorial_hagva \
  --mixvpr --download-weights
```

The framework must already be installed in a dedicated checkout, with its
normal dependencies. The helper installs the supplied demo, loader and two
framework extractors into that checkout. It clones MixVPR into
`phase3a/mixvpr/vendor/MixVPR` and saves its checkpoint under
`phase3a/mixvpr/assets/`; both directories are ignored by Git. Existing
mismatched source/weights produce an error rather than being replaced.

Provision the real images/GT, source `selected_images.txt` mappings and existing
400-view narrow/wide pools described in [PHASE_REPRODUCTION.md](PHASE_REPRODUCTION.md).
Also provide original intervention JPEGs in:

```text
/scratch/hagva_run/phase2/views/<gp|sfu|stl>/{warp_black,warp_telea,genwarp_black,genwarp_telea}/
/scratch/hagva_run/phase3b/views/<gp|sfu|stl>/{warp_black,warp_telea,genwarp_black,genwarp_telea}/
```

MixVPR also needs GardensPoint `day_left` real images for C2b. SFU requires
the clean baseline `VPR-Query-NEW/images/SFU/GT_original.npz` under the data
root. All assets must match their recorded image stems, target poses and source
associations. Images and model weights are not part of the Git repository.

## Execute validation, then complete evaluation

The examples use new run IDs so the published `*_20261020` results remain intact.
Run serially. Replace the paths with your data and framework locations.

```bash
python -u phase3a/scripts/run_cosplace.py \
  --project /scratch/hagva_run --framework /scratch/VPR_Tutorial_hagva \
  --data-root /data/genwarp --render-root /data/genwarp_regenerated \
  --run-id cosplace_reproduction --mode validate
python -u phase3a/scripts/run_cosplace.py \
  --project /scratch/hagva_run --framework /scratch/VPR_Tutorial_hagva \
  --data-root /data/genwarp --render-root /data/genwarp_regenerated \
  --run-id cosplace_reproduction --mode all
python phase3a/scripts/summarize_cosplace.py --run-id cosplace_reproduction

python -u phase3a/scripts/run_mixvpr.py \
  --project /scratch/hagva_run --framework /scratch/VPR_Tutorial_hagva \
  --data-root /data/genwarp --render-root /data/genwarp_regenerated \
  --run-id mixvpr_reproduction --mode validate
python -u phase3a/scripts/run_mixvpr.py \
  --project /scratch/hagva_run --framework /scratch/VPR_Tutorial_hagva \
  --data-root /data/genwarp --render-root /data/genwarp_regenerated \
  --run-id mixvpr_reproduction --mode all
python phase3a/scripts/summarize_mixvpr.py \
  --run-id mixvpr_reproduction --cosplace-run-id cosplace_reproduction
python phase3a/scripts/verify_mixvpr.py --run-id mixvpr_reproduction
```

Initial validation checks all three real-only baselines and a 20-view
GardensPoint injection against the framework demo, including raw similarities,
GT and AUC. MixVPR additionally checks official preprocessing/forward source
on six real/synthetic examples. Full-run replay checks cover SFU narrow C5
and StLucia wide C7. Validation failure stops execution.

The full local-run verifier requires the newly produced binary caches and
absolute-path input/preservation manifests. Those large/local artifacts are
not published here. To check this repository's recorded CSVs offline instead,
use `python3 tools/verify_phase_snapshot.py`.

## Outputs and coverage

| Descriptor | Combined gates | Combined causal | Narrow causal | Wide causal |
|---|---:|---:|---:|---:|
| CosPlace | 186 | 126 | 63 | 63 |
| MixVPR | 186 | 196 | 98 | 98 |

Combined outputs live under `phase3a/<descriptor>/<run-id>/`. Wide-only
`gate_results.csv` and `causal_results.csv` also live under the corresponding
`phase3b/<descriptor>/<run-id>/` folder. Each gate table covers 180
method/budget cells plus six all-400 checks. Methods are `w100`, `w000f`,
`w050f`, `rand_s0`–`rand_s4`, `brisque` and `clipiqa_neg`; budgets are 20/64/145.

CosPlace includes C0 and five-seed C1/C2/C5/C7. MixVPR includes C0 and
five-seed C1/C2/C4/C5/C6/C7, plus GardensPoint-only C2b. Non-C0 draws contain
50 injected views; C0 is measured real-only AUC. The additional MixVPR
controls are not claimed as completed CosPlace measurements.

The six strict-cutoff tie settings in the reproduction guide apply identically
to each descriptor. Actual counts are preserved in every gate row. These
extensions do not impose exact-count tie-breaking or change random rankings.

Both descriptors preserve three-decimal run AUCs for compatibility with prior
phases and also record `auc_full_precision`. Causal summaries provide paired
two-sided t-tests on both representations. Results are unadjusted for multiple
comparisons. The borderline MixVPR StLucia wide C7−C5 contrast changes from
p=0.0560 at recorded precision to p=0.0474 at full precision; report both.

Recorded `model_provenance.json` files retain historical local paths. Those
paths document the original runs and are not instructions to use those paths
on a new machine. Published CSV values and original phase scripts are retained
without scientific changes.
