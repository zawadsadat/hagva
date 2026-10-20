# Hallucination-Aware Multi-View Fusion with Foundation
Models: Robust Learning from Low-Quality Synthetic Views

Research code for *Hallucination-Aware Multi-View Fusion with Foundation
Models: Robust Learning from Low-Quality Synthetic Views*.

HAGVA studies how diffusion-generated novel views affect visual place
recognition (VPR). It ranks synthetic views by **DINOv2 patch-level semantic
drift** relative to their source images and admits the lowest-scoring views
at a fixed budget. Selected views augment the **query set**, alongside the
original real queries. The ranking is training-free and does not use retrieval
AUC or place labels to choose its cutoff.

The main budgets are **20, 64 and 145 views out of 400 candidates**.
Geometric consistency and semantic–geometric fusion are comparative
ablations. View generation uses GenWarp and ZoeDepth, following the workflow
in [nvs-vpr-eval](https://github.com/zawadsadat/nvs-vpr-eval).

## Experiments

Evaluation covers **GardensPoint, SFU and StLucia**, with **PatchNetVLAD**,
**EigenPlaces**, **CosPlace** and **MixVPR** descriptors. Each dataset has 400 candidate views in each of
two viewpoint regimes:

| Regime | Azimuth (degrees) | Elevation choices (degrees) | Radius parameter choices |
|---|---:|---|---|
| Narrow (`low`) | 0 | 1, 2, 3, 4, 5 | 0.10, 0.15, 0.20 |
| Wide | 0 | 8, 12, 16 | 0.30, 0.40, 0.50 |

The name `low` in scripts and filenames refers to the narrow regime.
Radius denotes the generation code's camera-radius parameter.

The experiment groups are:

| Group | Scope |
|---|---|
| Phase 1, including 1b/1c | Narrow-pool PatchNetVLAD selection, IQA and fusion comparisons, including corrected geometry |
| Phase 2 | Pose-matched causal controls, auxiliary NetVLAD controls, mask/geometry diagnostics and stratified analysis |
| Phase 3a | Narrow-pool EigenPlaces selection and causal controls; descriptor baselines |
| Phase 3b | Wide-pool PatchNetVLAD/EigenPlaces selection and causal controls |
| CosPlace extension | Narrow/wide selection and five-seed C1/C2/C5/C7 controls, with independently measured baselines |
| MixVPR extension | Narrow/wide selection and five-seed C1/C2/C4/C5/C6/C7 controls, plus GardensPoint C2b, with independently measured baselines |

## Results

AUC at **64 accepted views** is shown below. Baseline uses no synthetic
injection; naive injects all 400 candidates. Random is the mean of five seeded
selections. Semantic and random selections in this table have exactly 64
accepted views.

| Dataset | Descriptor | Baseline | Narrow naive | Narrow semantic | Narrow random | Wide naive | Wide semantic | Wide random |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| GardensPoint | PatchNetVLAD | 0.802 | 0.755 | 0.807 | 0.7830 | 0.707 | 0.805 | 0.7624 |
| SFU | PatchNetVLAD | 0.729 | 0.651 | 0.737 | 0.7044 | 0.624 | 0.742 | 0.6948 |
| StLucia | PatchNetVLAD | 0.594 | 0.448 | 0.589 | 0.5508 | 0.421 | 0.558 | 0.5352 |
| GardensPoint | EigenPlaces | 0.675 | 0.594 | 0.652 | 0.6432 | 0.550 | 0.654 | 0.6262 |
| SFU | EigenPlaces | 0.746 | 0.465 | 0.689 | 0.6668 | 0.456 | 0.687 | 0.6642 |
| StLucia | EigenPlaces | 0.736 | 0.579 | 0.724 | 0.6728 | 0.581 | 0.688 | 0.6746 |
| GardensPoint | CosPlace | 0.593 | 0.490 | 0.543 | 0.5502 | 0.461 | 0.559 | 0.5426 |
| SFU | CosPlace | 0.489 | 0.288 | 0.441 | 0.4320 | 0.287 | 0.443 | 0.4318 |
| StLucia | CosPlace | 0.766 | 0.602 | 0.730 | 0.6966 | 0.607 | 0.702 | 0.7006 |
| GardensPoint | MixVPR | 0.693 | 0.718 | 0.716 | 0.7026 | 0.662 | 0.719 | 0.6812 |
| SFU | MixVPR | 0.548 | 0.440 | 0.529 | 0.5140 | 0.418 | 0.522 | 0.5100 |
| StLucia | MixVPR | 0.878 | 0.845 | 0.902 | 0.8654 | 0.830 | 0.876 | 0.8620 |

At wide-viewpoint budgets of 64 and 145, semantic selection exceeds the
five-seed random mean in all **24 dataset–descriptor–budget comparisons**.
These are comparisons of observed means, not a blanket significance claim.
Selected synthetic views do not consistently exceed the real-query baseline.
At n=64, semantic selection exceeds each descriptor's own baseline in 4/6
PatchNetVLAD cells, 0/6 EigenPlaces cells, 0/6 CosPlace cells and 3/6 MixVPR cells.
Each baseline is measured independently for its descriptor.

The recorded drivers convert each requested budget into a strict score
cutoff. Across the four descriptors, six tied settings per descriptor produce
**24 main-method rows** that accept one fewer view than requested. Use `n_passed` as the actual count and `n_requested` as the
nominal budget. The [reproduction guide](docs/PHASE_REPRODUCTION.md#selection-counts)
lists the affected settings.

The primary result tables are:

- [Phase 1 gates](phase1/phase1_results.csv).
- [Phase 2 causal controls](phase2/phase2_results.csv).
- [Phase 3a gates](phase3a/phase3a_gate_results.csv) and [causal controls](phase3a/phase3a_causal_results.csv).
- [Phase 3b gates](phase3b/phase3b_gate_results.csv) and [causal controls](phase3b/phase3b_causal_results.csv).


## Causal controls

The controls use five seeded draws of 50 candidate views per dataset. Each
draw fixes source identities; synthetic and warp variants use the recorded
target poses.

| Condition | Injected images |
|---|---|
| C0 | No injection |
| C1 | GenWarp synthetic views |
| C2 | Corresponding real source images |
| C2b | Different-traverse real views, GardensPoint only |
| C4 / C5 | Warp-only views with black / Telea-filled holes |
| C6 / C7 | Generated views with black / matched Telea-filled holes |

`C2 − C1` compares generation with matched source reinjection. `C7 − C5`
compares visible-region content under matched hole fill, and `C7 − C1` tests
replacing the hole content. C5/C7 share their hole fill before separate JPEG
encoding; decoded hole pixels need not be identical. Effects depend on
dataset, descriptor and viewpoint regime. Synthetic queries inherit source
place labels rather than independently annotated target-view labels.

The paired tables report `C2 − C1` and `C7 − C5` over five matched seeds.
For consistency with the original phases, primary p-values use the recorded
three-decimal run AUCs; CosPlace and MixVPR also retain full-precision AUCs
and paired tests. Tests are two-sided and unadjusted for multiple comparisons.
MixVPR has positive C7−C5 means on GardensPoint in both regimes, whereas
EigenPlaces and CosPlace have negative means there. The MixVPR StLucia wide
C7−C5 p-value is sensitive to rounding: 0.0560 at recorded precision versus
0.0474 at full precision. Effects therefore require dataset- and
descriptor-specific interpretation.

## Repository structure

```text
view_synthesis/   GenWarp + ZoeDepth generation notebook
module_a/scores/  Original 400-view scores and source/synthetic mappings
phase1/          Narrow PatchNetVLAD studies and extended scores
phase2/          Causal controls, draws, manifests and geometry diagnostics
phase3a/         EigenPlaces studies; CosPlace/MixVPR runners and combined results
phase3b/         Wide scores, selection studies and descriptor-specific wide results
patches/         VPR demo, dataset loader and CosPlace/EigenPlaces extractors
tools/           Workspace/model setup, data verification and GT diagnostics
docs/            Reproduction guide, setup notes and experiment checksums
```

Recorded result CSVs, raw gate CSVs, precomputed scores, seeded draws and
variant manifests are included. Dataset images, generated image pools,
intervention images, model weights, virtual environments and per-run
similarity matrices are supplied separately. Paths inside recorded provenance
and variant manifests identify their original experiment locations.

## Requirements

- GardensPoint, SFU Mountain and StLucia_small, with the source mappings and
  ground-truth files described in [the reproduction guide](docs/PHASE_REPRODUCTION.md).
- The [VPR_Tutorial framework](https://github.com/stschubert/VPR_Tutorial), with
  this repository's `patches/demo.py` and `patches/load_dataset.py` applied.
- PatchNetVLAD, EigenPlaces, CosPlace and MixVPR implementations and weights;
  full Phase 2 reproduction also uses NetVLAD.
- Python with NumPy, SciPy, Matplotlib, Pillow, PyTorch and the framework's
  descriptor dependencies.
- For rescoring: torchvision and DINOv2 ViT-S/14. IQA comparisons use `pyiqa`;
  intervention preprocessing uses OpenCV.
- For view generation: the GenWarp/ZoeDepth environment and model checkpoints.

Recorded scoring provenance reports PyTorch 2.6.0, NumPy 1.26.4 and pyiqa
0.1.15.post2. DINOv2 is pinned in the scripts to commit
`85a24602099d397264d5b30461ad7f3bfd726ca1`. These records are not a complete
environment lockfile; use the framework's installation instructions and
verify compatibility with the selected descriptor.

## Getting started

Verify the recorded scripts and data without loading models or running VPR:

```bash
python3 tools/verify_phase_snapshot.py
```

The experiment scripts retain the paths used for the recorded runs. Prepare
a separate working directory with paths for your machine:

```bash
python3 tools/prepare_phase_workspace.py \
  --workspace /scratch/hagva_run \
  --data-root /data/genwarp \
  --render-root /data/genwarp_regenerated \
  --framework /scratch/VPR_Tutorial_hagva \
  --vpr-python /opt/envs/vpr/bin/python3 \
  --iqa-python /opt/envs/iqa/bin/python3
```

Replace the example inputs with existing directories and Python executables.
Use a new workspace and a dedicated framework checkout. Preparation copies
and configures scripts and score/draw inputs without starting experiments.
It leaves recorded result tables in this repository intact.

Follow [the reproduction guide](docs/PHASE_REPRODUCTION.md) for input layout,
complete gate schedules, causal controls and optional rescoring. Read
[the setup notes](docs/SETUP_NOTES.md) before changing injection or GT handling.
Run evaluations serially: some runners temporarily modify the framework
loader, and SFU loading writes staged GT files.

## CosPlace and MixVPR entry points

The descriptor extensions reuse existing images, scores, candidate mappings and
causal draw lists. Both evaluate all three datasets and both regimes at budgets
20, 64 and 145, with five random seeds, BRISQUE, CLIP-IQA, corrected geometry
and equal-weight semantic–geometry fusion. They compute fresh descriptor-specific
baselines and use the unchanged multi-match PR/AUC metric.

| Script | Purpose |
|---|---|
| [`run_cosplace.py`](phase3a/scripts/run_cosplace.py) | Validate and execute the complete CosPlace gate/causal workload |
| [`summarize_cosplace.py`](phase3a/scripts/summarize_cosplace.py) | Summarize CosPlace and compare with PatchNetVLAD/EigenPlaces |
| [`run_mixvpr.py`](phase3a/scripts/run_mixvpr.py) | Validate and execute the complete MixVPR gate/causal workload |
| [`feature_extractor_mixvpr.py`](phase3a/scripts/feature_extractor_mixvpr.py) | Official ResNet-50/4096 inference adapter with 320 × 320 bicubic preprocessing |
| [`summarize_mixvpr.py`](phase3a/scripts/summarize_mixvpr.py) | Budget/causal tables and all four descriptor comparisons |
| [`verify_mixvpr.py`](phase3a/scripts/verify_mixvpr.py) | Audit a completed local run, including caches, original assets and paired statistics |
| [`setup_descriptor_extensions.py`](tools/setup_descriptor_extensions.py) | Install framework patches and obtain pinned MixVPR source/weights |

CosPlace uses ResNet-50/2048 with the recorded shorter-side Resize(480)
preprocessing. MixVPR uses its official ResNet-50/4096 checkpoint, fixed
320 × 320 bicubic resize and ImageNet normalization. Extraction uses batch
size one, eval mode and float32. Each run has separate caches and framework
staging, and verifies similarities/GT/AUC against the framework demo.

Use the dedicated runners rather than setting `HAGVA_DESCRIPTOR` in the older
shell drivers; their baseline guards cover the original descriptors. See
[CosPlace and MixVPR reproduction](docs/DESCRIPTOR_EXTENSIONS.md) for setup,
checkpoint checksums, complete commands and the different causal coverage.
The MixVPR training-only Lightning/metric-learning dependencies are not required.

## View generation and scoring

View synthesis follows the existing GenWarp/ZoeDepth procedure in the
[generation notebook](view_synthesis/genwarp_view_generation.ipynb). Configure
the source/output paths and `ELE_OPTIONS`, `RADIUS_OPTIONS` and `azi_deg`
using the regime table above. Preserve full synthetic-view IDs, source
mappings and pose metadata. Saved example settings are not the specification
of every evaluated pool.

Corrected scoring from existing source/synthetic/warp/mask inputs uses
[`phase3b/scripts/score_pool.py`](phase3b/scripts/score_pool.py). Warp-only images
and masks must match each synthetic view's recorded pose. Existing prepared
assets and precomputed scores avoid regenerating views or rescoring for gate
reproduction; retrieval evaluation still needs images and descriptor extraction.

## Scope and attribution

The evaluation covers one synthesis backend, three datasets and four
descriptors. It does not establish transfer to other synthesis methods.
Offline consistency checks do not replace end-to-end evaluation with the
original assets, GT, weights and environment.

GenWarp, ZoeDepth, DINOv2, PatchNetVLAD, EigenPlaces, CosPlace, MixVPR and
VPR_Tutorial are
third-party components. See [LICENSE](LICENSE) for this repository and the
respective upstream projects for their licenses and attribution requirements.
