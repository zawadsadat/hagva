# Reproducing the Phase 1–3b experiments

This guide covers input preparation, complete experiment schedules and
optional rescoring. "Narrow" in the README means the `low` pools used by the
scripts and filenames. Run commands from a configured working directory.

## View synthesis

View generation retains the earlier repository's GenWarp/ZoeDepth procedure
and setup. Configure the dataset source/output folders and pose choices in
the view-generation notebook:

| Regime | `azi_deg` | `ELE_OPTIONS` | `RADIUS_OPTIONS` |
|---|---:|---|---|
| Narrow (`low`) | 0.0 | `[1, 2, 3, 4, 5]` | `[0.10, 0.15, 0.20]` |
| Wide | 0.0 | `[8, 12, 16]` | `[0.30, 0.40, 0.50]` |

Use the query-source pools `garden_400`, `sfu_400` and `st_400` and their
recorded source mappings. Current notebook settings can reflect another
dataset or an exploratory run; inspect their source/output paths before use.
Warp-only images and masks for the controls must match existing synthesized
view IDs and poses, rather than newly sampled poses. The pose and export
settings configure the existing GenWarp/ZoeDepth generation procedure.

Existing synth/warp/mask assets suffice for reproducing the recorded gate and
causal studies. The remaining steps assume those assets already exist.

## Preparation

The phase scripts retain their recorded experiment paths. Create a working
copy with paths for your machine and keep its output away from recorded
results. From the repository root, for example:

```bash
python3 tools/verify_phase_snapshot.py
python3 tools/prepare_phase_workspace.py \
  --workspace /scratch/hagva_run \
  --data-root /data/genwarp \
  --render-root /data/genwarp_regenerated \
  --framework /scratch/VPR_Tutorial_hagva \
  --vpr-python /opt/envs/vpr/bin/python3 \
  --iqa-python /opt/envs/iqa/bin/python3
```

Replace these example paths with existing inputs and Python executables. The
workspace must be new and must not overlap the repository, data, renders or
framework. Historical shell scripts require paths without spaces or shell
metacharacters. Preparation validates snapshot hashes, copies scripts and
score/draw inputs, rebases paths in copied Python/shell code and records both
hashes in `workspace_manifest.json`. It isolates `/tmp/` scratch paths below
the new workspace. It neither runs scripts nor changes external inputs.

Recorded AUC tables and raw gate outputs are deliberately not seeded into the
working directory: resumable drivers would otherwise skip completed cells.
Variant images and their manifests are not copied. Supply them separately,
or build variants from the existing synth/warp/mask files with
`phase2/scripts/fill_holes.py`; this CPU preprocessing does not invoke GenWarp.
The Phase 1 summary script is `phase1/scripts/summarize.py`; its optional
`checkpoint2_verification.csv` footnote evidence is included. Prepared copies
contain the six original 400-view score CSVs and `tools/gt_audit.py` under
`repository/`, where the rebased scripts expect them.

Apply `patches/demo.py` and `patches/load_dataset.py` to the dedicated framework
copy. Use a disposable framework checkout because GardensPoint/SFU runners
temporarily edit the loader. Run all evaluations **serially**, including across
different prepared workspaces. Keep baseline data untouched and GT writes
confined to the staged dataset. See [setup notes](SETUP_NOTES.md).

## Inputs and environments

The data root replaces `/mnt/sda/Zawad/genwarp` and must retain the expected
relative layout: `garden_400`, `sfu_400`, `st_400`,
their `*_400_out_low` pools and `VPR-Query-NEW/images/SFU/GT_original.npz`.
Inspect the dataset-specific configs in `r2_gate_*.py`, `r3_gate_*.py` and
`inject_eval.py` if your layout differs. SFU staging selects 385 original
queries. Original GardensPoint staging must have 200 queries.

Original dataset images and StLucia GT are read from the dedicated framework's
`images/` tree: `GardensPoint/{night_right,day_right,day_left}`,
`SFU/{jan,dry}` and `StLucia_small/{180809_1545,100909_0845,GT.npz}`.
Each 400-source folder also needs the recorded `selected_images.txt` mapping.

The render root replaces `/home/newview/genwarp`. It contains:

- Low regenerated `<garden|sfu|st>_400_warp_only` and `*_400_warp_masks`.
- Wide `<garden|sfu|st>_400_out_wide` synthetic pools.
- GardensPoint/SFU wide `<garden|sfu>_wide_400_warp_only` and `*_warp_masks`.
- StLucia wide `st_wide_ref400_warp_only` and `st_wide_ref400_warp_masks`.

Despite their names, those last two folders contain the **query-side** StLucia
wide renders/masks: their 400 filenames match `st_400_out_wide`. Do not substitute
any separate reference-side pool. The source trees are not included in Git.

Existing low scores are under `phase1/scores/*_400_extended.csv`; wide scores
are under `phase3b/scores/*_wide_extended.csv`. Main columns are `w100`
(semantic), `w000f` (corrected geometry), `w050f` (corrected equal fusion),
`rand_s0`–`rand_s4`, `clipiqa_neg`, and `brisque`. All sort ascending.
Unsuffixed `w000`/`w050` use historical geometry.

VPR uses the framework's dependencies, including NumPy, SciPy, PyTorch,
Matplotlib and the selected descriptor package/checkpoint (PatchNetVLAD or
EigenPlaces). Scoring additionally uses Pillow, torchvision and DINOv2;
intervention preprocessing uses OpenCV. IQA uses `pyiqa`. Recorded provenance
reports PyTorch 2.6.0, NumPy 1.26.4 and pyiqa 0.1.15.post2. DINOv2 is pinned
in the scripts to `85a24602099d397264d5b30461ad7f3bfd726ca1`.
These provenance entries are not a complete environment lockfile; framework
and dependency compatibility must be checked before future evaluations.

## Complete gate coverage

Only run the following after provisioning and auditing inputs. Set your
workspace path, then execute from it:

```bash
cd /scratch/hagva_run

# Low pool, PatchNetVLAD: 93 main-comparison cells.
bash phase1/scripts/run_phase1.sh "rand_s0" "20 64 145 400"
bash phase1/scripts/run_phase1.sh \
  "w100 w000f w050f rand_s1 rand_s2 rand_s3 rand_s4 clipiqa_neg brisque" "20 64 145"

# Low pool, EigenPlaces: 93 main-comparison cells.
HAGVA_DESCRIPTOR=EigenPlaces bash phase3a/scripts/run_phase3a_gate.sh "rand_s0" "20 64 145 400"
HAGVA_DESCRIPTOR=EigenPlaces bash phase3a/scripts/run_phase3a_gate.sh \
  "w100 w000f w050f rand_s1 rand_s2 rand_s3 rand_s4 clipiqa_neg brisque" "20 64 145"

# Wide pool, both descriptors: GardensPoint/SFU, followed by StLucia.
# These chains also build variants and run the causal controls.
bash phase3b/scripts/run_3b_all.sh
bash phase3b/scripts/run_3b_stl_chain.sh
```

The low PatchNetVLAD result table has 309 rows because it also contains the
additional Phase 1b/1c fusion/IQA ablations. Use the explicit columns above
for the manuscript's main comparisons. Do not use the default `ALL_COLS` as
a specification of the revised method: several wrappers retain older lists
and skip unavailable columns. The `run_phase3a_all.sh` wrapper omits completed
random/IQA budgets; the explicit commands above cover all recorded main cells.
`run_3b_all.sh` covers GardensPoint/SFU only; the StLucia command is required.

Gate drivers preserve the original strict-threshold behavior and report
`n_passed`. Do not replace it with exact-count tie-breaking when reproducing
the stored AUCs: that would select additional images and require new evaluation.

## Selection counts

Drivers use the midpoint between the requested rank and the next score as a
strict `< tau` cutoff. When those two scores tie, the accepted count can be
one below the nominal budget. Each setting below occurs for both PatchNetVLAD
and EigenPlaces, giving 12 main-comparison rows:

| Pool | Dataset | Method | Requested | Accepted |
|---|---|---|---:|---:|
| Narrow (`low`) | SFU | Semantic (`w100`) | 145 | 144 |
| Narrow (`low`) | StLucia | Semantic (`w100`) | 145 | 144 |
| Narrow (`low`) | StLucia | Corrected geometry (`w000f`) | 20 | 19 |
| Wide | GardensPoint | Corrected geometry (`w000f`) | 20 | 19 |
| Wide | GardensPoint | Corrected fusion (`w050f`) | 64 | 63 |
| Wide | SFU | Corrected fusion (`w050f`) | 64 | 63 |

Keep the recorded `n_passed` and AUC for these cells. Exact-count tie-breaking
would select additional images and require new retrieval measurements.
Semantic/random comparisons at wide budgets 64 and 145 are count-matched.

## Low-pool causal controls

The prepared workspace contains the recorded draws, five seeds of 50 views per
dataset. Populate `phase2/views/<gp|sfu|stl>/` with the four intervention folders
(`warp_black`, `warp_telea`, `genwarp_black`, `genwarp_telea`) before evaluation.
`fill_holes.py --genwarp-fill paste` shares the Telea hole content before JPEG
encoding. Input synth, warp and mask dimensions must already agree; this
script asserts alignment. Use existing aligned experimental assets.

```bash
# Phase 2 includes both NetVLAD and PatchNetVLAD in the recorded 196-row table.
bash phase2/scripts/run_phase2.sh
HAGVA_DESCRIPTOR=EigenPlaces bash phase3a/scripts/run_phase3a_causal.sh
```

For the manuscript's narrower C1/C2/C5/C7 subset, the drivers accept explicit
`CONDS_GP` and `CONDS_OTHER` overrides. Restricting conditions yields a subset
of the archived result tables, not their full recorded row counts. Audit an
individual staging path with `inject_eval.py --audit` before a full run.

## Optional rescoring from existing assets

The supplied scores make rescoring unnecessary for gate reproduction.
For new pools, use `phase3b/scripts/score_pool.py` with explicit source, synth,
warp, mask and output paths. It resizes warp images bilinearly and masks with
nearest-neighbor interpolation; it does not crop a 512-pixel render to the
source canvas. Its semantic score matches the notebook's patch-level method.

`run_3bA_chain.sh` scores and extends only GardensPoint/SFU. If new StLucia
wide scores are needed, the missing separate scoring steps are:

```bash
# Use your configured Python and render/data roots; these are example paths.
/opt/envs/vpr/bin/python3 phase3b/scripts/score_pool.py \
  --synth-dir /data/genwarp_regenerated/st_400_out_wide \
  --source-dir /data/genwarp/st_400 \
  --warp-dir /data/genwarp_regenerated/st_wide_ref400_warp_only \
  --mask-dir /data/genwarp_regenerated/st_wide_ref400_warp_masks \
  --out /scratch/hagva_run/phase3b/scores/stlucia_wide_scores.csv
/opt/envs/iqa/bin/python3 phase3b/scripts/build_wide_extended_stl.py
```

For low corrected geometry, `phase2/scripts/recompute_geo.py` also reads the
old Module A masks to compute diagnostic dead bands; those historical assets
are required by that script even when using regenerated inputs. Add the
`geofix` group explicitly to `phase1/scripts/build_scores.py` to extend low
CSVs from corrected scores; it is not in the default groups. Use this current
builder for the Phase 1b/1c groups as well.

Run summaries only in the prepared workspace. `pool_stats.py` writes the
GardensPoint/SFU table and `pool_stats_stl.py` appends StLucia: run them in
that order, once per rebuilt table, to avoid repeated appended sections.
