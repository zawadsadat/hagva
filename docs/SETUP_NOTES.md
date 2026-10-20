# Setup and ground-truth notes

Start with [README](../README.md) and the
[Phase 1–3b reproduction guide](PHASE_REPRODUCTION.md). This package retains
the phase scripts, measured results and required input scores.

## Framework patches

Apply `patches/demo.py` as the framework's `demo.py` and
`patches/load_dataset.py` as `datasets/load_dataset.py` in a dedicated scratch
framework checkout. The patches support `--images_dir`, the inherited-query
GT logic and EigenPlaces. `HAGVA_DESCRIPTOR` supplies the default descriptor;
an explicit `--descriptor` takes precedence. `HAGVA_SAVE_S=1` saves raw
reference-by-query similarity values as `S.npy` in the requested output
directory, before the visualization normalization.

The patched demo prints AUC with **three** decimal places. Existing phase
baseline guards compare those formatted values, so increasing precision
requires updating the parsers/guards together. Stored phase AUCs retain their
recorded precision.

## Query ordering and inherited labels

Synthetic views are appended to the **query** set and inherit the GT column
of their source image. They do not have independent target-view labels.
Original query ordering must remain consistent with GT column ordering.
An injected filename that sorts before the originals silently shifts the
association. An all-zero added column makes an injected view unmatchable.

| Dataset | Evaluated mechanism | Injected filename |
|---|---|---|
| GardensPoint | Runtime additions to `extra_query_gt` before the loader's basename loop | `Image200.jpg` onward |
| SFU | Runtime additions to `copy_from` before GT resizing | `z_hagva_tmp_*` |
| StLucia | Explicit hard/soft GT extension in staging | `zzz_inject_NNNN.jpg` |

The loader contains legacy static mappings that were present in the evaluated
version. It also contains exact insertion anchors used by the runners.
Changing those mappings or anchors can alter GT or break patch insertion.
The supplied loader is the version used for the recorded experiments.

## Staging and serial execution

GP/SFU runners temporarily rewrite the framework loader and attempt cleanup
after evaluation. An interrupted process can leave injected patch lines.
Run serially on a disposable framework copy; inspect the loader after an
interruption before restarting. Isolation of scratch directories does not
make simultaneous runs against one loader safe.

The SFU loader backs up and rewrites `GT.npz` during loading. Staging must use
a copied, writable GT file, not a symlink to an original GT. The runners
restore it from `VPR-Query-NEW/images/SFU/GT_original.npz`. Keep that baseline
backup intact. The GT audit invokes the loader too, so audit staged data.

GardensPoint's baseline uses 200 original queries; SFU staging uses the first
385 original query JPEGs. Source pool folders also require their original
`selected_images.txt` mappings. Framework image folders must contain the
expected original images. A contaminated query directory can change AUC
without a clear error.

`tools/gt_audit.py` checks the staged query GT without descriptor extraction:

```bash
python3 tools/gt_audit.py <framework_dir> StLuciaDataset <staged_images_root>
```

For StLucia, `<staged_images_root>` is the staged `StLucia_small/` directory
containing `180809_1545/`, `100909_0845/` and `GT.npz`; the causal
runner's `--audit` option supplies the appropriate dataset path automatically.
Compare empty-column counts with the original GT, rather than requiring
zero: the original StLucia data legitimately has 35 empty columns.

Treat this legacy helper as a diagnostic, not an independent inheritance
certificate. It reads its `original_all_zero` comparator from staged GT after
loading, when that file may already be extended or rewritten. Its `STATUS`
does not prove that each injected column equals its source column. A strict
audit must compare against an untouched baseline and the injection mapping.

## Scoring entry points

Semantic drift and corrected geometry are implemented in
`phase3b/scripts/score_pool.py`. Geometry uses spatially aligned inputs; a
512-pixel render must not be cropped onto a larger source canvas.

Use `phase3b/scripts/score_pool.py` on existing source/synthetic/warp/mask
assets. It computes DINOv2 patch-level semantic drift and geometry
(bilinear warp resizing, nearest-neighbor mask resizing). Use the exact
synthetic view ID to associate assets. Low-pool geometry diagnostics are in
`phase2/scripts/recompute_geo.py`, which also needs the historical stored
Module A masks. `phase1/scripts/build_scores.py --groups geofix` adds corrected
columns. This builder also contains the Phase 1b fusion rules.

The main selection column is `w100` (semantic drift). Corrected geometry
`w000f` and equal fusion `w050f` are comparison methods. Unsuffixed geometry
columns and the notebook's combined-score exports are historical.
