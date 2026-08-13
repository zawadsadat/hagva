# HAGVA — Hallucination-Aware Generative View Augmentation

Code and precomputed scores for studying why diffusion-synthesized novel views
degrade Visual Place Recognition (VPR), and for filtering the harmful ones
before injection.

Synthesized views are scored per-pixel by two training-free signals — geometric
consistency derived from GenWarp's own warp and visibility outputs, and semantic
drift measured with DINOv2 patch features. Views whose mean combined score
exceeds a threshold are rejected; the rest are injected into the VPR query set.

View synthesis itself builds on
[zawadsadat/nvs-vpr-eval](https://github.com/zawadsadat/nvs-vpr-eval) and its
accompanying preprint.

---

## Results

400 synthesized views per dataset, PatchNetVLAD descriptor. Recovery is
`(HAGVA − Naive) / (Baseline − Naive)`.

| Dataset | Baseline | Naive | HAGVA | τ | Views accepted | Recovery |
|---|---|---|---|---|---|---|
| GardensPoint | 0.802 | 0.755 | 0.792 | 0.42 | 64 | 78.7% |
| SFU | 0.729 | 0.651 | 0.743 | 0.44 | 71 | 117.9% |
| StLucia | 0.594 | 0.448 | 0.599 | 0.36 | 20 | 103.4% |

SFU and StLucia exceed the no-injection baseline, i.e. filtered injection
provides a net retrieval benefit rather than only avoiding harm.

All reported numbers, including the signal and descriptor ablations, are in
[`results/RESULTS.md`](results/RESULTS.md).

**Threshold caveat.** The operating points above come from sweeping τ on each
evaluation dataset, which is selection on the test data. Several selection
rules requiring no access to retrieval performance were tried — fixed absolute
threshold, fixed acceptance rate, and a distribution-derived cutoff — and none
transferred cleanly across datasets. See `ablation/threshold_rule_*.py`.

---

## Repository layout

```
view_synthesis/   GenWarp + ZoeDepth view generation (run first)
module_a/         Signal 1 + Signal 2 scoring notebook
module_a/scores/  precomputed per-view scores for all datasets
gate/             400-view threshold sweep, one script per dataset
exp5/             per-image single-view injection correlation
ablation/         signal ablation, NetVLAD ablation, threshold-rule studies
patches/          modified demo.py and load_dataset.py — required
tools/            gt_audit.py, figure generation
results/          RESULTS.md
docs/             SETUP_NOTES.md — read before modifying injection code
```

---

## Reproducing the results

The precomputed scores in `module_a/scores/` mean you do **not** need GenWarp,
ZoeDepth, DINOv2, or a GPU to reproduce any gate, ablation, or Exp 5 number.
You need the VPR framework, the three datasets, and PatchNetVLAD.

**1. Framework.** Clone
[VPR_Tutorial](https://github.com/stschubert/VPR_Tutorial) and replace
`demo.py` and `datasets/load_dataset.py` with the versions in `patches/`.
The stock files will not work: they lack the `--images_dir` override and the
ground-truth extension logic that injected views require.

**2. Datasets.** GardensPoint Walking, SFU Mountain, and StLucia (small).
Query folders must contain only the original images — see the hygiene note
below.

**3. Environment.** `patchnetvlad`, `torch`, `numpy`, `scipy`, `matplotlib`.
DINOv2 loads via `torch.hub` and is only needed to regenerate scores.

**4. Paths.** Every script has an absolute-path config block at the top
pointing at the original machine. Edit these before running.

**5. Run.** For example:

```bash
HAGVA_THRESHOLDS="0.30,0.32,0.34,0.36,0.38,0.40,0.42,0.44,0.46,0.48,0.50" \
  python3 gate/gate_gardenspoint.py
```

Scripts accept `HAGVA_THRESHOLDS` (comma-separated) and, where relevant,
`HAGVA_SCORE_COL` (`mean_geo_score`, `mean_sem_score`, or
`mean_combined_score`).

---

## Ground truth for injected views — read this

Injecting a synthetic view into a VPR query set requires two things, and
getting either wrong produces plausible but wrong AUC values:

1. **Filename sort order.** The injected file must sort *after* every original
   query. A file named `01_Image001_....jpg` sorts before `Image000.jpg` and
   shifts the entire query set against an identity ground-truth matrix.
2. **A ground-truth column.** The injected view needs the ground truth of its
   source image. A column left all-zero makes the view an unmatchable query,
   which lowers AUC for reasons unrelated to hallucination.

Three mechanisms are used here, one per dataset, because the loaders differ:

| Dataset | Mechanism | Injected filename |
|---|---|---|
| GardensPoint | runtime patch adding `extra_query_gt[idx] = ref` | `Image200.jpg` … |
| SFU | runtime patch adding `copy_from['<inj>'] = '<src>'` | `z_hagva_tmp_*` |
| StLucia | explicit GT matrix extension written to staging | `zzz_inject_NNNN.jpg` |

The StLucia approach is the only one that does not rewrite framework source at
runtime, and is the pattern to prefer.

**`tools/gt_audit.py`** verifies that a staged, injected dataset has complete
ground truth:

```bash
python3 tools/gt_audit.py <framework_dir> StLuciaDataset <staging_images_dir>
```

Compare the all-zero column count against the *original* GT, not against zero —
StLucia legitimately has 35 empty columns before any injection.

**Query folder hygiene.** Interrupted runs leave injected files behind, and a
contaminated query folder silently changes the baseline: GardensPoint reads
0.802 with 200 queries and 0.782 with 250. Check counts before trusting a
baseline.

**AUC precision.** `patches/demo.py` prints AUC at six decimals. Gate results
are reported at three, where differences are ~0.05. Experiment 5 needs six —
its effects span roughly 0.005 in total, and three-decimal rounding collapses
the data onto about five discrete values.

---

## Notes

`results/RESULTS.md` records four ground-truth handling defects found and fixed
during this work, along with the numbers they affected. They are documented
rather than quietly corrected so that anyone extending the code knows which
failure modes to check for — `gt_audit.py` exists for exactly this reason.

Per-run CSVs are not included. The scripts overwrite their output file on each
invocation, so the copies on disk did not reliably correspond to the reported
sweeps. All values are in `RESULTS.md` and regenerate from the scripts.
