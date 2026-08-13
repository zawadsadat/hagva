# HAGVA — corrected results (all under VPR-GIT, clean GT)

## Baselines (PatchNetVLAD, clean query folders)
GardensPoint 0.802 (0.801858) | SFU 0.729 (0.729180) | StLucia 0.594 (0.593812)
NetVLAD GardensPoint 0.199

## 400-view gate
| Dataset | Base | Naive | HAGVA | tau | views | Recovery |
|---|---|---|---|---|---|---|
| GardensPoint | 0.802 | 0.755 | 0.792 | 0.42 | 64  | 78.7% |
| SFU          | 0.729 | 0.651 | 0.743 | 0.44 | 71  | 117.9% |
| StLucia      | 0.594 | 0.448 | 0.599 | 0.36 | 20  | 103.4% |

GardensPoint full sweep: 0.30-0.38 zero-pass (0.802); 0.40/20/0.791;
0.42/64/0.792; 0.44/145/0.790; 0.46/257/0.767; 0.48/331/0.771; 0.50/377/0.759

## Exp 5 (36 views, single injection, 6-decimal AUC)
| Dataset | Pearson r | p | Spearman r | p |
|---|---|---|---|---|
| GardensPoint | +0.269 | 0.113 | +0.350 | 0.036 |
| SFU          | -0.119 | 0.491 | -0.052 | 0.763 |
| StLucia      | +0.096 | 0.576 | +0.199 | 0.246 |

## Signal ablation (matched accepted-view counts, PatchNetVLAD)
GardensPoint (base 0.802, naive 0.755)
  n=20  geo 0.794 | sem 0.803 | comb 0.791
  n=64  geo 0.734 | sem 0.807 | comb 0.792
  n=145 geo 0.733 | sem 0.812 | comb 0.790
SFU (base 0.729, naive 0.651)
  n=20  geo 0.727 | sem 0.731 | comb 0.731
  n=64  geo 0.713 | sem 0.737 | comb 0.741
  n=145 geo 0.695 | sem 0.740 | comb 0.732
StLucia (base 0.594, naive 0.448)
  n=20  geo 0.589 | sem 0.597 | comb 0.599
  n=64  geo 0.561 | sem 0.589 | comb 0.578
  n=145 geo 0.540 | sem 0.532 | comb 0.546

## NetVLAD ablation (GardensPoint, base 0.199, naive 0.167)
  n=20  0.184 (53.1%) | n=64 0.183 (50.0%) | n=145 0.161 (-18.8%)

## Known issues fixed
- GardensPoint under VPR-Query-NEW: NEW_QUERY_GT_MAP covered only indices
  200-235, so 364/400 injected views had zero GT. Produced the false
  naive=0.184 / 77.1% degradation figures.
- GardensPoint Exp 5: injected file kept its original name, sorted to query
  index 0, shifting all 200 queries against identity GT.
- SFU Exp 5: injected column never received GT (no patch_loader call).
- VPR-GIT GardensPoint night_right had 50 stray Image200-249 files; moved to
  _strays_200_249/. Baseline only reads 0.802 with them removed.
- demo.py AUC print raised to 6 decimals for Exp 5; gate numbers unaffected.

## Note on result CSVs
Per-run CSVs are not included. The scripts overwrite their output file on each
invocation, so the copies on disk did not reliably match the reported sweeps.
All reported values are recorded above; regenerate with the scripts in gate/,
ablation/, and exp5/.

Precomputed Module A scores ARE included (module_a/scores/) because regenerating
them requires GenWarp, ZoeDepth, DINOv2 and the synthesized view folders. With
those scores, every gate/ablation/Exp5 result can be reproduced using only the
VPR framework and the public datasets.
