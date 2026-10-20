#!/bin/bash
# Phase 3b: gate runs (both descriptors) on the wide pools, then draws + filled views, then causal runs. Each driver is resumable.
set -u
P3=/mnt/sda/HAGVA_R2/phase3b; S=$P3/scripts; PY=/home/newview/miniconda3/envs/vprtutorial/bin/python3
for D in PatchNetVLAD EigenPlaces; do
  HAGVA_DESCRIPTOR=$D bash $S/run_phase3b_gate.sh "rand_s0" "20 64 145 400" || exit 1
  HAGVA_DESCRIPTOR=$D bash $S/run_phase3b_gate.sh "w100 w000f w050f rand_s1 rand_s2 rand_s3 rand_s4 clipiqa_neg brisque" "20 64 145" || exit 1
done
echo "=== GATE DONE $(date '+%F %T')"
$PY $S/make_draws_wide.py || exit 1
for spec in "gp garden" "sfu sfu"; do set -- $spec
  nice -n 10 $PY /mnt/sda/HAGVA_R2/phase2/scripts/fill_holes.py --ds $1 --synth-dir /home/newview/genwarp/$2_400_out_wide --warp-dir /home/newview/genwarp/$2_wide_400_warp_only --mask-dir /home/newview/genwarp/$2_wide_400_warp_masks --name-by synth --genwarp-fill paste --out-root $P3/views 2>&1 | grep -vE '^\s+[0-9]+/400$' || exit 1
done
echo "=== VIEWS DONE $(date '+%F %T')"
bash $S/run_phase3b_causal.sh && echo "ALL DONE"
