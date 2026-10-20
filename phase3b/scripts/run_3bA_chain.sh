#!/bin/bash
set -e
cd /mnt/sda/HAGVA_R2/phase3b
PY=/home/newview/miniconda3/envs/vprtutorial/bin/python3; VENV=/mnt/sda/HAGVA_R2/phase1/iqa_venv/bin/python3
for spec in "gardenspoint garden" "sfu sfu"; do set -- $spec
  echo "=== score $1 wide $(date '+%T')"
  $PY scripts/score_pool.py --synth-dir /home/newview/genwarp/$2_400_out_wide --source-dir /mnt/sda/Zawad/genwarp/$2_400 --warp-dir /home/newview/genwarp/$2_wide_400_warp_masks/../$2_wide_400_warp_only --mask-dir /home/newview/genwarp/$2_wide_400_warp_masks --out scores/$1_wide_scores.csv 2>&1 | grep -vE 'xFormers|warn|Using cache'
done
echo "=== build extended (IQA on CPU, venv) $(date '+%T')"
nice -n 10 $VENV scripts/build_wide_extended.py 2>&1 | grep -vE 'UserWarning|warnings.warn|%\|'
echo "=== CHAIN DONE $(date '+%T')"
