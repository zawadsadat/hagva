#!/bin/bash
# Phase 3a: no-injection baselines for every framework descriptor other than NetVLAD/PatchNetVLAD.
# Serial; loader check before/after; staging rebuilt from the real folders with NOTHING injected
# (the SFU reference path restores GT_original.npz into staging, as the gate scripts do).
set -u
PY=/home/newview/miniconda3/envs/vprtutorial/bin/python3
P3=/mnt/sda/HAGVA_R2/phase3a; P2=/mnt/sda/HAGVA_R2/phase2
LOADER=/mnt/sda/Zawad/genwarp/VPR-GIT/datasets/load_dataset.py
RES=$P3/baselines.csv
[ -f "$RES" ] || echo "descriptor,dataset,auc,rc,seconds,loader_before,loader_after" > "$RES"
declare -A STAGE=( [gp]=/tmp/hagva_p2_gp_stage [sfu]=/tmp/hagva_p2_sfu_stage [stl]=/tmp/hagva_p2_stl_stage )
declare -A DSNAME=( [gp]=GardensPoint [sfu]=SFU [stl]=StLucia )
lc(){ echo "$(grep -c 'extra_query_gt\[' $LOADER)/$(grep -c "copy_from\['z_hagva_tmp_" $LOADER)"; }
for D in AlexNet SAD CosPlace EigenPlaces HDC-DELF; do
  for ds in gp sfu stl; do
    grep -q "^$D,${DSNAME[$ds]}," "$RES" && { echo "skip $D ${DSNAME[$ds]}"; continue; }
    b=$(lc); rm -rf "${STAGE[$ds]}"; t0=$(date +%s)
    echo "[$(date '+%F %T')] RUN $D ${DSNAME[$ds]} (loader $b)"
    env -u HAGVA_SAVE_S "$PY" "$P2/scripts/inject_eval.py" --ds $ds --condition C0 --descriptor "$D" --out-dir "$P3/baselines/${ds}_${D}" > "$P3/logs/${ds}_${D}.log" 2>&1; rc=$?
    a=$(lc); auc=$(grep -m1 '^RESULT' "$P3/logs/${ds}_${D}.log" | sed -n 's/.*auc=\([0-9.A-Z]*\).*/\1/p')
    echo "[$(date '+%F %T')]   rc=$rc auc=${auc:-NONE} $(( $(date +%s)-t0 ))s loader after $a"
    echo "$D,${DSNAME[$ds]},${auc:-ERROR},$rc,$(( $(date +%s)-t0 )),$b,$a" >> "$RES"
  done
done
echo DONE
