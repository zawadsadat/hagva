#!/bin/bash
# Radius-stratified naive injection: for each dataset inject ALL views at r=0.10, then r=0.15, then r=0.20
# (no scoring, no gate), PatchNetVLAD. Serial, resumable, loader checks, staging rm. Refuses to start unless
# phase2/run_phase2.log ends with DONE and no VPR/GPU process is alive.
# Output: phase2/strata_results.csv  dataset,stratum,n_injected,auc,baseline,naive400,mean_holefrac
set -u
PY=/home/newview/miniconda3/envs/vprtutorial/bin/python3
P2=/mnt/sda/HAGVA_R2/phase2
LOADER=/mnt/sda/Zawad/genwarp/VPR-GIT/datasets/load_dataset.py
RES=$P2/strata_results.csv
[ -f "$RES" ] || echo "dataset,stratum,n_injected,auc,baseline,naive400,mean_holefrac" > "$RES"
declare -A DSNAME=( [gp]=GardensPoint [sfu]=SFU [stl]=StLucia )
declare -A BASE=( [gp]=0.802 [sfu]=0.729 [stl]=0.594 )
declare -A NAIVE=( [gp]=0.755 [sfu]=0.651 [stl]=0.448 )
declare -A STAGE=( [gp]=/tmp/hagva_p2_gp_stage [sfu]=/tmp/hagva_p2_sfu_stage [stl]=/tmp/hagva_p2_stl_stage )
log(){ echo "[$(date '+%F %T')] $*" >&2; }
loader_check(){ local a b; a=$(grep -c 'extra_query_gt\[' "$LOADER"); b=$(grep -c "copy_from\['z_hagva_tmp_" "$LOADER"); if [ "$a" != 0 ] || [ "$b" != 0 ]; then log "LOADER DIRTY ($1) -- stripping"; sed -i -e '/extra_query_gt\[/d' -e "/copy_from\['z_hagva_tmp_/d" "$LOADER"; fi; }
[ "$(tail -1 $P2/run_phase2.log 2>/dev/null)" = "DONE" ] || { log "run_phase2.log does not end with DONE -- refusing"; exit 9; }
p=$(ps -eo args | grep -E 'python3 [^ ]*(r2_gate_|demo\.py|inject_eval\.py)' | grep -vc grep); g=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
[ "$p" = 0 ] && [ "$g" = 0 ] || { log "GUARD: $p VPR procs, $g GPU procs -- refusing"; exit 9; }
for ds in gp sfu stl; do
  for rad in 0.10 0.15 0.20; do
    grep -q "^${DSNAME[$ds]},r$rad," "$RES" && { log "skip ${DSNAME[$ds]} r$rad"; continue; }
    CSV=$P2/strata/${ds}_r${rad}.csv; n=$(($(wc -l < "$CSV")-1))
    hf=$("$PY" -c "
import csv; m={r['stem']:float(r['holefrac']) for r in csv.DictReader(open('$P2/views/$ds/manifest.csv'))}
s=[m[r['stem']] for r in csv.DictReader(open('$CSV'))]; print(f'{sum(s)/len(s):.4f}')")
    log "RUN ${DSNAME[$ds]} r=$rad n=$n (mean holefrac $hf)"
    loader_check pre; rm -rf "${STAGE[$ds]}"
    LOG=$P2/logs/strata_${ds}_r${rad}_$(date +%Y%m%d_%H%M%S).log
    "$PY" "$P2/scripts/inject_eval.py" --ds "$ds" --condition C1 --descriptor PatchNetVLAD --stems-csv "$CSV" --out-dir "$P2/runs/strata/${ds}_r${rad}" --save-s > "$LOG" 2>&1; rc=$?
    loader_check post
    auc=$(grep -m1 '^RESULT ' "$LOG" | sed -n 's/.*auc=\([0-9.A-Z]*\).*/\1/p'); ni=$(grep -m1 '^RESULT ' "$LOG" | sed -n 's/.*n_injected=\([0-9]*\).*/\1/p')
    log "  rc=$rc auc=${auc:-NONE} n_injected=${ni:-?} log=$LOG"
    [ -n "$auc" ] && [ "$auc" != ERROR ] && echo "${DSNAME[$ds]},r$rad,$ni,$auc,${BASE[$ds]},${NAIVE[$ds]},$hf" >> "$RES" || log "  not recorded (resumable)"
  done
done
log "DONE"; echo DONE
