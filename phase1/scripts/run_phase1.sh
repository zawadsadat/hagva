#!/bin/bash
# Phase 1 driver. Serial. Resumable. One gate-script invocation per (dataset, column)
# with all still-missing n values passed together as HAGVA_THRESHOLDS (saves the
# repeated baseline+naive runs). Appends one line per (ds, col, n) to phase1_results.csv:
#   dataset,method,n_requested,n_passed,tau,auc,baseline,naive
# Usage: run_phase1.sh [COLS] [NS] [DS]   (defaults: full sweep; DS e.g. "sfu")
#   COLS: space-separated column list, default all 14
#   NS  : space-separated n list, default "20 64 145"
# rand_s0 additionally gets n=400 in the full sweep.
set -u
PY=/home/newview/miniconda3/envs/vprtutorial/bin/python3
P1=/mnt/sda/HAGVA_R2/phase1
SCR=$P1/scripts
LOADER=/mnt/sda/Zawad/genwarp/VPR-GIT/datasets/load_dataset.py
RESULTS=$P1/phase1_results.csv
RAW=$P1/raw
mkdir -p "$RAW" "$P1/logs"
[ -f "$RESULTS" ] || echo "dataset,method,n_requested,n_passed,tau,auc,baseline,naive" > "$RESULTS"

ALL_COLS="w000 w025 w050 w075 w100 rand_s0 rand_s1 rand_s2 rand_s3 rand_s4 brisque niqe clipiqa_neg dino_global"
COLS=${1:-$ALL_COLS}
NS=${2:-"20 64 145"}
FULL_SWEEP=0; [ $# -eq 0 ] && FULL_SWEEP=1

declare -A DSNAME=( [gp]=GardensPoint [sfu]=SFU [stl]=StLucia )
declare -A CSVKEY=( [gp]=gardenspoint [sfu]=sfu [stl]=stlucia )
declare -A BASE=(   [gp]=0.802 [sfu]=0.729 [stl]=0.594 )
declare -A STAGE=(  [gp]=/tmp/vprgit_hagva_gp400_stage [sfu]=/tmp/ablate_sfu400_stage [stl]=/tmp/ablate_stlucia400_stage )
declare -A GDIR=(   [gp]=/tmp/vprgit_hagva_gp400 [sfu]=/tmp/ablate_sfu400 [stl]=/tmp/ablate_stlucia400 )

log(){ echo "[$(date '+%F %T')] $*"; }

loader_check() {
  local a b
  a=$(grep -c 'extra_query_gt\[' "$LOADER"); b=$(grep -c "copy_from\['z_hagva_tmp_" "$LOADER")
  if [ "$a" != "0" ] || [ "$b" != "0" ]; then
    log "LOADER DIRTY ($1): extra_query_gt=$a z_hagva_tmp=$b -- stripping"
    sed -i -e '/extra_query_gt\[/d' -e "/copy_from\['z_hagva_tmp_/d" "$LOADER"
    a=$(grep -c 'extra_query_gt\[' "$LOADER"); b=$(grep -c "copy_from\['z_hagva_tmp_" "$LOADER")
    [ "$a" = "0" ] && [ "$b" = "0" ] || { log "loader still dirty, abort"; exit 2; }
  fi
}

have_row() { grep -q "^$1,$2,$3," "$RESULTS"; }

for ds in ${3:-gp sfu stl}; do
  CSV=$P1/scores/${CSVKEY[$ds]}_400_extended.csv
  for col in $COLS; do
    ns="$NS"; [ $FULL_SWEEP = 1 ] && [ "$col" = rand_s0 ] && ns="$NS 400"
    missing=""
    for n in $ns; do have_row "${DSNAME[$ds]}" "$col" "$n" || missing="$missing $n"; done
    [ -z "$missing" ] && { log "skip ${DSNAME[$ds]} $col (all n present)"; continue; }
    head -1 "$CSV" | tr -d '\r' | tr ',' '\n' | grep -qx "$col" || { log "column $col not in $CSV -- skipping"; continue; }
    # taus for the missing n
    TAUINFO=$("$PY" "$SCR/tau_for.py" "$CSV" "$col" $missing) || { log "tau_for failed"; exit 3; }
    THR=$(echo "$TAUINFO" | awk '{print $2}' | paste -sd,)
    log "RUN ${DSNAME[$ds]} $col n=[$missing ] taus=$THR"
    echo "$TAUINFO" | sed 's/^/    n tau exp_pass tie: /'
    loader_check pre
    rm -rf "${STAGE[$ds]}"
    LOG=$P1/logs/${ds}_${col}_$(date +%Y%m%d_%H%M%S).log
    ( cd "$P1" && HAGVA_SCORES_CSV="$CSV" HAGVA_SCORE_COL="$col" HAGVA_THRESHOLDS="$THR" \
        "$PY" "$SCR/r2_gate_${ds}.py" ) > "$LOG" 2>&1
    rc=$?
    loader_check post
    base=$(grep -m1 'Baseline check AUC:' "$LOG" | awk '{print $NF}')
    naive=$(grep -m1 'Naive AUC:' "$LOG" | awk '{print $NF}')
    log "  rc=$rc baseline=$base naive=$naive log=$LOG"
    if [ "$base" != "${BASE[$ds]}" ]; then
      log "BASELINE MISMATCH for ${DSNAME[$ds]}: got '$base' expected ${BASE[$ds]} -- STOPPING (hard rule 4)"
      exit 4
    fi
    gcol=${col#mean_}; gcol=${gcol%_score}   # mirror gate_dir naming in the gate scripts
    GRES=${GDIR[$ds]}_${gcol}_results/gate_results.csv
    [ -f "$GRES" ] || { log "  no gate_results.csv at $GRES -- stopping"; exit 5; }
    cp "$GRES" "$RAW/${ds}_${col}_full_$(date +%Y%m%d_%H%M%S).csv"
    # one results line per n; match threshold value to n
    echo "$TAUINFO" | while read -r n tau exp tie; do
      row=$("$PY" - "$GRES" "$tau" <<'PYEOF'
import csv, sys
path, tau = sys.argv[1], float(sys.argv[2])
for r in csv.DictReader(open(path)):
    if abs(float(r['threshold']) - tau) < 1e-9:
        print(f"{r['n_passed']},{r['auc']}"); break
PYEOF
)
      if [ -z "$row" ]; then log "  n=$n: no row for tau=$tau in gate_results (ERROR in run?)"; continue; fi
      np=${row%,*}; auc=${row#*,}
      { echo "threshold,n_passed,n_rejected,auc,delta_from_baseline,delta_from_naive"; grep "^$tau," "$GRES"; } > "$RAW/${ds}_${col}_n${n}.csv"
      echo "${DSNAME[$ds]},$col,$n,$np,$tau,$auc,$base,$naive" >> "$RESULTS"
      log "  -> ${DSNAME[$ds]},$col,n=$n passed=$np auc=$auc$( [ "$tie" = 1 ] && echo ' (TIE at n-th/(n+1)-th value)')"
    done
  done
done
log "DONE"
echo DONE
