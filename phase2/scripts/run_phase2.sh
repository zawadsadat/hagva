#!/bin/bash
# Phase 2 driver (modelled on phase1/scripts/run_phase1.sh). Serial. Resumable. NOT to be run while any
# other gate/VPR process or a GenWarp GPU process is alive.
#   run_phase2.sh [DS ...]            DS in {gp sfu stl}; default all three
#   env SEEDS="0 1 2 3 4"  DESCS="NetVLAD PatchNetVLAD"  CONDS_GP="C1 C2 C2b C4 C5 C6 C7"  CONDS_OTHER="C1 C2 C4 C5 C6 C7"
# Per (ds, descriptor): C0 baseline first (once; must match the reference where one exists), then every
# (seed, condition). Each run: loader clean check -> rm -rf staging -> inject_eval.py -> loader clean check.
# Appends to phase2/phase2_results.csv:  dataset,seed,condition,descriptor,n_injected,auc,baseline
# Writes DONE as the last log line when the loop completes.
set -u
PY=/home/newview/miniconda3/envs/vprtutorial/bin/python3
P2=/mnt/sda/HAGVA_R2/phase2
LOADER=/mnt/sda/Zawad/genwarp/VPR-GIT/datasets/load_dataset.py
RESULTS=$P2/phase2_results.csv
mkdir -p "$P2/runs" "$P2/logs"
[ -f "$RESULTS" ] || echo "dataset,seed,condition,descriptor,n_injected,auc,baseline" > "$RESULTS"

DS_LIST=${*:-gp sfu stl}
SEEDS=${SEEDS:-"0 1 2 3 4"}
DESCS=${DESCS:-"NetVLAD PatchNetVLAD"}
CONDS_GP=${CONDS_GP:-"C1 C2 C2b C4 C5 C6 C7"}
CONDS_OTHER=${CONDS_OTHER:-"C1 C2 C4 C5 C6 C7"}

declare -A DSNAME=( [gp]=GardensPoint [sfu]=SFU [stl]=StLucia )
declare -A STAGE=( [gp]=/tmp/hagva_p2_gp_stage [sfu]=/tmp/hagva_p2_sfu_stage [stl]=/tmp/hagva_p2_stl_stage )
# reference baselines (RESULTS.md / Phase 1 / 2-0 dry run). Empty = no reference: C0 is measured and recorded.
declare -A REF=( [gp,PatchNetVLAD]=0.802 [sfu,PatchNetVLAD]=0.729 [stl,PatchNetVLAD]=0.594 [gp,NetVLAD]=0.199 [sfu,NetVLAD]="" [stl,NetVLAD]="" )

log(){ echo "[$(date '+%F %T')] $*" >&2; }   # stderr, so $(run_one ...) captures only the AUC

guard() {   # serial rule + GenWarp rule
  local p g
  p=$(ps -eo args | grep -E 'python3 [^ ]*(r2_gate_|demo\.py|inject_eval\.py)' | grep -vc grep)
  g=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
  if [ "$p" != "0" ] || [ "$g" != "0" ]; then log "GUARD: $p gate/VPR process(es), $g GPU compute process(es) running -- refusing to start"; exit 9; fi
}
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
have_row() { grep -q "^$1,$2,$3,$4," "$RESULTS"; }   # dataset,seed,condition,descriptor

run_one() {   # ds seed cond desc -> echoes auc or ERROR
  local ds=$1 seed=$2 cond=$3 desc=$4 extra="" LOG rc auc
  [ "$desc" = PatchNetVLAD ] && extra="--save-s"
  loader_check pre
  rm -rf "${STAGE[$ds]}"
  LOG=$P2/logs/${ds}_seed${seed}_${cond}_${desc}_$(date +%Y%m%d_%H%M%S).log
  "$PY" "$P2/scripts/inject_eval.py" --ds "$ds" --seed "$seed" --condition "$cond" --descriptor "$desc" $extra > "$LOG" 2>&1
  rc=$?
  loader_check post
  auc=$(grep -m1 '^RESULT ' "$LOG" | sed -n 's/.*auc=\([0-9.A-Z]*\).*/\1/p')
  log "  rc=$rc ${DSNAME[$ds]} seed=$seed $cond $desc auc=${auc:-NONE} log=$LOG"
  echo "${auc:-ERROR}"
}

guard
for ds in $DS_LIST; do
  conds=$CONDS_OTHER; [ "$ds" = gp ] && conds=$CONDS_GP
  for desc in $DESCS; do
    # ── C0 baseline (once per ds x descriptor) ─────────────────────────────────────────────
    if have_row "${DSNAME[$ds]}" "-" "C0" "$desc"; then
      base=$(grep "^${DSNAME[$ds]},-,C0,$desc," "$RESULTS" | head -1 | cut -d, -f6)
      log "skip ${DSNAME[$ds]} C0 $desc (present, baseline=$base)"
    else
      log "RUN ${DSNAME[$ds]} C0 baseline $desc"
      base=$(run_one "$ds" 0 C0 "$desc")
      [ "$base" = ERROR ] && { log "baseline run failed -- stopping"; exit 5; }
      echo "${DSNAME[$ds]},-,C0,$desc,0,$base,$base" >> "$RESULTS"
    fi
    ref=${REF[$ds,$desc]}
    if [ -n "$ref" ]; then
      [ "$base" = "$ref" ] || { log "BASELINE MISMATCH ${DSNAME[$ds]} $desc: got $base expected $ref -- STOPPING (hard rule 4)"; exit 4; }
    else
      log "  no reference baseline for ${DSNAME[$ds]} $desc: recorded $base"
    fi
    # ── conditions x seeds ────────────────────────────────────────────────────────────────
    for seed in $SEEDS; do
      for cond in $conds; do
        have_row "${DSNAME[$ds]}" "$seed" "$cond" "$desc" && { log "skip ${DSNAME[$ds]} seed=$seed $cond $desc"; continue; }
        log "RUN ${DSNAME[$ds]} seed=$seed $cond $desc"
        auc=$(run_one "$ds" "$seed" "$cond" "$desc")
        if [ "$auc" = ERROR ]; then log "  run failed; not recorded (resumable)"; continue; fi
        echo "${DSNAME[$ds]},$seed,$cond,$desc,50,$auc,$base" >> "$RESULTS"
      done
    done
  done
done
log "DONE"
echo DONE
