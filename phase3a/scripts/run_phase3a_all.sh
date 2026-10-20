#!/bin/bash
# Phase 3a full run: gate first (three column groups per the brief), then causal. Each driver is resumable.
set -u
S=/mnt/sda/HAGVA_R2/phase3a/scripts
export HAGVA_DESCRIPTOR=EigenPlaces
bash $S/run_phase3a_gate.sh "rand_s0" "20 400" \
 && bash $S/run_phase3a_gate.sh "w100 w000f w050f" "20 64 145" \
 && bash $S/run_phase3a_gate.sh "rand_s1 rand_s2 rand_s3 rand_s4 clipiqa_neg brisque" "20" \
 && bash $S/run_phase3a_causal.sh \
 && echo "ALL DONE"
