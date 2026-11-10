#!/usr/bin/env bash
# memexp26 driver: 3 batches of 9 chem E2E runs (r7, r8, r9); while the next batch runs, the previous one is processed
# in the background (rebuild arms -> drafts -> restructuring -> judges).
set -u
cd "$(dirname "$0")/.."
export MEMOSCHEMA_API_MAX_RETRIES=${MEMOSCHEMA_API_MAX_RETRIES:-6} MEMOSCHEMA_API_TIMEOUT=300
L=results/memexp26; mkdir -p $L results/e2e/e2e_v3
RUNS9="COF_01 COF_02 COF_03 MOF_01 MOF_02 MOF_03 ZIF_01 ZIF_02 ZIF_03"

e2e() {  # e2e <batch> <SUB_SET>...   (lanes waited by pgrep)
  b=$1; shift
  if [ -f results/e2e/e2e_v3/INPUT_PACKAGE.json ]; then   # minimal input package installed: its runs are the inputs
    echo "[m26] input package installed: using the packaged $b runs, no corpus construction"; return
  fi
  for pass in 1 2; do
    echo "[m26] e2e $b pass $pass $(date)"
    for x in "$@"; do
      s=${x%_*}; k=${x#*_}
      SET=$k scripts/run_e2e_local.sh e2e_v3 $b chem:$s >> results/e2e/e2e_v3/lane_${b}_${s}_$k.log 2>&1 &
    done
    sleep 5
    while pgrep -f "run_e2e_local.sh e2e_v3 $b " > /dev/null; do sleep 60; done
  done
  echo "[m26] e2e $b done $(date): $(ls results/e2e/e2e_v3/$b/chem/*/iterative_phase1_chem_*/final_schema_from_cumulative_corpus/final_schema.json 2>/dev/null | wc -l) final schemas"
}

process() {  # process <batch>   (runs in a background subshell; waits only on its own children)
  b=$1; tag=e2e${b#r}; export TARGETS=$b
  for pass in 1 2; do
    echo "[m26] $b build pass $pass $(date)"; P=""
    for i in 0 1 2 3; do ARMS=ORIG,ORIG2,ORIG3 SHARD=$i/4 python3 scripts/memexp26.py build >> $L/build_${b}_$i.log 2>&1 & P="$P $!"; done
    for i in 0 1; do ARMS=LEGACY SHARD=$i/2 python3 scripts/memexp26.py build >> $L/build_${b}_legacy$i.log 2>&1 & P="$P $!"; done
    wait $P
  done
  for pass in 1 2; do
    echo "[m26] $b schemas pass $pass $(date)"
    ARMS=ORIG,ORIG2,ORIG3 python3 scripts/memexp26.py schemas >> $L/schemas_$b.log 2>&1 &
    P=$!; ARMS=LEGACY python3 scripts/memexp26.py schemas >> $L/schemas_${b}_legacy.log 2>&1; wait $P
  done
  for pass in 1 2; do echo "[m26] $b merge pass $pass $(date)"; python3 scripts/memexp26.py merge >> $L/merge_$b.log 2>&1; done
  echo "[m26] $b judge $(date)"
  for r in $(ls -d $L/runs/$tag/s*); do scripts/eval_fine.sh "$r" 2 8; scripts/eval_replay.sh "$r" 2 8; done
  echo "[m26] $b PROCESSED $(date)"
}

e2e r7 $RUNS9
process r7 > $L/process_r7.log 2>&1 & P7=$!
e2e r8 $RUNS9
process r8 > $L/process_r8.log 2>&1 & P8=$!
e2e r9 $RUNS9
process r9 > $L/process_r9.log 2>&1 & P9=$!

wait $P7 $P8 $P9
echo "[m26] ALL_DONE $(date)"
