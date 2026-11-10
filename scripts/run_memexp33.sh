#!/usr/bin/env bash
# memexp33 driver: 9 live E2E runs (r10) with paper + merge + seed memory and exact usage logs; restructuring (Fig. 3b accounting).
# Run scripts/prefill_memory.py first to fill the merge/seed stores.
set -u
cd "$(dirname "$0")/.."
export MEMOSCHEMA_API_MAX_RETRIES=6 MEMOSCHEMA_API_TIMEOUT=300
R=$(pwd); L=results/memexp33; mkdir -p $L/usage results/e2e/e2e_v3
export MEMOSCHEMA_MERGE_CACHE_DIR=$R/results/memory_store/merge MEMOSCHEMA_SEED_CACHE_DIR=$R/results/memory_store/seed
lane() {  # lane <SUB_SET> <pass>
  s=${1%_*}; k=${1#*_}
  dirs=$(ls -d $R/results/e2e/e2e_v3/r[3-9]/chem/${s}_*/iterative_phase1_chem_*/schema_state/distill_cache 2>/dev/null | paste -sd: -)
  MEMOSCHEMA_PAPER_CACHE_DIRS=$dirs MEMOSCHEMA_USAGE_LOG=$R/$L/usage/${1}.jsonl SET=$k \
    scripts/run_e2e_local.sh e2e_v3 r10 chem:$s >> results/e2e/e2e_v3/lane_r10_$1.log 2>&1
}
for pass in 1 2; do
  echo "[m33] e2e pass $pass $(date)"
  for x in COF_01 COF_02 COF_03 MOF_01 MOF_02 MOF_03 ZIF_01 ZIF_02 ZIF_03; do lane $x $pass & done
  sleep 5; while pgrep -f "run_e2e_local.sh e2e_v3 r10 " > /dev/null; do sleep 60; done
done
echo "[m33] e2e done $(date): $(ls results/e2e/e2e_v3/r10/chem/*/iterative_phase1_chem_*/final_schema_from_cumulative_corpus/final_schema.json 2>/dev/null | wc -l) final schemas"
for pass in 1 2; do python3 scripts/memexp33_merge.py r10 >> $L/merge.log 2>&1; done
echo "[m33] ALL_DONE $(date)"
