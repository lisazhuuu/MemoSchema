#!/usr/bin/env bash
# memexp38 driver: REUSEC build (4 shards) -> anchors/schemas (samples 1-6) -> MERGE -> judge -> report.
set -u
cd "$(dirname "$0")/.."
export MEMEXP26_OUT=$(pwd)/results/memexp38
L=results/memexp38; mkdir -p $L
echo "[m38] build $(date)"
P=""; for i in 0 1 2 3; do SHARD=$i/4 python3 scripts/memexp38.py build >> $L/build_$i.log 2>&1 & P="$P $!"; done; wait $P
echo "[m38] built $(ls $L/canon/REUSEC/*/*/canonical_schema_candidates.json | wc -l)/27; schemas $(date)"
for pass in 1 2; do python3 scripts/memexp38.py schemas >> $L/schemas.log 2>&1; done
for pass in 1 2; do python3 scripts/memexp38.py merge >> $L/merge.log 2>&1; done
echo "[m38] merged $(ls $L/runs/*/s*/REUSEC_M/chem/*/final_schema.json | wc -l)/162; judge $(date)"
for r in $(ls -d $L/runs/e2e*/s*); do scripts/eval_fine.sh "$r" 2 8; scripts/eval_replay.sh "$r" 2 8; done
python3 scripts/memexp38.py report > /dev/null
echo "[m38] DONE $(date)"
