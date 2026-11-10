#!/usr/bin/env bash
# memexp32 driver: AutoSchema paired on 27 fresh corpora x 3 reps (9 parallel), judge, report.
set -u
cd "$(dirname "$0")/.."
L=results/memexp32; A=results/autoschema; mkdir -p $L
for pass in 1 2; do
  echo "[m32] generate pass $pass $(date)"
  for b in r7 r8 r9; do for run in COF_01 COF_02 COF_03 MOF_01 MOF_02 MOF_03 ZIF_01 ZIF_02 ZIF_03; do for rep in 1 2 3; do
    echo "$b $run $rep"; done; done; done | xargs -P 9 -L 1 sh -c 'python3 baselines/autoschema/run_paired.py --batch $0 --run $1 --rep $2 >> results/memexp32/gen_$0_$1_$2.log 2>&1 || echo "FAIL $0 $1 $2"'
done
echo "[m32] schemas: $(ls $A/runs/e2e*/s*/AUTOSCHEMA/chem/*/final_schema.json | wc -l)/81 $(date)"
for r in $(ls -d $A/runs/e2e*/s*); do scripts/eval_fine.sh "$r" 2 8; scripts/eval_replay.sh "$r" 2 8; done
echo "[m32] ALL_DONE $(date)"
