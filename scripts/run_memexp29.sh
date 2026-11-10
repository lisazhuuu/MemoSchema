#!/usr/bin/env bash
# memexp29 driver: ArxivDIGESTables baseline on the fresh runs whose PDFs are on disk (BATCHES), judges, report.
set -u
cd "$(dirname "$0")/.."
export MEMOSCHEMA_API_TIMEOUT=300
L=results/memexp29; A=results/arxivdigestables; BATCHES=${BATCHES:-r7,r8,r9}
mkdir -p "$L" "$A"
for pass in 1 2; do
  echo "[m29] generate pass $pass $(date)"
  for run in COF_01 COF_02 COF_03 MOF_01 MOF_02 MOF_03 ZIF_01 ZIF_02 ZIF_03; do
    python3 baselines/arxivdigestables/reproduce.py --batches $BATCHES --runs $run --reps 3 --execute >> $L/gen_$run.log 2>&1 &
  done
  wait
done
echo "[m29] schemas: $(ls $A/runs/e2e*/s*/ADT_*/chem/*/final_schema.json | wc -l) $(date)"
for r in $(ls -d $A/runs/e2e*/s*); do scripts/eval_fine.sh "$r" 2 8; scripts/eval_replay.sh "$r" 2 8; done
echo "[m29] ALL_DONE $(date)"
