#!/usr/bin/env bash
# memexp27 driver: LLMs4SD adapter on each batch once its E2E runs are done (PDFs must still be on disk; waits for
# "e2e <batch> done" in results/memexp26/driver.log, the stdout of run_memexp26.sh), then judges.
set -u
cd "$(dirname "$0")/.."
export MEMOSCHEMA_API_TIMEOUT=300
L=results/memexp27; mkdir -p $L
for b in r7 r8 r9; do
  tag=e2e${b#r}
  until grep -q "e2e $b done" results/memexp26/driver.log; do sleep 120; done
  for pass in 1 2; do
    echo "[m27] $b L4S pass $pass $(date)"
    while IFS= read -r a; do
      eval "python3 llm4schema/scripts/run_llms4schema_baseline.py $a" >> $L/l4s_$b.log 2>&1 &
    done < <(python3 scripts/memexp27.py jobs $b)
    wait
  done
  python3 scripts/memexp27.py collect $b
  echo "[m27] $b judge $(date)"
  for r in $(ls -d $L/runs/$tag/s*); do scripts/eval_fine.sh "$r" 2 8; scripts/eval_replay.sh "$r" 2 8; done
  echo "[m27] $b done $(date)"
done
echo "[m27] ALL_DONE $(date)"
