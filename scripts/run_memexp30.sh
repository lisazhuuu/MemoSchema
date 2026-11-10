#!/usr/bin/env bash
# memexp30 driver: wait for EVAPORATE stage 1 on each corpus, run the full author flow (9 lanes, one per run id,
# batches r7->r8->r9; up to two attempts each), then judge every EVAPORATE schema (hierarchical and strict F1, 2 reps).
# Summaries: scripts/paper_tables.py.
set -u
cd "$(dirname "$0")/.."
L=results/memexp30; PY=baselines/evaporate/.venv-full/bin/python; mkdir -p $L
topic() { case $1 in COF) echo "Covalent Organic Frameworks (COFs)";; MOF) echo "Metal-organic frameworks (MOFs)";; ZIF) echo "Zeolitic imidazolate frameworks (ZIFs)";; esac; }
lane() {
  run=$1
  for b in r7 r8 r9; do
    tag=e2e${b#r}; sd=results/evaporate/runs/$tag/s1/EVAPORATE8/chem/$run
    until grep -q '"complete"' $sd/status.json 2>/dev/null; do sleep 120; done
    out=results/evaporate_full/$tag/$run
    for pass in 1 2; do
      $PY baselines/evaporate/full_pipeline.py --papers results/evaporate/inputs/$b/$run/papers.jsonl --schema-dir $sd \
        --topic "$(topic ${run%_*})" --out $out --omit-stop --temperature omit --max-calls 20000 --execute >> $L/full_${tag}_$run.log 2>&1 && break
    done
    if grep -q '"complete"' $out/status.json 2>/dev/null; then
      d=results/evaporate/runs/$tag/s1/EVAPORATE_FULL8/chem/$run; mkdir -p $d; cp $out/final_schema.json $d/
      echo "[m30] full done $tag $run $(date)"
    else echo "[m30] full FAILED $tag $run $(date)"; fi
  done
}
echo "[m30] start $(date)"
for run in COF_01 COF_02 COF_03 MOF_01 MOF_02 MOF_03 ZIF_01 ZIF_02 ZIF_03; do lane $run & done
wait
echo "[m30] judge $(date)"
for r in $(ls -d results/evaporate/runs/e2e*/s1); do scripts/eval_fine.sh "$r" 2 8; scripts/eval_replay.sh "$r" 2 8; done
echo "[m30] ALL_DONE $(date)"
