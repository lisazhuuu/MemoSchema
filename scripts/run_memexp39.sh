#!/usr/bin/env bash
# memexp39 driver: EVAPORATE stage 1 + full author flow on the 9 r10 corpora (3 lanes); cost and call records for Fig. 3b.
set -u
cd "$(dirname "$0")/.."
L=results/memexp39; PY=baselines/evaporate/.venv-full/bin/python; mkdir -p $L
topic() { case $1 in COF) echo "Covalent Organic Frameworks (COFs)";; MOF) echo "Metal-organic frameworks (MOFs)";; ZIF) echo "Zeolitic imidazolate frameworks (ZIFs)";; esac; }
lane() {
  for run in "$@"; do
    sd=results/evaporate/runs/e2e10/s1/EVAPORATE8/chem/$run
    for pass in 1 2; do
      grep -q '"complete"' $sd/status.json 2>/dev/null && break
      $PY baselines/evaporate/reproduce.py batch --batches r10 --runs $run --omit-stop --temperature omit --execute >> $L/stage1_$run.log 2>&1
    done
    out=results/evaporate_full/e2e10/$run
    for pass in 1 2; do
      grep -q '"complete"' $out/status.json 2>/dev/null && break
      $PY baselines/evaporate/full_pipeline.py --papers results/evaporate/inputs/r10/$run/papers.jsonl --schema-dir $sd \
        --topic "$(topic ${run%_*})" --out $out --omit-stop --temperature omit --max-calls 20000 --execute >> $L/full_$run.log 2>&1
    done
    if grep -q '"complete"' $out/status.json 2>/dev/null; then
      d=results/evaporate/runs/e2e10/s1/EVAPORATE_FULL8/chem/$run; mkdir -p $d; cp $out/final_schema.json $d/; echo "[m39] done $run $(date)"
    else echo "[m39] FAILED $run $(date)"; fi
  done
}
echo "[m39] start $(date)"
lane COF_01 MOF_01 ZIF_01 & lane COF_02 MOF_02 ZIF_02 & lane COF_03 MOF_03 ZIF_03 & wait
echo "[m39] DONE $(date)"
