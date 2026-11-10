#!/usr/bin/env bash
# Local restartable E2E runs (frozen config: cluster,anchorschema, 12k) for fresh confirmation data.
# Usage: scripts/run_e2e_local.sh <out-root> <rep> <dom:sub>...   e.g. ... e2e_v3 r3 chem:MOF ad:tau
set -u
OUT=$1; REP=$2; shift 2
cd "$(dirname "$0")/.."
if [ -f "results/e2e/$OUT/INPUT_PACKAGE.json" ]; then
  echo "results/e2e/$OUT holds the minimal input package; new runs would mix with the packaged ones. Use another <out-root>." >&2
  exit 2
fi
mkdir -p "results/e2e/$OUT"
export MEMOSCHEMA_API_MAX_RETRIES=${MEMOSCHEMA_API_MAX_RETRIES:-6}
for ds in "$@"; do
  dom=${ds%%:*}; s=${ds##*:}
  case $s in
    amyloid) t="Amyloid production and APP processing";; neuroinflammation) t="Neuroinflammation microglia";; tau) t="Tau";;
    COF) t="Covalent Organic Frameworks (COFs)";; MOF) t="Metal-organic frameworks (MOFs)";; ZIF) t="Zeolitic imidazolate frameworks (ZIFs)";;
    llm) t="LLM agents";; prompt) t="Prompt engineering";; RLHF) t="Reinforcement learning from human feedback (RLHF)";;
  esac
  SET=${SET:-02}; root="results/e2e/$OUT/$REP/$dom/${s}_$SET"
  run=$(ls -dt "$root"/iterative_phase1_${dom}_* 2>/dev/null | head -1)
  if [ -n "$run" ] && [ -f "$run/final_schema_from_cumulative_corpus/final_schema.json" ]; then echo "[e2e] $ds done"; continue; fi
  resume=(); [ -n "$run" ] && ls -d "$run"/round0* >/dev/null 2>&1 && resume=(--resume-run-dir "$run")
  echo "[e2e] $ds start $(date)"
  python3 scripts/run_memoschema_pipeline.py --domain $dom --subdomain $s --last-tier "$t" --target-run-id $SET \
    --seed-dir "seeds/seed_$dom/${s}_$SET" --output-root "$root" --rounds 3 --sources core,openalex,nature \
    --target-total-papers 250 --fixes cluster,anchorschema ${resume[@]+"${resume[@]}"} > "results/e2e/$OUT/${REP}_${dom}_${s}_$SET.log" 2>&1 \
    && echo "[e2e] $ds finished $(date)" || echo "[e2e] $ds FAILED $(date)"
done
echo "[e2e] ALL_DONE $(date)"
