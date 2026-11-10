#!/usr/bin/env bash
# Usage: scripts/eval_replay.sh <replay-root> <n-reps> [parallel]
# Runs the ontology judge on every <replay-root>/<variant>/<domain>/<run>/final_schema.json,
# n-reps times, into <replay-root>/eval/<variant>/rep<k>/<domain>/<run>/. Optional ARMS="A B": judge only these variants.
set -u
ROOT=$1; REPS=$2; PAR=${3:-6}
sub() {
  case "$1" in
    amyloid) echo Amyloid ;; neuroinflammation) echo Neuroinflammation ;; tau) echo Tau ;;
    llm) echo LLM ;; prompt) echo Prompt ;; *) echo "$1" ;;
  esac
}
for f in "$ROOT"/*/*/*/final_schema.json; do
  rel=${f#"$ROOT"/}; variant=${rel%%/*}; [ "$variant" = eval ] && continue
  [ -n "${ARMS:-}" ] && case " $ARMS " in *" $variant "*) ;; *) continue;; esac
  run=$(basename "$(dirname "$f")"); dom=$(basename "$(dirname "$(dirname "$f")")")
  for k in $(seq 1 "$REPS"); do
    out="$ROOT/eval/$variant/rep$k/$dom/$run"
    [ -f "$out/metrics.csv" ] && continue
    echo "$f $(sub "${run%_*}") $out"
  done
done | xargs -P "$PAR" -L 1 sh -c 'mkdir -p "$2" && python3 Evaluation/evaluate_against_ontology.py --schema "$0" --subdomain "$1" --out-dir "$2" > "$2/judge.log" 2>&1 || echo "FAIL $2"'
