#!/usr/bin/env bash
# Usage: scripts/eval_fine.sh <replay-root> <n-reps> [parallel]
# Hierarchical-F1 judge on every <root>/<variant>/chem/<run>/final_schema.json
# into <root>/eval_fine/<variant>/rep<k>/chem/<run>/metrics_fine.json. Skips finished ones.
# Optional ARMS="A B": judge only these variants.
set -u
ROOT=$1; REPS=$2; PAR=${3:-6}
for f in "$ROOT"/*/chem/*/final_schema.json; do
  rel=${f#"$ROOT"/}; variant=${rel%%/*}
  case "$variant" in eval|eval_fine) continue;; esac
  [ -n "${ARMS:-}" ] && case " $ARMS " in *" $variant "*) ;; *) continue;; esac
  run=$(basename "$(dirname "$f")")
  for k in $(seq 1 "$REPS"); do
    out="$ROOT/eval_fine/$variant/rep$k/chem/$run"
    [ -f "$out/metrics_fine.json" ] && continue
    echo "$f ${run%_*} $out"
  done
done | xargs -P "$PAR" -L 1 sh -c 'mkdir -p "$2" && python3 Evaluation/evaluate_fine.py --schema "$0" --subdomain "$1" --out-dir "$2" > "$2/judge.log" 2>&1 || echo "FAIL $2"'
