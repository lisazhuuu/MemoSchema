#!/usr/bin/env bash
# memexp34: name-only judging of the first schema output of each method (project judge, one repetition), then report.
set -u
cd "$(dirname "$0")/.."
export MEMOSCHEMA_API_TIMEOUT=300
python3 scripts/memexp34.py build
for r in $(ls -d results/memexp34/runs/e2e*/s1); do scripts/eval_fine.sh "$r" 1 6; scripts/eval_replay.sh "$r" 1 6; done
python3 scripts/memexp34.py report > /dev/null && echo "[m34] report: results/memexp34/report.md $(date)"
