#!/usr/bin/env bash
# memexp37: MERGE on LEGACY (sample 1, 27 corpora), judge, report.
set -u
cd "$(dirname "$0")/.."
python3 - <<'PY'
import sys; sys.path.insert(0, "scripts")
import memexp26 as m
from concurrent.futures import ThreadPoolExecutor
jobs = [(b, r, "LEGACY", 1) for b in ("r7", "r8", "r9") for r in m.runs_of(b)]
with ThreadPoolExecutor(6) as ex: list(ex.map(m.merge_job, jobs))
print("merged", len(jobs))
PY
for t in e2e7 e2e8 e2e9; do scripts/eval_fine.sh results/memexp26/runs/$t/s1 2 8; scripts/eval_replay.sh results/memexp26/runs/$t/s1 2 8; done
echo "[m37] DONE $(date)"
