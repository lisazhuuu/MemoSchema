#!/usr/bin/env bash
# Regenerate the cross-run history batches used by the reuse replay (memexp38) and the r10 live run (memexp33):
# r3 = COF/MOF/ZIF seed set 02; r4, r5, r6 = all nine chem seed sets (same settings as r7-r10, see
# data/corpora_history/). Paid API calls and time-dependent retrieval: the result is a new history, not the paper's.
# Output: results/e2e/e2e_v3/<batch>/chem/<SUB>_<SET>/iterative_phase1_chem_<timestamp>/   (rerun to resume)
set -u
cd "$(dirname "$0")/.."
for s in COF MOF ZIF; do SET=02 scripts/run_e2e_local.sh e2e_v3 r3 chem:$s; done
for b in r4 r5 r6; do
  for k in 01 02 03; do
    for s in COF MOF ZIF; do SET=$k scripts/run_e2e_local.sh e2e_v3 $b chem:$s; done
  done
done
echo "[history] ALL_DONE $(date)"
