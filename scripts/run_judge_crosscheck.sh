#!/usr/bin/env bash
# Run the Gemini cross-judge sequentially; when the daily quota is hit, wait until after the reset (02:15 CT) and resume.
set -u
cd "$(dirname "$0")/.."
export JUDGE_PROVIDER=gemini GEMINI_JUDGE_MODEL=models/gemini-3.1-pro-preview
O=results/judge_crosscheck; mkdir -p $O; rm -f $O/ALL_SCHEMAS_DONE
until [ -f $O/ALL_SCHEMAS_DONE ]; do
  echo "[xj] start $(date)"
  python3 scripts/judge_crosscheck.py run >> $O/run.log 2>&1
  [ -f $O/ALL_SCHEMAS_DONE ] && break
  echo "[xj] daily quota hit $(date); fine $(find $O/gemini -name metrics_fine.json | wc -l)/162 coarse $(find $O/gemini -name metrics.csv | wc -l)/162"
  now=$(date +%s); t=$(date -j -v+1d -f "%Y-%m-%d %H:%M" "$(date +%Y-%m-%d) 02:15" +%s)
  [ $(date +%H%M) -lt 0215 ] && t=$(date -j -f "%Y-%m-%d %H:%M" "$(date +%Y-%m-%d) 02:15" +%s)
  echo "[xj] sleeping $(( (t-now)/60 )) min"; sleep $(( t-now ))
done
echo "[xj] ALL_DONE $(date)"
