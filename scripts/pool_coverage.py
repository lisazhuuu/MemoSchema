"""Introduction diagnostic, pool side (no API calls): the top-K candidates of each development run by the
final-selection rank (no anchor slots), written as an "induced schema" so the ontology judge can measure how much of
the reference the candidate pool covers. Diagnostic only, never used by the method.

Inputs: the nine runs of scripts/dev_diagnostic.py (Ablation/<domain>/<SUB>_02),
round03/schema_from_cumulative_corpus/canonical_schema_candidates.json, LEGACY selection (fix flags cleared).
Output: <out-root>/top<K>/<domain>/<run>/final_schema.json, judged by
  scripts/eval_replay.sh results/abl_pool_old 1   ->  <out-root>/eval/top40/rep1/<domain>/<run>/metrics.csv
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "Phase1"), str(REPO), str(REPO / "scripts")]
from dev_diagnostic import POOL_ROOT, legacy_env, round_file, run_dirs

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=40)
    ap.add_argument("--out-root", default=POOL_ROOT)
    a = ap.parse_args()
    runs = run_dirs()
    cands = {rd: round_file(rd, "canonical_schema_candidates.json") for rd in runs}   # check all before writing
    legacy_env()
    from schema_lib.schema_cluster_merge import select_final_schema
    from memexp26 import last_tier as _last_tier
    for rd in runs:
        out = REPO / a.out_root / f"top{a.k}" / rd.parent.name / rd.name
        out.mkdir(parents=True, exist_ok=True)
        select_final_schema(
            canonical_candidates=json.loads(cands[rd].read_text()), out_dir=out,
            domain=rd.parent.name, last_tier=_last_tier(rd), max_final_fields=a.k, min_support=2, anchors=[],
        )
        print(f"[pool] {rd.parent.name}/{rd.name}", flush=True)

if __name__ == "__main__":
    main()
