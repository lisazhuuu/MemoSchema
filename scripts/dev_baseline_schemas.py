"""Introduction diagnostic, baseline side (no API calls): 8-field schemas of the development-stage baseline configuration.

For each of the nine runs of scripts/dev_diagnostic.py (Ablation/<domain>/<SUB>_02, the same runs as the pool side):
LEGACY select_final_schema (fix flags cleared) on round03/schema_from_cumulative_corpus/canonical_schema_candidates.json
with the same round's discovered_anchors.json (lexical anchor matching, no embeddings), max 8 fields, min support 2.
Output: <out-root>/baseline/<domain>/<run>/final_schema.json, judged by
  scripts/eval_replay.sh results/abl_old 3      ->  <out-root>/eval/baseline/rep<k>/<domain>/<run>/metrics.csv"""
import argparse, json, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "Phase1"), str(REPO), str(REPO / "scripts")]
from dev_diagnostic import BASE_ROOT, legacy_env, round_file, run_dirs

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", default=BASE_ROOT)
    a = ap.parse_args()
    runs = run_dirs()
    inp = {rd: (round_file(rd, "canonical_schema_candidates.json"), round_file(rd, "discovered_anchors.json")) for rd in runs}
    legacy_env()
    from schema_lib.schema_cluster_merge import select_final_schema
    from memexp26 import last_tier
    for rd in runs:
        cand, anc = inp[rd]
        out = REPO / a.out_root / "baseline" / rd.parent.name / rd.name; out.mkdir(parents=True, exist_ok=True)
        final = select_final_schema(canonical_candidates=json.loads(cand.read_text()), out_dir=out, domain=rd.parent.name,
                                    last_tier=last_tier(rd), max_final_fields=8, min_support=2,
                                    anchors=json.loads(anc.read_text())["anchors"])
        print(f"[dev-baseline] {rd.parent.name}/{rd.name}: {[f['name'] for f in final]}", flush=True)

if __name__ == "__main__":
    main()
