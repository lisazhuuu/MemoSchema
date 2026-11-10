"""Apply restructuring (memexp26.merge_fields) to the live final schema of E2E runs (r10 efficiency comparison).
Usage: memexp33_merge.py <batch> [<batch> ...]  -> results/memexp33/runs/<tag>/s1/LIVE_M/chem/<run>/final_schema.json"""
import json, os, re, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "Phase1"), str(REPO), str(REPO / "scripts")]
os.environ["MEMEXP26_OUT"] = str(REPO / "results/memexp33")
if not sys.argv[1:] or not all(re.fullmatch(r"r\d+", b) for b in sys.argv[1:]):
    print(__doc__ if sys.argv[1:2] in (["-h"], ["--help"]) else "usage: python scripts/memexp33_merge.py <batch> [<batch> ...]")
    sys.exit(0 if sys.argv[1:2] in (["-h"], ["--help"]) else 2)
import memexp26 as m26
import input_package; input_package.check("replay", "memexp33_merge")   # minimal input package: out of scope, stop
for b in sys.argv[1:]:
    tag = "e2e" + b[1:]
    for runp in sorted((REPO / f"results/e2e/e2e_v3/{b}/chem").iterdir()):
        rds = sorted(runp.glob("iterative_phase1_chem_*"))
        if not rds or not (rds[-1] / "final_schema_from_cumulative_corpus/final_schema.json").exists():
            print("skip", b, runp.name); continue
        rd = rds[-1]; od = REPO / f"results/memexp33/runs/{tag}/s1/LIVE_M/chem/{runp.name}"
        if (od / "final_schema.json").exists():
            continue
        summ = json.loads((rd / "run_summary.json").read_text()); src = (summ.get("final_schema") or {}).get("copied_from")
        src = Path(src) if src else sorted(rd.glob("round0*/schema_from_cumulative_corpus"))[-1]
        G = json.loads((rd / "final_schema_from_cumulative_corpus/final_schema.json").read_text())
        m26.write_merged(od, m26.merge_fields(f"merge_live_{tag}_{runp.name}", G, json.loads((src / "canonical_schema_candidates.json").read_text()), runp.name.split("_")[0]))
        print("merged", tag, runp.name, flush=True)
