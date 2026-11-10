"""Fill the live memory stores from earlier runs (r3..r9) before a new batch: merge memory (identical member sets of
clusters merged by the LLM, all rounds) and seed-summary memory (round-1 seed PDFs, matched to the saved summary of
the latest earlier run with the same seed set by chunk count). The paper memory needs no prefill (it reads the earlier
runs' distill caches directly)."""
import glob, json, os, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
if __name__ == "__main__" and (sys.argv[1:]):                     # usage check before any work
    print(__doc__ if sys.argv[1:2] in (["-h"], ["--help"]) else "usage: python scripts/" + Path(__file__).name + " " + "(no arguments)")
    sys.exit(0 if sys.argv[1:2] in (["-h"], ["--help"]) else 2)
sys.path[:0] = [str(REPO / "Phase1"), str(REPO)]
import input_package; input_package.check("replay", "prefill_memory")   # minimal input package: out of scope, stop
STORE = REPO / "results/memory_store"
os.environ["MEMOSCHEMA_MERGE_CACHE_DIR"] = str(STORE / "merge"); os.environ["MEMOSCHEMA_SEED_CACHE_DIR"] = str(STORE / "seed")
from dotenv import load_dotenv; load_dotenv(REPO / ".env", override=False)
import schema_lib.schema_cluster_merge as scm
from queries_lib import seed_summarizer as ss
from queries_lib.domain import get_spec
from queries_lib.paper_model import pdf_to_text, chunk_text
B = ["r3", "r4", "r5", "r6", "r7", "r8", "r9"]
n = 0
for b in B:
    for f in glob.glob(str(REPO / f"results/e2e/e2e_v3/{b}/chem/*/iterative_phase1_chem_*/round0*/schema_from_cumulative_corpus/merged_field_clusters.json")):
        for cl in json.load(open(f)):
            if scm.support_count(cl["members"]) < 2:
                continue
            p = scm._merge_cache_path(cl["members"], "chem")
            if not p.exists():
                p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(cl["llm_merged"], ensure_ascii=False)); n += 1
print("merge memory entries:", n)
spec = get_spec("chem"); m = 0
for sd in sorted((REPO / "seeds/seed_chem").iterdir()):
    runs = sorted(glob.glob(str(REPO / f"results/e2e/e2e_v3/r9/chem/{sd.name}/iterative_phase1_chem_*/round01/papers.jsonl")))
    if not runs:
        continue
    rows = [json.loads(l)["extracted"] for l in open(runs[-1]) if l.strip()]
    for pdf in sorted(p for p in sd.iterdir() if p.suffix.lower() == ".pdf"):
        cc = len(chunk_text(pdf_to_text(pdf), chunk_size=4500, overlap=500))
        hit = [r for r in rows if r.get("chunk_count") == cc]
        if len(hit) == 1:
            path = ss._seed_cache_path(pdf, spec, 4500, 500)
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(hit[0], ensure_ascii=False)); m += 1
print("seed memory entries:", m)
