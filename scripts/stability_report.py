"""Cross-corpus schema stability (paper Table VI): name-only soft Jaccard, greedy one-to-one lexical matching,
term-overlap threshold 0.30 (Evaluation/stability_lib). First schema output (sample 1) of each method; the nine corpus
instances of each subdomain give 36 pairs per subdomain, 108 overall. Reported descriptively: pairs share schemas.

MemoSchema = final content-based reuse configuration: results/memexp38/runs/<tag>/s1/REUSEC_M/chem/<run>/final_schema.json;
a missing schema of any method stops the report."""
import itertools, statistics as st, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]; OUT = REPO / "results/stability"
if __name__ == "__main__" and (sys.argv[1:]):                     # usage check before any work
    print(__doc__ if sys.argv[1:2] in (["-h"], ["--help"]) else "usage: python scripts/" + Path(__file__).name + " " + "(no arguments)")
    sys.exit(0 if sys.argv[1:2] in (["-h"], ["--help"]) else 2)
sys.path.insert(0, str(REPO / "Evaluation"))
from stability_lib.schema_io import load_schema_file
from stability_lib.matching import align_schemas
from stability_lib.metrics import soft_jaccard
OURS = "REUSEC_M"
SRC = {OURS: ("results/memexp38", "REUSEC_M"), "LEGACY": ("results/memexp26", "LEGACY"),
       "AUTOSCHEMA": ("results/autoschema", "AUTOSCHEMA"), "L4S": ("results/memexp27", "L4S"),
       "ADT_topic": ("results/arxivdigestables", "ADT_topic"), "EVAPORATE_FULL8": ("results/evaporate", "EVAPORATE_FULL8")}
LABEL = {OURS: "MemoSchema (REUSEC_M)"}
CORP = [(t, f"{s}_{k}") for t in ("e2e7", "e2e8", "e2e9") for s in ("COF", "MOF", "ZIF") for k in ("01", "02", "03")]

def path(m, t, run):
    root, arm = SRC[m]
    return REPO / root / f"runs/{t}/s1/{arm}/chem/{run}/final_schema.json"

def load(m, t, run):
    S = load_schema_file(path(m, t, run), label=f"{t}_{run}")
    for f in S.fields:                                   # name only: signatures use name + aliases + member names
        f.aliases, f.member_field_names = [], []
    return S

def main():
    miss = [str(path(m, *c).relative_to(REPO)) for m in SRC for c in CORP if not path(m, *c).exists()]
    if miss:
        raise SystemExit(f"[stability] missing {len(miss)} first-output schemas:\n  " + "\n  ".join(miss[:12]))
    L = ["# Cross-corpus schema stability (name-only, first output, 36 pairs per subdomain; descriptive)\n",
         "| method | soft Jaccard | COF / MOF / ZIF soft Jaccard | pairs |", "|---|---|---|---|"]
    for m in SRC:
        S = {c: load(m, *c) for c in CORP}; V = {}
        for a, b in itertools.combinations(CORP, 2):
            if a[1][:3] != b[1][:3]: continue
            V[(a, b)] = soft_jaccard(align_schemas(S[a], S[b]), len(S[a].fields), len(S[b].fields))
        sub = " / ".join(f"{st.mean(v for k, v in V.items() if k[0][1].startswith(x)):.3f}" for x in ("COF", "MOF", "ZIF"))
        L.append(f"| {LABEL.get(m, m)} | {st.mean(V.values()):.3f} | {sub} | {len(V)} |")
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "report.md").write_text("\n".join(L)); print("\n".join(L))

if __name__ == "__main__":
    main()
