"""Slot analysis (no LLM calls): with a fixed budget of 8 fields, how many slots are wasted and how many reference concepts
do the slots cover, for LEGACY -> ORIG (cluster + anchors) -> ORIG_M (+ restructuring)? 27 corpora, reuse disabled (memexp26).
From the saved judge outputs (project judge, all reps):
  fine   idle slots = fields to which no reference concept is assigned; concepts covered (strong or partial) / all
         reference concepts; concepts per active slot
  coarse matched slots = fields matched one-to-one to a reference concept / all fields
Output: results/slot_analysis/report.md"""
import csv, glob, json, random, statistics as st, sys
from collections import defaultdict
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]; M = REPO / "results/memexp26"; OUT = REPO / "results/slot_analysis"
if __name__ == "__main__" and (sys.argv[1:]):                     # usage check before any work
    print(__doc__ if sys.argv[1:2] in (["-h"], ["--help"]) else "usage: python scripts/" + Path(__file__).name + " " + "(no arguments)")
    sys.exit(0 if sys.argv[1:2] in (["-h"], ["--help"]) else 2)
sys.path.insert(0, str(REPO / "Evaluation"))
from stability_lib.schema_io import load_schema_file
ARMS = {"LEGACY": ["LEGACY"], "LEGACY_M": ["LEGACY_M"], "ORIG": ["ORIG", "ORIG2", "ORIG3"], "ORIG_M": ["ORIG_M", "ORIG2_M", "ORIG3_M"]}
CORP = [(t, f"{s}_{k}") for t in ("e2e7", "e2e8", "e2e9") for s in ("COF", "MOF", "ZIF") for k in ("01", "02", "03")]

def schema_metrics(tag, run, arm):
    sp = M / f"runs/{tag}/s1/{arm}/chem/{run}/final_schema.json"
    if not sp.exists(): return None
    S = load_schema_file(sp); names = [f.name for f in S.fields]; n = len(names)
    fine = []
    for a in glob.glob(str(M / f"runs/{tag}/s1/eval_fine/{arm}/rep*/chem/{run}/alignment_fine.json")):
        A = json.load(open(a)); cov = [x for x in A if x["strength"] in ("strong", "partial") and x["field"]]
        used = {x["field"] for x in cov}
        fine.append({"idle": sum(nm not in used for nm in names) / n, "covered": len(cov) / max(1, len(A)),
                     "per_active": len(cov) / max(1, len(used & set(names)))})
    coarse = []
    for c in glob.glob(str(M / f"runs/{tag}/s1/eval/{arm}/rep*/chem/{run}/alignments/*.csv")):
        R = list(csv.DictReader(open(c)))
        matched = {r["induced_name"] for r in R if r.get("status") == "matched" and r.get("induced_name")}
        coarse.append(len(matched & set(names)) / n)
    if not fine: return None
    return {"idle": st.mean(x["idle"] for x in fine), "covered": st.mean(x["covered"] for x in fine),
            "per_active": st.mean(x["per_active"] for x in fine), "matched": st.mean(coarse) if coarse else None, "n": n}

def arm_values():
    V = defaultdict(dict)
    for name, arms in ARMS.items():
        for c in CORP:
            rows = [r for a in arms if (r := schema_metrics(*c, a))]
            if rows: V[name][c] = {k: st.mean(r[k] for r in rows if r[k] is not None) for k in rows[0]}
    return V

def contrast(V, A, B, k):
    D = {c: V[A][c][k] - V[B][c][k] for c in V[A] if c in V[B]}; by = defaultdict(list)
    for c in D: by[c[1][:3]].append(c)
    random.seed(0); bs = sorted(st.mean([D[c] for cs in by.values() for c in random.choices(cs, k=len(cs))]) for _ in range(5000))
    return f"{st.mean(D.values()):+.3f} [{bs[125]:+.3f}, {bs[4874]:+.3f}]"

def main():
    import completeness as C                    # first sample of each arm, both judge reps, 27 corpora
    C.require_files([f"results/memexp26/runs/{t}/s1/{a}/chem/{r}/final_schema.json" for n in ARMS.values() for a in n for t, r in CORP]
                    + [f"results/memexp26/runs/{t}/s1/eval_fine/{a}/rep{k}/chem/{r}/alignment_fine.json" for n in ARMS.values() for a in n
                       for t, r in CORP for k in (1, 2)]
                    + [f"results/memexp26/runs/{t}/s1/eval/{a}/rep{k}/chem/{r}/alignments/{r[:3]}.csv" for n in ARMS.values() for a in n
                       for t, r in CORP for k in (1, 2)], "schemas and judge alignments", "slot_analysis")
    V = arm_values(); OUT.mkdir(exist_ok=True)
    K = [("idle", "idle slots (fine: no concept assigned)"), ("covered", "reference concepts covered (fine)"),
         ("per_active", "concepts per active slot (fine)"), ("matched", "slots matched one-to-one (coarse)")]
    L = ["# Slot analysis: 8-field budget, 27 fresh corpora (memexp26), project judge, all reps\n",
         "| arm | " + " | ".join(d for _, d in K) + " | fields |", "|---|" + "---|" * (len(K) + 1)]
    for a in ARMS:
        L.append(f"| {a} | " + " | ".join(f"{st.mean(v[k] for v in V[a].values()):.3f}" for k, _ in K) + f" | {st.mean(v['n'] for v in V[a].values()):.1f} |")
    L += ["\nPaired contrasts (per corpus, bootstrap stratified by subdomain, 95%):", "| contrast | " + " | ".join(d for _, d in K) + " |", "|---|" + "---|" * len(K)]
    for A, B in (("ORIG", "LEGACY"), ("ORIG_M", "ORIG"), ("ORIG_M", "LEGACY_M"), ("LEGACY_M", "LEGACY")):
        L.append(f"| {A} - {B} | " + " | ".join(contrast(V, A, B, k) for k, _ in K) + " |")
    (OUT / "report.md").write_text("\n".join(L)); print("\n".join(L))

if __name__ == "__main__":
    main()
