"""Recompute the schema-quality numbers of the paper from the saved judge outputs in results/ (no LLM calls).

Table IV  mean hierarchical / strict F1 per method (27 corpora; repetitions and judge reps averaged within a corpus)
          and paired differences MemoSchema - baseline (95% bootstrap over corpora, stratified by subdomain).
Table V   organized draft vs LEGACY, restructured vs input draft (no-reuse control, memexp26 resampling),
          organized vs LEGACY draft with both restructured (first anchor sample), restructuring vs SR_DRAFT.
Usage: python scripts/paper_tables.py"""
import json, random, statistics as st, sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if __name__ == "__main__" and (sys.argv[1:]):                     # usage check before any work
    print(__doc__ if sys.argv[1:2] in (["-h"], ["--help"]) else "usage: python scripts/" + Path(__file__).name + " " + "(no arguments)")
    sys.exit(0 if sys.argv[1:2] in (["-h"], ["--help"]) else 2)
sys.path[:0] = [str(REPO / "scripts"), str(REPO / "Phase1")]
SUBS = ("COF", "MOF", "ZIF")
METHODS = {"MemoSchema": ("results/memexp38", "REUSEC_M"), "AutoSchema": ("results/autoschema", "AUTOSCHEMA"),
           "EVAPORATE": ("results/evaporate", "EVAPORATE_FULL8"), "LLMs4SchemaDiscovery": ("results/memexp27", "L4S"),
           "ArxivDIGESTables": ("results/arxivdigestables", "ADT_topic"), "IA_SR": ("results/memexp40", "IA_SR")}

def load(root, hierarchical):
    """(tag, run, arm, sample, rep) -> scores, for the 27 fresh corpora (batches e2e7-e2e9)."""
    pat = "runs/e2e[789]/s*/" + ("eval_fine/*/rep*/chem/*/metrics_fine.json" if hierarchical else "eval/*/rep*/chem/*/metrics.csv")
    S = defaultdict(list)
    for f in (REPO / root).glob(pat):
        p = f.parts; i = p.index("runs")
        v = json.load(open(f))["fine_f1"] if hierarchical else float(open(f).read().splitlines()[1].split(",")[11])
        S[(p[i + 1], p[i + 7], p[i + 4], p[i + 2], p[i + 5])].append(v)
    return S

def per_corpus(S, arms, samples=None):
    D = defaultdict(list)
    for (t, r, a, s, _), v in S.items():
        if a in arms and (samples is None or s in samples):
            D[(t, r)] += v
    return {k: st.mean(v) for k, v in D.items()}

def paired(A, B, draws=5000):
    # Fixed order (sorted corpus keys, subdomains COF, MOF, ZIF) so the bootstrap does not depend on directory listing order.
    D = {k: A[k] - B[k] for k in sorted(A) if k in B}
    by = defaultdict(list)
    for k in D:
        by[k[1][:3]].append(k)
    groups = [by[s] for s in sorted(by)]
    random.seed(0)
    bs = sorted(st.mean([D[k] for g in groups for k in random.choices(g, k=len(g))]) for _ in range(draws))
    return f"{st.mean(D.values()):+.3f} [{bs[int(.025 * draws)]:+.3f}, {bs[int(.975 * draws) - 1]:+.3f}] (n={len(D)})"

def main():
    import memexp26 as m26
    import completeness as C                    # paper protocol: 27 corpora x samples x 2 judge reps, every arm used below
    C.require_scores([("results/memexp38", "REUSEC_M"), ("results/autoschema", "AUTOSCHEMA"), ("results/evaporate", "EVAPORATE_FULL8"),
                      ("results/memexp27", "L4S"), ("results/arxivdigestables", "ADT_topic"), ("results/memexp40", "IA_SR"),
                      ("results/memexp40", "SR_DRAFT")] + [("results/memexp26", a) for a in
                     ("ORIG", "ORIG2", "ORIG3", "ORIG_M", "ORIG2_M", "ORIG3_M", "LEGACY", "LEGACY_M")], "paper_tables")
    out = []
    for hier in (True, False):
        name = "hierarchical F1" if hier else "strict one-to-one F1"
        cache = {}
        V = {}
        for m, (root, arm) in METHODS.items():
            if root not in cache:
                cache[root] = load(root, hier)
            V[m] = per_corpus(cache[root], [arm])
        out.append(f"\n## Table IV, {name}")
        out += [f"- {m}: {st.mean(V[m].values()):.3f} (n={len(V[m])})" for m in METHODS]
        out += [f"- MemoSchema - {m}: {paired(V['MemoSchema'], V[m])}" for m in METHODS if m != "MemoSchema"]

        m26_scores = cache.setdefault("results/memexp26", load("results/memexp26", hier))
        S = m26.scores(hier)
        runs = sorted({(k[0], k[1]) for k in S if k[0] in ("e2e7", "e2e8", "e2e9")})
        orig, orig_m = ["ORIG", "ORIG2", "ORIG3"], ["ORIG_M", "ORIG2_M", "ORIG3_M"]
        out.append(f"\n## Table V, {name}")
        for label, A, B in (("organized draft vs LEGACY", orig, ["LEGACY"]), ("restructured vs input draft", orig_m, orig)):
            c = m26.contrast(S, runs, A, B, 2000)
            out.append(f"- {label}: {c['mean']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}] (n={c['n']}; resampling corpora, reconstructions, anchor samples)")
        first = ["s1"]
        # Retain the original sample-1 contrast and its bootstrap draw sequence.
        saved_samples = m26.SAMPLES
        try:
            m26.SAMPLES = [1]
            c = m26.contrast(S, runs, ["ORIG_M"], ["LEGACY_M"], 5000)
        finally:
            m26.SAMPLES = saved_samples
        out.append("- organized vs LEGACY draft, both restructured (first anchor sample): "
                   + f"{c['mean']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}] (n={c['n']})")
        sr = per_corpus(load("results/memexp40", hier), ["SR_DRAFT"])
        out.append("- restructuring vs SR_DRAFT (same draft and candidates): " + paired(per_corpus(m26_scores, ["ORIG_M"], first), sr))
        out.append("- SR_DRAFT vs input draft: " + paired(sr, per_corpus(m26_scores, ["ORIG"], first)))
    print("\n".join(out))

if __name__ == "__main__":
    main()
