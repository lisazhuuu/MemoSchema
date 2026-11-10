"""memexp34: name-only judging of the first schema output (sample 1) of every (method, corpus). Stages: build | report

MemoSchema = final content-based reuse configuration: results/memexp38/runs/<tag>/s1/REUSEC_M/chem/<run>/final_schema.json
(anchor sample 1). It is required for all 27 corpora; a missing schema or score stops the stage.
Name-only copies: results/memexp34/runs/<tag>/s1/<method>/chem/<run>/final_schema.json; judged by scripts/run_memexp34.sh."""
import glob, json, random, statistics as st, sys
from collections import defaultdict
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]; OUT = REPO / "results/memexp34"
OURS = "REUSEC_M"
SRC = {OURS: ("results/memexp38", "REUSEC_M"), "LEGACY": ("results/memexp26", "LEGACY"),
       "AUTOSCHEMA": ("results/autoschema", "AUTOSCHEMA"), "L4S": ("results/memexp27", "L4S"),
       "ADT_topic": ("results/arxivdigestables", "ADT_topic"), "EVAPORATE_FULL8": ("results/evaporate", "EVAPORATE_FULL8")}
LABEL = {OURS: "MemoSchema (REUSEC_M)"}
CORP = [(t, f"{s}_{k}") for t in ("e2e7", "e2e8", "e2e9") for s in ("COF", "MOF", "ZIF") for k in ("01", "02", "03")]

def require(paths, what):
    miss = [str(p.relative_to(REPO)) for p in paths if not p.exists()]
    if miss:
        raise SystemExit(f"[memexp34] missing {len(miss)} {what} (run scripts/run_memexp38.sh first):\n  " + "\n  ".join(miss[:10]))

def source(m, tag, run):
    root, arm = SRC[m]
    return REPO / root / f"runs/{tag}/s1/{arm}/chem/{run}/final_schema.json"

def build():
    require([source(OURS, t, r) for t, r in CORP], "MemoSchema REUSEC_M first-output schemas")
    n = 0
    for m, (root, arm) in SRC.items():
        files = [source(m, t, r) for t, r in CORP] if m == OURS else \
                [Path(f) for f in glob.glob(str(REPO / root / f"runs/e2e*/s1/{arm}/chem/*/final_schema.json"))]
        for f in files:
            p = f.parts; tag, run = p[p.index("runs") + 1], p[-2]
            S = [{"name": x.get("name", ""), "description": ""} for x in json.load(open(f))]
            od = OUT / "runs" / tag / "s1" / m / "chem" / run; od.mkdir(parents=True, exist_ok=True)
            (od / "final_schema.json").write_text(json.dumps(S, ensure_ascii=False, indent=1)); n += 1
    print("name-only schemas", n)

def score(path, fine):
    if not Path(path).exists(): return None
    return json.load(open(path))["fine_f1"] if fine else float(open(path).read().splitlines()[1].split(",")[11])

def boot(d, keys):
    """Paired mean and 95% bootstrap interval over corpora, stratified by subdomain."""
    by = defaultdict(list)
    for k in keys: by[k[1][:3]].append(k)
    random.seed(0)
    bs = sorted(st.mean([d[k] for ks in by.values() for k in random.choices(ks, k=len(ks))]) for _ in range(5000))
    return st.mean(d[k] for k in keys), bs[125], bs[4874]

def report():
    L = ["# memexp34 report: name-only judging (first output, project judge rep 1)\n"]
    for fine in (True, False):
        ev, fn = ("eval_fine", "metrics_fine.json") if fine else ("eval", "metrics.csv")
        require([OUT / f"runs/{t}/s1/{ev}/{OURS}/rep1/chem/{r}/{fn}" for t, r in CORP], f"name-only {ev} scores of {OURS}")
        require([REPO / SRC[OURS][0] / f"runs/{t}/s1/{ev}/{OURS}/rep1/chem/{r}/{fn}" for t, r in CORP], f"full-field {ev} scores of {OURS}")
        N, F = defaultdict(dict), defaultdict(dict)
        for m, (root, arm) in SRC.items():
            for f in glob.glob(str(OUT / f"runs/e2e*/s1/{ev}/{m}/rep1/chem/*/{fn}")):
                p = Path(f).parts; tag, run = p[p.index("runs") + 1], p[-2]
                N[m][(tag, run)] = score(f, fine)
                F[m][(tag, run)] = score(REPO / root / f"runs/{tag}/s1/{ev}/{arm}/rep1/chem/{run}/{fn}", fine)
        L += [f"\n## {'hierarchical' if fine else 'strict'} F1", "| method | full fields | name only | n |", "|---|---|---|---|"]
        incomplete = [f"{m} ({name}): {sum(S[m].get(k) is not None for k in CORP)}/{len(CORP)}" for m in SRC
                      for name, S in (("full", F), ("name-only", N)) if any(S[m].get(k) is None for k in CORP)]
        if incomplete:                                    # paper setting: 27 corpora per method and setting, judge rep 1
            raise SystemExit(f"[memexp34] incomplete {ev} scores, no report written:\n  " + "\n  ".join(incomplete))
        if not fine:                                      # no failure placeholder may count as a score
            import completeness as C
            C.require_valid([str(Path(f).relative_to(REPO)) for m, (root, arm) in SRC.items() for t, r in CORP for f in (
                OUT / f"runs/{t}/s1/eval/{m}/rep1/chem/{r}/metrics.csv", REPO / root / f"runs/{t}/s1/eval/{arm}/rep1/chem/{r}/metrics.csv")],
                "memexp34")
        for m in SRC:
            ks = [k for k in CORP if N[m].get(k) is not None and F[m].get(k) is not None]
            if ks: L.append(f"| {LABEL.get(m, m)} | {st.mean(F[m][k] for k in ks):.3f} | {st.mean(N[m][k] for k in ks):.3f} | {len(ks)} |")
        L.append(f"\n{LABEL[OURS]} - baseline (paired over corpora, bootstrap stratified by subdomain, 95%):")
        for m in SRC:
            if m == OURS: continue
            for name, S in (("full", F), ("name-only", N)):
                ks = [k for k in CORP if S[m].get(k) is not None and S[OURS].get(k) is not None]   # fixed corpus order
                if len(ks) < 3: continue
                d = {k: S[OURS][k] - S[m][k] for k in ks}; mean, lo, hi = boot(d, ks)
                L.append(f"- vs {m} [{name}]: {mean:+.3f} [{lo:+.3f}, {hi:+.3f}] (n={len(ks)})")
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "report.md").write_text("\n".join(L)); print("\n".join(L))

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("build", "report"):
        print(__doc__ if sys.argv[1:2] in (["-h"], ["--help"]) else "usage: python scripts/memexp34.py build|report")
        sys.exit(0 if sys.argv[1:2] in (["-h"], ["--help"]) else 2)
    {"build": build, "report": report}[sys.argv[1]]()
