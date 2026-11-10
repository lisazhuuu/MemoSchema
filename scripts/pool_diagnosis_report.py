"""Introduction diagnostic numbers (no API calls), recomputed from saved judge outputs.
Corpora: the nine development runs of scripts/dev_diagnostic.py (seed set 02 of chem / AD / CS; pool and baseline
from the same Ablation/ run of each corpus). Project one-to-one (coarse) judge.
  pool      top-40 candidates by the final-selection rank   results/abl_pool_old/eval/top40/rep*/<domain>/<run>/metrics.csv
  baseline  8-field schema of the baseline configuration    results/abl_old/eval/baseline/rep*/<domain>/<run>/metrics.csv
Coverage = lenient recall. Scores are averaged over the judge repetitions within each corpus, then over the nine corpora.
Every corpus needs at least one judged repetition on both sides, otherwise the report stops.
Output: results/pool_diagnosis/report.md"""
import csv, glob, statistics as st, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
if __name__ == "__main__" and (sys.argv[1:]):                     # usage check before any work
    print(__doc__ if sys.argv[1:2] in (["-h"], ["--help"]) else "usage: python scripts/" + Path(__file__).name + " " + "(no arguments)")
    sys.exit(0 if sys.argv[1:2] in (["-h"], ["--help"]) else 2)
sys.path.insert(0, str(REPO / "scripts"))
from dev_diagnostic import BASE_ROOT, DEV_RUNS, POOL_ROOT

SIDES = {"pool": f"{POOL_ROOT}/eval/top40", "baseline": f"{BASE_ROOT}/eval/baseline"}
COLS = ("lenient_recall", "strict_recall", "strict_f1")

def scores(side):
    per, miss = {}, []
    for d, r in DEV_RUNS:
        ms = sorted(glob.glob(str(REPO / SIDES[side] / "rep*" / d / r / "metrics.csv")))
        if not ms:
            miss.append(f"{SIDES[side]}/rep*/{d}/{r}/metrics.csv"); continue
        rows = [next(csv.DictReader(open(m))) for m in ms]
        per[(d, r)] = {c: st.mean(float(x[c]) for x in rows) for c in COLS} | {"reps": len(rows)}
    return per, miss

def main():
    S, miss = {}, []
    for side in SIDES:
        S[side], m = scores(side); miss += m
    if miss:
        raise SystemExit(f"[diagnosis] missing judged schemas for {len(miss)} corpus/side pairs:\n  " + "\n  ".join(miss))
    import completeness as C                    # no failure placeholder may count as a score
    C.require_valid([str(Path(m).relative_to(REPO)) for side in SIDES for d, r in DEV_RUNS
                     for m in glob.glob(str(REPO / SIDES[side] / "rep*" / d / r / "metrics.csv"))], "diagnosis")
    L = ["# Introduction diagnostic (9 development corpora, seed set 02, same Ablation/ run on both sides; one-to-one judge)\n",
         "Per-corpus mean over judge repetitions, then mean over corpora.\n",
         "| corpus | pool lenient recall (reps) | baseline lenient recall (reps) |", "|---|---|---|"]
    L += [f"| {d}/{r} | {S['pool'][d, r]['lenient_recall']:.3f} ({S['pool'][d, r]['reps']}) | "
          f"{S['baseline'][d, r]['lenient_recall']:.3f} ({S['baseline'][d, r]['reps']}) |" for d, r in DEV_RUNS]
    for side, label in (("pool", "top-40 pool"), ("baseline", "baseline 8-field schema")):
        v = S[side].values()
        L.append(f"\n- {label}: coverage (lenient recall) {st.mean(x['lenient_recall'] for x in v):.3f}, "
                 f"strict recall {st.mean(x['strict_recall'] for x in v):.3f}, strict F1 {st.mean(x['strict_f1'] for x in v):.3f} "
                 f"(n={len(v)} corpora, {sum(x['reps'] for x in v)} judged schemas)")
    out = REPO / "results/pool_diagnosis"; out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text("\n".join(L) + "\n"); print("\n".join(L))

if __name__ == "__main__":
    main()
