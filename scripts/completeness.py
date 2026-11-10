"""Completeness checks for the summary scripts: the paper's corpus, sample and judge-repetition counts.

Every summary stops (no report written) when a judged score, record or input of its protocol is missing, instead of
averaging over what happens to be present. Counts (paper §VI-A): 27 evaluation corpora r7-r9 (e2e7-e2e9 x COF/MOF/ZIF
x 01-03); each schema judged twice; MemoSchema and the no-reuse control: 6 anchor samples; AutoSchema,
LLMs4SchemaDiscovery, ArxivDIGESTables, IA_SR and SR_DRAFT: 3 runs; EVAPORATE: 1 run; LEGACY_M: first sample only."""
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CORP = [(t, f"{s}_{k}") for t in ("e2e7", "e2e8", "e2e9") for s in ("COF", "MOF", "ZIF") for k in ("01", "02", "03")]
R10 = [f"{s}_{k}" for s in ("COF", "MOF", "ZIF") for k in ("01", "02", "03")]
# (results root, arm) -> (samples, judge repetitions)
PROTOCOL = {("results/memexp38", "REUSEC_M"): (6, 2),
            **{("results/memexp26", a): (6, 2) for a in ("ORIG", "ORIG2", "ORIG3", "ORIG_M", "ORIG2_M", "ORIG3_M", "LEGACY")},
            ("results/memexp26", "LEGACY_M"): (1, 2),
            ("results/autoschema", "AUTOSCHEMA"): (3, 2), ("results/memexp27", "L4S"): (3, 2),
            ("results/arxivdigestables", "ADT_topic"): (3, 2), ("results/memexp40", "IA_SR"): (3, 2),
            ("results/memexp40", "SR_DRAFT"): (3, 2), ("results/evaporate", "EVAPORATE_FULL8"): (1, 2)}
METRIC = {"eval_fine": "metrics_fine.json", "eval": "metrics.csv"}
# Placeholder written by the earlier judge code when every API attempt failed (all references "unmatched", F1 = 0).
# A score file next to such an alignment is not a judgement and must not be averaged.
FAILED_MARKERS = ("judge_call_failed",)

def failed_judgement(path):
    """True if a strict-judge output (metrics.csv, its directory, or an alignments/*.csv) is a failure placeholder."""
    p = REPO / path
    files = [p] if p.suffix == ".csv" and p.parent.name == "alignments" else \
        sorted((p.parent if p.name == "metrics.csv" else p).glob("alignments/*.csv"))
    return any(m in f.read_text(errors="replace") for f in files if f.exists() for m in FAILED_MARKERS)

def require_valid(paths, who):
    bad = sorted({str(Path(p).parent if Path(p).parent.name != "alignments" else Path(p).parent.parent) for p in paths
                  if (str(p).endswith("metrics.csv") or "/alignments/" in str(p)) and failed_judgement(p)})
    if bad:
        raise SystemExit(f"[{who}] {len(bad)} judge outputs are failure placeholders ({', '.join(FAILED_MARKERS)}: every API "
                         "attempt failed), not scores; no report written. Re-judge them first, e.g.\n  " + "\n  ".join(bad[:8]))

def missing_scores(root, arm, evals=("eval_fine", "eval")):
    n_s, n_r = PROTOCOL[(root, arm)]
    return [f"{root}/runs/{t}/s{s}/{ev}/{arm}/rep{r}/chem/{run}/{METRIC[ev]}" for ev in evals for t, run in CORP
            for s in range(1, n_s + 1) for r in range(1, n_r + 1)
            if not (REPO / root / f"runs/{t}/s{s}/{ev}/{arm}/rep{r}/chem/{run}/{METRIC[ev]}").exists()]

def require_files(paths, what, who, check_valid=True):
    miss = [str(p) for p in paths if not (REPO / p).exists()]
    if miss:
        raise SystemExit(f"[{who}] incomplete {what}: {len(miss)} missing, no report written, e.g.\n  " + "\n  ".join(miss[:8]))
    if check_valid:
        require_valid(paths, who)

def require_scores(arms, who, evals=("eval_fine", "eval")):
    """arms: iterable of (root, arm) keys of PROTOCOL; every corpus x sample x judge repetition must be scored
    (and no strict score may be a failure placeholder)."""
    arms = list(arms)
    miss = [m for key in arms for m in missing_scores(*key, evals=evals)]
    if miss:
        raise SystemExit(f"[{who}] incomplete judged scores (protocol: 27 corpora x samples x 2 judge reps): {len(miss)} "
                         f"missing, no report written, e.g.\n  " + "\n  ".join(miss[:8]))
    require_valid([f"{root}/runs/{t}/s{s}/eval/{arm}/rep{r}/chem/{run}/metrics.csv" for root, arm in arms
                   for t, run in CORP for s in range(1, PROTOCOL[(root, arm)][0] + 1)
                   for r in range(1, PROTOCOL[(root, arm)][1] + 1) if "eval" in evals], who)
