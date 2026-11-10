"""Inputs of the introduction diagnostic (shared by pool_coverage.py, dev_baseline_schemas.py, pool_diagnosis_report.py).

Nine development corpora: seed set 02 of the nine subdomains (chem COF/MOF/ZIF, ad amyloid/neuroinflammation/tau,
cs RLHF/llm/prompt). Both sides of the comparison come from the same run of each corpus, <repo>/Ablation/<domain>/<SUB>_02
(manifests: data/corpora_dev/<domain>/<SUB>_02.json):
  pool      top-40 candidates by the final-selection rank       -> results/abl_pool_old/top40/<domain>/<run>/
  baseline  8-field schema of the baseline (LEGACY) selection   -> results/abl_old/baseline/<domain>/<run>/
Both sides read round03/schema_from_cumulative_corpus/ of that run (the baseline also its discovered_anchors.json), with the
LEGACY selection (MEMOSCHEMA_FIX_* unset). Missing inputs raise an error; there is no fallback to other rounds or run sets."""
import os
from pathlib import Path
import input_package

REPO = Path(__file__).resolve().parents[1]
RUN_SET = "Ablation"
DEV_RUNS = [(d, f"{s}_02") for d, subs in (("chem", ("COF", "MOF", "ZIF")), ("ad", ("amyloid", "neuroinflammation", "tau")),
                                            ("cs", ("RLHF", "llm", "prompt"))) for s in subs]
POOL_ROOT, BASE_ROOT = "results/abl_pool_old", "results/abl_old"
ROUND = "round03"
FIX_FLAGS = ("MEMOSCHEMA_FIX_ANCHOR_SCHEMA", "MEMOSCHEMA_FIX_CLUSTER")

def legacy_env():
    """The diagnostic used the LEGACY selection; clear the fix flags whatever the caller's environment holds."""
    for k in FIX_FLAGS:
        os.environ.pop(k, None)

def round_file(rd: Path, name: str) -> Path:
    """<run>/round03/schema_from_cumulative_corpus/<name>; error if missing (no other round is used)."""
    f = rd / ROUND / "schema_from_cumulative_corpus" / name
    if not f.exists():
        raise SystemExit(f"[dev] missing {f.relative_to(REPO)}")
    return f

def run_dirs(run_set: str = RUN_SET) -> list[Path]:
    """The nine run directories; error if any is missing."""
    input_package.check("dev", "dev_diagnostic")
    dirs = [REPO / run_set / d / r for d, r in DEV_RUNS]
    miss = [str(p.relative_to(REPO)) for p in dirs if not (p / "schema_state/paper_level_field_proposals.jsonl").exists()]
    if miss:
        raise SystemExit(f"[dev] missing {len(miss)} development runs (see README, development corpora):\n  " + "\n  ".join(miss))
    return dirs
