"""Iterative corpus construction + per-round schema induction (r7-r10 corpora): seed summaries -> queries ->
retrieval -> paper proposals -> candidates -> anchors -> draft, for --rounds rounds. Cross-run reuse is opt-in via
MEMOSCHEMA_PAPER_CACHE_DIRS / MEMOSCHEMA_MERGE_CACHE_DIR / MEMOSCHEMA_SEED_CACHE_DIR (see scripts/run_memexp33.sh)."""
from __future__ import annotations
import argparse, os, re, subprocess, sys
from pathlib import Path
from typing import List, Optional

# Subprocess and run-directory lookup
def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]

def _run(label: str, cmd: List[str], cwd: Optional[Path] = None) -> None:
    print()
    print(f"{'=' * 70}")
    print(f"[Pipeline] STEP: {label}")
    print(f"[Pipeline] CWD : {cwd or Path.cwd()}")
    print(f"[Pipeline] CMD : {' '.join(str(c) for c in cmd)}")
    print(f"{'=' * 70}")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        print(
            f"\n[Pipeline] ERROR: '{label}' exited with code {result.returncode}. "
            "Stopping pipeline.",
            file=sys.stderr,
        )
        sys.exit(result.returncode)
    print(f"[Pipeline] OK: {label}")

def _scan_round_dirs(run_dir: Path) -> List[Path]:
    found = []
    for entry in run_dir.iterdir():
        m = re.match(r"^round_?(\d+)$", entry.name)
        if m and entry.is_dir():
            found.append((int(m.group(1)), entry))
    return [p for _, p in sorted(found)]

def _resolve_actual_run_dir(output_root: Path, domain: str) -> Path:
    # output_root itself is a run directory if it contains round directories.
    if _scan_round_dirs(output_root):
        return output_root

    # Find candidate subdirs
    prefix = f"iterative_phase1_{domain}_"
    candidates = [
        d for d in output_root.iterdir()
        if d.is_dir() and d.name.startswith(prefix) and _scan_round_dirs(d)
    ]
    if not candidates:
        print(
            f"[Pipeline] ERROR: Cannot find a timestamped run dir matching "
            f"'{prefix}*' with round dirs inside {output_root}",
            file=sys.stderr,
        )
        sys.exit(1)
    # Pick the most recently modified one
    best = max(candidates, key=lambda d: d.stat().st_mtime)
    return best

# CLI
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Identity
    p.add_argument("--domain", required=True,
                   choices=["ad", "chem", "cs"],
                   help="Top-level domain (ad | chem | cs).")
    p.add_argument("--subdomain", required=True,
                   help="Subdomain label recorded in the run config (e.g. 'ZIF').")
    p.add_argument("--last-tier", dest="last_tier", default=None,
                   help="Most-specific subdomain term passed to iterative_expand "
                        "(e.g. 'Amyloid beta plaque'). Defaults to --subdomain.")
    p.add_argument("--target-run-id", required=True, dest="target_run_id",
                   help="Seed-set ID recorded in the run config (e.g. '01').")

    # Paths
    p.add_argument("--seed-dir", required=True, dest="seed_dir",
                   help="Directory containing initial seed PDFs.")
    p.add_argument("--output-root", required=True, dest="output_root",
                   help="Base output directory (= run_dir) for iterative_expand.")
    # Retrieval controls (passed to iterative_expand)
    p.add_argument("--rounds", type=int, default=3,
                   help="Number of retrieval rounds (default: 3).")
    p.add_argument("--max-queries", type=int, default=15, dest="max_queries",
                   help="Max queries per round (default: 15).")
    p.add_argument("--results-per-source", type=int, default=20,
                   dest="results_per_source",
                   help="Results per source per query (default: 20).")
    p.add_argument("--sources", default="biorxiv,openalex,pubmed,core",
                   help="Comma-separated retrieval sources (default: biorxiv,openalex,pubmed,core).")
    p.add_argument("--target-total-papers", type=int, default=150,
                   dest="target_total_papers",
                   help="Target total papers to collect (default: 150).")
    p.add_argument("--min-corpus-for-induction", type=int, default=20,
                   dest="min_corpus_for_induction",
                   help="Min corpus size before schema induction runs (default: 20).")

    # Schema induction controls (passed to iterative_expand)
    p.add_argument("--embed-threshold", type=float, default=0.65,
                   dest="embed_threshold",
                   help="Embedding similarity threshold for field clustering (default: 0.65).")
    p.add_argument("--min-support", type=int, default=2, dest="min_support",
                   help="Min proposal support count for a field to survive (default: 2).")
    p.add_argument("--max-final-fields", type=int, default=8, dest="max_final_fields",
                   help="Max fields in the final schema (default: 8).")
    p.add_argument("--max-chars-per-paper", type=int, default=12000, dest="max_chars_per_paper",
                   help="Raw paper text read before distill and the length the distill "
                        "output is rendered to (default: 12000).")
    p.add_argument("--distill-input-chars", type=int, default=60000, dest="distill_input_chars",
                   help="Distill input limit; the text is already cut to --max-chars-per-paper "
                        "(default: 60000).")

    # Resume a crashed run in place
    p.add_argument(
        "--resume-run-dir",
        dest="resume_run_dir",
        default=None,
        help=(
            "Existing timestamped run directory (iterative_phase1_<domain>_*) to "
            "continue with resume_phase1.py."
        ),
    )

    p.add_argument(
        "--fixes",
        default="",
        help=(
            "Comma-separated options (default: none = LEGACY configuration). "
            "cluster: average-linkage field clustering without semantic-twin chaining; "
            "anchorschema: one anchor per schema slot, and anchors (grounded by their "
            "assigned candidates) become the draft fields. The paper uses cluster,anchorschema."
        ),
    )

    return p.parse_args()

_FIX_ENV = {"cluster": "MEMOSCHEMA_FIX_CLUSTER", "anchorschema": "MEMOSCHEMA_FIX_ANCHOR_SCHEMA"}

def _code_fingerprint(repo: Path) -> dict:
    # Provenance for run_config: SHA-256 over every source file that can affect a run (recorded only, not a cache key).
    import hashlib
    h = hashlib.sha256()
    files = sorted(f for d in ("Phase1", "scripts", "setup")
                   for f in (repo / d).rglob("*.py") if "__pycache__" not in f.parts)
    for f in files:
        h.update(str(f.relative_to(repo)).encode())
        h.update(f.read_bytes())
    return {"sha256": h.hexdigest(), "n_files": len(files)}

def _write_run_config(run_dir: Path, args, fixes: set, repo: Path) -> None:
    """Record what produced this run (no secrets: deployment names only)."""
    import datetime, json
    try:
        from dotenv import load_dotenv
        load_dotenv(repo / ".env", override=False)
    except ImportError:
        pass
    now = datetime.datetime.now()
    cfg = {
        "created": now.isoformat(timespec="seconds"),
        "code": _code_fingerprint(repo),
        "config_name": "+".join(sorted(fixes)) or "legacy (original behaviour, no fixes)",
        "fixes": sorted(fixes),
        "fix_env": {env: os.environ.get(env, "") for env in
                    list(_FIX_ENV.values()) + ["MEMOSCHEMA_API_MAX_RETRIES"]},
        "models": {"chat_deployment": os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT"),
                   "embed_deployment": os.environ.get("AZURE_OPENAI_EMBED_DEPLOYMENT", "text-embedding-3-large")},
        "text": {"read_chars": args.max_chars_per_paper,
                 "distill_input_chars": args.distill_input_chars,
                 "render_chars": args.max_chars_per_paper},
        "inputs": {"seed_dir": args.seed_dir, "sources": args.sources, "rounds": args.rounds,
                   "max_queries": args.max_queries, "results_per_source": args.results_per_source,
                   "target_total_papers": args.target_total_papers},
        "args": vars(args),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / f"run_config_{now:%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[Pipeline] run config: {out}")

_KNOWN_FIXES = set(_FIX_ENV)

# Main
def main() -> None:
    args = _parse_args()
    fixes = {f.strip() for f in args.fixes.split(",") if f.strip()}
    unknown = fixes - _KNOWN_FIXES
    if unknown:
        raise SystemExit(f"[Pipeline] ERROR: unknown --fixes {sorted(unknown)}")
    # Subprocess steps inherit the environment.
    for fix, env in _FIX_ENV.items():
        if fix in fixes:
            os.environ[env] = "1"

    repo = _repo_root()
    run_dir = Path(args.output_root).resolve()
    last_tier = args.last_tier or args.subdomain

    # Subprocess steps run with cwd=Phase1, so a relative --seed-dir would
    # resolve against Phase1/ instead of the invocation directory. Resolve
    # it to an absolute path up front.
    args.seed_dir = str(Path(args.seed_dir).resolve())
    if not Path(args.seed_dir).is_dir():
        raise SystemExit(f"[Pipeline] ERROR: seed dir not found: {args.seed_dir}")

    _write_run_config(run_dir, args, fixes, repo)

    print()
    print("=" * 70)
    print("[Pipeline] MemoSchema iterative construction — configuration")
    print(f"  repo root       : {repo}")
    print(f"  domain          : {args.domain}")
    print(f"  subdomain       : {args.subdomain}")
    print(f"  last_tier       : {last_tier}")
    print(f"  target_run_id   : {args.target_run_id}")
    print(f"  seed_dir        : {args.seed_dir}")
    print(f"  run_dir         : {run_dir}")
    print(f"  rounds          : {args.rounds}")
    print(f"  max_queries     : {args.max_queries}")
    print(f"  results/source  : {args.results_per_source}")
    print(f"  sources         : {args.sources}")
    print(f"  target_papers   : {args.target_total_papers}")
    print(f"  min_corpus_ind  : {args.min_corpus_for_induction}")
    print(f"  embed_threshold : {args.embed_threshold}")
    print(f"  min_support     : {args.min_support}")
    print(f"  max_final_fields: {args.max_final_fields}")
    if args.resume_run_dir:
        print(f"  resume_run_dir  : {args.resume_run_dir}")
    print("=" * 70)

    expand_cmd = [
        sys.executable,
        str(repo / "Phase1" / "iterative_expand.py"),
        "--domain", args.domain,
        "--last_tier", last_tier,
        "--seed_dir", args.seed_dir,
        "--output_root", str(run_dir),
        "--rounds", str(args.rounds),
        "--max_queries", str(args.max_queries),
        "--results_per_source", str(args.results_per_source),
        "--sources", args.sources,
        "--target_total_papers", str(args.target_total_papers),
        "--min_corpus_for_induction", str(args.min_corpus_for_induction),
        "--embed_threshold", str(args.embed_threshold),
        "--min_support", str(args.min_support),
        "--max_final_fields", str(args.max_final_fields),
        "--max_chars_per_paper", str(args.max_chars_per_paper),
        "--distill_input_chars", str(args.distill_input_chars),
    ]
    if args.resume_run_dir:
        # Continue a crashed run in place (same fixes via env).
        actual_run_dir = Path(args.resume_run_dir).resolve()
        resume_cmd = [sys.executable, str(repo / "Phase1" / "resume_phase1.py"),
                      "--resume_run_dir", str(actual_run_dir)]
        resume_cmd += [a for a in expand_cmd[2:] if a not in ("--output_root", str(run_dir))]
        _run("resume_phase1", resume_cmd, cwd=repo / "Phase1")
    else:
        _run("iterative_expand", expand_cmd, cwd=repo / "Phase1")
        actual_run_dir = _resolve_actual_run_dir(run_dir, args.domain)
    print()
    print("=" * 70)
    print("[Pipeline] Run complete.")
    print(f"  Run directory : {actual_run_dir}")
    print("=" * 70)

if __name__ == "__main__":
    main()
