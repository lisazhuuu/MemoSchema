from __future__ import annotations
import argparse, os, sys
from pathlib import Path
from typing import Optional

_THIS_DIR = Path(__file__).resolve().parent
_PHASE1_DIR = _THIS_DIR.parent
_REPO_ROOT = _THIS_DIR.parents[1]
if str(_PHASE1_DIR) not in sys.path:
    sys.path.insert(0, str(_PHASE1_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from .schema_cluster_merge import build_canonical_candidates, select_final_schema
from .subdomain_discovery import discover_subdomain_anchors

# Generate canonical candidates from per-paper proposals, discover corpus-driven anchors, then select final schema.
def cluster_merge_from_paper_level_jsonl(
    paper_level_jsonl: Path,
    out_dir: Path,
    domain: str,
    last_tier: Optional[str] = None,
    embed_threshold: float = 0.65,
    min_support: int = 2,
    max_final_fields: int = 8,
    n_anchors: int = 6,
    skip_discovery: bool = False,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    canonicals, stats = build_canonical_candidates(
        paper_level_jsonl=paper_level_jsonl,
        out_dir=out_dir,
        domain=domain,
        embed_threshold=embed_threshold,
    )

    # Default to None so that select_final_schema falls back to loading discovered_anchors.json from out_dir (written by an earlier round).
    anchors = None
    # With anchors used as the schema skeleton, discover one anchor per slot.
    if os.getenv("MEMOSCHEMA_FIX_ANCHOR_SCHEMA") == "1":
        n_anchors = max_final_fields
    if not skip_discovery and last_tier:
        # Lazy import to avoid a hard dependency when discovery is skipped
        from schema_induction import llm_chat
        anchors = discover_subdomain_anchors(
            llm_chat_fn=llm_chat,
            paper_level_jsonl=paper_level_jsonl,
            domain=domain,
            last_tier=last_tier,
            n_anchors=n_anchors,
            out_path=out_dir / "discovered_anchors.json",
        )
        print(f"[Refine][Discover] {len(anchors)} corpus anchors emerged for {last_tier!r}")

    final = select_final_schema(
        canonical_candidates=canonicals,
        out_dir=out_dir,
        domain=domain,
        last_tier=last_tier,
        max_final_fields=max_final_fields,
        min_support=min_support,
        anchors=anchors,
    )

    print(f"[Refine][ClusterMerge] stats={stats}")
    print(f"[Refine][ClusterMerge] canonical_candidates={len(canonicals)} final={len(final)}")
    return final

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--paper_level_jsonl", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--domain", required=True)
    p.add_argument("--last_tier", type=str, default=None)
    p.add_argument("--embed_threshold", type=float, default=0.65)
    p.add_argument("--min_support", type=int, default=2)
    p.add_argument("--max_final_fields", type=int, default=8)
    p.add_argument("--n_anchors", type=int, default=6)
    p.add_argument("--skip_discovery", action="store_true",
                   help="Skip corpus-driven anchor discovery (no LLM call).")
    return p.parse_args()

def main():
    args = _parse_args()
    cluster_merge_from_paper_level_jsonl(
        paper_level_jsonl=Path(args.paper_level_jsonl),
        out_dir=Path(args.out_dir),
        domain=args.domain,
        last_tier=args.last_tier,
        embed_threshold=args.embed_threshold,
        min_support=args.min_support,
        max_final_fields=args.max_final_fields,
        n_anchors=args.n_anchors,
        skip_discovery=args.skip_discovery,
    )

if __name__ == "__main__":
    main()
