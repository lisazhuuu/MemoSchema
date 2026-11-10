from __future__ import annotations

import hashlib, inspect, json, os, shutil
from pathlib import Path
from typing import Any, Dict, List, Set

# JSONL file operations
def _save_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def _append_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def _count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for ln in path.open("r", encoding="utf-8") if ln.strip())

# Cross-run paper memory: content-addressed fingerprint of one per-paper call.
# Covers the paper text the operator reads, the code that builds its prompts (proposal + distill module), the domain
# spec, every argument and the deployment, so a cached result is reused only for exactly the same input and operator.
def paper_memo_fingerprint(mods, args, text: str, raw_chars: int) -> str:
    dp = mods.design_prompt
    spec = dp.get_domain_spec(args.domain)
    op = [inspect.getsource(dp.propose_paper_level_candidate_fields),
          inspect.getsource(inspect.getmodule(dp.distill_paper_for_schema)),
          json.dumps(getattr(spec, "__dict__", str(spec)), default=str, sort_keys=True)]
    params = [args.domain, args.last_tier, args.max_candidate_fields, args.max_chars_per_paper, args.distill_input_chars,
              args.distill_quote_limit, bool(args.no_distill), raw_chars, os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "")]
    return hashlib.sha256(json.dumps([hashlib.sha256(text.encode("utf-8")).hexdigest(), op, params]).encode()).hexdigest()

def _read_fp(cache_dir: Path, pid: str) -> str:
    f = cache_dir / f"{pid}.memo_fp"
    return f.read_text(encoding="utf-8").strip() if f.exists() else ""

# Distill cache
def _cached_paper_ids(distill_cache_dir: Path) -> Set[str]:
    if not distill_cache_dir.exists():
        return set()
    return {p.stem for p in distill_cache_dir.glob("*.json")}

def _save_distill_rows(distill_cache_dir: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    distill_cache_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        pid = row.get("paper_id", "")
        if not pid:
            continue
        (distill_cache_dir / f"{pid}.json").write_text(
            json.dumps(row, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

# Main entry point used by iterative_expand
def run_round_schema_step(
    *,
    mods,
    args,
    round_idx: int,
    new_pdfs_dir: Path,
    schema_state_dir: Path,
    round_schema_out: Path,
    per_round_distill_cap: int,
) -> Dict[str, Any]:
    schema_state_dir.mkdir(parents=True, exist_ok=True)
    round_schema_out.mkdir(parents=True, exist_ok=True)

    cumulative_jsonl = schema_state_dir / "paper_level_field_proposals.jsonl"
    distill_cache_dir = schema_state_dir / "distill_cache"

    # Raw text is cut to max_chars_per_paper before distill.
    raw_chars = args.max_chars_per_paper

    # Per-paper distill debug fan-out (consumed inside schema_induction.py)
    os.environ["AUTOSCHEMA_DEBUG_DIR"] = str(round_schema_out / "paper_level_debug")

    round_pdfs = sorted(new_pdfs_dir.glob("*.pdf"))
    if not round_pdfs:
        return {
            "round": round_idx,
            "selected_papers_this_round": 0,
            "new_papers_distilled": 0,
            "cached_hits_this_round": 0,
            "cumulative_paper_rows": _count_jsonl_rows(cumulative_jsonl),
            "final_schema_fields": None,
            "final_schema_path": None,
            "skipped_reason": f"no PDFs in {new_pdfs_dir}",
        }

    # 1) Pick a representative subset of this round's new PDFs
    max_papers = min(per_round_distill_cap, args.max_schema_papers, len(round_pdfs))
    candidate_pool_size = min(
        max(args.schema_candidate_pool_size, max_papers),
        len(round_pdfs),
    )

    selected = mods.design_prompt.select_schema_papers(
        pdf_dir=new_pdfs_dir,
        domain=args.domain,
        last_tier=args.last_tier,
        max_papers=max_papers,
        candidate_pool_size=candidate_pool_size,
        rep_chars=args.schema_rep_chars,
        diversity_threshold=args.schema_diversity_threshold,
    )
    if not selected:
        return {
            "round": round_idx,
            "selected_papers_this_round": 0,
            "new_papers_distilled": 0,
            "cached_hits_this_round": 0,
            "cumulative_paper_rows": _count_jsonl_rows(cumulative_jsonl),
            "final_schema_fields": None,
            "final_schema_path": None,
            "skipped_reason": "select_schema_papers returned 0",
        }

    # Cross-run paper memory (opt-in): MEMOSCHEMA_PAPER_CACHE_DIRS = ":"-joined distill caches of
    # earlier runs. A selected paper found there is copied into this run's cache and then handled
    # like a resume hit, so it is not distilled/proposed again.
    cross_run_hits = cross_run_mismatch = 0
    xdirs = [Path(d) for d in os.getenv("MEMOSCHEMA_PAPER_CACHE_DIRS", "").split(":") if d]
    if xdirs:
        own = _cached_paper_ids(distill_cache_dir)
        cand = [p for p in selected if p.stem not in own and any((d / f"{p.stem}.json").exists() for d in xdirs)]
        texts = {x["paper_id"]: x["text"] for x in mods.design_prompt.collect_paper_texts_from_paths(cand, max_chars_per_paper=raw_chars)} if cand else {}
        for p in cand:
            if p.stem not in texts:
                continue
            fp = paper_memo_fingerprint(mods, args, texts[p.stem], raw_chars)
            src = next((d for d in xdirs if (d / f"{p.stem}.json").exists() and _read_fp(d, p.stem) == fp), None)
            if src is None:
                cross_run_mismatch += 1          # same paper id, different text/operator/config -> recompute
                continue
            distill_cache_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src / f"{p.stem}.json", distill_cache_dir / f"{p.stem}.json")
            (distill_cache_dir / f"{p.stem}.memo_fp").write_text(fp, encoding="utf-8")
            cross_run_hits += 1
        print(f"[Schema] cross-run paper memory: {cross_run_hits}/{len(selected)} selected papers reused "
              f"({cross_run_mismatch} same-id entries rejected: fingerprint differs)")

    # 2) Distill cache: skip paper_ids already processed in earlier rounds
    cached_ids = _cached_paper_ids(distill_cache_dir)
    fresh_pdfs: List[Path] = [p for p in selected if p.stem not in cached_ids]
    cached_hits: List[Path] = [p for p in selected if p.stem in cached_ids]

    new_rows: List[Dict[str, Any]] = []
    if fresh_pdfs:
        papers = mods.design_prompt.collect_paper_texts_from_paths(
            fresh_pdfs,
            max_chars_per_paper=raw_chars,
        )
        # One paper at a time, cached as soon as it is done, so a crash
        # mid-round does not repeat the papers already distilled.
        for paper in papers:
            rows = mods.design_prompt.propose_paper_level_candidate_fields(
                papers=[paper],
                domain=args.domain,
                last_tier=args.last_tier,
                max_fields_per_paper=args.max_candidate_fields,
                max_text_chars=args.max_chars_per_paper,
                distill_input_chars=args.distill_input_chars,
                distill_quote_limit=args.distill_quote_limit,
                disable_distill=args.no_distill,
            )
            _save_distill_rows(distill_cache_dir, rows)
            if rows:
                (distill_cache_dir / f"{paper['paper_id']}.memo_fp").write_text(
                    paper_memo_fingerprint(mods, args, paper["text"], raw_chars), encoding="utf-8")
            new_rows.extend(rows)

    # Resume: cached papers of this round that a crash kept out of the
    # cumulative jsonl (cached papers from earlier rounds are already in it).
    if cached_hits:
        in_cumulative = set()
        if cumulative_jsonl.exists():
            for ln in cumulative_jsonl.open("r", encoding="utf-8"):
                if ln.strip():
                    in_cumulative.add(json.loads(ln).get("paper_id"))
        recovered = {
            p.stem: json.loads((distill_cache_dir / f"{p.stem}.json").read_text(encoding="utf-8"))
            for p in cached_hits if p.stem not in in_cumulative
        }
        if recovered:
            print(f"[Schema] resume: {len(recovered)} papers of this round restored from distill cache")
            by_id = {r.get("paper_id"): r for r in new_rows}
            by_id.update(recovered)
            new_rows = [by_id[p.stem] for p in selected if p.stem in by_id]

    # 3) Append this round's new rows to the cumulative JSONL
    _append_jsonl(cumulative_jsonl, new_rows)

    if new_rows:
        _save_jsonl(round_schema_out / "paper_level_new_this_round.jsonl", new_rows)
    (round_schema_out / "selected_schema_papers.txt").write_text(
        "\n".join(str(p) for p in selected), encoding="utf-8"
    )

    # 4) Re-cluster and re-rank over the full cumulative JSONL
    if _count_jsonl_rows(cumulative_jsonl) == 0:
        return {
            "round": round_idx,
            "selected_papers_this_round": len(selected),
            "new_papers_distilled": len(new_rows),
            "cached_hits_this_round": len(cached_hits),
            "cumulative_paper_rows": 0,
            "final_schema_fields": None,
            "final_schema_path": None,
            "skipped_reason": "cumulative jsonl empty after append",
        }

    mods.schema_refine.cluster_merge_from_paper_level_jsonl(
        paper_level_jsonl=cumulative_jsonl,
        out_dir=round_schema_out,
        domain=args.domain,
        last_tier=args.last_tier,
        embed_threshold=args.embed_threshold,
        min_support=args.min_support,
        max_final_fields=args.max_final_fields,
    )

    final_schema_path = round_schema_out / "final_schema.json"
    if not final_schema_path.exists():
        return {
            "round": round_idx,
            "selected_papers_this_round": len(selected),
            "new_papers_distilled": len(new_rows),
            "cached_hits_this_round": len(cached_hits),
            "cumulative_paper_rows": _count_jsonl_rows(cumulative_jsonl),
            "final_schema_fields": None,
            "final_schema_path": None,
            "skipped_reason": "cluster_merge did not write final_schema.json",
        }

    final_schema = json.loads(final_schema_path.read_text(encoding="utf-8"))
    extraction_prompt = mods.design_prompt.build_extraction_prompt(
        final_fields=final_schema,
        domain=args.domain,
        last_tier=args.last_tier,
    )
    (round_schema_out / "extraction_prompt.txt").write_text(extraction_prompt, encoding="utf-8")

    return {
        "round": round_idx,
        "selected_papers_this_round": len(selected),
        "new_papers_distilled": len(new_rows),
        "cached_hits_this_round": len(cached_hits),
        "cross_run_hits_this_round": cross_run_hits,
        "cross_run_fingerprint_mismatch_this_round": cross_run_mismatch,
        "cumulative_paper_rows": _count_jsonl_rows(cumulative_jsonl),
        "final_schema_fields": len(final_schema),
        "final_schema_path": str(final_schema_path),
    }

def copy_latest_schema_as_final(
    round_schema_out: Path,
    final_dir: Path,
) -> Dict[str, Any]:
    final_dir.mkdir(parents=True, exist_ok=True)
    if not round_schema_out or not Path(round_schema_out).exists():
        return {"copied_from": None, "final_dir": str(final_dir), "copied": []}
    copied: List[str] = []
    for fname in (
        "final_schema.json",
        "candidate_fields.json",
        "discovered_anchors.json",
        "extraction_prompt.txt",
        "selected_schema_papers.txt",
        "paper_level_new_this_round.jsonl",
    ):
        src = Path(round_schema_out) / fname
        if src.exists():
            shutil.copy2(src, final_dir / fname)
            copied.append(fname)
    return {
        "copied_from": str(round_schema_out),
        "final_dir": str(final_dir),
        "copied": copied,
    }