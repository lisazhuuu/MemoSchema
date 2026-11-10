from __future__ import annotations
import argparse, json, os, re, shutil, sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set

PHASE1_DIR = Path(__file__).resolve().parent
if str(PHASE1_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE1_DIR))

from iterative_expand import (
    RuntimeModules, run_retrieval_round, copy_records_to_corpus,
    save_json, save_jsonl, list_seed_pdfs, ensure_clean_dir,
)

def _parse_args():
    p = argparse.ArgumentParser(description="Resume a partial iterative_expand run.")
    p.add_argument("--resume_run_dir", required=True)
    p.add_argument("--domain", required=True, choices=["ad", "chem", "cs"])
    p.add_argument("--seed_dir", required=True)
    p.add_argument("--last_tier", required=True)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--max_queries", type=int, default=15)
    p.add_argument("--results_per_source", type=int, default=20)
    p.add_argument("--sources", type=str, default="biorxiv,openalex,pubmed,core")
    p.add_argument("--target_total_papers", type=int, default=150)
    p.add_argument("--seeds_per_next_round", type=int, default=4)
    p.add_argument("--overlap_threshold", type=float, default=0.15)
    p.add_argument("--core_api_key", type=str, default=os.getenv("CORE_API_KEY"))
    p.add_argument(
        "--crossref_email",
        type=str,
        default=os.getenv("CROSSREF_MAILTO"),
    )
    p.add_argument(
        "--openalex_email",
        type=str,
        default=os.getenv("OPENALEX_MAILTO"),
    )
    p.add_argument("--min_corpus_for_induction", type=int, default=20)
    p.add_argument("--max_schema_papers", type=int, default=150)
    p.add_argument("--schema_candidate_pool_size", type=int, default=180)
    p.add_argument("--schema_rep_chars", type=int, default=2500)
    p.add_argument("--schema_diversity_threshold", type=float, default=0.85)
    p.add_argument("--max_chars_per_paper", type=int, default=12000)
    p.add_argument("--max_candidate_fields", type=int, default=25)
    p.add_argument("--distill_input_chars", type=int, default=60000)
    p.add_argument("--distill_quote_limit", type=int, default=10)
    p.add_argument("--no_distill", action="store_true")
    p.add_argument("--embed_threshold", type=float, default=0.65)
    p.add_argument("--min_support", type=int, default=2)
    p.add_argument("--max_final_fields", type=int, default=8)
    p.add_argument("--per_round_distill_cap", type=int, default=60)
    p.add_argument("--mmr_lambda", type=float, default=0.7)
    p.add_argument("--anchor_alpha", type=float, default=0.7)
    p.add_argument("--strict_tier_anchor", action="store_true")
    p.add_argument("--output_root", type=str, default=None)  # ignored (state lives in resume_run_dir)
    return p.parse_args()

def _norm_title(t):
    return re.sub(r"\s+", " ", (t or "").lower()).strip()

def _reconstruct_state(run_dir: Path, mods, rounds_max: int):
    history_keys: Set[str] = set()
    history_titles: Set[str] = set()
    last_round = 0
    for ri in range(1, rounds_max + 1):
        rd = run_dir / f"round{ri:02d}"
        meta = rd / "downloaded_metadata.jsonl"
        if not meta.exists():
            continue
        last_round = ri
        with open(meta, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                k = mods.retrieve_literature.get_record_key(rec)
                if k:
                    history_keys.add(k)
                t = _norm_title(rec.get("title"))
                if t:
                    history_titles.add(t)
    return history_keys, history_titles, last_round

def _finish_round_schema(mods, args, run_dir: Path, round_idx: int) -> bool:
    schema_out = run_dir / f"round{round_idx:02d}" / "schema_from_cumulative_corpus"
    if (schema_out / "final_schema.json").exists():
        return True
    cumulative = run_dir / "schema_state" / "paper_level_field_proposals.jsonl"
    paper_level_new = schema_out / "paper_level_new_this_round.jsonl"

    # Case 2: distill never completed for this round -> re-run full schema step.
    if not paper_level_new.exists():
        print(f"[resume] round {round_idx} distill incomplete (no paper_level_new_this_round.jsonl); "
              f"re-running full run_round_schema_step")
        download_dir = run_dir / f"round{round_idx:02d}" / "downloads"
        result = mods.schema_iter.run_round_schema_step(
            mods=mods, args=args, round_idx=round_idx,
            new_pdfs_dir=download_dir,
            schema_state_dir=run_dir / "schema_state",
            round_schema_out=schema_out,
            per_round_distill_cap=args.per_round_distill_cap,
        )
        return bool(result and result.get("final_schema_path"))

    # Case 1: distill completed, cluster_merge crashed -> re-run cluster_merge only.
    if not cumulative.exists():
        print(f"[resume] cumulative jsonl missing at {cumulative}; cannot finish round {round_idx} schema")
        return False
    print(f"[resume] finishing round {round_idx} cluster_merge on cumulative jsonl ...")
    schema_out.mkdir(parents=True, exist_ok=True)
    mods.schema_refine.cluster_merge_from_paper_level_jsonl(
        paper_level_jsonl=cumulative,
        out_dir=schema_out,
        domain=args.domain,
        last_tier=args.last_tier,
        embed_threshold=args.embed_threshold,
        min_support=args.min_support,
        max_final_fields=args.max_final_fields,
    )
    fs_path = schema_out / "final_schema.json"
    if not fs_path.exists():
        print(f"[resume] cluster_merge did not produce final_schema.json")
        return False
    final_schema = json.loads(fs_path.read_text(encoding="utf-8"))
    prompt = mods.design_prompt.build_extraction_prompt(
        final_fields=final_schema, domain=args.domain, last_tier=args.last_tier
    )
    (schema_out / "extraction_prompt.txt").write_text(prompt, encoding="utf-8")
    print(f"[resume] round {round_idx} schema done, {len(final_schema)} fields")
    return True

def main():
    args = _parse_args()
    run_dir = Path(args.resume_run_dir).resolve()
    if not run_dir.exists():
        sys.exit(f"resume_run_dir not found: {run_dir}")
    os.chdir(PHASE1_DIR)
    seed_dir = Path(args.seed_dir).resolve()

    corpus_dir = run_dir / "cumulative_corpus"
    seeds_root = run_dir / "round_seeds"
    stats_dir = run_dir / "stats"
    schema_state_dir = run_dir / "schema_state"
    for d in (corpus_dir, seeds_root, stats_dir, schema_state_dir):
        d.mkdir(parents=True, exist_ok=True)

    mods = RuntimeModules(args.domain, args.last_tier)
    # Patch schema_induction.llm_chat to skip papers that trip OpenAI's content
    # policy filter (single paper failure should not kill the entire distill batch).
    import openai
    _orig_llm_chat = mods.design_prompt.llm_chat
    def _safe_llm_chat(messages, force_json=False):
        try:
            return _orig_llm_chat(messages, force_json=force_json)
        except openai.BadRequestError as e:
            err = str(e).lower()
            if "violating our usage policy" in err or "invalid_prompt" in err:
                print(f"[resume][SAFE-LLM] content policy flagged; skipping this paper")
                return "{}" if force_json else ""
            raise
    mods.design_prompt.llm_chat = _safe_llm_chat
    
    # schema_lib.schema_cluster_merge imports llm_chat at module load time via
    # `from schema_induction import llm_chat`, so it has its own bound
    # reference that the design_prompt patch above doesn't touch. Patch it too.
    import schema_lib.schema_cluster_merge as _scm
    _scm.llm_chat = _safe_llm_chat

    history_keys, history_titles, last_round = _reconstruct_state(run_dir, mods, args.rounds)
    seen_seed_keys: Set[str] = set()  # restart this set; original seeds aren't keyed anyway
    print(f"[resume] state rebuilt: history_keys={len(history_keys)} "
          f"history_titles={len(history_titles)} last_round_with_retrieve={last_round}")

    if last_round > 0:
        _finish_round_schema(mods, args, run_dir, last_round)

    # rebuild round_summaries from existing rounds
    round_summaries: List[Dict[str, Any]] = []
    for ri in range(1, last_round + 1):
        rd = run_dir / f"round{ri:02d}"
        fs_path = rd / "schema_from_cumulative_corpus" / "final_schema.json"
        si = None
        if fs_path.exists():
            try:
                si = {"final_schema_fields": len(json.loads(fs_path.read_text())),
                      "final_schema_path": str(fs_path)}
            except Exception:
                si = None
        round_summaries.append({"round": ri, "round_dir": str(rd),
                                "download_dir": str(rd / "downloads"),
                                "schema_induction": si})

    next_round = last_round + 1
    if next_round <= args.rounds:
        # ensure current_seeds_dir for next_round exists; if not, run seed selection
        current_seeds_dir = seeds_root / f"round{next_round:02d}_seeds"
        if not current_seeds_dir.exists() or not list(current_seeds_dir.glob("*.pdf")):
            current_seeds_dir.mkdir(parents=True, exist_ok=True)
            prev_seeds_dir = seeds_root / f"round{last_round:02d}_seeds"
            if not prev_seeds_dir.exists():
                # last_round must be 0 -> use original --seed_dir
                prev_seeds_dir = run_dir / "_orig_seeds_for_resume"
                prev_seeds_dir.mkdir(parents=True, exist_ok=True)
                for pdf in list_seed_pdfs(seed_dir):
                    shutil.copy2(pdf, prev_seeds_dir / pdf.name)
            print(f"[resume] picking seeds for round {next_round} from round {last_round} downloads")
            mods.seed_mod.select_new_seeds_by_similarity(
                round_dir=run_dir / f"round{last_round:02d}",
                download_dir=run_dir / f"round{last_round:02d}" / "downloads",
                seeds_dir=prev_seeds_dir, next_seed_dir=current_seeds_dir,
                max_seed_pdfs=args.seeds_per_next_round, seen_seed_keys=seen_seed_keys,
                domain=args.domain, original_seeds_dir=seed_dir, run_dir=run_dir,
                mmr_lambda=args.mmr_lambda, anchor_alpha=args.anchor_alpha,
            )

        # main remaining-rounds loop (mirrors iterative_expand.main)
        for round_idx in range(next_round, args.rounds + 1):
            out = run_retrieval_round(
                mods=mods, args=args, round_idx=round_idx, base_run_dir=run_dir,
                seeds_dir=current_seeds_dir, history_keys=history_keys,
                history_titles=history_titles,
            )
            round_dir = Path(out["round_dir"]); download_dir = Path(out["download_dir"])
            history_before_round = set(out.get("history_before_round", set()))
            new_unique_keys = set(out.get("new_unique_keys", set()))
            copied = copy_records_to_corpus(
                round_dir=round_dir, download_dir=download_dir, corpus_dir=corpus_dir,
                history_before_round=history_before_round,
                get_record_key=mods.retrieve_literature.get_record_key,
                fname_with_tag=mods.utils.fname_with_tag,
            )
            save_jsonl(round_dir / "new_unique_papers.jsonl", copied)
            history_titles |= mods.dedup_mod.collect_round_titles(round_dir)

            schema_summary = None
            corpus_count = len(list(corpus_dir.glob("*.pdf")))
            if corpus_count >= args.min_corpus_for_induction:
                schema_summary = mods.schema_iter.run_round_schema_step(
                    mods=mods, args=args, round_idx=round_idx,
                    new_pdfs_dir=download_dir, schema_state_dir=schema_state_dir,
                    round_schema_out=round_dir / "schema_from_cumulative_corpus",
                    per_round_distill_cap=args.per_round_distill_cap,
                )
            round_summaries.append({"round": round_idx, "round_dir": str(round_dir),
                                    "download_dir": str(download_dir),
                                    "new_unique_papers_this_round": len(new_unique_keys),
                                    "cumulative_unique_papers": corpus_count,
                                    "schema_induction": schema_summary})
            save_json(stats_dir / "round_summaries.json", round_summaries)
            print(f"[resume] round={round_idx} new_unique={len(new_unique_keys)} cumulative={corpus_count}")

            if corpus_count >= args.target_total_papers:
                print(f"[resume] reached target_total_papers={args.target_total_papers}"); break
            if round_idx < args.rounds:
                next_seed_dir = seeds_root / f"round{round_idx + 1:02d}_seeds"
                ensure_clean_dir(next_seed_dir)
                chosen = mods.seed_mod.select_new_seeds_by_similarity(
                    round_dir=round_dir, download_dir=download_dir,
                    seeds_dir=current_seeds_dir, next_seed_dir=next_seed_dir,
                    max_seed_pdfs=args.seeds_per_next_round, seen_seed_keys=seen_seed_keys,
                    domain=args.domain, original_seeds_dir=seed_dir, run_dir=run_dir,
                    mmr_lambda=args.mmr_lambda, anchor_alpha=args.anchor_alpha,
                )
                if not chosen:
                    print("[resume] no new seeds; stopping early"); break
                current_seeds_dir = next_seed_dir

    # final alias
    last_round_with_schema = None
    for s in reversed(round_summaries):
        si = s.get("schema_induction") or {}
        fp = si.get("final_schema_path")
        if fp:
            last_round_with_schema = Path(fp).parent; break
    final_out = run_dir / "final_schema_from_cumulative_corpus"
    if last_round_with_schema is not None:
        mods.schema_iter.copy_latest_schema_as_final(
            round_schema_out=last_round_with_schema, final_dir=final_out,
        )

    save_json(run_dir / "run_summary.json", {
        "domain": args.domain, "seed_dir": str(seed_dir), "run_dir": str(run_dir),
        "actual_total_papers": len(list(corpus_dir.glob("*.pdf"))),
        "rounds_completed": len(round_summaries),
        "round_summaries": round_summaries,
        "resumed_at": datetime.now().isoformat(),
    })
    print(f"\n[resume] done. run_dir: {run_dir}")

if __name__ == "__main__":
    main()
