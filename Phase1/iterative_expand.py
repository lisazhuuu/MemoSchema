from __future__ import annotations

import argparse, importlib, json, os, shutil
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Iterative Phase I orchestrator: retrieval + cumulative schema induction."
    )
    p.add_argument("--domain", type=str, required=True, choices=["ad", "chem", "cs"])
    p.add_argument("--seed_dir", type=str, required=True, help="Directory containing initial seed PDFs")
    p.add_argument("--output_root", type=str, required=True, help="Base output directory for this run")
    p.add_argument(
        "--last_tier",
        type=str,
        required=True,
        help="Most-specific subdomain term (e.g. 'Neuroinflammation microglia'). "
             "Appended to SPEC.tiers at runtime; queries_lib/domain.py only "
             "stores tiers up to the second-to-last layer.",
    )

    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--max_queries", type=int, default=15)
    p.add_argument("--results_per_source", type=int, default=20)
    p.add_argument(
        "--sources",
        type=str,
        default="biorxiv,openalex,pubmed,core",
        help="Comma-separated sources. Use pubmed, not PMC.",
    )
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

    # schema induction controls
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
    p.add_argument(
        "--per_round_distill_cap",
        type=int,
        default=60,
        help="Cap on how many of THIS round's new PDFs to distill into the "
             "cumulative paper-level proposals jsonl. Default 60 matches the "
             "incremental design ('top 60 of this round's 80').",
    )
    p.add_argument(
        "--mmr_lambda",
        type=float,
        default=0.7,
        help="MMR relevance weight when picking next-round seeds: "
             "score = lambda*rel + (1-lambda)*novelty.",
    )
    p.add_argument(
        "--anchor_alpha",
        type=float,
        default=0.7,
        help="Weight on the ORIGINAL --seed_dir centroid when forming the "
             "next-round-seed anchor: anchor = alpha*orig + (1-alpha)*current_seeds.",
    )
    p.add_argument(
        "--strict_tier_anchor",
        action="store_true",
        help="Opt in to the stricter subdomain anchoring in query generation: "
             "append a hard 'must contain bottom-two-tier keywords' instruction "
             "to the LLM prompt AND post-hoc reject queries that miss those "
             "keywords. Off by default — turn on if multi-round drift "
             "(off-subdomain papers leaking in across rounds) becomes a problem.",
    )

    return p.parse_args()

class RuntimeModules:
    def __init__(self, domain: str, last_tier: str):
        os.environ["LAST_TIER"] = last_tier

        # Import after DOMAIN is set so queries_lib.domain resolves correctly.
        self.queries_generation = importlib.import_module("queries_generation")
        self.retrieve_literature = importlib.import_module("retrieve_literature")
        self.dedup_mod = importlib.import_module("iterative_lib.dedup")
        self.seed_mod = importlib.import_module("iterative_lib.seed")
        self.schema_iter = importlib.import_module("iterative_lib.schema")
        # Queries_lib.domain holds DomainSpec / DOMAINS / get_spec 
        self.domain_args = importlib.import_module("queries_lib.domain")
        self.utils = importlib.import_module("retrieve_lib.utils")
        self.design_prompt = importlib.import_module("schema_induction")
        self.schema_refine = importlib.import_module("schema_lib.prompt_refine")

        from dataclasses import replace
        base_spec = self.queries_generation.get_spec(domain)
        self.effective_spec = replace(
            base_spec, tiers=list(base_spec.tiers) + [last_tier]
        )

        self.queries_generation.DOMAIN = domain
        self.queries_generation.SPEC = self.effective_spec

def ensure_clean_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def list_seed_pdfs(seed_dir: Path) -> List[Path]:
    pdfs = sorted(seed_dir.glob("*.pdf"))
    if not pdfs:
        raise RuntimeError(f"No seed PDFs found in {seed_dir}")
    return pdfs

def copy_records_to_corpus(
    round_dir: Path,
    download_dir: Path,
    corpus_dir: Path,
    history_before_round: Set[str],
    get_record_key,
    fname_with_tag,
) -> List[Dict[str, Any]]:
    search_results_path = round_dir / "search_results.jsonl"
    if not search_results_path.exists():
        return []

    copied_rows: List[Dict[str, Any]] = []
    with open(search_results_path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            rec = json.loads(raw)
            key = get_record_key(rec)
            if not key or key in history_before_round:
                continue

            src_pdf = download_dir / fname_with_tag(rec)
            if not src_pdf.exists():
                continue

            dst_pdf = corpus_dir / src_pdf.name
            if not dst_pdf.exists():
                shutil.copy2(src_pdf, dst_pdf)

            copied_rows.append(
                {
                    "paper_key": key,
                    "title": rec.get("title"),
                    "source": rec.get("source"),
                    "doi": rec.get("doi"),
                    "id": rec.get("id"),
                    "pdf_filename": src_pdf.name,
                }
            )
    return copied_rows

def save_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def save_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def _read_previous_queries(prev_round_dir: Path, max_n: int = 15) -> List[str]:
    qpath = prev_round_dir / "queries.json"
    if not qpath.exists():
        return []
    try:
        obj = json.loads(qpath.read_text(encoding="utf-8"))
    except Exception:
        return []
    items = obj.get("queries", []) if isinstance(obj, dict) else obj
    out: List[str] = []
    for q in items:
        if isinstance(q, dict):
            nl = q.get("query_nl") or q.get("query_bool") or ""
            if nl:
                out.append(nl)
        elif isinstance(q, str):
            out.append(q)
        if len(out) >= max_n:
            break
    return out

def _read_previous_round_titles(prev_round_dir: Path, max_n: int = 10) -> List[str]:
    p = prev_round_dir / "search_results.jsonl"
    if not p.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            t = (rec.get("title") or "").strip()
            if not t:
                continue
            rows.append(rec)
    rows.sort(
        key=lambda r: (r.get("domain_score", r.get("ad_score", 0)) or 0),
        reverse=True,
    )
    return [r["title"] for r in rows[:max_n]]

def run_retrieval_round(
    mods: RuntimeModules,
    args: argparse.Namespace,
    round_idx: int,
    base_run_dir: Path,
    seeds_dir: Path,
    history_keys: Set[str],
    history_titles: Set[str],
) -> Dict[str, Any]:
    round_dir = base_run_dir / f"round{round_idx:02d}"
    ensure_clean_dir(round_dir)

    seed_pdfs = list_seed_pdfs(seeds_dir)
    print(f"\n{'=' * 70}")
    print(f"[orchestrator] round={round_idx} seeds={len(seed_pdfs)} dir={seeds_dir}")
    print(f"{'=' * 70}")

    # Round >= 2: feed the LLM the previous round's queries + a small set of top corpus titles
    previous_queries: List[str] = []
    avoid_titles: List[str] = []
    if round_idx > 1:
        prev_round_dir = base_run_dir / f"round{round_idx - 1:02d}"
        previous_queries = _read_previous_queries(prev_round_dir)
        avoid_titles = _read_previous_round_titles(prev_round_dir)
        if previous_queries or avoid_titles:
            print(
                f"[orchestrator] round={round_idx} feeding query-gen "
                f"prev_queries={len(previous_queries)} avoid_titles={len(avoid_titles)}"
            )

    queries_path = mods.queries_generation.build_queries_from_seed_pdfs(
        seed_pdfs=seed_pdfs,
        out_dir=round_dir,
        max_queries=args.max_queries,
        previous_queries=previous_queries,
        avoid_titles=avoid_titles,
        strict_tier_anchor=args.strict_tier_anchor,
    )

    domain_spec = mods.effective_spec
    contact_emails = {
        "CROSSREF_MAILTO": args.crossref_email,
        "OPENALEX_MAILTO": args.openalex_email,
    }
    sources_tuple = tuple(s.strip() for s in args.sources.split(",") if s.strip())

    result = mods.retrieve_literature.run_one_round_search_and_download(
        queries_path=queries_path,
        round_dir=round_dir,
        domain_spec=domain_spec,
        results_per_source=args.results_per_source,
        sources=sources_tuple,
        contact_emails=contact_emails,
        core_api_key=args.core_api_key,
        exclude_keys=set(history_keys),
    )
    download_dir = Path(result["download_dir"])

    history_before_round = set(history_keys)

    if history_titles:
        mods.dedup_mod.prune_by_history_title(
            round_idx=round_idx,
            round_dir=round_dir,
            download_dir=download_dir,
            history_titles=history_titles,
        )

    current_keys_before, _ = mods.dedup_mod.load_round_paper_keys(round_dir, download_dir)
    if not current_keys_before:
        return {
            "round_dir": round_dir,
            "download_dir": download_dir,
            "queries_path": queries_path,
            "new_unique_keys": set(),
            "remaining_keys": set(),
        }

    remaining_keys, _ = mods.dedup_mod.dedup_against_history(
        round_idx=round_idx,
        round_dir=round_dir,
        download_dir=download_dir,
        history_keys=history_before_round,
        overlap_threshold=args.overlap_threshold,
    )

    current_new_unique = remaining_keys - history_before_round
    per_round_target = args.results_per_source * len(sources_tuple)
    desired_new_this_round = min(per_round_target, max(0, args.target_total_papers - len(history_before_round)))
    need_more = max(0, desired_new_this_round - len(current_new_unique))

    if need_more > 0:
        extra_new = mods.dedup_mod.top_off_with_new_pdfs(
            round_idx=round_idx,
            round_dir=round_dir,
            download_dir=download_dir,
            queries_path=queries_path,
            domain=args.domain,
            sources=args.sources,
            crossref_email=args.crossref_email,
            openalex_email=args.openalex_email,
            already_seen_keys=history_before_round | current_new_unique,
            needed_new=need_more,
            base_results_per_source=args.results_per_source,
            core_api_key=args.core_api_key,
        )
        remaining_keys = remaining_keys | extra_new

    final_new_unique = remaining_keys - history_before_round
    history_keys |= final_new_unique

    return {
        "round_dir": round_dir,
        "download_dir": download_dir,
        "queries_path": queries_path,
        "new_unique_keys": final_new_unique,
        "remaining_keys": remaining_keys,
        "history_before_round": history_before_round,
    }

def main() -> None:
    args = parse_args()

    phase1_dir = Path(__file__).resolve().parent
    os.chdir(phase1_dir)

    seed_dir = Path(args.seed_dir).resolve()
    output_root = Path(args.output_root).resolve()
    run_dir = output_root / f"iterative_phase1_{args.domain}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ensure_clean_dir(run_dir)

    mods = RuntimeModules(args.domain, args.last_tier)

    corpus_dir = run_dir / "cumulative_corpus"
    seeds_root = run_dir / "round_seeds"
    stats_dir = run_dir / "stats"
    ensure_clean_dir(corpus_dir)
    ensure_clean_dir(seeds_root)
    ensure_clean_dir(stats_dir)

    # Persistent state for incremental schema induction
    schema_state_dir = run_dir / "schema_state"
    ensure_clean_dir(schema_state_dir)

    current_seeds_dir = seeds_root / "round01_seeds"
    ensure_clean_dir(current_seeds_dir)
    for pdf in list_seed_pdfs(seed_dir):
        shutil.copy2(pdf, current_seeds_dir / pdf.name)

    history_keys: Set[str] = set()
    # Cumulative set of normalized titles seen across all rounds
    history_titles: Set[str] = set()
    seen_seed_keys: Set[str] = set()
    round_summaries: List[Dict[str, Any]] = []

    for round_idx in range(1, args.rounds + 1):
        out = run_retrieval_round(
            mods=mods,
            args=args,
            round_idx=round_idx,
            base_run_dir=run_dir,
            seeds_dir=current_seeds_dir,
            history_keys=history_keys,
            history_titles=history_titles,
        )

        round_dir = Path(out["round_dir"])
        download_dir = Path(out["download_dir"])
        history_before_round = set(out.get("history_before_round", set()))
        new_unique_keys = set(out.get("new_unique_keys", set()))

        copied_rows = copy_records_to_corpus(
            round_dir=round_dir,
            download_dir=download_dir,
            corpus_dir=corpus_dir,
            history_before_round=history_before_round,
            get_record_key=mods.retrieve_literature.get_record_key,
            fname_with_tag=mods.utils.fname_with_tag,
        )
        save_jsonl(round_dir / "new_unique_papers.jsonl", copied_rows)

        # Update cumulative title history from this round's surviving records.
        new_round_titles = mods.dedup_mod.collect_round_titles(round_dir)
        history_titles |= new_round_titles

        schema_summary: Dict[str, Any] | None = None
        corpus_count = len(list(corpus_dir.glob("*.pdf")))
        if corpus_count >= args.min_corpus_for_induction:
            schema_out_dir = round_dir / "schema_from_cumulative_corpus"
            schema_summary = mods.schema_iter.run_round_schema_step(
                mods=mods,
                args=args,
                round_idx=round_idx,
                new_pdfs_dir=download_dir,
                schema_state_dir=schema_state_dir,
                round_schema_out=schema_out_dir,
                per_round_distill_cap=args.per_round_distill_cap,
            )

        round_summary = {
            "round": round_idx,
            "seed_dir": str(current_seeds_dir),
            "round_dir": str(round_dir),
            "download_dir": str(download_dir),
            "new_unique_papers_this_round": len(new_unique_keys),
            "cumulative_unique_papers": corpus_count,
            "schema_induction": schema_summary,
        }
        round_summaries.append(round_summary)
        save_json(stats_dir / "round_summaries.json", round_summaries)

        print(
            f"[orchestrator] round={round_idx} new_unique={len(new_unique_keys)} "
            f"cumulative={corpus_count}"
        )

        if corpus_count >= args.target_total_papers:
            print(f"[orchestrator] reached target_total_papers={args.target_total_papers}")
            break

        next_seed_dir = seeds_root / f"round{round_idx + 1:02d}_seeds"
        ensure_clean_dir(next_seed_dir)
        chosen = mods.seed_mod.select_new_seeds_by_similarity(
            round_dir=round_dir,
            download_dir=download_dir,
            seeds_dir=current_seeds_dir,
            next_seed_dir=next_seed_dir,
            max_seed_pdfs=args.seeds_per_next_round,
            seen_seed_keys=seen_seed_keys,
            domain=args.domain,
            original_seeds_dir=seed_dir,
            run_dir=run_dir,
            mmr_lambda=args.mmr_lambda,
            anchor_alpha=args.anchor_alpha,
        )
        if not chosen:
            print("[orchestrator] no new seeds selected; stopping early")
            break
        current_seeds_dir = next_seed_dir

    # Find the most recent round that successfully produced a schema and alias its output as the final schema
    last_schema_round_dir: Path | None = None
    for s in reversed(round_summaries):
        si = s.get("schema_induction") or {}
        fp = si.get("final_schema_path")
        if fp:
            last_schema_round_dir = Path(fp).parent
            break

    final_schema_out = run_dir / "final_schema_from_cumulative_corpus"
    if last_schema_round_dir is not None:
        final_schema_summary = mods.schema_iter.copy_latest_schema_as_final(
            round_schema_out=last_schema_round_dir,
            final_dir=final_schema_out,
        )
    else:
        final_schema_summary = {
            "skipped": True,
            "reason": "no round produced a schema (corpus may be below min_corpus_for_induction)",
            "final_dir": str(final_schema_out),
        }

    final_summary = {
        "domain": args.domain,
        "seed_dir": str(seed_dir),
        "run_dir": str(run_dir),
        "target_total_papers": args.target_total_papers,
        "actual_total_papers": len(list(corpus_dir.glob("*.pdf"))),
        "rounds_completed": len(round_summaries),
        "final_schema": final_schema_summary,
        "round_summaries": round_summaries,
    }
    save_json(run_dir / "run_summary.json", final_summary)

    print("\n[orchestrator] done")
    print(f"[orchestrator] run dir: {run_dir}")
    print(f"[orchestrator] cumulative corpus dir: {corpus_dir}")
    print(f"[orchestrator] final schema dir: {final_schema_out}")

if __name__ == "__main__":
    main()
