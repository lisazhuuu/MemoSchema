from __future__ import annotations

import os, json, uuid, argparse, time, sys
from dataclasses import asdict
from pathlib import Path
from typing import List
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
except ImportError:
    pass

from openai.types.chat.chat_completion import ChatCompletion
from openai import RateLimitError

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from setup.test_azure_openai import chat_client, AZURE_OPENAI_CHAT_DEPLOYMENT
from queries_lib.domain import get_spec
from queries_lib.tiers import validate_tiers
from queries_lib.paper_model import Paper
from queries_lib.seed_summarizer import summarize_seed_pdf
from queries_lib.query_pruner import apply_keep_floor
from queries_lib.query_scorer import score_and_mark_queries

OPEN_AI_MODEL = AZURE_OPENAI_CHAT_DEPLOYMENT
MODE = "azure"
print(f"[queries_generation] Mode: {MODE.upper()} | deployment={OPEN_AI_MODEL}")

MAX_SEED_PDFS_FOR_QUERY_GEN = None

# Call API to get LLM response with extraction & search prompts
def llm_chat(messages, force_json: bool = False) -> str:
    MAX_RETRIES = 3
    for attempt in range(MAX_RETRIES):
        kwargs = {"model": OPEN_AI_MODEL, "messages": messages}
        if force_json:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp: ChatCompletion = chat_client.chat.completions.create(**kwargs)
            if os.getenv("MEMOSCHEMA_USAGE_LOG"):
                try:
                    with open(os.environ["MEMOSCHEMA_USAGE_LOG"], "a", encoding="utf-8") as _f:
                        _f.write(json.dumps({"t": time.time(), "module": "queries_generation",
                                             "usage": resp.usage.model_dump() if resp.usage else {}}) + "\n")
                except Exception:
                    pass
            return resp.choices[0].message.content or ""
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait_time = 2 ** (attempt + 2)
                if isinstance(e, RateLimitError):
                    wait_time = 60
                print(
                    f"[LLM Retry] API call failed ({e}). Waiting {wait_time} seconds before retrying... "
                    f"(Attempt {attempt + 1})"
                )
                time.sleep(wait_time)
            else:
                print(f"[ERROR] LLM API call failed: Max retries ({MAX_RETRIES}) reached. Error: {e}")
                return json.dumps({"queries": []})

# Arxiv style category filters
def inject_arxiv_cats(qb: str) -> str:
    qb = (qb or "").strip()
    cats = SPEC.arxiv_bool
    if not cats or "cat:" in qb.lower():
        return qb
    return f"({qb}) AND {cats}" if qb else cats

# Require both domain tiers in each query.
def _build_tier_constraint_block(spec) -> str:
    tiers = list(getattr(spec, "tiers", []) or [])
    if not tiers:
        return ""
    last = tiers[-1]

    # Generic sibling-subdomain negative instruction
    drift_note = (
        "\n\nAvoid drifting into sibling subdomains unless the query explicitly "
        "ties them back to the requested last-tier focus."
    )

    if len(tiers) >= 2:
        penult = tiers[-2]
        return (
            "\n\n[Hard subdomain anchoring]\n"
            f"EVERY query you generate MUST contain (a) a near-synonym of "
            f"\"{last}\" AND (b) a near-synonym of \"{penult}\" inside the "
            f"`query_nl` and `query_bool` strings themselves — not only in the "
            f"synonyms list, and not only in the `tiers` metadata field. Use "
            f"the actual subdomain words, NOT abbreviations (write 'Alzheimer "
            f"disease' not 'AD', 'Reinforcement learning from human feedback' "
            f"not 'RLHF' alone, etc.). Queries that fail this rule will be "
            f"rejected post-hoc and waste the budget."
            + drift_note
        )
    return (
        "\n\n[Hard subdomain anchoring]\n"
        f"EVERY query you generate MUST contain a near-synonym of \"{last}\" "
        f"inside the `query_nl` and `query_bool` strings themselves — not only "
        f"in the synonyms list, and not in abbreviated form. Queries that fail "
        f"this rule will be rejected post-hoc and waste the budget."
        + drift_note
    )

# Build optional exclusions for the query prompt.
def _build_avoid_block(
    previous_queries: List[str] | None,
    avoid_titles: List[str] | None,
    max_queries_in_prompt: int = 15,
    max_titles_in_prompt: int = 10,
) -> str:
    parts: list[str] = []
    if previous_queries:
        prev = [q.strip() for q in previous_queries if q and q.strip()]
        prev = prev[:max_queries_in_prompt]
        if prev:
            prev_lines = "\n".join(f"- {q}" for q in prev)
            parts.append(
                "[Previously issued queries]\n"
                "These queries were already used in earlier rounds and have already "
                "pulled their share of papers. Generate NEW queries that explore "
                "complementary angles within the SAME subdomain. Do NOT produce "
                "semantic near-duplicates of these:\n"
                f"{prev_lines}"
            )
    if avoid_titles:
        titles = [t.strip() for t in avoid_titles if t and t.strip()]
        titles = titles[:max_titles_in_prompt]
        if titles:
            title_lines = "\n".join(f"- {t}" for t in titles)
            parts.append(
                "[Already-retrieved papers]\n"
                "The following papers are already in the corpus. Do NOT design "
                "queries whose primary intent is to retrieve more papers like these. "
                "Aim for adjacent angles within the same subdomain (different "
                "mechanisms, models, cohorts, or methodological lenses):\n"
                f"{title_lines}"
            )
    if not parts:
        return ""
    return "\n\n" + "\n\n".join(parts)

# Build queries from the seed PDFs of a round directory; returns the path of queries.json
def build_queries_from_seed_pdfs(
    seed_pdfs: List[Path],
    out_dir: Path,
    max_queries: int = 8,
    previous_queries: List[str] | None = None,
    avoid_titles: List[str] | None = None,
    strict_tier_anchor: bool = False,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds_dir = out_dir / "seeds"
    if seeds_dir.exists():
        for old_file in seeds_dir.glob("*"):
            try:
                old_file.unlink()
            except IsADirectoryError:
                pass
    else:
        seeds_dir.mkdir(parents=True, exist_ok=True)

    papers: List[Paper] = []
    for pdf in seed_pdfs[:MAX_SEED_PDFS_FOR_QUERY_GEN]:
        dest = seeds_dir / pdf.name
        if not dest.exists():
            dest.write_bytes(pdf.read_bytes())

        extracted = summarize_seed_pdf(
            pdf_path=pdf,
            spec=SPEC,
            chat_fn=llm_chat,
            chunk_size=4500,
            overlap=500,
        )

        if not extracted:
            continue

        papers.append(
            Paper(
                paper_id=f"seed_{uuid.uuid4().hex[:8]}",
                title=extracted.get("title"),
                authors=extracted.get("authors"),
                year=extracted.get("year"),
                venue=extracted.get("venue"),
                doi=extracted.get("doi") or extracted.get("doi_or_id"),
                source="seed",
                raw_text_path=None,
                extracted=extracted,
            )
        )

    with open(out_dir / "papers.jsonl", "w", encoding="utf-8") as f:
        for p in papers:
            f.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")

    snapshot_items = []
    for p in papers:
        ex = p.extracted or {}
        snapshot_items.append(
            {
                "title": ex.get("title"),
                "core_subtopic": ex.get("core_subtopic"),
                "key_mechanisms_or_entities": ex.get("key_mechanisms_or_entities"),
                "important_proteins_genes_pathways": ex.get("important_proteins_genes_pathways"),
                "model_system_or_study_type": ex.get("model_system_or_study_type"),
                "main_experimental_setting_or_evidence_type": ex.get("main_experimental_setting_or_evidence_type"),
                "likely_retrieval_keywords": ex.get("likely_retrieval_keywords"),
                "off_topic_or_should_avoid": ex.get("off_topic_or_should_avoid"),
                "concise_paper_summary": ex.get("concise_paper_summary"),
                "why_this_seed_is_useful_for_retrieval": ex.get("why_this_seed_is_useful_for_retrieval"),
            }
        )

    snapshot = json.dumps(snapshot_items, ensure_ascii=False, indent=2)
    avoid_block = _build_avoid_block(previous_queries, avoid_titles)
    # Tier-constraint prompt is only attached when caller opts in.
    tier_constraint_block = _build_tier_constraint_block(SPEC) if strict_tier_anchor else ""

    def run_query_prompt(prompt_text: str):
        raw = llm_chat(
            [
                {"role": "system", "content": "Return ONLY valid JSON."},
                {"role": "user", "content": prompt_text + f"\n\nExtracted snapshot:\n{snapshot}" + avoid_block + tier_constraint_block},
            ],
            force_json=True,
        )
        time.sleep(1)
        try:
            obj = json.loads(raw)
            return obj.get("queries", [])
        except Exception:
            return []

    q_domain = run_query_prompt(SPEC.prompt_gen_domain)

    def norm_qb(s: str) -> str:
        return " ".join((s or "").lower().split())

    seen = set()
    merged = []
    for item in q_domain:
        if SPEC.tiers and "tiers" not in item:
            item["tiers"] = SPEC.tiers

        qb_raw = item.get("query_bool") or item.get("query_nl") or ""
        qb_final = inject_arxiv_cats(qb_raw)

        if not validate_tiers(item, qb_final, DOMAIN, SPEC, strict_anchor=strict_tier_anchor):
            continue

        key = norm_qb(qb_final)
        if key in seen:
            continue
        seen.add(key)

        item["query_bool"] = qb_final
        merged.append(item)

    if len(merged) < max_queries:
        print(
            f"[queries_generation] Domain prompt returned {len(merged)} queries, generating general queries to reach {max_queries}."
        )
        q_general = run_query_prompt(SPEC.prompt_gen_general)
        for item in q_general:
            if SPEC.tiers and "tiers" not in item:
                item["tiers"] = SPEC.tiers

            qb_raw = item.get("query_bool") or item.get("query_nl") or ""
            qb_final = inject_arxiv_cats(qb_raw)

            if not validate_tiers(item, qb_final, DOMAIN, SPEC, strict_anchor=strict_tier_anchor):
                continue

            key = norm_qb(qb_final)
            if key in seen:
                continue
            seen.add(key)

            item["query_bool"] = qb_final
            merged.append(item)
            if len(merged) >= max_queries:
                break

    merged = merged[:max_queries]

    embedding_model = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")

    scored_queries = score_and_mark_queries(
        queries=merged,
        seed_summaries=[p.extracted or {} for p in papers],
        embed_client=chat_client,
        embedding_model=embedding_model,
    )

    final_queries = apply_keep_floor(scored_queries)
    
    scored_path = out_dir / "queries_scored.json"
    scored_path.write_text(
        json.dumps({"queries": scored_queries}, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    queries_path = out_dir / "queries.json"
    queries_path.write_text(
        json.dumps({"queries": final_queries}, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"[queries_generation] Wrote {len(final_queries)} queries to {queries_path}")
    return queries_path

if __name__ == "__main__":
    from dataclasses import replace

    parser = argparse.ArgumentParser(description="Generate per-round queries.json from seed PDFs.")
    parser.add_argument("--domain", choices=["cs", "chem", "ad"], default="ad")
    parser.add_argument(
        "--last_tier",
        required=True,
        help="Most-specific subdomain term (e.g. 'Neuroinflammation microglia'). "
             "Appended to SPEC.tiers at runtime so the same domain config can be "
             "reused across different subdomains without editing queries_lib/domain.py.",
    )
    parser.add_argument("--seed_dir", required=True, help="Directory containing seed PDFs")
    parser.add_argument("--out_dir", required=True, help="Output round directory")
    parser.add_argument("--max_queries", type=int, default=8, help="Maximum number of queries to produce")
    parser.add_argument(
        "--strict_tier_anchor",
        action="store_true",
        help="Opt in to the stricter subdomain anchoring: append a hard "
             "constraint to the LLM prompt AND post-hoc reject queries that "
             "don't mention a keyword from each of the bottom-two tiers. "
             "Off by default — turn on if you observe multi-round drift.",
    )
    args = parser.parse_args()

    # Add a domain switch and append the CLI-supplied last_tier so SPEC.tiers
    # has the full chain at runtime.
    DOMAIN = args.domain
    SPEC = get_spec(DOMAIN)
    SPEC = replace(SPEC, tiers=list(SPEC.tiers) + [args.last_tier])

    seed_dir = Path(args.seed_dir)
    seed_pdfs = sorted(seed_dir.glob("*.pdf"))
    if not seed_pdfs:
        raise FileNotFoundError(f"No PDFs found under {seed_dir}")

    build_queries_from_seed_pdfs(
        seed_pdfs=seed_pdfs,
        out_dir=Path(args.out_dir),
        max_queries=args.max_queries,
        strict_tier_anchor=args.strict_tier_anchor,
    )
