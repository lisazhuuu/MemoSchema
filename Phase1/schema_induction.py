from __future__ import annotations
import argparse, json, os, re, sys, time
from collections import Counter
from datetime import datetime
from math import sqrt
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from schema_lib.domain_registry import get_domain_spec
from queries_lib.paper_model import pdf_to_text
from schema_lib.paper_distill import distill_paper_for_schema, render_distill_to_text

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from setup.test_azure_openai import chat_client, AZURE_OPENAI_CHAT_DEPLOYMENT
OPENAI_MODEL = AZURE_OPENAI_CHAT_DEPLOYMENT

# LLM call, usage log and JSON/text file operations
def _sanitize_text(s: str) -> str:
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    return s.encode("utf-8", "replace").decode("utf-8")

def llm_chat(messages: List[Dict[str, str]], force_json: bool = False) -> str:
    safe_messages = [
        {"role": m.get("role", "user"), "content": _sanitize_text(m.get("content", ""))}
        for m in messages
    ]
    kwargs: Dict[str, Any] = {"model": OPENAI_MODEL, "messages": safe_messages}
    if force_json:
        kwargs["response_format"] = {"type": "json_object"}
    resp = chat_client.chat.completions.create(**kwargs)
    _log_usage("schema_induction", resp)
    return (resp.choices[0].message.content or "").strip()

def _log_usage(module: str, resp) -> None:
    """Opt-in exact token accounting: MEMOSCHEMA_USAGE_LOG = absolute jsonl path (one line per chat call)."""
    path = os.getenv("MEMOSCHEMA_USAGE_LOG", "")
    if not path:
        return
    try:
        u = resp.usage.model_dump() if getattr(resp, "usage", None) else {}
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"t": time.time(), "module": module, "usage": u}) + "\n")
    except Exception:
        pass

def save_json(path: Path, obj: Any):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def save_jsonl(path: Path, rows: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def save_text(path: Path, text: str):
    path.write_text(text, encoding="utf-8")

# Lightweight tokenization for selector cosine similarity
_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")

def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall((text or "").lower())

def _cosine_sim_from_tokens(text1: str, text2: str) -> float:
    c1 = Counter(_tokenize(text1))
    c2 = Counter(_tokenize(text2))
    if not c1 or not c2:
        return 0.0
    common = set(c1) & set(c2)
    dot = sum(c1[t] * c2[t] for t in common)
    n1 = sqrt(sum(v * v for v in c1.values()))
    n2 = sqrt(sum(v * v for v in c2.values()))
    if n1 == 0 or n2 == 0:
        return 0.0
    return dot / (n1 * n2)

# Selector target text — generic, parameterized by last_tier only
def build_sampling_target_text(
    domain: str,
    last_tier: Optional[str] = None,
    queries: Optional[List[str]] = None,
    seed_summaries: Optional[List[str]] = None,
) -> str:
    spec = get_domain_spec(domain)
    last_tier = last_tier or spec.default_last_tier

    parts = [
        f"Domain: {domain}",
        f"Subdomain focus: {last_tier}",
        f"Schema goal: {spec.schema_task_description}",
        "Task boundaries: infer fields from the corpus; avoid leaking a predefined checklist.",
    ]
    if queries:
        parts.append("Current search queries:")
        parts.extend(queries)
    if seed_summaries:
        parts.append("Seed paper summaries:")
        parts.extend(seed_summaries[:5])
    return "\n".join(parts)

# Abstract-priority paper representation
_ABSTRACT_HEADER_RE = re.compile(
    r"(?:^|\n)[ \t]*(?:abstract|summary)[ \t]*[:.\-]?[ \t]*\n",
    re.IGNORECASE,
)

_ABSTRACT_END_RE = re.compile(
    r"\n[ \t]*(?:"
    r"introduction|background|keywords?|key\s*words?|"
    r"main\s*text|main\s*body|significance\s*statement|"
    r"methods?|materials?\s*and\s*methods?|"
    r"results?|"
    r"\d+\.?\s+introduction|"
    r"\d+\.?\s+background|"
    r"\d+\.?\s+methods?"
    r")\b",
    re.IGNORECASE,
)

def _extract_pdf_abstract(text: str, max_chars: int = 4000) -> str:
    if not text:
        return ""
    head = text[:20000]
    m = _ABSTRACT_HEADER_RE.search(head)
    if not m:
        return ""
    start = m.end()
    rest = head[start:]
    e = _ABSTRACT_END_RE.search(rest)
    end = e.start() if e else min(len(rest), max_chars)
    abstract = rest[:end].strip()
    if len(abstract) < 80:
        return ""
    return abstract[:max_chars]

def _load_metadata_abstracts(metadata_jsonl: Optional[Path]) -> Dict[str, str]:
    if not metadata_jsonl:
        return {}
    metadata_jsonl = Path(metadata_jsonl)
    if not metadata_jsonl.exists():
        print(f"[PromptDesign][WARNING] metadata file not found: {metadata_jsonl}")
        return {}
    out: Dict[str, str] = {}
    try:
        with open(metadata_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                abstract = (row.get("abstract") or "").strip()
                if not abstract or len(abstract) < 40:
                    continue
                for k in ("paper_id", "pdf_filename", "pdf_path",
                          "source_file", "filename", "doi", "id"):
                    val = row.get(k)
                    if not val:
                        continue
                    val = str(val)
                    out[val] = abstract
                    stem = Path(val).stem if (val.endswith(".pdf") or "/" in val) else val
                    out[stem] = abstract
    except Exception as e:
        print(f"[PromptDesign][WARNING] Failed to load metadata abstracts: {e}")
    return out

def build_paper_representation(
    pdf_path: Path,
    metadata_abstracts: Optional[Dict[str, str]] = None,
    max_chars: int = 2500,
) -> Tuple[str, str]:
    if metadata_abstracts:
        for k in (pdf_path.stem, pdf_path.name, str(pdf_path)):
            if k in metadata_abstracts:
                rep = metadata_abstracts[k]
                return rep[: max_chars * 2], "metadata_abstract"
    try:
        text = pdf_to_text(pdf_path)
    except Exception as e:
        print(f"[PromptDesign][WARNING] Failed to read {pdf_path.name}: {e}")
        return "", "empty"
    if not text or not text.strip():
        return "", "empty"
    pdf_abs = _extract_pdf_abstract(text)
    if pdf_abs:
        return pdf_abs[: max_chars * 2], "pdf_abstract"
    return text[:max_chars], "fallback"

# Selector
def select_schema_papers(
    pdf_dir: Path,
    domain: str,
    last_tier: Optional[str] = None,
    queries: Optional[List[str]] = None,
    seed_summaries: Optional[List[str]] = None,
    max_papers: int = 24,
    candidate_pool_size: int = 40,
    rep_chars: int = 2500,
    diversity_threshold: float = 0.85,
    metadata_jsonl: Optional[Path] = None,
) -> List[Path]:
    all_pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not all_pdfs:
        print(f"[PromptDesign][WARNING] No PDFs found in {pdf_dir}")
        return []

    if metadata_jsonl is None:
        for cand in [
            pdf_dir / "downloaded_metadata.jsonl",
            pdf_dir.parent / "downloaded_metadata.jsonl",
            pdf_dir.parent / "metadata" / "downloaded_metadata.jsonl",
        ]:
            if cand.exists():
                metadata_jsonl = cand
                print(f"[PromptDesign] Auto-discovered metadata: {cand}")
                break

    metadata_abstracts = _load_metadata_abstracts(metadata_jsonl)
    target_text = build_sampling_target_text(
        domain=domain, last_tier=last_tier,
        queries=queries, seed_summaries=seed_summaries,
    )

    scored: List[Tuple[Path, float]] = []
    reps: Dict[Path, str] = {}
    source_counts: Counter = Counter()

    for pdf_path in all_pdfs:
        rep, source = build_paper_representation(
            pdf_path, metadata_abstracts=metadata_abstracts, max_chars=rep_chars,
        )
        source_counts[source] += 1
        if not rep:
            continue
        score = _cosine_sim_from_tokens(rep, target_text)
        scored.append((pdf_path, score))
        reps[pdf_path] = rep

    print(f"[PromptDesign] Representation sources: {dict(source_counts)}")
    if not scored:
        print("[PromptDesign][WARNING] No usable paper representations found.")
        return []

    scored.sort(key=lambda x: x[1], reverse=True)
    top_candidates = [p for p, _ in scored[:candidate_pool_size]]

    selected: List[Path] = []
    for pdf_path in top_candidates:
        rep = reps[pdf_path]
        if any(_cosine_sim_from_tokens(rep, reps[chosen]) >= diversity_threshold for chosen in selected):
            continue
        selected.append(pdf_path)
        if len(selected) >= max_papers:
            break

    if len(selected) < max_papers:
        for pdf_path in top_candidates:
            if pdf_path not in selected:
                selected.append(pdf_path)
            if len(selected) >= max_papers:
                break

    print(f"[PromptDesign] Selected {len(selected)} schema papers from {len(all_pdfs)} PDFs.")
    return selected

def collect_paper_texts_from_paths(pdf_paths: List[Path], max_chars_per_paper: int) -> List[Dict[str, str]]:
    papers: List[Dict[str, str]] = []
    total = len(pdf_paths)
    for i, pdf_path in enumerate(pdf_paths, start=1):
        try:
            print(f"[PromptDesign][ReadPDF] {i}/{total} start {pdf_path.name}", flush=True)
            text = pdf_to_text(pdf_path)
            if not text or not text.strip():
                print(f"[PromptDesign][ReadPDF] {i}/{total} empty {pdf_path.name}", flush=True)
                continue
            raw_len = len(text)
            if max_chars_per_paper and len(text) > max_chars_per_paper:
                text = text[:max_chars_per_paper]
            papers.append({"paper_id": pdf_path.stem, "source_file": pdf_path.name, "text": text,
                           "raw_text_chars": raw_len})
            print(f"[PromptDesign][ReadPDF] {i}/{total} done {pdf_path.name} chars={len(text)} "
                  f"(extracted={raw_len})", flush=True)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"[PromptDesign][WARNING] Failed to read {pdf_path.name}: {e}", flush=True)
    return papers

def sample_text_windows(full_text: str, window: int) -> str:
    t = (full_text or "").strip()
    if len(t) <= window * 3:
        return t
    mid = len(t) // 2
    head = t[:window]
    midw = t[mid : mid + window]
    tail = t[-window:]
    return head + "\n\n[...MIDDLE...]\n\n" + midw + "\n\n[...TAIL...]\n\n" + tail

# Per-paper field proposal — fully generic prompt, only `{last_tier}` varies
def propose_paper_level_candidate_fields(
    papers: List[Dict[str, str]],
    domain: str,
    last_tier: Optional[str] = None,
    max_fields_per_paper: int = 25,
    max_text_chars: int = 12000,
    distill_input_chars: int = 60000,
    distill_quote_limit: int = 10,
    disable_distill: bool = False,
) -> List[Dict[str, Any]]:
    spec = get_domain_spec(domain)
    last_tier = (last_tier or spec.default_last_tier).strip()
    avoid_lines = "\n".join(f"- {x}" for x in spec.avoid_fields)

    debug_dir = Path(os.getenv("AUTOSCHEMA_DEBUG_DIR", "._paper_level_debug"))
    debug_dir.mkdir(exist_ok=True)

    results: List[Dict[str, Any]] = []
    total = len(papers)
    for i, p in enumerate(papers, start=1):
        paper_id = p.get("paper_id", "")
        source_file = p.get("source_file", "")
        full_text = p.get("text") or ""

        print(f"[PromptDesign][Paper] {i}/{total} start paper_id={paper_id} file={source_file}", flush=True)

        distill = None
        if disable_distill:
            paper_text = sample_text_windows(full_text, window=max_text_chars // 3)
        else:
            print(f"[PromptDesign][Distill] {i}/{total} paper_id={paper_id} start", flush=True)
            distill = distill_paper_for_schema(
                llm_chat_fn=llm_chat,
                domain=domain,
                paper_id=paper_id,
                full_text=full_text,
                input_chars=distill_input_chars,
                quote_limit=distill_quote_limit,
                last_tier=last_tier,
            )
            print(f"[PromptDesign][Distill] {i}/{total} paper_id={paper_id} done", flush=True)
            paper_text = render_distill_to_text(distill, max_chars=max_text_chars)

        (debug_dir / f"{paper_id}.distilled.txt").write_text(paper_text, encoding="utf-8")
        if distill is not None:
            (debug_dir / f"{paper_id}.distill.raw.json").write_text(distill.get("_raw", ""), encoding="utf-8")

        if not paper_text.strip():
            results.append({"paper_id": paper_id, "source_file": source_file, "fields": []})
            print(f"[PromptDesign][Paper] {i}/{total} empty -> 0 fields", flush=True)
            continue

        print(f"[PromptDesign][Extract] {i}/{total} paper_id={paper_id} start", flush=True)

        user_prompt = f"""
            You are an expert researcher designing a reusable extraction schema.

            Domain         : {domain} ({spec.task_noun})
            Subdomain focus: {last_tier}
            Goal           : {spec.schema_task_description}

            You are proposing schema fields inferred ONLY from THIS single paper.
            Every field must be grounded in evidence snippets from the paper text.

            Use your own subject-matter knowledge of "{last_tier}" research to identify
            variables that are likely reusable across papers in this subdomain — but
            only emit fields you can ground in this paper's text.

            Avoid:
                {avoid_lines}

            Output: propose up to {max_fields_per_paper} candidate fields found in this paper.

            For EACH field:
                - name: concise snake_case
                - description: short and precise
                - value_type: one of ["string","number","integer","boolean","enum","array","object"]
                - example_values: 1-3 examples from the paper
                - evidence_snippets: 1-3 verbatim snippets (<=40 words each)
                - scope: one of
                    "global"          (reusable across the whole {domain} domain),
                    "subdomain"       (reusable across {last_tier} papers but not generic to {domain}),
                    "paper_specific"  (only meaningful in this single paper)
                - possible_aliases: 3-8 alternate names for this field
                - estimated_coverage: 0..1 (how often this is reported across {last_tier} papers)
                - importance_score: 0..1

            Normalization rules (CRITICAL):
                - Field names must be GENERAL and reusable.
                - DO NOT put specific entity / compound / protein names into the FIELD NAME.
                  Bad: "zif8_bet_surface_area" or "apoE_fibril_elongation_rate"
                  Good: "surface_area" or "aggregation_rate"
                - If a concept is entity-specific, put the entity as a VALUE.
                - If the field is only meaningful in this paper, set scope="paper_specific".

            Return ONLY valid JSON:
            {{
            "paper_id": "{paper_id}",
            "fields": [
                {{
                "name": "snake_case",
                "description": "...",
                "value_type": "string",
                "example_values": ["..."],
                "evidence_snippets": ["..."],
                "scope": "global",
                "possible_aliases": ["..."],
                "estimated_coverage": 0.5,
                "importance_score": 0.7
                }}
            ]
            }}

            Paper text:
            {paper_text}
        """.strip()

        raw = llm_chat(
            [{"role": "system", "content": "Return ONLY valid JSON. No markdown, no commentary."},
             {"role": "user", "content": user_prompt}],
            force_json=True,
        )
        (debug_dir / f"{paper_id}.raw.json").write_text(raw, encoding="utf-8")

        fields: List[Dict[str, Any]] = []
        try:
            data = json.loads(raw)
            fields = data.get("fields", []) if isinstance(data, dict) else []
            if not isinstance(fields, list):
                fields = []
        except Exception as e:
            print(f"[PromptDesign][WARNING] Failed to parse paper-level fields for {paper_id}: {e}", flush=True)
            fields = []

        if not fields:
            retry_prompt = user_prompt + (
                "\n\nIMPORTANT: If any study variables appear in the text, you MUST output at "
                "least 8 fields. Only output an empty list if the paper truly contains none."
            )
            raw2 = llm_chat(
                [{"role": "system", "content": "Return ONLY valid JSON matching the schema. No markdown."},
                 {"role": "user", "content": retry_prompt}],
                force_json=True,
            )
            (debug_dir / f"{paper_id}.raw.retry.json").write_text(raw2, encoding="utf-8")
            try:
                data2 = json.loads(raw2)
                fields2 = data2.get("fields", []) if isinstance(data2, dict) else []
                if isinstance(fields2, list) and fields2:
                    fields = fields2
            except Exception:
                pass

        results.append({"paper_id": paper_id, "source_file": source_file, "fields": fields})
        print(f"[PromptDesign][Paper] {i}/{total} done paper_id={paper_id} fields={len(fields)}", flush=True)

    return results

# Final extraction prompt — only domain-level meta + corpus-discovered focus
def build_extraction_prompt(final_fields: List[Dict[str, Any]], domain: str, last_tier: Optional[str]) -> str:
    spec = get_domain_spec(domain)
    last_tier = last_tier or spec.default_last_tier

    col_names = [f.get("name", "") for f in final_fields]
    header_line = "| " + " | ".join(col_names) + " |"

    bullet_lines = [f"- **{f.get('name','')}**: {f.get('description','')}" for f in final_fields]
    bullets = "\n".join(bullet_lines)

    return f"""
        [ROLE] You are a {spec.extractor_role}.

        [TASK]
            Extract structured information about "{last_tier}" from the paper text into the
            specified Markdown table format.

        [STRICT EXTRACTION RULES]
            1. Preserve numerical values and units exactly when present.
            2. Match each value to the correct entity, cohort, model, intervention, or condition.
            3. If a value is missing, write "N/A". Never guess.
            4. If the paper contains multiple cohorts/arms/conditions, create separate rows when appropriate.
            5. Keep extraction faithful to the source text.
            6. Do not invent fields outside the schema.

        [SCHEMA]
            Columns:
                {header_line}

        [FIELD DEFINITIONS]
            {bullets}

        [OUTPUT FORMAT]
            Return ONLY the Markdown table. No explanation before or after.
    """.strip()

# CLI entry — discovery + selection orchestration
def main():
    parser = argparse.ArgumentParser(description="Design schema and extraction prompt from PDFs.")
    parser.add_argument("--domain", type=str, required=True)
    parser.add_argument("--pdf_dir", type=str, required=True)
    parser.add_argument("--last_tier", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)

    parser.add_argument("--max_papers", type=int, default=60)
    parser.add_argument("--candidate_pool_size", type=int, default=60)
    parser.add_argument("--rep_chars", type=int, default=2500)
    parser.add_argument("--diversity_threshold", type=float, default=0.85)

    parser.add_argument("--metadata_jsonl", type=str, default=None)
    parser.add_argument("--max_chars_per_paper", type=int, default=12000)
    parser.add_argument("--max_candidate_fields", type=int, default=25)
    parser.add_argument("--max_final_fields", type=int, default=8)
    parser.add_argument("--min_coverage", type=float, default=0.0)

    parser.add_argument("--no_distill", action="store_true")
    parser.add_argument("--distill_input_chars", type=int, default=60000)
    parser.add_argument("--distill_quote_limit", type=int, default=10)

    parser.add_argument("--n_anchors", type=int, default=6)
    parser.add_argument("--skip_discovery", action="store_true")
    parser.add_argument("--embed_threshold", type=float, default=0.65)
    parser.add_argument("--min_support", type=int, default=2)

    args = parser.parse_args()

    spec = get_domain_spec(args.domain)
    pdf_dir = Path(args.pdf_dir)
    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

    base_dir = Path(__file__).resolve().parent.parent
    default_runs = base_dir / "runs" / args.domain
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = default_runs / f"design_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[PromptDesign] Domain: {args.domain}")
    print(f"[PromptDesign] Subdomain (last_tier): {args.last_tier!r}")
    print(f"[PromptDesign] Task: {spec.schema_task_description}")
    print(f"[PromptDesign] Output dir: {out_dir}")

    selected = select_schema_papers(
        pdf_dir=pdf_dir,
        domain=args.domain,
        last_tier=args.last_tier,
        max_papers=args.max_papers,
        candidate_pool_size=args.candidate_pool_size,
        rep_chars=args.rep_chars,
        diversity_threshold=args.diversity_threshold,
        metadata_jsonl=Path(args.metadata_jsonl) if args.metadata_jsonl else None,
    )
    if not selected:
        print("[PromptDesign][ERROR] No schema papers selected.")
        return

    papers = collect_paper_texts_from_paths(selected, max_chars_per_paper=args.max_chars_per_paper)
    if not papers:
        print("[PromptDesign][ERROR] No usable paper texts found.")
        return

    paper_level = propose_paper_level_candidate_fields(
        papers=papers,
        domain=args.domain,
        last_tier=args.last_tier,
        max_fields_per_paper=args.max_candidate_fields,
        max_text_chars=args.max_chars_per_paper,
        distill_input_chars=args.distill_input_chars,
        distill_quote_limit=args.distill_quote_limit,
        disable_distill=args.no_distill,
    )
    paper_level_path = out_dir / "paper_level_field_proposals.jsonl"
    save_jsonl(paper_level_path, paper_level)

    # Cluster / merge / discover anchors / select final
    from schema_lib.schema_cluster_merge import build_canonical_candidates, select_final_schema
    from schema_lib.subdomain_discovery import discover_subdomain_anchors

    canonicals, stats = build_canonical_candidates(
        paper_level_jsonl=paper_level_path,
        out_dir=out_dir,
        domain=args.domain,
        embed_threshold=args.embed_threshold,
    )

    anchors = []
    if not args.skip_discovery and args.last_tier:
        anchors = discover_subdomain_anchors(
            llm_chat_fn=llm_chat,
            paper_level_jsonl=paper_level_path,
            domain=args.domain,
            last_tier=args.last_tier,
            n_anchors=args.n_anchors,
            out_path=out_dir / "discovered_anchors.json",
        )
        print(f"[PromptDesign][Discover] {len(anchors)} corpus anchors emerged for {args.last_tier!r}")
        for a in anchors:
            print(f"  - {a.get('name')} (centrality={a.get('centrality_to_subdomain')})")

    final_fields = select_final_schema(
        canonical_candidates=canonicals,
        out_dir=out_dir,
        domain=args.domain,
        last_tier=args.last_tier,
        max_final_fields=args.max_final_fields,
        min_support=args.min_support,
        anchors=anchors,
    )

    extraction_prompt = build_extraction_prompt(final_fields, domain=args.domain, last_tier=args.last_tier)

    save_json(out_dir / "candidate_fields.json", canonicals)
    save_json(out_dir / "final_fields.json", final_fields)
    save_text(out_dir / "extraction_prompt.txt", extraction_prompt)
    save_text(out_dir / "selected_schema_papers.txt", "\n".join(str(p) for p in selected))
    save_text(
        out_dir / "sample_texts_preview.txt",
        "\n\n==========\n\n".join([f"[SOURCE: {p['source_file']}]\n{p['text']}" for p in papers]),
    )

    print(f"[PromptDesign] Stats: {stats}")
    print("[PromptDesign] Saved discovered_anchors.json, final_schema.json, extraction_prompt.txt")

if __name__ == "__main__":
    main()