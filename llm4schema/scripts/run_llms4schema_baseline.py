from __future__ import annotations
import argparse, json, math, os, re, sys, time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)
except Exception:
    pass
ROOT = Path(__file__).resolve().parents[2]
PHASE1 = ROOT / "Phase1"
for p in (ROOT, PHASE1):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
try:
    from setup.test_azure_openai import chat_client, AZURE_OPENAI_CHAT_DEPLOYMENT
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Could not import setup.test_azure_openai. Run this script from the repo "
        "where your existing Azure OpenAI setup module is available."
    ) from e
try:
    from queries_lib.paper_model import pdf_to_text
except Exception:
    pdf_to_text = None
try:
    from schema_lib.domain_registry import get_domain_spec
except Exception:
    get_domain_spec = None

# File I/O, text cleanup, JSON parsing, LLM call and PDF reading
def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path or not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows

def clean_text(s: str, max_chars: int) -> str:
    s = re.sub(r"\s+", " ", s or "").strip()
    return s[:max_chars]

def strip_code_fence(s: str) -> str:
    s = (s or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.I)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()

def parse_json_object(raw: str) -> Dict[str, Any]:
    raw = strip_code_fence(raw)
    try:
        return json.loads(raw)
    except Exception:
        pass
    m = re.search(r"\{.*\}", raw, flags=re.S)
    if not m:
        return {"schema": [], "raw": raw}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {"schema": [], "raw": raw}

def llm_json(system_prompt: str, user_prompt: str, max_retries: int = 3) -> Dict[str, Any]:
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = chat_client.chat.completions.create(
                model=AZURE_OPENAI_CHAT_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )
            return parse_json_object(resp.choices[0].message.content or "{}")
        except Exception as e:
            last_err = e
            wait = 10 * (attempt + 1)
            print(f"[LLMs4SchemaBaseline] LLM call failed: {e}; retry in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"LLM failed after retries: {last_err}")

def read_pdf_text(path: Path, max_chars: int) -> str:
    if pdf_to_text is not None:
        try:
            return clean_text(pdf_to_text(path), max_chars)
        except Exception as e:
            print(f"[LLMs4SchemaBaseline] pdf_to_text failed for {path.name}: {e}")
    return ""

@dataclass
class PaperText:
    source: str
    title: str
    text: str

def paper_title_from_path(p: Path) -> str:
    return re.sub(r"[_\-]+", " ", p.stem).strip()

def collect_seed_papers(seed_dir: Path, max_chars_per_paper: int) -> List[PaperText]:
    pdfs = sorted(seed_dir.glob("*.pdf"))
    out: List[PaperText] = []
    for p in pdfs:
        text = read_pdf_text(p, max_chars_per_paper)
        if text:
            out.append(PaperText(source=str(p), title=paper_title_from_path(p), text=text))
    return out

def collect_stage3_papers(
    corpus_dir: Optional[Path],
    metadata_jsonl: Optional[Path],
    max_papers: int,
    max_chars_per_paper: int,
) -> List[PaperText]:
    papers: List[PaperText] = []

    # Prefer metadata if provided because it contains titles/abstracts and local paths.
    rows = load_jsonl(metadata_jsonl) if metadata_jsonl else []
    seen: set[str] = set()
    for r in rows:
        if len(papers) >= max_papers:
            break
        title = str(r.get("title") or r.get("paper_title") or "").strip()
        abstract = str(r.get("abstract") or r.get("summary") or "").strip()
        local = r.get("local_pdf_path") or r.get("pdf_path")
        key = local or title
        if not key or key in seen:
            continue
        seen.add(key)
        text = ""
        if abstract:
            text = clean_text(f"Title: {title}\nAbstract: {abstract}", max_chars_per_paper)
        elif local and Path(local).exists():
            text = read_pdf_text(Path(local), max_chars_per_paper)
        if text:
            papers.append(PaperText(source=str(local or metadata_jsonl), title=title or str(key), text=text))

    # Fallback: read PDFs from cumulative_corpus.
    if corpus_dir and corpus_dir.exists() and len(papers) < max_papers:
        import random

        pdfs = sorted(corpus_dir.glob("*.pdf"))
        if len(pdfs) > max_papers:
            rng = random.Random(42)
            pdfs = rng.sample(pdfs, max_papers)
        for p in pdfs:
            if len(papers) >= max_papers:
                break
            if str(p) in seen:
                continue
            text = read_pdf_text(p, max_chars_per_paper)
            if text:
                papers.append(PaperText(source=str(p), title=paper_title_from_path(p), text=text))
                seen.add(str(p))

    return papers[:max_papers]

def render_papers(papers: List[PaperText], max_total_chars: int) -> str:
    chunks: List[str] = []
    used = 0
    for i, p in enumerate(papers, start=1):
        block = f"[Paper {i}]\nTitle: {p.title}\nSource: {p.source}\nText: {p.text}\n"
        if used + len(block) > max_total_chars:
            remain = max_total_chars - used
            if remain <= 500:
                break
            block = block[:remain]
        chunks.append(block)
        used += len(block)
    return "\n---\n".join(chunks)

def chunked(xs: List[Any], n: int) -> Iterable[List[Any]]:
    for i in range(0, len(xs), n):
        yield xs[i:i + n]

def schema_list(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    s = obj.get("schema") or obj.get("fields") or obj.get("properties") or []
    if isinstance(s, dict):
        # JSON-schema style: properties dict -> list.
        props = s.get("properties") if "properties" in s else s
        if isinstance(props, dict):
            out = []
            for name, v in props.items():
                if isinstance(v, dict):
                    row = dict(v)
                    row.setdefault("name", name)
                    out.append(row)
            return out
    if isinstance(s, list):
        return [x for x in s if isinstance(x, dict)]
    return []

def normalize_field_name(name: str) -> str:
    name = (name or "").strip()
    name = re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()
    return name

def normalize_schema(obj: Dict[str, Any], max_final_fields: int, method_tag: str) -> List[Dict[str, Any]]:
    fields = schema_list(obj)
    out: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for f in fields:
        raw_name = f.get("name") or f.get("property") or f.get("field") or f.get("field_name")
        name = normalize_field_name(str(raw_name or ""))
        if not name or name in seen:
            continue
        seen.add(name)
        aliases = f.get("aliases") or f.get("synonyms") or []
        if isinstance(aliases, str):
            aliases = [aliases]
        desc = str(f.get("description") or f.get("definition") or f.get("desc") or "").strip()
        out.append({
            "name": name,
            "description": desc,
            "aliases": [str(a).strip() for a in aliases if str(a).strip()],
            "data_type": str(f.get("data_type") or f.get("type") or "string_or_numeric").strip(),
            "constraints": f.get("constraints") or [],
            "relationships": f.get("relationships") or [],
            "support_count": int(f.get("support_count") or f.get("evidence_count") or 0),
            "evidence_snippets": f.get("evidence_snippets") or [],
            "source": method_tag,
        })
    return out[:max_final_fields]

def get_process_spec_text(domain: str, subdomain: str, last_tier: str) -> str:
    base = ""
    if get_domain_spec is not None:
        try:
            spec = get_domain_spec(domain)
            base = str(getattr(spec, "schema_task_description", "") or "")
        except Exception:
            base = ""
    return (
        f"Domain: {domain}\n"
        f"Subdomain label: {subdomain}\n"
        f"Target focus / last tier: {last_tier}\n"
        f"Scientific schema induction goal: {base or 'Discover reusable extraction fields from scientific papers.'}\n"
        "The schema should contain concise extraction fields suitable for downstream evidence-grounded information extraction."
    )

# Prompts
SYSTEM = """You are an expert system for scientific schema discovery.
Return only valid JSON. Do not include markdown.
The schema should be compact, general across papers in the target subdomain, and useful for evidence-grounded extraction.
Each field must have: name, description, data_type, constraints, relationships, aliases, and evidence_snippets.
Use snake_case field names. Avoid paper-specific fields, author names, and overly generic fields such as title or abstract.
"""

def stage1_prompt(process_spec: str, max_fields: int) -> str:
    return f"""Stage 1 of LLMs4SchemaDiscovery: generate an initial schema from the process specification.

Process specification:
{process_spec}

Return JSON with this shape:
{{"schema": [{{"name": "...", "description": "...", "data_type": "...", "constraints": [], "relationships": [], "aliases": [], "evidence_snippets": []}}], "notes": "..."}}
Limit to at most {max_fields} candidate fields. The later stages may refine it.
"""

def refine_prompt(stage_name: str, prev_schema: List[Dict[str, Any]], process_spec: str,
                  papers: List[PaperText], max_fields: int, max_total_chars: int) -> str:
    corpus = render_papers(papers, max_total_chars=max_total_chars)
    return f"""{stage_name} of LLMs4SchemaDiscovery under a no-human-feedback protocol.

Process specification:
{process_spec}

Previous schema:
{json.dumps(prev_schema, ensure_ascii=False, indent=2)}

Scientific papers for this refinement batch:
{corpus}

Instructions:
1. Refine the schema using only the papers and process specification.
2. Add missing reusable fields if they are well supported by this batch.
3. Merge redundant fields and keep concise names.
4. Remove fields that are too generic, off-topic, or paper-specific.
5. Preserve fields that remain useful even if this batch does not mention every detail.

Return JSON with this shape:
{{"schema": [{{"name": "...", "description": "...", "data_type": "...", "constraints": [], "relationships": [], "aliases": [], "evidence_snippets": ["short evidence phrase"]}}], "notes": "..."}}
Limit final schema to at most {max_fields} fields.
"""
# Main workflow
def run(args: argparse.Namespace) -> None:
    out = Path(args.output_root).resolve()
    out.mkdir(parents=True, exist_ok=True)

    process_spec = get_process_spec_text(args.domain, args.subdomain, args.last_tier)
    save_text(out / "process_specification.txt", process_spec)

    seed_papers = collect_seed_papers(Path(args.seed_dir), args.max_chars_per_paper)
    if not seed_papers:
        raise RuntimeError(f"No readable seed PDFs found in {args.seed_dir}")

    stage3_papers = collect_stage3_papers(
        corpus_dir=Path(args.stage3_corpus_dir) if args.stage3_corpus_dir else None,
        metadata_jsonl=Path(args.stage3_metadata_jsonl) if args.stage3_metadata_jsonl else None,
        max_papers=args.max_stage3_papers,
        max_chars_per_paper=args.max_chars_per_paper,
    )
    if not stage3_papers:
        print("[LLMs4SchemaBaseline] WARNING: no stage-3 papers found; stage 3 will reuse seed papers only.")
        stage3_papers = seed_papers

    manifest = {
        "method": "LLMs4SchemaDiscovery_no_human_feedback_adapter",
        "domain": args.domain,
        "subdomain": args.subdomain,
        "last_tier": args.last_tier,
        "seed_dir": str(args.seed_dir),
        "stage3_corpus_dir": str(args.stage3_corpus_dir or ""),
        "stage3_metadata_jsonl": str(args.stage3_metadata_jsonl or ""),
        "n_seed_papers": len(seed_papers),
        "n_stage3_papers": len(stage3_papers),
        "batch_size": args.batch_size,
        "max_final_fields": args.max_final_fields,
        "model": AZURE_OPENAI_CHAT_DEPLOYMENT,
    }
    save_json(out / "run_config.json", manifest)

    print(f"[LLMs4SchemaBaseline] Stage 1: process specification -> initial schema")
    stage1 = llm_json(SYSTEM, stage1_prompt(process_spec, args.max_intermediate_fields))
    save_json(out / "stage1_initial_schema_raw.json", stage1)
    current = normalize_schema(stage1, args.max_intermediate_fields, "llms4schema_stage1")
    save_json(out / "stage1_initial_schema.json", current)

    print(f"[LLMs4SchemaBaseline] Stage 2: refine with {len(seed_papers)} curated seed papers")
    stage2 = llm_json(
        SYSTEM,
        refine_prompt(
            "Stage 2 preliminary schema refinement",
            current,
            process_spec,
            seed_papers,
            args.max_intermediate_fields,
            args.max_total_chars_per_call,
        ),
    )
    save_json(out / "stage2_refined_schema_raw.json", stage2)
    current = normalize_schema(stage2, args.max_intermediate_fields, "llms4schema_stage2")
    save_json(out / "stage2_refined_schema.json", current)

    print(f"[LLMs4SchemaBaseline] Stage 3: finalize with {len(stage3_papers)} broader-corpus papers")
    stage3_steps: List[Dict[str, Any]] = []
    batches = list(chunked(stage3_papers, args.batch_size))
    for bi, batch in enumerate(batches, start=1):
        print(f"[LLMs4SchemaBaseline]   Stage 3 batch {bi}/{len(batches)}: {len(batch)} papers")
        raw = llm_json(
            SYSTEM,
            refine_prompt(
                f"Stage 3 final schema refinement batch {bi}/{len(batches)}",
                current,
                process_spec,
                batch,
                args.max_intermediate_fields,
                args.max_total_chars_per_call,
            ),
        )
        step_schema = normalize_schema(raw, args.max_intermediate_fields, f"llms4schema_stage3_batch_{bi}")
        stage3_steps.append({"batch": bi, "n_papers": len(batch), "raw": raw, "schema": step_schema})
        current = step_schema
        save_json(out / f"stage3_batch_{bi:02d}_schema_raw.json", raw)
        save_json(out / f"stage3_batch_{bi:02d}_schema.json", step_schema)

    final_schema = normalize_schema({"schema": current}, args.max_final_fields, "llms4schema_final")
    final_dir = out / "final_schema_from_cumulative_corpus"
    save_json(final_dir / "final_schema.json", final_schema)
    save_json(final_dir / "final_fields.json", final_schema)
    save_json(out / "stage3_steps_summary.json", [
        {"batch": s["batch"], "n_papers": s["n_papers"], "field_names": [f["name"] for f in s["schema"]]}
        for s in stage3_steps
    ])
    save_json(out / "run_summary.json", {**manifest, "final_schema_path": str(final_dir / "final_schema.json")})
    print(f"[LLMs4SchemaBaseline] DONE: {final_dir / 'final_schema.json'}")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LLMs4SchemaDiscovery no-human-feedback baseline for MemoSchema.")
    p.add_argument("--domain", required=True, choices=["ad", "chem", "cs"])
    p.add_argument("--subdomain", required=True)
    p.add_argument("--last-tier", required=True, dest="last_tier")
    p.add_argument("--seed-dir", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--stage3-corpus-dir", default=None,
                   help="Usually the target run's cumulative_corpus directory.")
    p.add_argument("--stage3-metadata-jsonl", default=None,
                   help="Optional downloaded_metadata.jsonl/new_unique_papers.jsonl; abstracts are preferred when present.")
    p.add_argument("--max-stage3-papers", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=5)
    p.add_argument("--max-final-fields", type=int, default=8)
    p.add_argument("--max-intermediate-fields", type=int, default=14)
    p.add_argument("--max-chars-per-paper", type=int, default=4500)
    p.add_argument("--max-total-chars-per-call", type=int, default=24000)
    return p.parse_args()

if __name__ == "__main__":
    run(parse_args())
