import hashlib, json, os, time
from pathlib import Path
from typing import Callable, Dict, Any, List

from queries_lib.paper_model import pdf_to_text, chunk_text

JsonChatFn = Callable[[list, bool], str]

def llm_json(chat_fn: JsonChatFn, system_prompt: str, user_prompt: str, sleep_s: float = 1.0) -> Dict[str, Any]:
    raw = chat_fn(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        force_json=True,
    )
    time.sleep(sleep_s)
    try:
        return json.loads(raw)
    except Exception:
        return {"_raw": raw}

def _seed_cache_path(pdf_path: Path, spec, chunk_size: int, overlap: int):
    """Cross-run seed-summary memory (opt-in): MEMOSCHEMA_SEED_CACHE_DIR = absolute directory. Keyed by the PDF bytes
    and everything that changes the summary (prompts, system role, chunking, deployment), so a hit is only ever an
    earlier summary of the same file under the same configuration."""
    root = os.getenv("MEMOSCHEMA_SEED_CACHE_DIR", "")
    if not root:
        return None
    h = hashlib.sha256(Path(pdf_path).read_bytes()).hexdigest()
    cfg = json.dumps([spec.system_role, spec.chunk_extract_prompt, spec.paper_merge_prompt, chunk_size, overlap,
                      os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "")], ensure_ascii=False)
    return Path(root) / hashlib.sha256(cfg.encode()).hexdigest()[:16] / f"{h}.json"

def summarize_seed_pdf(pdf_path: Path, spec, chat_fn: JsonChatFn, chunk_size: int = 4500, overlap: int = 500) -> Dict[str, Any]:
    cache = _seed_cache_path(pdf_path, spec, chunk_size, overlap)
    if cache is not None and cache.exists():
        merged = json.loads(cache.read_text(encoding="utf-8"))
        print(f"[SeedMemory] reused summary of {Path(pdf_path).name} ({merged.get('chunk_count')} chunks)", flush=True)
        return merged
    merged = _summarize_seed_pdf(pdf_path, spec, chat_fn, chunk_size, overlap)
    if cache is not None and merged:
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(f".tmp{os.getpid()}")
        tmp.write_text(json.dumps(merged, ensure_ascii=False), encoding="utf-8"); os.replace(tmp, cache)
    return merged

def _summarize_seed_pdf(pdf_path: Path, spec, chat_fn: JsonChatFn, chunk_size: int = 4500, overlap: int = 500) -> Dict[str, Any]:
    full_text = pdf_to_text(pdf_path)
    chunks = chunk_text(full_text, chunk_size=chunk_size, overlap=overlap)

    if not chunks:
        return {}

    chunk_summaries: List[Dict[str, Any]] = []
    for i, chunk in enumerate(chunks):
        obj = llm_json(
            chat_fn=chat_fn,
            system_prompt=f"You are a {spec.system_role} extraction assistant. Return ONLY valid JSON.",
            user_prompt=spec.chunk_extract_prompt.replace("{paper_text}", chunk).replace("{chunk_id}", str(i)),
        )
        chunk_summaries.append(obj)

    merged_input = json.dumps(
        {
            "paper_name": pdf_path.name,
            "chunk_summaries": chunk_summaries,
        },
        ensure_ascii=False,
    )

    merged = llm_json(
        chat_fn=chat_fn,
        system_prompt=f"You are a {spec.system_role} extraction assistant. Return ONLY valid JSON.",
        user_prompt=spec.paper_merge_prompt.replace("{chunk_summaries}", merged_input),
    )

    merged["chunk_count"] = len(chunks)
    return merged