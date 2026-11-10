from __future__ import annotations

import hashlib, json, math, os, re, shutil, sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Same embedding client schema_lib already uses
_PHASE1_DIR = Path(__file__).resolve().parents[1]
if str(_PHASE1_DIR) not in sys.path:
    sys.path.insert(0, str(_PHASE1_DIR))

from setup.test_azure_embed import embed_client, AZURE_OPENAI_EMBED_DEPLOYMENT
from queries_lib.paper_model import pdf_to_text
from retrieve_lib.utils import fname_with_tag
from retrieve_literature import get_record_key  # canonical key: doi > id > norm title

# Vector operations: cosine similarity, mean vector, blend
def _cos(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)

def _mean_vector(vectors: List[List[float]]) -> List[float]:
    if not vectors:
        return []
    dim = len(vectors[0])
    out = [0.0] * dim
    for v in vectors:
        if len(v) != dim:
            continue
        for i, x in enumerate(v):
            out[i] += x
    return [x / len(vectors) for x in out]

def _blend(a: List[float], b: List[float], alpha: float) -> List[float]:
    if not a:
        return list(b)
    if not b:
        return list(a)
    if len(a) != len(b):
        return list(a)
    return [alpha * x + (1.0 - alpha) * y for x, y in zip(a, b)]

# Embedding cache (append-only jsonl)
class _EmbedCache:
    def __init__(self, cache_path: Path):
        self.cache_path = Path(cache_path)
        self._mem: Dict[str, List[float]] = {}
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            row = json.loads(line)
                        except Exception:
                            continue
                        k = row.get("key")
                        e = row.get("embedding")
                        if isinstance(k, str) and isinstance(e, list) and e:
                            self._mem[k] = e
            except Exception as ex:
                print(f"[seed-embed] embed cache read failed: {ex}")

    def get(self, key: str) -> Optional[List[float]]:
        return self._mem.get(key)

    def put(self, key: str, emb: List[float]) -> None:
        if not key or key in self._mem:
            return
        self._mem[key] = emb
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "embedding": emb}, ensure_ascii=False) + "\n")

def _embed_batch(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    resp = embed_client.embeddings.create(
        model=AZURE_OPENAI_EMBED_DEPLOYMENT,
        input=texts,
    )
    return [item.embedding for item in resp.data]

def _embed_with_cache(
    items: List[Tuple[str, str]],   # (cache_key, text)
    cache: _EmbedCache,
    batch_size: int = 64,
) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {}
    pending_keys: List[str] = []
    pending_texts: List[str] = []
    for key, text in items:
        if not key:
            continue
        cached = cache.get(key)
        if cached is not None:
            out[key] = cached
            continue
        if not text or not text.strip():
            continue
        pending_keys.append(key)
        pending_texts.append(text)

    for i in range(0, len(pending_texts), batch_size):
        chunk_texts = pending_texts[i:i + batch_size]
        chunk_keys = pending_keys[i:i + batch_size]
        try:
            embs = _embed_batch(chunk_texts)
        except Exception as e:
            print(f"[seed-embed] embedding batch failed (n={len(chunk_texts)}): {e}")
            continue
        for k, emb in zip(chunk_keys, embs):
            out[k] = emb
            cache.put(k, emb)
    return out

# Text representation
def _abstract_or_pdf_text(
    rec: dict,
    pdf_path: Optional[Path],
    max_pdf_chars: int = 4000,
) -> str:
    title = (rec.get("title") or "").strip()
    for k in ("abstract", "summary", "description"):
        v = rec.get(k)
        if v and isinstance(v, str) and len(v.strip()) >= 80:
            text = v.strip()
            return f"{title}\n\n{text}" if title else text
    if pdf_path and Path(pdf_path).exists():
        try:
            text = pdf_to_text(Path(pdf_path))
            if text and text.strip():
                head = text[:max_pdf_chars]
                return f"{title}\n\n{head}" if title else head
        except Exception:
            pass
    return title

# Original seed centroid
def _compute_original_seed_centroid(
    original_seeds_dir: Path,
    cache: _EmbedCache,
    state_dir: Path,
    max_pdf_chars: int = 4000,
) -> List[float]:
    centroid_path = state_dir / "original_seed_centroid.json"
    if centroid_path.exists():
        try:
            obj = json.loads(centroid_path.read_text(encoding="utf-8"))
            emb = obj.get("centroid")
            if isinstance(emb, list) and emb:
                return emb
        except Exception:
            pass

    seeds = sorted(Path(original_seeds_dir).glob("*.pdf"))
    if not seeds:
        return []

    items: List[Tuple[str, str]] = []
    for p in seeds:
        try:
            text = pdf_to_text(p)
        except Exception as e:
            print(f"[seed-embed] failed reading original seed {p.name}: {e}")
            continue
        if not text or not text.strip():
            continue
        # Prefix avoids cache-key collision with retrieved-paper keys
        items.append((f"orig_seed_pdf:{p.stem}", text[:max_pdf_chars]))

    embs_map = _embed_with_cache(items, cache=cache)
    embs = [embs_map[k] for k, _ in items if k in embs_map]
    centroid = _mean_vector(embs)
    if centroid:
        state_dir.mkdir(parents=True, exist_ok=True)
        centroid_path.write_text(
            json.dumps({"centroid": centroid, "n_seeds": len(embs)}, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[seed-embed] computed original_seed_centroid from {len(embs)} seeds, "
              f"cached to {centroid_path.name}")
    return centroid

def _compute_seeds_dir_centroid(
    seeds_dir: Path,
    cache: _EmbedCache,
    max_pdf_chars: int = 4000,
) -> List[float]:
    seed_pdfs = sorted(Path(seeds_dir).glob("*.pdf"))
    items: List[Tuple[str, str]] = []
    for p in seed_pdfs:
        try:
            text = pdf_to_text(p)
        except Exception:
            continue
        if not text:
            continue
        # Round seeds are cached under the round-seed key, so a round seed that is a copy of an original seed
        # (round 1) does not reuse the "orig_seed_pdf:..." entry; in round 1 both sets are identical, so the
        # centroid is the same.
        items.append((f"round_seed_pdf:{p.stem}", text[:max_pdf_chars]))
    embs_map = _embed_with_cache(items, cache=cache)
    return _mean_vector(list(embs_map.values()))

# Corpus history (already-downloaded papers across all rounds)
def _collect_corpus_records(run_dir: Path) -> Tuple[List[dict], Dict[str, Path]]:
    by_key: Dict[str, dict] = {}
    pdf_lookup: Dict[str, Path] = {}
    if not Path(run_dir).exists():
        return [], {}
    for round_dir in sorted(Path(run_dir).glob("round*")):
        if not round_dir.is_dir():
            continue
        download_dir = round_dir / "downloads"
        meta_files = []
        for fname in ("downloaded_metadata.jsonl", "search_results.jsonl"):
            p = round_dir / fname
            if p.exists():
                meta_files.append(p)
        for p in meta_files:
            try:
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except Exception:
                            continue
                        key = get_record_key(rec)
                        if not key:
                            continue
                        # only keep records whose PDF physically exists
                        try:
                            fname2 = fname_with_tag(rec)
                        except Exception:
                            fname2 = None
                        if fname2:
                            pdf_path = download_dir / fname2
                            if pdf_path.exists():
                                pdf_lookup.setdefault(key, pdf_path)
                                by_key.setdefault(key, rec)
                                continue
                        # fallback: keep record without pdf path
                        by_key.setdefault(key, rec)
            except Exception:
                continue
    return list(by_key.values()), pdf_lookup

def _embed_records(
    records: List[dict],
    pdf_lookup: Dict[str, Path],
    cache: _EmbedCache,
    max_pdf_chars: int = 4000,
) -> Dict[str, List[float]]:
    items: List[Tuple[str, str]] = []
    for rec in records:
        key = get_record_key(rec)
        if not key:
            continue
        text = _abstract_or_pdf_text(rec, pdf_lookup.get(key), max_pdf_chars=max_pdf_chars)
        if not text or not text.strip():
            continue
        items.append((key, text))
    return _embed_with_cache(items, cache=cache)

# Main entry: MMR-based seed selection
def select_new_seeds_by_similarity(
    round_dir: Path,
    download_dir: Path,
    seeds_dir: Path,
    next_seed_dir: Path,
    max_seed_pdfs: int = 5,
    seen_seed_keys: Set[str] | None = None,
    domain: str | None = None,
    *,
    original_seeds_dir: Path | None = None,
    run_dir: Path | None = None,
    mmr_lambda: float = 0.7,
    anchor_alpha: float = 0.7,
    max_pdf_chars: int = 4000,
    seed_state_dir: Path | None = None,
) -> List[Path]:
    seen_seed_keys = seen_seed_keys if seen_seed_keys is not None else set()
    next_seed_dir = Path(next_seed_dir)
    next_seed_dir.mkdir(parents=True, exist_ok=True)

    if run_dir is None:
        # round_dir is run_dir/roundNN
        run_dir = Path(round_dir).resolve().parent
    else:
        run_dir = Path(run_dir)

    if seed_state_dir is None:
        seed_state_dir = run_dir / "seed_state"
    seed_state_dir = Path(seed_state_dir)
    seed_state_dir.mkdir(parents=True, exist_ok=True)

    cache = _EmbedCache(seed_state_dir / "embed_cache.jsonl")

    # 1) anchor = alpha * original_seed_centroid + (1-alpha) * current_seeds_centroid
    original_seeds_dir = Path(original_seeds_dir) if original_seeds_dir else Path(seeds_dir)
    original_centroid = _compute_original_seed_centroid(
        original_seeds_dir=original_seeds_dir,
        cache=cache,
        state_dir=seed_state_dir,
        max_pdf_chars=max_pdf_chars,
    )
    current_centroid = _compute_seeds_dir_centroid(
        seeds_dir=Path(seeds_dir),
        cache=cache,
        max_pdf_chars=max_pdf_chars,
    )

    if original_centroid and current_centroid:
        anchor = _blend(original_centroid, current_centroid, anchor_alpha)
        anchor_src = f"alpha={anchor_alpha} (original + current blend)"
    elif original_centroid:
        anchor = original_centroid
        anchor_src = "original only"
    elif current_centroid:
        anchor = current_centroid
        anchor_src = "current only (original missing)"
    else:
        anchor = []
        anchor_src = "no anchor (all seed embeddings failed)"

    # 2) read this round's downloaded candidates
    search_results_path = Path(round_dir) / "search_results.jsonl"
    if not search_results_path.exists():
        raise FileNotFoundError(f"search_results.jsonl not found: {search_results_path}")

    candidate_records: List[dict] = []
    cand_pdf_lookup: Dict[str, Path] = {}
    with open(search_results_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            key = get_record_key(rec)
            if not key:
                continue
            try:
                fname = fname_with_tag(rec)
            except Exception:
                continue
            pdf_path = Path(download_dir) / fname
            if not pdf_path.exists():
                continue
            candidate_records.append(rec)
            cand_pdf_lookup[key] = pdf_path

    if not candidate_records:
        print("[seed-embed] No downloaded PDFs in this round to choose from.")
        return []

    # 3) embed candidates
    cand_embs_map = _embed_records(
        candidate_records,
        pdf_lookup=cand_pdf_lookup,
        cache=cache,
        max_pdf_chars=max_pdf_chars,
    )

    # 4) embed already-seen corpus papers (across all rounds, including this one)
    corpus_records, corpus_pdf_lookup = _collect_corpus_records(run_dir)
    corpus_embs_map = _embed_records(
        corpus_records,
        pdf_lookup=corpus_pdf_lookup,
        cache=cache,
        max_pdf_chars=max_pdf_chars,
    )

    # 5) iterative MMR
    print(f"[seed-embed] anchor={anchor_src}  lambda={mmr_lambda}  alpha={anchor_alpha}  "
          f"corpus_embs={len(corpus_embs_map)}  candidates={len(candidate_records)}")

    # Pre-compute relevance for each viable candidate
    cand_pool: List[Dict] = []
    for rec in candidate_records:
        key = get_record_key(rec)
        if not key or key in seen_seed_keys:
            continue
        emb = cand_embs_map.get(key)
        if emb is None:
            continue
        rel = _cos(emb, anchor) if anchor else 0.0
        cand_pool.append({
            "key": key,
            "emb": emb,
            "rec": rec,
            "pdf": cand_pdf_lookup[key],
            "rel": rel,
        })

    if not cand_pool:
        print("[seed-embed] All candidates either lack an embedding or were already used as seeds.")
        return []

    chosen_paths: List[Path] = []
    selected_embs: List[List[float]] = []
    while len(chosen_paths) < max_seed_pdfs and cand_pool:
        best_idx = -1
        best_score = -1e18
        best_metrics: Tuple[float, float] = (0.0, 0.0)
        for i, c in enumerate(cand_pool):
            emb = c["emb"]
            # max similarity to (corpus minus self) plus already-selected seeds
            max_sim = 0.0
            for k, c_emb in corpus_embs_map.items():
                if k == c["key"]:
                    continue
                s = _cos(emb, c_emb)
                if s > max_sim:
                    max_sim = s
            for sel_emb in selected_embs:
                s = _cos(emb, sel_emb)
                if s > max_sim:
                    max_sim = s
            nov = 1.0 - max_sim
            score = mmr_lambda * c["rel"] + (1.0 - mmr_lambda) * nov
            if score > best_score:
                best_score = score
                best_idx = i
                best_metrics = (c["rel"], nov)
        if best_idx < 0:
            break
        picked = cand_pool.pop(best_idx)
        rel, nov = best_metrics
        dest = next_seed_dir / picked["pdf"].name
        if not dest.exists():
            shutil.copy2(picked["pdf"], dest)
        chosen_paths.append(dest)
        selected_embs.append(picked["emb"])
        seen_seed_keys.add(picked["key"])
        print(f"  pick score={best_score:.4f}  rel={rel:.4f}  nov={nov:.4f}  "
              f"-> {picked['pdf'].name}")

    if not chosen_paths:
        print("[seed-embed][CRITICAL] No new seeds could be selected.")
    elif len(chosen_paths) < max_seed_pdfs:
        print(f"[seed-embed][WARNING] Only selected {len(chosen_paths)}/{max_seed_pdfs} seeds.")

    return chosen_paths
