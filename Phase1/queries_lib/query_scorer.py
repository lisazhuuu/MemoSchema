import os, math, re
from typing import List, Dict, Any

def cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)

def mean_vector(vectors: List[List[float]]) -> List[float]:
    if not vectors:
        return []
    dim = len(vectors[0])
    out = [0.0] * dim
    for v in vectors:
        for i, x in enumerate(v):
            out[i] += x
    return [x / len(vectors) for x in out]

def clean_query_bool(qb: str) -> str:
    qb = qb or ""
    qb = re.sub(r"\(?cat:[^)]+\)?", " ", qb)
    qb = re.sub(r"\s+", " ", qb).strip()
    return qb

def build_query_text(q: Dict[str, Any]) -> str:
    parts = [
        q.get("query_nl", ""),
        " ".join(q.get("synonyms", []) or []),
        clean_query_bool(q.get("query_bool", "")),
        " ".join(q.get("tiers", []) or []),
    ]
    return "\n".join([p for p in parts if p]).strip()

def build_seed_text(seed: Dict[str, Any]) -> str:
    parts = [
        seed.get("title", ""),
        seed.get("core_subtopic", ""),
        " ".join(seed.get("key_mechanisms_or_entities", []) or []),
        " ".join(seed.get("important_proteins_genes_pathways", []) or []),
        " ".join(seed.get("model_system_or_study_type", []) or []),
        " ".join(seed.get("main_experimental_setting_or_evidence_type", []) or []),
        " ".join(seed.get("likely_retrieval_keywords", []) or []),
        seed.get("concise_paper_summary", ""),
        seed.get("why_this_seed_is_useful_for_retrieval", ""),
    ]
    return "\n".join([p for p in parts if p]).strip()

def embed_texts(embed_client, embedding_model: str, texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    resp = embed_client.embeddings.create(
        model=embedding_model,
        input=texts,
    )
    return [item.embedding for item in resp.data]

def score_and_mark_queries(
    queries: List[Dict[str, Any]],
    seed_summaries: List[Dict[str, Any]],
    embed_client,
    embedding_model: str,
) -> List[Dict[str, Any]]:
    if not queries:
        return []

    if not embedding_model:
        # fallback: no embedding deployment configured
        for q in queries:
            q["seed_alignment_score"] = 1.0
            q["max_query_similarity"] = 0.0
            q["diversity_score"] = 1.0
            q["final_score"] = 1.0
            q["is_bad"] = False
        return queries

    query_texts = [build_query_text(q) for q in queries]
    seed_texts = [build_seed_text(s) for s in seed_summaries if s]

    if not seed_texts:
        for q in queries:
            q["seed_alignment_score"] = 1.0
            q["max_query_similarity"] = 0.0
            q["diversity_score"] = 1.0
            q["final_score"] = 1.0
            q["is_bad"] = False
        return queries

    query_embs = embed_texts(embed_client, embedding_model, query_texts)
    seed_embs = embed_texts(embed_client, embedding_model, seed_texts)
    seed_centroid = mean_vector(seed_embs)

    scored = []
    for i, q in enumerate(queries):
        q_emb = query_embs[i]

        # query-seed alignment
        seed_sims = [cosine_similarity(q_emb, s_emb) for s_emb in seed_embs]
        align_max = max(seed_sims) if seed_sims else 0.0
        align_centroid = cosine_similarity(q_emb, seed_centroid) if seed_centroid else 0.0
        seed_alignment = 0.6 * align_max + 0.4 * align_centroid

        # query-query redundancy
        qq_sims = []
        for j, other_emb in enumerate(query_embs):
            if i == j:
                continue
            qq_sims.append(cosine_similarity(q_emb, other_emb))
        max_query_similarity = max(qq_sims) if qq_sims else 0.0
        diversity_score = 1.0 - max_query_similarity

        # final score
        final_score = 0.75 * seed_alignment + 0.25 * diversity_score

        # bad query rules
        is_bad = False
        if seed_alignment < 0.58:
            is_bad = True
        if max_query_similarity > 0.93 and seed_alignment < 0.72:
            is_bad = True
        if final_score < 0.52:
            is_bad = True

        q["seed_alignment_score"] = round(seed_alignment, 4)
        q["max_query_similarity"] = round(max_query_similarity, 4)
        q["diversity_score"] = round(diversity_score, 4)
        q["final_score"] = round(final_score, 4)
        q["is_bad"] = is_bad
        scored.append(q)

    scored.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
    return scored