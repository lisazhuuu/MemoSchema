from __future__ import annotations
import json, math, os, time, sys, re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_PHASE1_DIR = Path(__file__).resolve().parents[1]
if str(_PHASE1_DIR) not in sys.path:
    sys.path.insert(0, str(_PHASE1_DIR))

from schema_induction import llm_chat
from setup.test_azure_embed import embed_client, AZURE_OPENAI_EMBED_DEPLOYMENT
EMBED_MODEL = AZURE_OPENAI_EMBED_DEPLOYMENT

from .domain_registry import DomainSpec, get_domain_spec
from .subdomain_discovery import load_anchors

# I/O
def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def save_json(path: Path, obj: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=True, indent=2)

# Text normalization, tokenization and cosine similarity
_TOKEN_RE = re.compile(r"[a-zA-Z0-9_βΒαΑ+-]+")

def _snake_to_words(s: str) -> str:
    return (s or "").strip().replace("_", " ").replace("-", " ")

def _norm_text(s: str, domain_spec: Optional[DomainSpec] = None) -> str:
    s = (s or "").lower()
    if domain_spec:
        for old, new in domain_spec.text_normalizations:
            s = s.replace(old, new)
    return s

def _tokenize(s: str, domain_spec: Optional[DomainSpec] = None) -> List[str]:
    return _TOKEN_RE.findall(_norm_text(s, domain_spec=domain_spec))

def _cosine_text_sim(a: str, b: str, domain_spec: Optional[DomainSpec] = None) -> float:
    c1 = Counter(_tokenize(a, domain_spec=domain_spec))
    c2 = Counter(_tokenize(b, domain_spec=domain_spec))
    if not c1 or not c2:
        return 0.0
    common = set(c1) & set(c2)
    dot = sum(c1[t] * c2[t] for t in common)
    n1 = math.sqrt(sum(v * v for v in c1.values()))
    n2 = math.sqrt(sum(v * v for v in c2.values()))
    if n1 == 0 or n2 == 0:
        return 0.0
    return dot / (n1 * n2)

def _field_text(f: Dict[str, Any]) -> str:
    name = _snake_to_words(str(f.get("name", "")))
    desc = str(f.get("description", "") or "").strip()
    aliases = f.get("possible_aliases") or f.get("aliases") or []
    if not isinstance(aliases, list):
        aliases = []
    alias_text = " ".join(_snake_to_words(str(a)) for a in aliases[:10])
    return f"{name}\n{alias_text}\n{desc}".strip()

def _candidate_full_text(c: Dict[str, Any]) -> str:
    parts = [
        str(c.get("name", "") or ""),
        str(c.get("description", "") or ""),
        " ".join(c.get("aliases", []) or []),
        " ".join(c.get("member_field_names", []) or []),
        " ".join(c.get("evidence_snippets", []) or []),
    ]
    return " ".join(parts)

# Embedding + clustering
def embed_texts(texts: List[str], batch_size: int = 128) -> List[List[float]]:
    cleaned = []
    for t in texts:
        t = (t or "").strip() or "unknown field"
        cleaned.append(t[:4000])

    all_vecs: List[List[float]] = []
    for i in range(0, len(cleaned), batch_size):
        batch = cleaned[i:i + batch_size]
        resp = embed_client.embeddings.create(model=EMBED_MODEL, input=batch)
        all_vecs.extend([d.embedding for d in resp.data])
        print(f"[Embed] processed {min(i + batch_size, len(cleaned))}/{len(cleaned)}", flush=True)

    return all_vecs

class DSU:
    def __init__(self, n: int):
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.r[ra] < self.r[rb]:
            self.p[ra] = rb
        elif self.r[ra] > self.r[rb]:
            self.p[rb] = ra
        else:
            self.p[rb] = ra
            self.r[ra] += 1

def cluster_by_embedding(fields: List[Dict[str, Any]], threshold: float) -> List[List[int]]:
    texts = [_field_text(f) for f in fields]
    vecs = embed_texts(texts)

    X = np.asarray(vecs, dtype=np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X = X / norms

    n = X.shape[0]
    dsu = DSU(n)

    chunk = 256
    for i0 in range(0, n, chunk):
        i1 = min(i0 + chunk, n)
        sims = X[i0:i1] @ X.T
        for ii in range(i1 - i0):
            i = i0 + ii
            row = sims[ii]
            js = np.where(row[i + 1:] >= threshold)[0] + (i + 1)
            for j in js.tolist():
                dsu.union(i, j)
        print(f"[Cluster] processed {i1}/{n}", flush=True)

    comps: Dict[int, List[int]] = {}
    for i in range(n):
        comps.setdefault(dsu.find(i), []).append(i)
    return list(comps.values())

# Average-linkage clustering (MEMOSCHEMA_FIX_CLUSTER=1; the LEGACY configuration uses single-linkage union at 0.65).
# Single-linkage union at 0.65 chains
# a domain's fields into one component holding 50-80% of all proposals, which
# the chunked LLM merge then turns into mega-canonicals with hundreds of
# unrelated aliases. Average linkage on name+description (aliases excluded,
# they are what makes unrelated fields look alike) yields concept-sized
# clusters instead.
_CLUSTER_AVG_SIM = 0.60
_MERGE_WORKERS = int(os.getenv("MEMOSCHEMA_MERGE_WORKERS", "8"))

def cluster_by_average_linkage(fields: List[Dict[str, Any]], min_sim: float = _CLUSTER_AVG_SIM) -> List[List[int]]:
    from scipy.cluster.hierarchy import fcluster, linkage
    if len(fields) < 2:
        return [list(range(len(fields)))]
    texts = [f"{_snake_to_words(str(f.get('name', '')))}: {f.get('description', '') or ''}" for f in fields]
    X = np.asarray(embed_texts(texts), dtype=np.float32)
    X /= np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-9, None)
    labels = fcluster(linkage(X, method="average", metric="cosine"), t=1 - min_sim, criterion="distance")
    comps: Dict[int, List[int]] = {}
    for i, lab in enumerate(labels):
        comps.setdefault(int(lab), []).append(i)
    return list(comps.values())

def _unmerged_canonical(members: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Single-paper cluster: cannot pass min_support, so skip the LLM merge and
    # keep it as a candidate only for memory matching / auditing.
    f = members[0]
    aliases = f.get("possible_aliases") if isinstance(f.get("possible_aliases"), list) else []
    return {
        "canonical_name": f.get("name"),
        "canonical_description": f.get("description"),
        "value_type": f.get("value_type"),
        "scope": f.get("scope"),
        "aliases": [str(a) for a in aliases][:8],
        "member_field_names": list(dict.fromkeys(str(m.get("name")) for m in members if m.get("name"))),
        "supporting_paper_ids": list(dict.fromkeys(m.get("paper_id") for m in members if m.get("paper_id"))),
        "evidence_snippets": (f.get("evidence_snippets") or [])[:2] if isinstance(f.get("evidence_snippets"), list) else [],
    }

def support_count(members: List[Dict[str, Any]]) -> int:
    return len({m.get("paper_id") for m in members if m.get("paper_id")})


# In-cluster LLM merge
def _merge_compact(f: Dict[str, Any]) -> Dict[str, Any]:
    """The part of a member field that the merge prompt sees (also the content hashed by merge_cache_key)."""
    return {
        "name": f.get("name"),
        "description": f.get("description"),
        "value_type": f.get("value_type"),
        "scope": f.get("scope"),
        "possible_aliases": (f.get("possible_aliases") or [])[:8]
        if isinstance(f.get("possible_aliases"), list) else [],
        "paper_id": f.get("paper_id"),
        "example_values": (f.get("example_values") or [])[:3]
        if isinstance(f.get("example_values"), list) else [],
        "evidence_snippets": (f.get("evidence_snippets") or [])[:2]
        if isinstance(f.get("evidence_snippets"), list) else [],
    }

def llm_merge_within_cluster(cluster_fields: List[Dict[str, Any]], domain: str) -> Dict[str, Any]:
    compact: List[Dict[str, Any]] = [_merge_compact(f) for f in cluster_fields]

    prompt = f"""
        You are merging schema field proposals for domain "{domain}".

        Input is a cluster of roughly similar field proposals.
        Your job:
            - Merge true aliases/synonyms.
            - Split over-broad fields if they represent different concepts.
            - Mark narrow quirks as scope="paper_specific".

        Return ONLY JSON:
            {{
            "canonicals": [
                {{
                "canonical_name": "snake_case",
                "canonical_description": "...",
                "value_type": "string|number|integer|boolean|enum|array|object",
                "scope": "global|subdomain|paper_specific",
                "aliases": ["..."],
                "member_field_names": ["..."],
                "supporting_paper_ids": ["..."],
                "evidence_snippets": ["<=40 words each"]
                }}
            ]
            }}

        Rules:
            - supporting_paper_ids MUST be the union of member paper_id values.
            - evidence_snippets must be verbatim from members.
    """.strip()

    raw = llm_chat(
        [
            {"role": "system", "content": "Return ONLY valid JSON. No markdown."},
            {"role": "user", "content": prompt + "\n\nField proposals:\n" + json.dumps(compact, ensure_ascii=True)},
        ],
        force_json=True,
    )
    return json.loads(raw)

_MERGE_CHUNK_THRESHOLD = 80   # clusters larger than this are merged in chunks
_MERGE_CHUNK_SIZE = 70        # maximum number of fields per merge chunk

def merge_cache_key_v1(members: List[Dict[str, Any]], domain: str) -> str:
    """Key used in the experiments up to 2026-10-11: set of (paper_id, field name) + domain + deployment.
    Kept for the audit (scripts/memo_audit.py); no longer used for lookups."""
    import hashlib
    pairs = sorted({(str(m.get("paper_id", "")), str(m.get("name", ""))) for m in members})
    return hashlib.sha256(json.dumps([domain, os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", ""), pairs]).encode()).hexdigest()

_MERGE_OP_VERSION = None

def merge_operator_version() -> str:
    """Hash of everything that turns a member set into the merge prompt (prompt code, member view, chunking)."""
    global _MERGE_OP_VERSION
    if _MERGE_OP_VERSION is None:
        import hashlib, inspect
        src = [inspect.getsource(llm_merge_within_cluster), inspect.getsource(_merge_compact),
               inspect.getsource(_chunked_llm_merge), _MERGE_CHUNK_THRESHOLD, _MERGE_CHUNK_SIZE]
        _MERGE_OP_VERSION = hashlib.sha256(json.dumps(src).encode()).hexdigest()
    return _MERGE_OP_VERSION

def merge_member_content(members: List[Dict[str, Any]]) -> List[str]:
    """Order-insensitive multiset of the members exactly as the merge prompt sees them."""
    return sorted(json.dumps(_merge_compact(m), ensure_ascii=True, sort_keys=True) for m in members)

def merge_cache_key(members: List[Dict[str, Any]], domain: str) -> str:
    """Content-addressed key: the full member content seen by the prompt + domain + operator version + deployment.
    A hit is only ever an earlier merge of exactly the same input under the same prompt and model."""
    import hashlib
    return hashlib.sha256(json.dumps([domain, os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", ""), merge_operator_version(),
                                      merge_member_content(members)]).encode()).hexdigest()

def _merge_cache_path(members: List[Dict[str, Any]], domain: str):
    """Cross-run merge memory (opt-in): MEMOSCHEMA_MERGE_CACHE_DIR = absolute directory."""
    root = os.getenv("MEMOSCHEMA_MERGE_CACHE_DIR", "")
    if not root:
        return None
    return Path(root) / f"{merge_cache_key(members, domain)}.json"

def _chunked_llm_merge(
    members: List[Dict[str, Any]],
    domain: str,
    cid: int,
) -> Dict[str, Any]:
    n = len(members)
    if n <= _MERGE_CHUNK_THRESHOLD:
        return llm_merge_within_cluster(members, domain=domain)

    num_chunks = (n + _MERGE_CHUNK_SIZE - 1) // _MERGE_CHUNK_SIZE
    print(
        f"[Merge] cluster {cid} size={n} > {_MERGE_CHUNK_THRESHOLD}, "
        f"splitting into {num_chunks} chunks of <= {_MERGE_CHUNK_SIZE}",
        flush=True,
    )

    all_canonicals: List[Dict[str, Any]] = []
    for ci in range(num_chunks):
        lo = ci * _MERGE_CHUNK_SIZE
        hi = min(lo + _MERGE_CHUNK_SIZE, n)
        chunk = members[lo:hi]
        print(
            f"[Merge] cluster {cid} chunk {ci + 1}/{num_chunks} size={len(chunk)}",
            flush=True,
        )
        try:
            sub = llm_merge_within_cluster(chunk, domain=domain)
        except Exception as e:
            # A failed chunk is skipped so that the rest of the cluster
            # is still merged.
            print(
                f"[Merge] cluster {cid} chunk {ci + 1} FAILED: "
                f"{type(e).__name__}: {e}",
                flush=True,
            )
            continue

        for c in (sub.get("canonicals") or []):
            all_canonicals.append(c)

    return {"canonicals": all_canonicals}

def _norm_name_for_alias(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (s or "").lower()).strip("_")

_GENERIC_TOKENS: set = {
    # role / structural
    "method", "methods", "type", "types", "name", "names", "format", "formats",
    "level", "levels", "status", "presence", "value", "values",
    "id", "identifier", "approach", "approaches", "strategy", "strategies",
    "way", "ways", "kind", "kinds", "category", "categories",
    "class", "classes", "group", "groups", "result", "results",
    "info", "information", "description", "label", "tag",
    "study", "studies", "field", "fields", "variable", "variables",
    "score", "metric", "measure", "data", "scope",
    # filler / function words
    "the", "of", "and", "or", "in", "to", "for", "with", "by", "on", "at",
    "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those", "a", "an", "as", "from", "into",
    "such", "any", "all", "each", "per", "via", "used", "use", "using",
}

def _content_tokens(text: str, domain_spec: Optional[DomainSpec] = None) -> List[str]:
    return [
        t for t in _tokenize(text, domain_spec=domain_spec)
        if t not in _GENERIC_TOKENS and len(t) >= 3
    ]

def _canon_core_tokens(c: Dict[str, Any], domain_spec: Optional[DomainSpec] = None) -> set:
    name = _snake_to_words(str(c.get("name", "") or ""))
    desc = _snake_to_words(str(c.get("description", "") or ""))
    return set(_content_tokens(name, domain_spec=domain_spec)) | \
           set(_content_tokens(desc, domain_spec=domain_spec))

def _canon_identity_text(c: Dict[str, Any], top_n_aliases: int = 5) -> str:
    parts: List[str] = [
        _snake_to_words(str(c.get("name", "") or "")),
        str(c.get("description", "") or "").strip(),
    ]
    for a in (c.get("aliases") or [])[:top_n_aliases]:
        parts.append(_snake_to_words(str(a)))
    return " ".join(p for p in parts if p).strip()

def _canons_share_content_tokens(
    a: Dict[str, Any],
    b: Dict[str, Any],
    domain_spec: Optional[DomainSpec] = None,
    min_overlap: int = 1,
) -> bool:
    ta = _canon_core_tokens(a, domain_spec=domain_spec)
    tb = _canon_core_tokens(b, domain_spec=domain_spec)
    if not ta or not tb:
        # No content tokens at all — refuse to bridge, since we have no signal.
        return False
    return len(ta & tb) >= min_overlap

def _alias_fits_canonical(
    alias: str,
    canonical: Dict[str, Any],
    domain_spec: Optional[DomainSpec] = None,
) -> bool:
    a_tokens = set(_content_tokens(_snake_to_words(alias), domain_spec=domain_spec))
    if not a_tokens:
        return True
    return bool(a_tokens & _canon_core_tokens(canonical, domain_spec=domain_spec))

def _merge_canonical_group(members: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not members:
        return {}
    if len(members) == 1:
        return members[0]
    scope_rank = {"subdomain": 2, "global": 1, "paper_specific": 0}
    rep = max(members, key=lambda c: int(c.get("support_count", 0)))
    spids: List[Any] = []
    evid: List[str] = []
    aliases: List[str] = []
    member_names: List[str] = []
    seen_p, seen_e, seen_a, seen_m = set(), set(), set(), set()
    for m in members:
        for pid in m.get("supporting_paper_ids", []) or []:
            if pid not in seen_p:
                seen_p.add(pid); spids.append(pid)
        for e in m.get("evidence_snippets", []) or []:
            if e not in seen_e:
                seen_e.add(e); evid.append(e)
        for a in m.get("aliases", []) or []:
            if a not in seen_a:
                seen_a.add(a); aliases.append(a)
        for mn in m.get("member_field_names", []) or []:
            if mn not in seen_m:
                seen_m.add(mn); member_names.append(mn)
    best_scope = max(
        (m.get("scope") or "global" for m in members),
        key=lambda s: scope_rank.get((s or "global").lower(), 1),
    )
    return {
        "name": rep.get("name"),
        "description": rep.get("description"),
        "value_type": rep.get("value_type"),
        "scope": best_scope,
        "aliases": aliases,
        "member_field_names": member_names,
        "supporting_paper_ids": spids,
        "support_count": len(spids),
        "evidence_snippets": evid[:8],
    }

def _consolidate_canonicals_by_alias(
    canonicals: List[Dict[str, Any]],
    max_aliases_per_bucket: int = 30,
    domain_spec: Optional[DomainSpec] = None,
) -> List[Dict[str, Any]]:
    if not canonicals:
        return canonicals
    n = len(canonicals)

    # Per-canonical normalized alias keys (name + aliases)
    canon_alias_keys: List[set] = []
    for c in canonicals:
        s: set = set()
        for nm in [c.get("name", "")] + list(c.get("aliases") or []):
            key = _norm_name_for_alias(str(nm))
            if key:
                s.add(key)
        canon_alias_keys.append(s)

    name_to_idx: Dict[str, List[int]] = {}
    for i, s in enumerate(canon_alias_keys):
        for k in s:
            name_to_idx.setdefault(k, []).append(i)

    dsu = DSU(n)
    # group_aliases[root] = union of normalized alias keys in that group
    group_aliases: Dict[int, set] = {i: set(canon_alias_keys[i]) for i in range(n)}

    # Sort keys for deterministic union order
    for _key in sorted(name_to_idx.keys()):
        idxs = name_to_idx[_key]
        if len(idxs) <= 1:
            continue
        first = idxs[0]
        for j in idxs[1:]:
            ra = dsu.find(first)
            rb = dsu.find(j)
            if ra == rb:
                continue
            # Anti-pollution gate: refuse merging unrelated concepts that
            # only share an incidental generic alias key.
            if not _canons_share_content_tokens(
                canonicals[first], canonicals[j], domain_spec=domain_spec,
            ):
                continue
            combined = group_aliases.get(ra, set()) | group_aliases.get(rb, set())
            # Refuse the union if it would create a super-bucket
            if len(combined) > max_aliases_per_bucket:
                continue
            dsu.union(first, j)
            new_root = dsu.find(first)
            group_aliases[new_root] = combined
            if new_root != ra and ra in group_aliases and ra != new_root:
                group_aliases.pop(ra, None)
            if new_root != rb and rb in group_aliases and rb != new_root:
                group_aliases.pop(rb, None)

    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(dsu.find(i), []).append(i)

    return [_merge_canonical_group([canonicals[i] for i in idxs])
            for _, idxs in groups.items()]

def _filter_polluted_aliases(
    canonicals: List[Dict[str, Any]],
    domain_spec: Optional[DomainSpec] = None,
) -> List[Dict[str, Any]]:
    if not canonicals:
        return canonicals
    cleaned: List[Dict[str, Any]] = []
    for c in canonicals:
        kept_aliases = [
            a for a in (c.get("aliases") or [])
            if _alias_fits_canonical(a, c, domain_spec=domain_spec)
        ]
        cc = dict(c)
        cc["aliases"] = kept_aliases
        cleaned.append(cc)
    return cleaned

def _merge_semantic_twins(
    canonicals: List[Dict[str, Any]],
    domain_spec: Optional[DomainSpec] = None,
    twin_threshold: float = 0.45,
    twin_threshold_with_evidence: float = 0.18,
) -> List[Dict[str, Any]]:
    if len(canonicals) <= 1:
        return canonicals
    n = len(canonicals)
    identity_texts = [_canon_identity_text(c) for c in canonicals]

    def _alias_key_set(c: Dict[str, Any]) -> set:
        out: set = set()
        for nm in [c.get("name", "")] + list(c.get("aliases") or []):
            k = _norm_name_for_alias(str(nm))
            if k:
                out.add(k)
        return out

    keysets = [_alias_key_set(c) for c in canonicals]
    paper_sets = [set(c.get("supporting_paper_ids") or []) for c in canonicals]
    evid_sets = [set(c.get("evidence_snippets") or []) for c in canonicals]
    value_types = [(c.get("value_type") or "").lower() for c in canonicals]

    dsu = DSU(n)
    for i in range(n):
        if not identity_texts[i]:
            continue
        for j in range(i + 1, n):
            if not identity_texts[j]:
                continue
            sim = _cosine_text_sim(
                identity_texts[i], identity_texts[j], domain_spec=domain_spec,
            )
            # Hard gate: value_types must match (or one is unspecified).
            # boolean / string / array carry semantic intent we don't override.
            vi, vj = value_types[i], value_types[j]
            if vi and vj and vi != vj:
                continue
            shared_evidence = bool(evid_sets[i] & evid_sets[j])
            shared_alias = bool(keysets[i] & keysets[j])
            shared_paper = bool(paper_sets[i] & paper_sets[j])
            if not (shared_evidence or shared_alias or shared_paper):
                continue
            # Verbatim shared evidence is a strong same-axis signal — accept
            # at a much lower cosine threshold. Otherwise require standard.
            threshold = twin_threshold_with_evidence if shared_evidence else twin_threshold
            if sim < threshold:
                continue
            dsu.union(i, j)

    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(dsu.find(i), []).append(i)
    return [_merge_canonical_group([canonicals[i] for i in idxs])
            for _, idxs in groups.items()]

def _canonicals_from_llm_merge(merged: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for c in merged.get("canonicals", []) or []:
        spids = list(dict.fromkeys(c.get("supporting_paper_ids", []) or []))
        out.append({
            "name": c.get("canonical_name"),
            "description": c.get("canonical_description"),
            "value_type": c.get("value_type"),
            "scope": c.get("scope"),
            "aliases": c.get("aliases", []) or [],
            "member_field_names": c.get("member_field_names", []) or [],
            "supporting_paper_ids": spids,
            "support_count": len(spids),
            "evidence_snippets": c.get("evidence_snippets", []) or [],
        })
    return out

def consolidate_canonicals(
    canonical_candidates: List[Dict[str, Any]],
    domain_spec: Optional[DomainSpec] = None,
    fix_cluster: bool = False,
) -> List[Dict[str, Any]]:
    # Four-pass anti-pollution / anti-redundancy consolidation.
    # 1) Alias-based union with pollution gate (refuses incidental shared-alias
    #    bridges between unrelated canonicals).
    canonical_candidates = _consolidate_canonicals_by_alias(
        canonical_candidates, domain_spec=domain_spec,
    )
    # 2) Pre-filter: drop within-canonical aliases that don't share content
    #    tokens with the canonical's name+description. Reduces noise in the
    #    identity texts that step 3 compares.
    canonical_candidates = _filter_polluted_aliases(
        canonical_candidates, domain_spec=domain_spec,
    )
    # 3) Force-merge near-duplicate canonicals that didn't share an alias key
    #    (e.g. feedback_type <-> reward_model_training_signal).
    #    Skipped with average-linkage clustering: its union-find over "shares a paper
    #    and cosine >= 0.45" chains coherent concepts back into mega-canonicals.
    if not fix_cluster:
        canonical_candidates = _merge_semantic_twins(
            canonical_candidates, domain_spec=domain_spec,
        )
    # 4) Post-filter: re-run alias filter against the FINAL merged canonical's name+description. 
    canonical_candidates = _filter_polluted_aliases(
        canonical_candidates, domain_spec=domain_spec,
    )
    return canonical_candidates

def build_canonical_candidates(
    paper_level_jsonl: Path,
    out_dir: Path,
    domain: str,
    embed_threshold: float = 0.65,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows = load_jsonl(paper_level_jsonl)
    domain_spec = get_domain_spec(domain)

    all_fields: List[Dict[str, Any]] = []
    for r in rows:
        pid = r.get("paper_id")
        sfile = r.get("source_file")
        for f in (r.get("fields") or []):
            if not isinstance(f, dict):
                continue
            ff = dict(f)
            ff["paper_id"] = pid
            ff["source_file"] = sfile
            all_fields.append(ff)

    fix_cluster = os.getenv("MEMOSCHEMA_FIX_CLUSTER") == "1"
    if fix_cluster:
        clusters = cluster_by_average_linkage(all_fields)
    else:
        clusters = cluster_by_embedding(all_fields, threshold=embed_threshold)

    merged_clusters: List[Dict[str, Any]] = []
    canonical_candidates: List[Dict[str, Any]] = []

    total = len(clusters)
    start = time.time()

    premerged: Dict[int, Dict[str, Any]] = {}
    if fix_cluster:
        from concurrent.futures import ThreadPoolExecutor

        def _merge_one(cid_idxs):
            cid, idxs = cid_idxs
            members = [all_fields[i] for i in idxs]
            if support_count(members) < 2:
                return cid, {"canonicals": [_unmerged_canonical(members)]}
            cache = _merge_cache_path(members, domain)
            if cache is not None and cache.exists():
                return cid, json.loads(cache.read_text(encoding="utf-8"))
            for attempt in range(4):
                try:
                    out = _chunked_llm_merge(members, domain=domain, cid=cid)
                    if cache is not None:
                        cache.parent.mkdir(parents=True, exist_ok=True)
                        tmp = cache.with_suffix(f".tmp{os.getpid()}_{cid}")
                        tmp.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8"); os.replace(tmp, cache)
                    return cid, out
                except Exception as e:
                    print(f"[Merge] cluster {cid} attempt {attempt + 1} failed: {type(e).__name__}: {e}", flush=True)
                    time.sleep(5 * 2 ** attempt)
            return cid, {"canonicals": []}

        with ThreadPoolExecutor(max_workers=_MERGE_WORKERS) as ex:
            premerged = dict(ex.map(_merge_one, enumerate(clusters, start=1)))

    for cid, idxs in enumerate(clusters, start=1):
        members = [all_fields[i] for i in idxs]
        if fix_cluster:
            merged = premerged[cid]
        else:
            print(f"[Merge] cluster {cid}/{total} size={len(idxs)}", flush=True)
            merged = _chunked_llm_merge(members, domain=domain, cid=cid)

        merged_clusters.append({
            "cluster_id": cid - 1,
            "support_count": support_count(members),
            "members": members,
            "llm_merged": merged,
        })

        canonical_candidates.extend(_canonicals_from_llm_merge(merged))

        if cid % 10 == 0 or cid == total:
            dt = time.time() - start
            print(f"[Merge] progress {cid}/{total} elapsed={dt:.1f}s", flush=True)

    canonical_candidates = consolidate_canonicals(
        canonical_candidates, domain_spec=domain_spec, fix_cluster=fix_cluster,
    )

    save_json(out_dir / "merged_field_clusters.json", merged_clusters)
    save_json(out_dir / "canonical_schema_candidates.json", canonical_candidates)

    return canonical_candidates, {
        "total_raw_fields": len(all_fields),
        "total_coarse_clusters": len(clusters),
    }

# Anchor-driven scoring (anchors come from corpus, not from human registry)
def _build_anchor_target_blob(anchors: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for a in anchors:
        parts.append(str(a.get("name", "")))
        parts.append(str(a.get("description", "")))
        parts.extend(a.get("representative_keywords", []) or [])
        parts.extend(a.get("representative_variable_names_seen", []) or [])
    return " ".join(p for p in parts if p)

def _anchor_match_score(c: Dict[str, Any], anchor: Dict[str, Any], domain_spec: DomainSpec) -> float:
    name = str(c.get("name", "") or "").lower()
    blob = _candidate_full_text(c).lower()

    seen_names = [str(s).lower() for s in (anchor.get("representative_variable_names_seen") or [])]
    anchor_name = str(anchor.get("name", "") or "").lower()

    score = 0.0
    if name and name in seen_names:
        score += 3.0
    elif name and anchor_name and (anchor_name in name or name in anchor_name):
        score += 1.5
    elif any(name and (sn in name or name in sn) for sn in seen_names):
        score += 1.0

    kws = [str(k).lower() for k in (anchor.get("representative_keywords") or [])]
    kw_hits = sum(1 for k in kws if k and k in blob)
    score += min(kw_hits, 4) * 0.5

    return score

def _subdomain_relevance_score(
    c: Dict[str, Any],
    *,
    last_tier: Optional[str],
    anchors: List[Dict[str, Any]],
    domain_spec: DomainSpec,
) -> float:
    if not anchors and not last_tier:
        return 0.0

    blob = _candidate_full_text(c)
    target_parts: List[str] = []
    if last_tier:
        target_parts.append(last_tier)
    target_parts.append(_build_anchor_target_blob(anchors))
    target = " ".join(p for p in target_parts if p)

    score = _cosine_text_sim(blob, target, domain_spec=domain_spec)

    # Anchor-keyword direct hits, weighted by anchor centrality
    blob_l = blob.lower()
    anchor_kw_signal = 0.0
    for a in anchors:
        cent = float(a.get("centrality_to_subdomain", 0.5) or 0.5)
        kws = [str(k).lower() for k in (a.get("representative_keywords") or [])]
        hits = sum(1 for k in kws if k and k in blob_l)
        if hits:
            anchor_kw_signal += cent * min(hits, 4) / 4.0
    score += 0.4 * min(anchor_kw_signal, 1.5) / 1.5

    if (c.get("scope") or "").lower() == "subdomain":
        score += 0.15

    return score

# MEMOSCHEMA_FIX_ANCHOR_SCHEMA=1 (organized drafting): anchors (corpus-grounded core concepts) become the draft
# fields: each candidate is assigned to its most similar anchor (sim >= _ANCHOR_SIM_MIN); an anchor with no grounded
# support is dropped; the field carries the anchor name/description and the union of its candidates' papers, aliases
# and evidence. Remaining slots are filled by rank with candidates not assigned to a chosen anchor.
# Unset (LEGACY): keyword anchor matching, one slot per anchor, then fill by rank.
_ANCHOR_SIM_MIN = 0.60

def _identity_vecs(items: List[Dict[str, Any]], anchor: bool = False) -> np.ndarray:
    texts = []
    for c in items:
        t = f"{_snake_to_words(str(c.get('name', '')))}: {c.get('description', '') or ''}"
        if anchor:
            t += " " + ", ".join(c.get("representative_keywords") or [])
        texts.append(t)
    if not texts:
        return np.zeros((0, 1), dtype=np.float32)
    X = np.asarray(embed_texts(texts), dtype=np.float32)
    return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-9, None)

def _merge_selected(members: List[Dict[str, Any]], name: str, description: str) -> Dict[str, Any]:
    pids: List[Any] = []
    evid: List[str] = []
    aliases: List[str] = []
    member_names: List[str] = []
    for m in members:
        for pid in m.get("supporting_paper_ids") or []:
            if pid not in pids:
                pids.append(pid)
        for e in m.get("evidence_snippets") or []:
            if e not in evid:
                evid.append(e)
        for a in [m.get("name")] + list(m.get("aliases") or []):
            if a and a != name and a not in aliases:
                aliases.append(a)
        for mn in m.get("member_field_names") or []:
            if mn not in member_names:
                member_names.append(mn)
    rep = max(members, key=lambda c: int(c.get("support_count", 0)))
    return {
        "name": name or rep.get("name"),
        "description": description or rep.get("description"),
        "value_type": rep.get("value_type"),
        "scope": rep.get("scope"),
        "aliases": aliases[:30],
        "member_field_names": member_names,
        "supporting_paper_ids": pids,
        "support_count": len(pids) or int(rep.get("support_count", 0)),
        "evidence_snippets": evid[:8],
        "merged_from": [m.get("name") for m in members],
    }

def _anchor_schema(
    kept: List[Dict[str, Any]], anchors: List[Dict[str, Any]], k: int, min_support: int, rank_fn,
) -> List[Dict[str, Any]]:
    C = _identity_vecs(kept)
    A = _identity_vecs(anchors, anchor=True)
    S = C @ A.T                                   # candidates x anchors
    best = S.argmax(axis=1)
    members: Dict[int, List[int]] = {}
    for i in range(len(kept)):
        if S[i, best[i]] >= _ANCHOR_SIM_MIN:
            members.setdefault(int(best[i]), []).append(i)
    fields: List[Tuple[float, Dict[str, Any]]] = []
    for ai, a in enumerate(anchors):
        idxs = members.get(ai, [])
        if not idxs:
            continue
        ms = [kept[i] for i in idxs]
        f = _merge_selected(ms, str(a.get("name") or ""), str(a.get("description") or ""))
        for v in a.get("representative_variable_names_seen") or []:
            if v not in f["aliases"] and v != f["name"]:
                f["aliases"].append(v)
        f["aliases"] = f["aliases"][:30]
        f["scope"] = "subdomain"
        f["_anchor"] = a.get("name")
        if int(f.get("support_count", 0)) < min_support:
            continue
        fields.append((max(rank_fn(m) for m in ms), f))
    fields.sort(key=lambda t: t[0], reverse=True)
    final = [f for _, f in fields[:k]]
    covered = {i for ai, idxs in members.items() for i in idxs
               if any(f.get("_anchor") == anchors[ai].get("name") for f in final)}
    for i, c in enumerate(kept):
        if len(final) >= k:
            break
        if i not in covered:
            final.append(c)
    return final

# Final selection — corpus-anchored, no hand-coded categories or blacklists
def select_final_schema(
    canonical_candidates: List[Dict[str, Any]],
    out_dir: Path,
    domain: str,
    last_tier: Optional[str] = None,
    max_final_fields: int = 8,
    min_support: int = 2,
    anchors: Optional[List[Dict[str, Any]]] = None,
    anchors_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    domain_spec = get_domain_spec(domain)

    # Prefer in-memory anchors; fall back to disk if a path was given
    if anchors is None:
        if anchors_path is None:
            anchors_path = out_dir / "discovered_anchors.json"
        anchors = load_anchors(anchors_path)

    # 1) drop paper_specific; 2) support filter (same min_support for subdomain-tagged candidates)
    kept: List[Dict[str, Any]] = [
        c for c in canonical_candidates
        if (c.get("scope") or "global") != "paper_specific" and int(c.get("support_count", 0)) >= int(min_support)
    ]

    # 3) rank score
    # Corpus-adaptive support normaliser: log-scale so that a field with 2x
    # the support of another does not score 2x higher at the top end.
    # Floor at 20 prevents over-amplification in very small corpora.
    _max_sup = max((int(c.get("support_count", 0)) for c in kept), default=20)
    _max_sup = max(_max_sup, 20)
    _log_max = math.log1p(_max_sup)

    def _final_rank(x: Dict[str, Any]) -> float:
        # Log-normalised, corpus-adaptive support score in [0, 1.5].
        support = min(math.log1p(int(x.get("support_count", 0))) / _log_max, 1.5)
        evidence = min(len(x.get("evidence_snippets", []) or []), 6) / 6.0
        subrel = _subdomain_relevance_score(
            x, last_tier=last_tier, anchors=anchors or [], domain_spec=domain_spec,
        )
        scope_bonus = 0.25 if (x.get("scope") or "").lower() == "subdomain" else 0.0
        return 1.0 * support + 0.5 * evidence + 2.0 * subrel + scope_bonus

    kept.sort(key=_final_rank, reverse=True)

    if os.getenv("MEMOSCHEMA_FIX_ANCHOR_SCHEMA") == "1" and anchors and kept:
        final = _anchor_schema(kept, anchors, max_final_fields, min_support, _final_rank)
        save_json(out_dir / "final_schema.json", final)
        return final

    final: List[Dict[str, Any]] = []
    chosen: set = set()

    def _take(i: int, anchor_name: Optional[str] = None) -> None:
        c = kept[i]
        if anchor_name is not None:
            c = dict(c)
            c["_anchor"] = anchor_name
        final.append(c)
        chosen.add(str(kept[i].get("name", "") or ""))

    # 4) LEGACY anchor coverage — one slot per discovered anchor when a candidate matches
    if anchors:
        for anchor in anchors:
            if len(final) >= max_final_fields:
                break
            best = None
            best_score: Optional[Tuple[float, float]] = None
            for i, c in enumerate(kept):
                name = str(c.get("name", "") or "")
                if not name or name in chosen:
                    continue
                a_score = _anchor_match_score(c, anchor, domain_spec)
                if a_score <= 0:
                    continue
                tup = (a_score, _final_rank(c))
                if best is None or tup > best_score:
                    best = i
                    best_score = tup
            if best is not None:
                _take(best, anchor.get("name"))

    # 5) Fill remaining slots by raw rank
    for i, c in enumerate(kept):
        if len(final) >= max_final_fields:
            break
        name = str(c.get("name", "") or "")
        if not name or name in chosen:
            continue
        _take(i)

    final.sort(key=_final_rank, reverse=True)
    save_json(out_dir / "final_schema.json", final)
    return final
