import re

# Subdomain anchoring checks for validate_tiers().
_TIERS_STOPWORDS = {
    "the", "for", "and", "with", "from", "into", "via", "via",
    "that", "this", "these", "those",
}

_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]+")

def _normalize(s: str) -> str:
    if not s:
        return ""
    return _NON_ALNUM_RE.sub(" ", s.lower())

def _tier_keyword_tokens(tier_label: str) -> set:
    norm = _normalize(tier_label)
    base = {t for t in norm.split() if len(t) >= 4 and t not in _TIERS_STOPWORDS}
    expanded = set(base)
    for t in base:
        if t.endswith("s") and len(t) >= 4:
            expanded.add(t[:-1])
    return expanded

def _query_text_for_validation(item: dict, qb_final: str) -> str:
    return " ".join([
        qb_final or "",
        item.get("query_nl") or "",
        item.get("query_bool") or "",
        " ".join(item.get("synonyms") or []),
    ])

def _contains_any_token(text: str, tokens: set) -> bool:
    if not tokens:
        return True  # nothing to require
    norm_text = _normalize(text)
    return any(tok in norm_text for tok in tokens)

# Query validation: required tiers, block terms and (optional) tier-anchor keywords
def validate_tiers(item, qb_final: str, DOMAIN: str, SPEC=None, strict_anchor: bool = False) -> bool:
    tiers = item.get("tiers") or []

    required_tiers = getattr(SPEC, "tiers", None) if SPEC is not None else None

    # If SPEC defines required tiers, enforce exact match
    if required_tiers:
        norm_req   = [t.strip().lower() for t in required_tiers]
        norm_tiers = [t.strip().lower() for t in tiers] if tiers else []
        if norm_tiers != norm_req:
            return False

    # If SPEC defines block terms, check if any appear in the query or synonyms
    block_terms = getattr(SPEC, "block_terms", None) if SPEC is not None else None
    if block_terms:
        text = " ".join([
            qb_final or "",
            item.get("query_nl") or "",
            " ".join(item.get("synonyms") or []),
        ]).lower()
        for term in block_terms:
            if term.lower() in text:
                return False

    # Tier-anchor check, only with strict_anchor=True (queries_generation.py --strict_tier_anchor): the query text
    # must contain a keyword of the last tier and, if there are two or more tiers, of the second-to-last tier
    if strict_anchor and required_tiers and len(required_tiers) >= 1:
        text = _query_text_for_validation(item, qb_final)
        last_keywords = _tier_keyword_tokens(required_tiers[-1])
        if last_keywords and not _contains_any_token(text, last_keywords):
            return False
        if len(required_tiers) >= 2:
            penult_keywords = _tier_keyword_tokens(required_tiers[-2])
            if penult_keywords and not _contains_any_token(text, penult_keywords):
                return False

    return True