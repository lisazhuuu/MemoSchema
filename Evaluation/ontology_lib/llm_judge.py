from __future__ import annotations
import json, os, re, time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)
from ontology_lib.config import (
    DEFAULT_MODEL_DEPLOYMENT,
    LLM_RETRIES,
    LLM_RETRY_BACKOFF_SEC,
    LLM_TEMPERATURE,
    MATCH_LABELS,
    MAX_ALIASES_IN_PROMPT,
    MAX_COMPLETION_TOKENS,
    MAX_MEMBER_FIELDS_IN_PROMPT,
)
from ontology_lib.schema_io import SchemaField

# Azure client
def build_azure_client():
    """Judge client. Default: Azure (the project's judge). JUDGE_PROVIDER=gemini switches to Google's
    OpenAI-compatible endpoint with GEMINI_API_KEY (cross-family judge robustness check) with the same prompts."""
    if os.getenv("JUDGE_PROVIDER", "").lower() == "gemini":
        from openai import OpenAI
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("JUDGE_PROVIDER=gemini but GEMINI_API_KEY is not set in .env")
        return OpenAI(api_key=key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
    from openai import AzureOpenAI
    endpoint = os.getenv("AZURE_OPENAI_CHAT_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_CHAT_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_CHAT_API_VERSION", "2024-12-01-preview")
    if not endpoint or not api_key:
        raise RuntimeError(
            "Missing AZURE_OPENAI_CHAT_ENDPOINT or AZURE_OPENAI_CHAT_API_KEY env var."
        )
    return AzureOpenAI(
        azure_endpoint=endpoint, api_key=api_key, api_version=api_version
    )

def get_model_deployment() -> str:
    if os.getenv("JUDGE_PROVIDER", "").lower() == "gemini":
        model = os.getenv("GEMINI_JUDGE_MODEL", "")
        if not model:
            raise RuntimeError("Set GEMINI_JUDGE_MODEL (e.g. a Gemini Pro model id) for JUDGE_PROVIDER=gemini")
        return model
    return os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", DEFAULT_MODEL_DEPLOYMENT)

# Prompt
_SYSTEM_PROMPT = (
    "You are a careful domain-expert evaluator. You compare a hand-curated "
    "reference ontology against an induced schema produced by an automated "
    "system, and decide which reference concepts the induced schema captured. "
    "You think in terms of *concepts*, not surface words — a different name "
    "with the same meaning still counts."
)

_INSTRUCTIONS = """[TASK]
For each REFERENCE CONCEPT below, identify the SINGLE best-matching field
in the INDUCED SCHEMA (or "none" if no induced field captures it).
Use the induced field's description and aliases to understand what concept
it represents — do not rely on surface name matching alone.

Label each pair:
  - "strong":   the induced field clearly captures the same concept as the
                reference, even if the wording differs. Roughly: an expert
                would treat them as equivalent.
  - "partial":  the induced field captures a closely related but not
                identical concept — a more specific or more general version,
                or a different aspect of the same underlying thing.
  - "none":     no induced field captures this reference concept.

[CONSTRAINTS]
1. Each induced field can be the match for AT MOST ONE reference concept
   (1-to-1 alignment). If two references both want the same induced field,
   assign it to the closer one and mark the other as "none" (or find it a
   different match).
2. Prefer concept-level alignment over surface lexical matching. If two
   names share a word but mean different things, label them "none" /
   "partial" — not "strong".
3. Be honest with "none". Better to mark it as missing than to force a
   bad match.

[OUTPUT FORMAT — return a single JSON object, nothing else]
The key for each reference entry MUST be the bare concept name exactly as
listed below (e.g. ``"abeta_species"``), NOT the ``R1``/``R2`` index label.
The ``matched_induced_field`` value MUST be the bare induced field name
exactly as listed below (e.g. ``"amyloid_beta_species"``), NOT the
``I1``/``I2`` index label.

{
  "alignment": {
    "<reference_concept_name>": {
      "matched_induced_field": "<induced_field_name>" or null,
      "match_strength": "strong" | "partial" | "none",
      "reasoning": "<one short sentence explaining the call>"
    },
    ...   (one entry per reference concept, exactly N entries)
  }
}
"""

def _render_induced_field(idx: int, f: SchemaField) -> str:
    lines = [f"  I{idx}. {f.name}"]
    if f.description:
        lines.append(f"      description: {f.description}")
    if f.value_type:
        lines.append(f"      value_type:  {f.value_type}")
    if f.aliases:
        shown = list(dict.fromkeys(f.aliases))[:MAX_ALIASES_IN_PROMPT]
        lines.append(f"      aliases (up to {MAX_ALIASES_IN_PROMPT}): " + ", ".join(shown))
    if f.member_field_names:
        shown_m = list(dict.fromkeys(f.member_field_names))[:MAX_MEMBER_FIELDS_IN_PROMPT]
        lines.append(f"      member fields: " + ", ".join(shown_m))
    return "\n".join(lines)

def build_alignment_prompt(
    subdomain: str,
    references: List[str],
    induced: List[SchemaField],
) -> str:
    parts: List[str] = []
    parts.append(_INSTRUCTIONS.strip())
    parts.append("")
    parts.append(f"SUBDOMAIN: {subdomain}")
    parts.append("")
    parts.append(f"REFERENCE CONCEPTS ({len(references)}):")
    for i, r in enumerate(references, 1):
        parts.append(f"  R{i}. {r}")
    parts.append("")
    parts.append(f"INDUCED SCHEMA FIELDS ({len(induced)}):")
    for j, f in enumerate(induced, 1):
        parts.append(_render_induced_field(j, f))
        parts.append("")
    return "\n".join(parts)

# Response parsing / sanitization
_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)

def _robust_json_parse(raw: str) -> Optional[Dict]:
    if not raw:
        return None
    text = _CODE_FENCE_RE.sub("", raw).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    candidate = re.sub(r",(\s*[}\]])", r"\1", m.group(0))
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None

def _coerce_label(raw_label: str) -> str:
    if not raw_label:
        return "none"
    s = raw_label.strip().lower()
    if s in MATCH_LABELS:
        return s
    if "strong" in s:
        return "strong"
    if "partial" in s or "weak" in s:
        return "partial"
    return "none"

_INDEX_PREFIX_RE = re.compile(r"^[RI]\d+[.\s:)\-]*", re.IGNORECASE)

def _strip_index_prefix(label: str) -> str:
    """Remove leading ``R1.``, ``I3:``, ``R12 - `` etc. so we can match
    by bare concept name even when the LLM kept the index label."""
    return _INDEX_PREFIX_RE.sub("", label.strip()).strip()

def _build_lookup(names):
    out = {}
    for n in names:
        if not n:
            continue
        out.setdefault(n, n)
        out.setdefault(n.lower(), n)
        squashed = re.sub(r"[^a-z0-9]", "", n.lower())
        if squashed:
            out.setdefault(squashed, n)
    return out

def _resolve_name(raw: str, lookup) -> Optional[str]:
    if not raw:
        return None
    candidates = []
    s = raw.strip()
    candidates.append(s)
    s2 = _strip_index_prefix(s)
    if s2 != s:
        candidates.append(s2)
    for c in list(candidates):
        candidates.append(c.lower())
        candidates.append(re.sub(r"[^a-z0-9]", "", c.lower()))
    for c in candidates:
        if c in lookup:
            return lookup[c]
    return None

# One alignment call
@dataclass
class AlignmentRow:
    ref_name: str
    matched_induced: Optional[str]   # None means "no match"
    strength: str                    # one of MATCH_LABELS
    reasoning: str

def _enforce_one_to_one(rows: List[AlignmentRow]) -> List[AlignmentRow]:
    rank = {"strong": 2, "partial": 1, "none": 0}
    best_for_field: Dict[str, int] = {}    # induced_name -> winning row index
    for i, r in enumerate(rows):
        if not r.matched_induced or r.strength == "none":
            continue
        prev = best_for_field.get(r.matched_induced)
        if prev is None or rank[r.strength] > rank[rows[prev].strength]:
            best_for_field[r.matched_induced] = i
    keep_ok = set(best_for_field.values())
    fixed: List[AlignmentRow] = []
    for i, r in enumerate(rows):
        if r.matched_induced and r.strength != "none" and i not in keep_ok:
            # downgrade: same induced field already taken by a stronger pair
            fixed.append(
                AlignmentRow(
                    ref_name=r.ref_name,
                    matched_induced=None,
                    strength="none",
                    reasoning=(r.reasoning + " | downgraded: induced field already assigned").strip(" |"),
                )
            )
        else:
            fixed.append(r)
    return fixed

class JudgeCallFailed(RuntimeError):
    """Every attempt of the judge call failed (API error, unparsable or malformed response)."""

def call_alignment(
    client,
    model: str,
    subdomain: str,
    references: List[str],
    induced: List[SchemaField],
) -> List[AlignmentRow]:
    """Ask the judge LLM to align reference concepts to induced fields. Raises JudgeCallFailed when every attempt fails."""
    prompt = build_alignment_prompt(subdomain, references, induced)
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    ref_lookup = _build_lookup(references)
    ind_lookup = _build_lookup([f.name for f in induced])

    last_err: Optional[str] = None
    for attempt in range(1, LLM_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=MAX_COMPLETION_TOKENS,
                temperature=LLM_TEMPERATURE,
                response_format={"type": "json_object"},
            )
            raw = (resp.choices[0].message.content or "").strip()
            parsed = _robust_json_parse(raw)
            if parsed is None:
                last_err = "json_parse_failed"
                print(f"   WARNING: attempt {attempt}: JSON parse failed for {subdomain}")
                print(f"      raw (first 500 chars): {raw[:500]!r}")
                continue

            alignment_raw = parsed.get("alignment", {}) or {}
            if not isinstance(alignment_raw, dict):
                last_err = "alignment_not_dict"
                print(f"   WARNING: attempt {attempt}: 'alignment' is not a dict ({type(alignment_raw).__name__})")
                continue

            # Re-key the LLM's alignment dict by resolving each key to a
            # real reference name (handles "R1. abeta_species" / case diffs).
            resolved: Dict[str, Dict] = {}
            unresolved_keys: List[str] = []
            for k, v in alignment_raw.items():
                canonical = _resolve_name(str(k), ref_lookup)
                if canonical:
                    resolved[canonical] = v if isinstance(v, dict) else {}
                else:
                    unresolved_keys.append(str(k))
            if unresolved_keys:
                print(f"   WARNING: could not resolve {len(unresolved_keys)} key(s) to a reference: "
                      f"{unresolved_keys[:5]}")

            rows: List[AlignmentRow] = []
            for ref in references:
                entry = resolved.get(ref, {})
                raw_matched = entry.get("matched_induced_field")
                matched = _resolve_name(str(raw_matched), ind_lookup) if raw_matched else None
                strength = _coerce_label(str(entry.get("match_strength", "")))
                if not matched:
                    strength = "none"
                rows.append(
                    AlignmentRow(
                        ref_name=ref,
                        matched_induced=matched,
                        strength=strength,
                        reasoning=str(entry.get("reasoning", "")).strip(),
                    )
                )
            # Warn when all references are unmatched; check for parsing errors.
            n_matched = sum(1 for r in rows if r.matched_induced)
            if n_matched == 0 and references:
                print(f"   WARNING: parser produced 0 matches — printing raw response (first 1500 chars) for debugging:")
                print(f"      {raw[:1500]!r}")
            return _enforce_one_to_one(rows)

        except Exception as e:
            last_err = repr(e)
            print(f"   WARNING: attempt {attempt} failed for {subdomain}: {e}")
            time.sleep(LLM_RETRY_BACKOFF_SEC * attempt)

    print(f"   ERROR: judge giving up on {subdomain} (last: {last_err})")
    # A failed judge call is not a score: raise instead of returning an all-unmatched alignment.
    raise JudgeCallFailed(f"judge call failed for {subdomain} after {LLM_RETRIES} attempts (last: {last_err})")
