from __future__ import annotations
import json
from typing import Any, Dict, List, Optional
from .domain_registry import get_domain_spec

_GENERIC_DISTILL_LABELS = [
    "variable",
    "method",
    "result",
    "context",
    "evidence",
    "other",
]

def build_distill_prompt(
    domain: str,
    paper_id: str,
    text: str,
    quote_limit: int = 10,
    last_tier: Optional[str] = None,
) -> List[Dict[str, str]]:
    spec = get_domain_spec(domain)
    last_tier = (last_tier or spec.default_last_tier).strip()

    label_options = "|".join(_GENERIC_DISTILL_LABELS)

    user = f"""
        You are reading ONE paper for the purpose of schema induction.

        Domain: {domain} ({spec.task_noun})
        Subdomain focus: {last_tier}

        A schema-relevant variable is something that:
          - is reported across many papers about "{last_tier}", not unique to this one
          - has a measurable, categorical, or otherwise structured value
          - is grounded in a verbatim quote from the paper text below

        Your job is to distill ONLY schema-relevant variables and their supporting evidence.
        Do NOT summarize the paper. Do NOT add concepts that are not in the paper.

        Use your own subject-matter knowledge of "{last_tier}" research to decide what
        counts as a recurring variable in this subdomain — but only emit variables for
        which you can cite a verbatim quote from this paper.

        Return ONLY valid JSON:
            {{
            "paper_id": "{paper_id}",
            "key_points": [
                {{
                "label": "{label_options}",
                "variable_name": "short reusable snake_case field name",
                "notes": "1-2 sentences. Why is this likely reusable across {last_tier} papers?",
                "quotes": ["verbatim quote <=40 words", "..."]
                }}
            ]
            }}

        Rules:
            - Quotes MUST be copied verbatim from the provided text.
            - Each quote <= 40 words.
            - Up to {quote_limit} total quotes across all key_points.
            - Do NOT include bibliographic metadata (title/authors/doi).
            - Do NOT propose fields useful only for this single paper.
            - If nothing schema-relevant is found, return an empty key_points list.

        Paper text:
            {text}
    """.strip()

    return [
        {"role": "system", "content": "Return ONLY valid JSON. No markdown, no commentary."},
        {"role": "user", "content": user},
    ]

def distill_paper_for_schema(
    *,
    llm_chat_fn,
    domain: str,
    paper_id: str,
    full_text: str,
    input_chars: int = 60000,
    quote_limit: int = 10,
    last_tier: Optional[str] = None,
) -> Dict[str, Any]:
    text = (full_text or "").strip()[:input_chars]

    messages = build_distill_prompt(
        domain=domain,
        paper_id=paper_id,
        text=text,
        quote_limit=quote_limit,
        last_tier=last_tier,
    )
    raw = llm_chat_fn(messages, force_json=True)

    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            return {"paper_id": paper_id, "key_points": [], "_raw": raw}
        data["_raw"] = raw
        return data
    except Exception:
        return {"paper_id": paper_id, "key_points": [], "_raw": raw}

def render_distill_to_text(distill: Dict[str, Any], max_chars: int = 12000) -> str:
    paper_id = distill.get("paper_id", "")
    kps = distill.get("key_points") or []
    lines: List[str] = [f"[PAPER_ID] {paper_id}", "[DISTILLED_SCHEMA_RELEVANT_POINTS]"]

    total = 0
    for kp in kps:
        if not isinstance(kp, dict):
            continue
        label = kp.get("label", "other")
        var = kp.get("variable_name", "")
        notes = kp.get("notes", "")
        quotes = kp.get("quotes") or []
        if not isinstance(quotes, list):
            quotes = []

        block = [
            f"- label: {label}",
            f"  variable_name: {var}",
            f"  notes: {notes}",
        ]
        for q in quotes:
            block.append(f'  quote: "{q}"')

        chunk = "\n".join(block).strip() + "\n"
        if total + len(chunk) > max_chars:
            break
        lines.append(chunk)
        total += len(chunk)

    out = "\n".join(lines).strip()
    return out[:max_chars]
