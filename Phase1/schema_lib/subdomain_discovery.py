from __future__ import annotations
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional
from .domain_registry import get_domain_spec

# Corpus-driven subdomain anchor discovery.
def _aggregate_proposals(
    paper_level_jsonl: Path,
    max_field_groups: int = 200,
    max_chars: int = 30000,
) -> str:
    if not paper_level_jsonl.exists():
        return ""

    grouped: Dict[str, Dict[str, Any]] = {}
    with open(paper_level_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            pid = row.get("paper_id", "")
            for fld in row.get("fields", []) or []:
                if not isinstance(fld, dict):
                    continue
                name = str(fld.get("name", "") or "").strip().lower()
                if not name:
                    continue
                g = grouped.setdefault(
                    name,
                    {
                        "name": name,
                        "papers": set(),
                        "descriptions": [],
                        "evidence": [],
                        "aliases": Counter(),
                        "scope": Counter(),
                    },
                )
                if pid:
                    g["papers"].add(pid)
                desc = str(fld.get("description", "") or "").strip()
                if desc and len(g["descriptions"]) < 3:
                    g["descriptions"].append(desc)
                evi = fld.get("evidence_snippets") or []
                if isinstance(evi, list) and evi and len(g["evidence"]) < 2:
                    g["evidence"].append(str(evi[0]))
                for a in fld.get("possible_aliases", []) or []:
                    g["aliases"][str(a).strip().lower()] += 1
                sc = str(fld.get("scope", "") or "").strip().lower()
                if sc:
                    g["scope"][sc] += 1

    ranked = sorted(grouped.values(), key=lambda g: len(g["papers"]), reverse=True)
    ranked = ranked[:max_field_groups]

    out_lines: List[str] = []
    used = 0
    for g in ranked:
        n_papers = len(g["papers"])
        block_lines = [
            f"- name: {g['name']}",
            f"  paper_count: {n_papers}",
        ]
        if g["descriptions"]:
            block_lines.append(f"  description: {g['descriptions'][0]}")
        if g["evidence"]:
            block_lines.append(f"  example_quote: \"{g['evidence'][0][:200]}\"")
        if g["aliases"]:
            top_aliases = ", ".join(a for a, _ in g["aliases"].most_common(5))
            block_lines.append(f"  aliases_seen: {top_aliases}")
        if g["scope"]:
            top_scope = g["scope"].most_common(1)[0][0]
            block_lines.append(f"  most_common_scope: {top_scope}")

        block = "\n".join(block_lines) + "\n"
        if used + len(block) > max_chars:
            break
        out_lines.append(block)
        used += len(block)

    return "\n".join(out_lines)

def _aggregate_canonicals(
    canonicals: List[Dict[str, Any]],
    max_field_groups: int = 200,
    max_chars: int = 30000,
) -> str:
    # Concept-level digest: one entry per canonical candidate, counted by the
    # distinct papers of all its variants (the name-level digest splits one
    # concept's count across its many spellings).
    ranked = sorted(
        (c for c in canonicals if (c.get("scope") or "") != "paper_specific"),
        key=lambda c: int(c.get("support_count", 0)), reverse=True,
    )[:max_field_groups]
    out_lines: List[str] = []
    used = 0
    for c in ranked:
        lines = [f"- name: {c.get('name')}", f"  paper_count: {int(c.get('support_count', 0))}"]
        if c.get("description"):
            lines.append(f"  description: {c.get('description')}")
        ev = c.get("evidence_snippets") or []
        if ev:
            lines.append(f'  example_quote: "{str(ev[0])[:200]}"')
        aliases = [str(a) for a in (c.get("aliases") or [])[:5]]
        if aliases:
            lines.append(f"  aliases_seen: {', '.join(aliases)}")
        block = "\n".join(lines) + "\n"
        if used + len(block) > max_chars:
            break
        out_lines.append(block)
        used += len(block)
    return "\n".join(out_lines)

def _build_discovery_prompt(
    digest: str,
    domain: str,
    last_tier: str,
    n_anchors: int = 6,
) -> List[Dict[str, str]]:
    spec = get_domain_spec(domain)

    user = f"""
        You are summarizing recurring extraction concepts in a paper corpus.

        Domain     : {domain} ({spec.task_noun})
        Subdomain  : {last_tier}
        Corpus size: aggregated field proposals from many papers about "{last_tier}".

        Below is a digest of the most frequently proposed schema fields across
        the corpus, including the count of papers each field appeared in and
        a representative description / quote / alias list.

        Your task: identify the {n_anchors} most prominent recurring extraction
        concepts in this corpus. A "recurring concept" is one that:
          - appears across multiple papers under different exact wordings,
          - is centrally about "{last_tier}" rather than generic to the
            "{domain}" domain as a whole,
          - is useful as a SCHEMA FIELD NAME (general, reusable, not a specific
            value or entity).

        STRICT RULES:
          - You MUST ground every output in the digest below. Do NOT invent
            keywords or variable names that are not present in the input.
          - Keywords and representative_variable_names_seen MUST be drawn
            verbatim (or near-verbatim) from the digest.
          - Do NOT use external taxonomies of "{last_tier}" — only what you see.
          - If you cannot find {n_anchors} recurring concepts, return fewer.

        For EACH concept output:
          - name: snake_case canonical name suggested by the digest itself
          - description: 1 sentence
          - representative_keywords: 5-8 keywords drawn from the digest text
          - representative_variable_names_seen: 3-5 actual field names seen above
          - centrality_to_subdomain: 0..1 (1 = highly specific to "{last_tier}",
                                            0 = generic to "{domain}" overall)

        Return ONLY valid JSON:
            {{
              "subdomain": "{last_tier}",
              "anchors": [
                {{
                  "name": "snake_case",
                  "description": "...",
                  "representative_keywords": ["..."],
                  "representative_variable_names_seen": ["..."],
                  "centrality_to_subdomain": 0.5
                }}
              ]
            }}

        Corpus digest:
        {digest}
    """.strip()

    return [
        {"role": "system", "content": "Return ONLY valid JSON. No markdown, no commentary."},
        {"role": "user", "content": user},
    ]

def discover_subdomain_anchors(
    *,
    llm_chat_fn,
    paper_level_jsonl: Path,
    domain: str,
    last_tier: str,
    n_anchors: int = 6,
    out_path: Optional[Path] = None,
    canonicals: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    digest = _aggregate_canonicals(canonicals) if canonicals else _aggregate_proposals(paper_level_jsonl)
    if not digest:
        if out_path:
            out_path.write_text(json.dumps({"subdomain": last_tier, "anchors": []}, indent=2),
                                encoding="utf-8")
        return []

    messages = _build_discovery_prompt(digest, domain=domain, last_tier=last_tier, n_anchors=n_anchors)
    raw = llm_chat_fn(messages, force_json=True)

    try:
        data = json.loads(raw)
        anchors = data.get("anchors", []) if isinstance(data, dict) else []
        if not isinstance(anchors, list):
            anchors = []
    except Exception:
        anchors = []

    cleaned: List[Dict[str, Any]] = []
    for a in anchors:
        if not isinstance(a, dict):
            continue
        cleaned.append(
            {
                "name": str(a.get("name", "") or "").strip(),
                "description": str(a.get("description", "") or "").strip(),
                "representative_keywords": [
                    str(k).strip() for k in (a.get("representative_keywords") or []) if str(k).strip()
                ][:10],
                "representative_variable_names_seen": [
                    str(k).strip() for k in (a.get("representative_variable_names_seen") or []) if str(k).strip()
                ][:8],
                "centrality_to_subdomain": float(a.get("centrality_to_subdomain", 0.5) or 0.5),
            }
        )

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps({"subdomain": last_tier, "anchors": cleaned}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return cleaned

def load_anchors(path: Path) -> List[Dict[str, Any]]:
    if not path or not Path(path).exists():
        return []
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        anchors = data.get("anchors", []) if isinstance(data, dict) else []
        if not isinstance(anchors, list):
            return []
        return [a for a in anchors if isinstance(a, dict)]
    except Exception:
        return []