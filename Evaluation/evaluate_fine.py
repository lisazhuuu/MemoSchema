"""Hierarchical F1 (paper §VI-A): judge an induced schema against the detailed reference inventory of its
chem subdomain (ontology_lib/references_fine.py).

Scoring:
  For each fine reference concept the judge names the induced field that records
  it (or none) and a strength:
    strong   the field records this attribute as its main content (same attribute,
             or a format/unit variant of it);
    partial  the field is broader and would hold this attribute as one part of
             its values, or is a narrower version of it;
    none     no field records it.
  One induced field MAY cover several fine concepts (unlike the coarse judge),
  because a schema with 8 fields is compared with ~25 fine concepts.
  fine recall    = (strong + 0.5 * partial) / n_fine
  fine precision = induced fields covering >= 1 fine concept / n_induced
  fine F1        = harmonic mean of the two (metrics_fine.json "fine_f1": the hierarchical F1 reported in the paper)
Outputs <out-dir>/metrics_fine.json and <out-dir>/alignment_fine.json.
"""
from __future__ import annotations
import argparse, json, re, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from ontology_lib.llm_judge import build_azure_client, get_model_deployment, _robust_json_parse, _render_induced_field
from ontology_lib.config import LLM_TEMPERATURE
from ontology_lib.references_fine import get_fine_reference
from ontology_lib.schema_io import load_schema

SYSTEM = ("You are a careful domain-expert evaluator. You decide which reference attributes an "
          "induced extraction schema records. Think in terms of what values a curator would put "
          "in each field, not surface words.")

TASK = """[TASK]
For EACH reference attribute below, name the induced field whose values would record it,
or null if none would. Label the strength:
  "strong":  the field records this attribute as its main content (same attribute, or a
             unit/format variant of it);
  "partial": the field is broader and this attribute would be only one part of its values
             (e.g. a general "synthesis_method" field for "modulator_use"), or the field is a
             narrower version of the attribute;
  "none":    no induced field would record it.
One induced field MAY be named for several reference attributes. Do not stretch: if a curator
filling the field would normally not record this attribute there, answer "none".

[OUTPUT: one JSON object]
{"alignment": {"<reference_attribute_name>": {"field": "<induced_field_name>" or null,
 "strength": "strong" | "partial" | "none", "reason": "<short>"}, ...}}
(one entry per reference attribute, keys exactly as listed)
"""

def build_prompt(sub, fine, induced):
    lines = [TASK, f"SUBDOMAIN: {sub}", "", f"REFERENCE ATTRIBUTES ({len(fine)}):"]
    for x in fine:
        lines.append(f"  - {x['name']}: {x['definition']}")
    lines += ["", f"INDUCED SCHEMA FIELDS ({len(induced)}):"]
    for j, f in enumerate(induced, 1):
        lines.append(_render_induced_field(j, f))
    return "\n".join(lines)

LAST_RAW = {"response": None, "unresolved": []}

def resolve_field(fld, induced):
    """Map the judge's field reference to an induced field name. Accepts the exact name, any case, and the
    prompt's index forms ("I3", "I3. name", "I3: name"), so an index-form answer is not scored as 'none';
    returns None if unresolvable."""
    if not fld:
        return None
    names = [f.name for f in induced]; lower = {n.lower(): n for n in names}
    s = str(fld).strip().strip("`'\"")
    if s in names or s.lower() in lower:
        return lower[s.lower()]
    m = re.match(r"^I?(\d+)\s*[.:)\-]?\s*(.*)$", s)
    if m:
        rest = m.group(2).strip().strip("`'\"").lower()
        if rest in lower:
            return lower[rest]
        i = int(m.group(1))
        if not rest and 1 <= i <= len(names):
            return names[i - 1]
    return None

def judge(client, model, sub, fine, induced, retries=3):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": build_prompt(sub, fine, induced)}]
    for a in range(retries):
        try:
            r = client.chat.completions.create(model=model, messages=msgs, temperature=LLM_TEMPERATURE,
                                               max_completion_tokens=6000, response_format={"type": "json_object"})
            raw = r.choices[0].message.content or ""
            LAST_RAW["response"] = raw
            al = (_robust_json_parse(raw) or {}).get("alignment")
            if not isinstance(al, dict):
                raise ValueError("no alignment dict")
            al = {str(k).strip().lower(): v for k, v in al.items()}
            rows, unresolved = [], []
            for x in fine:
                e = al.get(x["name"].lower()) or {}
                fld = resolve_field(e.get("field"), induced)
                if e.get("field") and fld is None and str(e.get("strength", "none")).lower() in ("strong", "partial"):
                    unresolved.append(str(e.get("field")))
                s = str(e.get("strength", "none")).lower()
                s = s if s in ("strong", "partial") and fld else "none"
                rows.append({"concept": x["name"], "parent": x["parent"], "field": fld if s != "none" else None,
                             "strength": s, "reason": str(e.get("reason", ""))[:300]})
            LAST_RAW["unresolved"] = unresolved
            if unresolved and a < retries - 1:
                raise ValueError(f"unresolved field references {unresolved[:3]}")
            return rows
        except Exception as ex:
            print(f"   judge attempt {a + 1} failed: {ex}", flush=True); time.sleep(3 * (a + 1))
    return None

def metrics(rows, n_induced):
    n = len(rows); st = sum(r["strength"] == "strong" for r in rows); pa = sum(r["strength"] == "partial" for r in rows)
    used = {r["field"] for r in rows if r["field"]}
    rec = (st + 0.5 * pa) / n
    prec = len(used) / n_induced if n_induced else 0.0
    f1 = lambda p, r: 2 * p * r / (p + r) if p + r else 0.0
    return {"n_fine": n, "n_induced": n_induced, "strong": st, "partial": pa, "fields_used": len(used),
            "fine_recall": rec, "fine_precision": prec, "fine_f1": f1(prec, rec)}

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--schema", type=Path, required=True)
    ap.add_argument("--subdomain", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    fine = get_fine_reference(a.subdomain); induced = load_schema(a.schema)
    rows = judge(build_azure_client(), get_model_deployment(), a.subdomain, fine, induced)
    if rows is None:
        print("FAILED"); return 1
    a.out_dir.mkdir(parents=True, exist_ok=True)
    m = metrics(rows, len(induced))
    (a.out_dir / "alignment_fine.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    (a.out_dir / "judge_raw.json").write_text(json.dumps(LAST_RAW, indent=1, ensure_ascii=False))
    (a.out_dir / "metrics_fine.json").write_text(json.dumps(m, indent=1))
    print(f"{a.subdomain}: fine F1 {m['fine_f1']:.3f} (R {m['fine_recall']:.3f}, P {m['fine_precision']:.3f})")
    return 0

if __name__ == "__main__":
    sys.exit(main())
