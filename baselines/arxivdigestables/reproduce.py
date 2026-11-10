#!/usr/bin/env python3
"""ArxivDIGESTables schema (column) generation on frozen MemoSchema inputs.

Author code: https://github.com/bnewm0609/arxivDIGESTables (pinned commit below). The prompt file and the helper
functions (paper formatting, output parsing, validation) are loaded from the author's files and verified by SHA256.
Only the column/schema step is reproduced; the per-paper value generation is not run.

Setting "topic": the author's caption prompt ("gold_caption_in_text_ref") with caption = the research topic that our
pipeline also receives, and no in-text references (no reference-table information is used).
Input papers: the paper set our pipeline induced from (schema_state paper ids), title + abstract parsed from the PDF.
Reference answers are never loaded. Dry-run by default; --execute calls the API.
"""
from __future__ import annotations
import argparse, ast, hashlib, json, os, re, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
UP = HERE / "upstream_repo" / "experiment" / "paper_comparison"
COMMIT = "018d8f5602f8b9235ddee27b868b11c2cab9d4d5"
sys.path[:0] = [str(HERE), str(REPO / "baselines" / "evaporate")]   # HERE: section_diag.py
TOPICS = {"COF": "Covalent Organic Frameworks (COFs)", "MOF": "Metal-organic frameworks (MOFs)",
          "ZIF": "Zeolitic imidazolate frameworks (ZIFs)"}
N_FIELDS = 8
ABSTRACT_CHARS = 2000

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def load_upstream():
    ns = {"json": json, "os": os}
    tree = ast.parse((UP / "generation.py").read_text())
    want = {"make_paper_list_input", "str_to_list", "validate_list_scheme"}
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in want]
    assert {n.name for n in body} == want, "author helper missing"
    exec(compile(ast.Module(body=body, type_ignores=[]), "generation.py", "exec"), ns)
    prompts = json.loads((UP / "data" / "prompt_ver3.json").read_text())
    prov = {"commit": COMMIT, "generation.py": sha(UP / "generation.py"), "prompt_ver3.json": sha(UP / "data" / "prompt_ver3.json")}
    return ns, prompts, prov

def title_of(pid):
    return re.sub(r"\s+", " ", re.sub(r"__[a-z]+$", "", pid).replace("_", " ")).strip()

def abstract_of(text):
    from section_diag import sections
    first, _ = sections(text)
    a, i = first.get("abstract"), first.get("intro")
    if a is not None:
        end = i if i is not None and i > a else a + ABSTRACT_CHARS
        return text[a: min(end, a + ABSTRACT_CHARS)].split("\n", 1)[-1].strip(), "abstract_heading"
    return text[:ABSTRACT_CHARS].strip(), "first_chars"

def prepare(run_dir, out):
    import importlib.util                                   # EVAPORATE adapter: same paper identities and parser
    spec = importlib.util.spec_from_file_location("evaporate_reproduce", REPO / "baselines" / "evaporate" / "reproduce.py")
    ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
    source_papers, extract_text = ev.source_papers, ev.extract_text
    papers, src = source_papers(run_dir)
    recs = []
    for pid, path in papers:
        raw = extract_text(path)
        ab, how = abstract_of(raw)
        recs.append({"paper_id": pid, "title": title_of(pid), "abstract": ab, "abstract_source": how, "pdf_sha256": sha(path)})
    out.mkdir(parents=True, exist_ok=True)
    (out / "papers.json").write_text(json.dumps({"source": src, "papers": recs}, ensure_ascii=False, indent=1))
    return recs

def chat_factory():
    from dotenv import load_dotenv
    load_dotenv(REPO / ".env", override=False)
    sys.path.insert(0, str(REPO))
    from setup.test_azure_openai import chat_client, AZURE_OPENAI_CHAT_DEPLOYMENT
    def chat(messages, max_tokens, temperature):
        kw = {"model": AZURE_OPENAI_CHAT_DEPLOYMENT, "messages": messages, "max_completion_tokens": max_tokens}
        if temperature is not None:
            kw["temperature"] = temperature
        r = chat_client.chat.completions.create(**kw)
        u = getattr(r, "usage", None)
        return r.choices[0].message.content or "", r.choices[0].finish_reason, (u.model_dump() if u else {}), AZURE_OPENAI_CHAT_DEPLOYMENT
    return chat

def parse(text, tpl, ns):
    """Author parser first (str_to_list for the list prompt). The current model may omit the ```list fence. Fallback (format only, content unchanged): the last fenced block, else the raw text."""
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        res = ns["str_to_list"](text.strip(), tpl["parse_str"])
    if not isinstance(res, str):
        return res
    blocks = re.findall(r"```(?:json|list)?\s*(.*?)```", text, re.S)
    for cand in ([blocks[-1]] if blocks else []) + [text.strip()]:
        try:
            return json.loads(cand)
        except Exception:
            pass
    return res

def run_one(recs, topic, rep, out, ns, prompts, chat, args):
    f = out / f"rep{rep}" / "final_schema.json"
    if f.exists():
        return
    # Author helper formats ONE paper; at commit 018d8f5 it returns only that paper (the paper_text argument is
    # discarded), so the author loop would pass just the last paper. We concatenate, as the paper describes
    # ("given a list of papers"). Documented deviation in favour of the baseline.
    paper_text = "".join(ns["make_paper_list_input"]("", i, p, "abstract", "multiple") for i, p in enumerate(recs))
    tpl = prompts["gold_caption_in_text_ref"]
    prompt = tpl["prompt_abstract"].format(num_attributes=N_FIELDS, input_info=f"[Paper Content]\n{paper_text}",
                                           caption=f"Comparison of research papers on {topic}", in_text_ref="")
    if len(prompt) > args.max_prompt_chars:
        raise SystemExit(f"prompt {len(prompt)} chars > {args.max_prompt_chars}; decide on an input policy first")
    messages = [{"role": "assistant", "content": tpl["system_instruction"]}, {"role": "user", "content": prompt}]   # author's message layout
    (out / f"rep{rep}").mkdir(parents=True, exist_ok=True)
    if not args.execute:
        print(f"[dry-run] {out.name} topic rep{rep}: {len(recs)} papers, prompt {len(prompt)} chars"); return
    log = []
    for attempt in range(5):                                           # author: up to 5 tries on invalid schema
        text, fin, usage, model = chat(messages, args.max_tokens, args.temperature)
        res = parse(text, tpl, ns)
        ok, err = ns["validate_list_scheme"](res)
        log.append({"attempt": attempt, "finish_reason": fin, "usage": usage, "valid": ok, "error": err, "response": text})
        if ok:
            break
    (out / f"rep{rep}" / "calls.json").write_text(json.dumps({"messages": messages, "model": model, "attempts": log}, ensure_ascii=False, indent=1))
    if not ok:
        (out / f"rep{rep}" / "FAILED").write_text(err); print(f"[fail] {out} rep{rep}: {err}"); return
    cols = res
    fields = [{"name": str(c), "description": "", "source": "arxivdigestables"} for c in cols][:N_FIELDS]
    f.write_text(json.dumps(fields, ensure_ascii=False, indent=2))
    (out / f"rep{rep}" / "meta.json").write_text(json.dumps({"n_columns_returned": len(cols)}, ensure_ascii=False, indent=1))
    print(f"[done] {out} rep{rep}: {[x['name'] for x in fields]}")

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--batches", default="r9"); p.add_argument("--runs", default="COF_01")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--out", default=str(REPO / "results" / "arxivdigestables"))
    p.add_argument("--max-tokens", type=int, default=3000, help="author: 3000")
    p.add_argument("--temperature", type=float, default=1.0, help="author: 1")
    p.add_argument("--max-prompt-chars", type=int, default=600000)
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    ns, prompts, prov = load_upstream()
    chat = chat_factory() if a.execute else None
    import glob
    for b in a.batches.split(","):
        tag = "e2e" + b[1:]
        for run in a.runs.split(","):
            hits = sorted(glob.glob(str(REPO / f"results/e2e/e2e_v3/{b}/chem/{run}/iterative_phase1_chem_*")))
            if not hits:
                print(f"[skip] {b} {run}: no run dir"); continue
            base = Path(a.out) / "inputs" / tag / run
            recs = json.loads((base / "papers.json").read_text())["papers"] if (base / "papers.json").exists() else prepare(Path(hits[-1]), base)
            out = Path(a.out) / "raw" / tag / "topic" / run
            out.mkdir(parents=True, exist_ok=True)
            (out / "provenance.json").write_text(json.dumps({**prov, "setting": "topic", "max_tokens": a.max_tokens, "temperature": a.temperature,
                                                              "n_fields": N_FIELDS, "papers": len(recs)}, indent=1))
            for r in range(1, a.reps + 1):
                run_one(recs, TOPICS[run.split("_")[0]], r, out, ns, prompts, chat, a)
                sf = out / f"rep{r}" / "final_schema.json"
                if sf.exists():                                       # judge layout: runs/<tag>/s<rep>/<ARM>/chem/<run>
                    d = Path(a.out) / "runs" / tag / f"s{r}" / "ADT_topic" / "chem" / run; d.mkdir(parents=True, exist_ok=True)
                    (d / "final_schema.json").write_text(sf.read_text())

if __name__ == "__main__":
    main()
