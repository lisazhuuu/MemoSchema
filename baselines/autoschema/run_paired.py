#!/usr/bin/env python3
"""AutoSchema Phase-I schema induction (unmodified artifact) on the papers of one MemoSchema run, round by round.

Only change: design_extraction_prompt.load_pdf_texts is replaced by a loader that returns the frozen first-12k texts
(same PyMuPDF page-join parser as the artifact's pdf_to_text). Env names are mapped from the project .env at runtime;
no key is printed. Every chat call's usage is logged."""
from __future__ import annotations
import argparse, glob, json, os, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
UP = HERE / "upstream"
TIER_KEY = {"COF": "cof", "MOF": "mof", "ZIF": "zif"}

def env_setup(out_root):
    from dotenv import load_dotenv
    load_dotenv(REPO / ".env", override=False)
    m = {"CHAT_DEPLOYMENT": "AZURE_OPENAI_CHAT_DEPLOYMENT", "CHAT_API_KEY": "AZURE_OPENAI_CHAT_API_KEY",
         "CHAT_ENDPOINT": "AZURE_OPENAI_CHAT_ENDPOINT", "CHAT_VERSION": "AZURE_OPENAI_CHAT_API_VERSION",
         "EMBEDDING_DEPLOYMENT": "AZURE_OPENAI_EMBED_DEPLOYMENT", "EMBEDDING_API_KEY": "AZURE_OPENAI_EMBED_API_KEY",
         "EMBEDDING_ENDPOINT": "AZURE_OPENAI_EMBED_ENDPOINT", "EMBEDDING_VERSION": "AZURE_OPENAI_EMBED_API_VERSION"}
    for k, v in m.items():
        if os.environ.get(v):
            os.environ[k] = os.environ[v]
    os.environ["LLM_PROVIDER"] = "azure_openai"
    os.environ["AUTOSCHEMA_OUTPUT_DIR"] = str(out_root / "_artifact_outputs")

def round_papers(rd):
    """paper ids our pipeline induced from, per round."""
    R = []
    for f in sorted(rd.glob("round0*/schema_from_cumulative_corpus/paper_level_new_this_round.jsonl")):
        ids = [json.loads(l)["paper_id"] for l in open(f) if l.strip()]
        if ids:
            R.append((f.parts[-3], ids))
    return R

def main():
    p = argparse.ArgumentParser(); p.add_argument("--batch", required=True); p.add_argument("--run", required=True)
    p.add_argument("--rep", type=int, required=True); p.add_argument("--out", type=Path, default=REPO / "results/autoschema")
    p.add_argument("--max-final-fields", type=int, default=8)
    a = p.parse_args()
    tag = "e2e" + a.batch[1:]
    final = a.out / "runs" / tag / f"s{a.rep}" / "AUTOSCHEMA" / "chem" / a.run / "final_schema.json"
    if final.exists():
        print("done already"); return
    env_setup(a.out)
    sys.path.insert(0, str(UP)); os.chdir(UP)
    # prompt_lib.domain resolves domain/last tier from argv at import time
    sys.argv = ["design_extraction_prompt.py", "--domain", "chem", "--last_tier", TIER_KEY[a.run.split("_")[0]]]
    import design_extraction_prompt as dep
    import prompt_lib.llm_client as lc
    usage_log = a.out / "raw" / tag / f"rep{a.rep}" / a.run / "usage.jsonl"; usage_log.parent.mkdir(parents=True, exist_ok=True)
    orig_get = lc.get_chat_client
    def logged_client():
        c = orig_get(); create = c.chat.completions.create
        if getattr(create, "_logged", False):
            return c
        def wrapped(**kw):
            r = create(**kw)
            u = r.usage.model_dump() if r.usage else {}
            with open(usage_log, "a") as f:
                f.write(json.dumps({"t": time.time(), "usage": u, "messages": kw.get("messages"),
                                    "response": r.choices[0].message.content}, ensure_ascii=False) + "\n")
            return r
        wrapped._logged = True; c.chat.completions.create = wrapped
        return c
    lc.get_chat_client = logged_client
    texts = {json.loads(l)["paper_id"]: json.loads(l)["text"] for l in open(REPO / f"results/evaporate/inputs/{a.batch}/{a.run}/papers.jsonl")}
    rd = sorted((REPO / f"results/e2e/e2e_v3/{a.batch}/chem/{a.run}").glob("iterative_phase1_chem_*"))[-1]
    prev = None
    for rname, ids in round_papers(rd):
        cur = [i for i in ids if i in texts]
        dep.load_pdf_texts = lambda pdf_dir, relevance_chars=4000, proposal_chars=8000, _c=cur: [
            {"source_file": i + ".pdf", "text_for_relevance": texts[i][:relevance_chars], "text_for_proposal": texts[i][:proposal_chars]} for i in _c]
        od = a.out / "raw" / tag / f"rep{a.rep}" / a.run / rname
        dummy = od / "_pdf_dir"; dummy.mkdir(parents=True, exist_ok=True)
        argv = ["design_extraction_prompt.py", "--domain", "chem", "--last_tier", TIER_KEY[a.run.split("_")[0]],
                "--pdf_dir", str(dummy), "--output", str(od), "--max_final_fields", str(a.max_final_fields)]
        if prev:
            argv += ["--prev_schema", str(prev)]
        sys.argv = argv
        print(f"[autoschema] {tag} {a.run} rep{a.rep} {rname}: {len(cur)} papers ({'refine' if prev else 'initialize'})", flush=True)
        dep.main()
        prev = od / "final_schema.json"
    final.parent.mkdir(parents=True, exist_ok=True)
    final.write_text(prev.read_text())
    print("[autoschema] COMPLETE", final, flush=True)

if __name__ == "__main__":
    main()
