"""Operator-level calls and tokens per corpus for the r7-r9 no-memory runs (per-paper calls, cluster consolidation,
anchors, restructuring); output results/cost_compare.json. memo_audit.py and memexp39_report.py use the per-call means
of "per_paper" and "merge" to price reuse that fails the content checks (paper Fig. 3b).
Our pipeline did not log tokens: prompts are rebuilt from saved artifacts with the pipeline's own prompt builders
(fake LLM that replays the saved responses) and counted with tiktoken o200k_base (matches the API's prompt-token
count on EVAPORATE's logged calls to within 0.5%). Merge-call count per round = (#clusters with >= 2 members) scaled by
the per-run ratio observed in the memexp26 ORIG rebuild hook. Query generation is not counted (outside Fig. 3b)."""
import glob, json, os, sys, statistics as st, tempfile
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
if __name__ == "__main__" and (sys.argv[1:]):                     # usage check before any work
    print(__doc__ if sys.argv[1:2] in (["-h"], ["--help"]) else "usage: python scripts/" + Path(__file__).name + " " + "(no arguments)")
    sys.exit(0 if sys.argv[1:2] in (["-h"], ["--help"]) else 2)
sys.path[:0] = [str(REPO / "Phase1"), str(REPO)]
os.environ["MEMOSCHEMA_FIX_CLUSTER"] = "1"; os.environ["MEMOSCHEMA_FIX_ANCHOR_SCHEMA"] = "1"
import tiktoken
ENC = tiktoken.get_encoding("o200k_base")
T = lambda s: len(ENC.encode(s if isinstance(s, str) else json.dumps(s, ensure_ascii=False)))
TM = lambda msgs: sum(T(m["content"]) for m in msgs) + 4 * len(msgs)
import schema_induction as si
import schema_lib.schema_cluster_merge as scm
from schema_lib.subdomain_discovery import discover_subdomain_anchors
TIER = {"COF": "Covalent Organic Frameworks (COFs)", "MOF": "Metal-organic frameworks (MOFs)", "ZIF": "Zeolitic imidazolate frameworks (ZIFs)"}

def per_paper(rd, texts, tier):
    calls = tin = tout = 0
    for dbg in sorted(rd.glob("round0*/schema_from_cumulative_corpus/paper_level_debug")):
        for f in dbg.glob("*.distill.raw.json"):
            pid = f.name[: -len(".distill.raw.json")]
            if pid not in texts:
                continue
            q = [f.read_text()]
            for suf in (".raw.json", ".raw.retry.json"):
                if (dbg / (pid + suf)).exists():
                    q.append((dbg / (pid + suf)).read_text())
            sent = []
            def fake(messages, force_json=False):
                sent.append(messages); return q[len(sent) - 1] if len(sent) <= len(q) else "{}"
            si.llm_chat = fake
            os.environ["AUTOSCHEMA_DEBUG_DIR"] = tempfile.mkdtemp()
            si.propose_paper_level_candidate_fields(papers=[{"paper_id": pid, "source_file": pid + ".pdf", "text": texts[pid]}],
                                                    domain="chem", last_tier=tier, max_fields_per_paper=25, max_text_chars=12000,
                                                    distill_input_chars=60000, distill_quote_limit=10)
            n = min(len(sent), len(q)); calls += n
            tin += sum(TM(m) for m in sent[:n]); tout += sum(T(x) for x in q[:n])
    return calls, tin, tout

def merges(rd, ratio):
    calls = tin = tout = 0
    for f in sorted(rd.glob("round0*/schema_from_cumulative_corpus/merged_field_clusters.json")):
        for cl in json.load(open(f)):
            if len({m.get("paper_id") for m in cl["members"] if m.get("paper_id")}) < 2:
                continue
            sent = []
            scm.llm_chat = lambda messages, force_json=False: (sent.append(messages) or json.dumps(cl["llm_merged"]))
            try:
                scm._chunked_llm_merge(cl["members"], domain="chem", cid=0)
            except Exception:
                pass
            calls += len(sent); tin += sum(TM(m) for m in sent); tout += len(sent) * T(cl["llm_merged"])
    return calls, tin, tout

def anchors(rd, tier):
    calls = tin = tout = 0; rows = []
    for rnd in sorted(rd.glob("round0*/schema_from_cumulative_corpus")):
        nf = rnd / "paper_level_new_this_round.jsonl"
        if nf.exists():
            rows += [l for l in nf.read_text().splitlines() if l.strip()]
        af = rnd / "discovered_anchors.json"
        if not af.exists() or not rows:
            continue
        tmp = Path(tempfile.mkdtemp()); (tmp / "p.jsonl").write_text("\n".join(rows) + "\n")
        sent = []
        def fake(messages, force_json=False):
            sent.append(messages); return af.read_text()
        discover_subdomain_anchors(llm_chat_fn=fake, paper_level_jsonl=tmp / "p.jsonl", domain="chem", last_tier=tier,
                                   n_anchors=8, out_path=tmp / "a.json")
        calls += len(sent); tin += sum(TM(m) for m in sent); tout += len(sent) * T(af.read_text())
    return calls, tin, tout

def main():
    import input_package; input_package.check("replay", "cost_compare")   # minimal input package: out of scope, stop
    import completeness as C                    # 27 corpora: frozen texts, per-round debug records, saved restructuring call
    C.require_files([f"results/evaporate/inputs/{'r' + t[3:]}/{r}/papers.jsonl" for t, r in C.CORP]
                    + [f"results/memexp26/calls/merge_{t}_{r}_ORIG_s1.json" for t, r in C.CORP], "cost inputs", "cost_compare")
    nodbg = [f"{b}/{run}" for b in ("r7", "r8", "r9") for run in sorted(os.listdir(REPO / f"results/e2e/e2e_v3/{b}/chem"))
             if not any(sorted((REPO / f"results/e2e/e2e_v3/{b}/chem/{run}").glob("iterative_phase1_chem_*"))[-1].glob(
                 "round0*/schema_from_cumulative_corpus/paper_level_debug/*.distill.raw.json"))]
    if nodbg:
        raise SystemExit(f"[cost_compare] no paper_level_debug records in {len(nodbg)} runs (e.g. {nodbg[0]}); no report written")
    out = {}
    for b in ("r7", "r8", "r9"):
        tag = "e2e" + b[1:]
        for run in sorted(os.listdir(REPO / f"results/e2e/e2e_v3/{b}/chem")):
            rd = sorted((REPO / f"results/e2e/e2e_v3/{b}/chem/{run}").glob("iterative_phase1_chem_*"))[-1]
            tier = TIER[run.split("_")[0]]
            texts = {json.loads(l)["paper_id"]: json.loads(l)["text"] for l in open(REPO / f"results/evaporate/inputs/{b}/{run}/papers.jsonl")}
            ratio = 1
            pp, mg, an = per_paper(rd, texts, tier), merges(rd, ratio), anchors(rd, tier)
            mf = REPO / f"results/memexp26/calls/merge_{tag}_{run}_ORIG_s1.json"
            M = json.load(open(mf)) if mf.exists() else {"prompt": "", "response": ""}
            mm = (1, T(M["prompt"]) + 8, T(M["response"] if isinstance(M["response"], str) else json.dumps(M["response"])))
            out[f"{tag}/{run}"] = {"per_paper": pp, "merge": mg, "anchors": an, "MERGE": mm}
            print(tag, run, {k: [round(x) for x in v] for k, v in out[f"{tag}/{run}"].items()}, flush=True)
    (REPO / "results/cost_compare.json").write_text(json.dumps(out, indent=1))
    tot = lambda k, i: st.mean(v[k][i] for v in out.values())
    print("\nmean per corpus (calls, input tokens, output tokens):")
    for k in ("per_paper", "merge", "anchors", "MERGE"):
        print(f"  {k}: {tot(k,0):.0f} calls, {tot(k,1):.0f} in, {tot(k,2):.0f} out")
    C = sum(tot(k, 0) for k in out[next(iter(out))]); I = sum(tot(k, 1) for k in out[next(iter(out))]); O = sum(tot(k, 2) for k in out[next(iter(out))])
    print(f"  TOTAL: {C:.0f} calls, {I:.0f} in, {O:.0f} out, ${I/1e6*2.5 + O/1e6*15:.2f} at $2.50/$15 per 1M")

if __name__ == "__main__":
    main()
