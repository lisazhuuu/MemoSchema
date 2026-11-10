"""memexp40: intent-aware self-refinement baselines.
Stages: part2 | part1 | report. Author prompts are imported from the pinned upstream prompts.py (not executed otherwise)."""
import ast, json, os, re, statistics as st, sys, threading, time, random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]; OUT = REPO / "results/memexp40"
if __name__ == "__main__" and (len(sys.argv) != 2 or sys.argv[1] not in ('part1', 'part2', 'report')):                     # usage check before any work
    print(__doc__ if sys.argv[1:2] in (["-h"], ["--help"]) else "usage: python scripts/" + Path(__file__).name + " " + 'part1 | part2 | report')
    sys.exit(0 if sys.argv[1:2] in (["-h"], ["--help"]) else 2)
sys.path[:0] = [str(REPO / "Phase1"), str(REPO)]
if not (len(sys.argv) > 1 and sys.argv[1] == "report"):
    _P = {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse(open(REPO / "baselines/intentaware/upstream/src/prompts.py").read()).body
          if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
    SYSTEM, GEN, CRIT, REV = _P["S2_SYSTEM_PROMPT"], _P["SCITABLES_GOALS_OPEN_LENGTH_SCHEMA"], _P["SINGLE_CRITIQUE_SCHEMA_TO_GOAL"], _P["UPDATE_SCHEMA_FROM_SINGLE_CRITIQUE"]
    GEN_ANCHOR = "Remember, the table should answer the question from the user goal."
    REV_ANCHOR = "Your goal is to incorporate all the feedback and improve on the original schema:"
    assert GEN_ANCHOR in GEN and REV_ANCHOR in REV
    GEN_B = GEN.replace(GEN_ANCHOR, GEN_ANCHOR + " The table must have exactly 8 attributes.")
    REV_B = REV.replace(REV_ANCHOR, REV_ANCHOR + " The updated schema must have exactly 8 attributes.")
NAMES = {"COF": "covalent organic frameworks (COFs)", "MOF": "metal-organic frameworks (MOFs)", "ZIF": "zeolitic imidazolate frameworks (ZIFs)"}
INTENT = {s: f"What are the key aspects along which research papers on {n} can be compared?" for s, n in NAMES.items()}
CORP = [(t, f"{s}_{k}") for t in ("e2e7", "e2e8", "e2e9") for s in ("COF", "MOF", "ZIF") for k in ("01", "02", "03")]
REPS = [1, 2, 3]
_lock = threading.Lock()

def fill(tpl, **kw):
    for k, v in kw.items():
        assert "{" + k + "}" in tpl, k
        tpl = tpl.replace("{" + k + "}", v)
    return tpl.replace("{{", "{").replace("}}", "}")

def call(prompt, log):
    from dotenv import load_dotenv; load_dotenv(REPO / ".env", override=False)
    from setup.test_azure_openai import chat_client, AZURE_OPENAI_CHAT_DEPLOYMENT as DEP
    for a in range(6):
        try:
            t = time.time()
            r = chat_client.chat.completions.create(model=DEP, messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                                                    temperature=0.7, max_completion_tokens=3000)
            u = r.usage.model_dump() if r.usage else {}
            log.append({"prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"), "seconds": time.time() - t,
                        "finish_reason": r.choices[0].finish_reason})
            return r.choices[0].message.content or ""
        except Exception as e:
            log.append({"api_error": f"{type(e).__name__}: {str(e)[:200]}"}); time.sleep(10 * (a + 1))
    raise RuntimeError("API failed")

def parse(text):
    t = re.sub(r"```(?:json)?", "", text or "").strip()
    for cand in (t, t[t.find("{"): t.rfind("}") + 1] if "{" in t else ""):
        try:
            x = json.loads(cand)
            if isinstance(x, dict) and x: return x
        except Exception: pass
    return None

def schema_call(prompt, log):
    """generation/revision: <= 3 attempts on unparsable output; returns (dict or None, attempts)."""
    for a in range(3):
        x = parse(call(prompt, log))
        if x is not None: return x, a + 1
    return None, 3

def budget(x):
    items = [(k, v) for k, v in x.items() if isinstance(k, str)]
    return dict(items[:8]), len(items)

def to_fields(x):
    return [{"name": k, "description": (v.get("definition", "") if isinstance(v, dict) else str(v)),
             "output_format": (v.get("output_format", "") if isinstance(v, dict) else "")} for k, v in x.items()]

def save(arm, tag, rep, run, fields, rec):
    d = OUT / f"runs/{tag}/s{rep}/{arm}/chem/{run}"; d.mkdir(parents=True, exist_ok=True)
    (d / "final_schema.json").write_text(json.dumps(fields, ensure_ascii=False, indent=1))
    (d / "record.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1))

def papers_text(tag, run):
    P = json.load(open(REPO / f"results/arxivdigestables/inputs/{tag}/{run}/papers.json"))["papers"]
    return "".join(f"Paper {i + 1}) {p['title']}\n{p['abstract']}\n" for i, p in enumerate(P))

def refine(intent, schema_json, info, rec):
    rec["critique_log"], rec["revision_log"] = [], []
    fb = call(fill(CRIT, intent=intent, schema=schema_json, papers=info), rec["critique_log"]); rec["model_feedback"] = fb
    x, a = schema_call(fill(REV_B, intent=intent, org_schema=schema_json, feedback=fb, papers=info), rec["revision_log"])
    rec["revision_attempts"] = a
    return x

def part1_job(job):
    tag, run, rep = job
    if (OUT / f"runs/{tag}/s{rep}/IA_SR/chem/{run}/final_schema.json").exists() or (OUT / f"runs/{tag}/s{rep}/IA_SR/chem/{run}/FAILED").exists(): return
    intent, info = INTENT[run[:3]], papers_text(tag, run)
    g = {"intent": intent, "gen_log": []}
    gf = OUT / f"runs/{tag}/s{rep}/IA_GEN/chem/{run}/final_schema.json"
    if gf.exists():
        G = {f["name"]: {"definition": f["description"], "output_format": f.get("output_format", "")} for f in json.load(open(gf))}
    else:
        x, a = schema_call(fill(GEN_B, user_goal=intent, papers=info), g["gen_log"]); g["gen_attempts"] = a
        if x is None:
            d = OUT / f"runs/{tag}/s{rep}/IA_GEN/chem/{run}"; d.mkdir(parents=True, exist_ok=True); (d / "FAILED").write_text(json.dumps(g)); return
        G, n = budget(x); g["n_raw_fields"] = n
        save("IA_GEN", tag, rep, run, to_fields(G), g)
    r = {"intent": intent, "start": G}
    x = refine(intent, json.dumps(G, ensure_ascii=False), info, r)
    if x is None:
        d = OUT / f"runs/{tag}/s{rep}/IA_SR/chem/{run}"; d.mkdir(parents=True, exist_ok=True); (d / "FAILED").write_text(json.dumps(r)); return
    S, n = budget(x); r["n_raw_fields"] = n
    save("IA_SR", tag, rep, run, to_fields(S), r)

def draft_and_candidates(tag, run):
    p = json.load(open(REPO / f"results/memexp26/calls/merge_{tag}_{run}_ORIG_s1.json"))["prompt"]
    d = p[p.index("DRAFT SCHEMA:") + len("DRAFT SCHEMA:"): p.index("CANDIDATE FIELDS:")].strip()
    c = p[p.index("CANDIDATE FIELDS:") + len("CANDIDATE FIELDS:"):].strip()
    return d, c

def part2_job(job):
    tag, run, rep = job
    d0 = OUT / f"runs/{tag}/s{rep}/SR_DRAFT/chem/{run}"
    if (d0 / "final_schema.json").exists() or (d0 / "FAILED").exists(): return
    draft, cands = draft_and_candidates(tag, run)
    info = "Candidate fields of the same corpus (name: description (number of supporting papers)):\n" + cands
    r = {"intent": INTENT[run[:3]], "draft": draft}
    x = refine(INTENT[run[:3]], draft, info, r)
    if x is None:
        d0.mkdir(parents=True, exist_ok=True); (d0 / "FAILED").write_text(json.dumps(r)); return
    S, n = budget(x); r["n_raw_fields"] = n
    save("SR_DRAFT", tag, rep, run, to_fields(S), r)

def report():
    sys.path.insert(0, str(REPO / "scripts")); from memexp27 import scores
    import completeness as C
    C.require_scores([("results/memexp40", "IA_SR"), ("results/memexp40", "SR_DRAFT"), ("results/memexp26", "ORIG"),
                      ("results/memexp26", "ORIG_M"), ("results/memexp38", "REUSEC_M"), ("results/arxivdigestables", "ADT_topic")], "memexp40")
    C.require_files([f"results/memexp40/runs/{t}/s{s}/{a}/chem/{r}/record.json" for a in ("IA_SR", "SR_DRAFT")
                     for t, r in C.CORP for s in (1, 2, 3)], "IA_SR / SR_DRAFT generation records", "memexp40")
    usd = lambda i, o: (i or 0) * 2.5e-6 + (o or 0) * 15e-6
    L = ["# memexp40 report: intent-aware self-refinement baselines (27 corpora r7-r9)\n"]
    # compliance / cost
    for arm in ("IA_SR", "SR_DRAFT"):
        recs = [json.load(open(f)) for f in OUT.glob(f"runs/*/s*/{arm}/chem/*/record.json")]
        fails = len(list(OUT.glob(f"runs/*/s*/{arm}/chem/*/FAILED")))
        n = [x.get("n_raw_fields") for x in recs if x.get("n_raw_fields") is not None]
        if arm == "IA_SR":
            # Include the initial generation stage in the full self-refinement cost.
            for rec, rf in zip(recs, OUT.glob(f"runs/*/s*/{arm}/chem/*/record.json")):
                initial = Path(str(rf).replace("/IA_SR/", "/IA_GEN/"))
                if initial.exists():
                    rec["gen_log"] = json.load(open(initial)).get("gen_log", [])
        cost = defaultdict(float); calls = defaultdict(int)
        for x in recs:
            for k in ("gen_log", "critique_log", "revision_log"):
                for c in x.get(k, []):
                    if "prompt_tokens" in c: calls[k] += 1; cost[k] += usd(c["prompt_tokens"], c["completion_tokens"])
        L.append(f"- {arm}: outputs {len(recs)}, failed {fails}, raw field count = 8: {sum(v == 8 for v in n)}, >8: {sum(v > 8 for v in n)}, <8: {sum(v < 8 for v in n)}; "
                 + ", ".join(f"{k[:-4]} calls/output {calls[k] / max(1, len(recs)):.2f} ${cost[k] / max(1, len(recs)):.3f}" for k in calls))
    roots = {"IA_SR": OUT, "SR_DRAFT": OUT, "ORIG": REPO / "results/memexp26", "ORIG_M": REPO / "results/memexp26",
             "REUSEC_M": REPO / "results/memexp38", "ADT_topic": REPO / "results/arxivdigestables"}
    for fine in (False, True):
        cache, V = {}, {}
        for arm, root in roots.items():
            if root not in cache:
                S = {}
                for f in (glob_ := list(root.glob("runs/e2e[789]/s*/" + ("eval_fine/*/rep*/chem/*/metrics_fine.json" if fine else "eval/*/rep*/chem/*/metrics.csv")))):
                    p = f.parts; i = p.index("runs")
                    v = json.load(open(f))["fine_f1"] if fine else float(open(f).read().splitlines()[1].split(",")[11])
                    S.setdefault((p[i + 1], p[i + 7], p[i + 4], p[i + 2]), []).append(v)
                cache[root] = S
            S = cache[root]
            smp = {"ORIG": ["s1"], "ORIG_M": ["s1"]}.get(arm)
            per = defaultdict(list)
            for (t, run, a, s), v in S.items():
                if a == arm and (smp is None or s in smp): per[(t, run)].append(st.mean(v))
            V[arm] = {k: st.mean(v) for k, v in per.items()}
        nm = "hierarchical F1" if fine else "strict one-to-one F1"
        L.append(f"\n## {nm}: " + ", ".join(f"{a} {st.mean(V[a].values()):.3f} (n={len(V[a])})" for a in roots if V[a]))
        for A, B in (("SR_DRAFT", "ORIG"), ("ORIG_M", "ORIG"), ("ORIG_M", "SR_DRAFT"), ("REUSEC_M", "IA_SR")):
            D = {k: V[A][k] - V[B][k] for k in sorted(V[A]) if k in V[B]}   # sorted: independent of directory order
            if len(D) < 3: continue
            by = defaultdict(list)
            for k in D: by[k[1][:3]].append(k)
            groups = [by[s] for s in sorted(by)]
            random.seed(0); bs = sorted(st.mean([D[k] for ks in groups for k in random.choices(ks, k=len(ks))]) for _ in range(5000))
            L.append(f"- {A} - {B}: {st.mean(D.values()):+.3f} [{bs[125]:+.3f}, {bs[4874]:+.3f}] ({sum(x > 0 for x in D.values())}/{sum(x < 0 for x in D.values())}) n={len(D)}")
    (OUT / "report.md").write_text("\n".join(L)); print("\n".join(L))

if __name__ == "__main__":
    stage = sys.argv[1]
    only = os.environ.get("ONLY", "")
    jobs = [(t, r, k) for t, r in CORP for k in REPS if not only or r == only and t == os.environ.get("ONLY_TAG", t)]
    if stage == "part2":
        with ThreadPoolExecutor(8) as ex: list(ex.map(part2_job, jobs))
    elif stage == "part1":
        with ThreadPoolExecutor(6) as ex: list(ex.map(part1_job, jobs))
    else:
        report()
