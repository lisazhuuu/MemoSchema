"""memexp36: downstream extraction utility (useful-value rate) of the first schema output. Stages: extract | judge | report

9 r8 corpora x 5 papers (shuffle of results/evaporate/inputs/r8/<run>/papers.jsonl with seed 0, first 5), shared
AutoSchema Phase-II extractor with field names + descriptions. MemoSchema = final content-based reuse configuration:
results/memexp38/runs/e2e8/s1/REUSEC_M/chem/<run>/final_schema.json (anchor sample 1); missing inputs stop the stage."""
import json, random, re, statistics as st, sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]; OUT = REPO / "results/memexp36"
sys.path[:0] = [str(REPO)]
OURS = "REUSEC_M"
SRC = {OURS: ("results/memexp38", "REUSEC_M"), "LEGACY": ("results/memexp26", "LEGACY"),
       "AUTOSCHEMA": ("results/autoschema", "AUTOSCHEMA"), "L4S": ("results/memexp27", "L4S"),
       "ADT_topic": ("results/arxivdigestables", "ADT_topic"), "EVAPORATE_FULL8": ("results/evaporate", "EVAPORATE_FULL8")}
RUNS = [f"{s}_{k}" for s in ("COF", "MOF", "ZIF") for k in ("01", "02", "03")]
TAG, B = "e2e8", "r8"
JUDGE = """For each extracted value below (from one paper), classify it:
- "specific": a concrete reported fact (a number with unit, a named material/compound/linker, a concrete synthesis
  method or condition, a measured property or outcome) that answers the field;
- "generic": a vague summary, a category name, or a list of topics rather than a concrete fact;
- "unsupported": not supported by its quote.
Return ONLY JSON: {{"labels": [{{"i": <index>, "label": "specific" | "generic" | "unsupported"}}]}}

VALUES:
{items}"""

def chat(msgs):
    from dotenv import load_dotenv; load_dotenv(REPO / ".env", override=False)
    from setup.test_azure_openai import chat_client, AZURE_OPENAI_CHAT_DEPLOYMENT as DEP
    for a in range(4):
        try:
            r = chat_client.chat.completions.create(model=DEP, messages=msgs, response_format={"type": "json_object"})
            return r.choices[0].message.content, (r.usage.model_dump() if r.usage else {})
        except Exception as e:
            import time; time.sleep(5 * 2 ** a); err = e
    raise err

def papers_file(run): return REPO / f"results/evaporate/inputs/{B}/{run}/papers.jsonl"

def papers(run):
    rows = [json.loads(l) for l in open(papers_file(run))]
    random.Random(0).shuffle(rows)
    return rows[:5]

def schema_path(m, run):
    root, arm = SRC[m]
    return REPO / root / f"runs/{TAG}/s1/{arm}/chem/{run}/final_schema.json"

def require_inputs():
    miss = [papers_file(r) for r in RUNS if not papers_file(r).exists()] + \
           [schema_path(OURS, r) for r in RUNS if not schema_path(OURS, r).exists()]
    if miss:
        raise SystemExit("[memexp36] missing inputs (paper texts: baselines/evaporate/reproduce.py prepare/batch; "
                         "REUSEC_M schemas: scripts/run_memexp38.sh):\n  " + "\n  ".join(str(p.relative_to(REPO)) for p in miss))

VARIANTS = ["strip"]  # Field names and available descriptions.

def p2_dir(m, run, v): return OUT / "phase2" / v / m / run

def extract_job(j):
    m, run, v = j
    d = p2_dir(m, run, v)
    if (d / "extractions.jsonl").exists() and sum(1 for _ in open(d / "extractions.jsonl")) >= len(papers(run)): return
    import subprocess
    root, arm = SRC[m]
    cmd = [sys.executable, str(REPO / "baselines/autoschema/run_phase2.py"), "--schema", str(schema_path(m, run)),
           "--papers", str(papers_file(run)), "--ids", "|".join(p["paper_id"] for p in papers(run)), "--out", str(d)]
    cmd.append("--strip")
    d.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(cmd, capture_output=True, text=True); (d / "run.log").write_text(r.stdout[-6000:] + r.stderr[-6000:])

def norm(s): return re.sub(r"\s+", " ", re.sub(r"[^\w%./+-]", " ", (s or "").lower())).strip()

def records(m, run, v):
    f = p2_dir(m, run, v) / "extractions.jsonl"
    R = {}
    if f.exists():
        for l in open(f):
            if l.strip():
                x = json.loads(l); R[x["paper_id"]] = x["extractions"]
    return R

def slots(ex, fields, text):
    out = []
    for name in fields:
        e = ex.get(name) or {}; vals = [str(v) for v in (e.get("values") or []) if str(v).strip() and str(v).strip().upper() != "N/A"]
        ev = e.get("evidence") or []; ev = [x if isinstance(x, str) else (x.get("quote") or "") if isinstance(x, dict) else json.dumps(x, ensure_ascii=False) for x in (ev if isinstance(ev, list) else [ev])]
        filled = bool(vals)
        out.append({"field": name, "value": " | ".join(vals), "quote": " | ".join(ev)[:600], "filled": filled,
                    "grounded": bool(filled and any(norm(q) and norm(q)[:200] in norm(text) for q in ev))})
    return out

def judge_job(j):
    m, run, v = j
    R = records(m, run, v)
    fields = [x["name"] for x in json.load(open(p2_dir(m, run, v) / "schema_used.json"))] if (p2_dir(m, run, v) / "schema_used.json").exists() else []
    for p in papers(run):
        g = OUT / "judge" / v / m / run / f"{p['paper_id'][:150]}.json"
        if g.exists() or p["paper_id"] not in R: continue
        S = slots(R[p["paper_id"]], fields, p["text"]); items = [(i, s) for i, s in enumerate(S) if s["filled"]]
        g.parent.mkdir(parents=True, exist_ok=True)
        if not items:
            g.write_text(json.dumps({"labels": [], "response": "{}"})); continue
        msgs = [{"role": "user", "content": JUDGE.format(items="\n".join(f"[{i}] field: {s['field']} | value: {s['value'][:400]} | quote: {s['quote'][:300]}" for i, s in items))}]
        raw, u = chat(msgs)
        g.write_text(json.dumps({"messages": msgs, "response": raw, "usage": u}, ensure_ascii=False))

def jobs():
    return [(m, run, v) for v in VARIANTS for m in SRC for run in RUNS]

def report():
    import completeness as C                    # 6 methods x 9 r8 corpora x 5 papers (45 per method)
    C.require_files([f"results/memexp36/phase2/{v}/{m}/{r}/{x}" for v in VARIANTS for m in SRC for r in RUNS
                     for x in ("schema_used.json", "extractions.jsonl")]
                    + [f"results/memexp36/judge/{v}/{m}/{r}/{p['paper_id'][:150]}.json" for v in VARIANTS for m in SRC for r in RUNS
                       for p in papers(r)], "extraction records and judgements", "memexp36")
    L = ["# memexp36 report: downstream extraction utility, AutoSchema Phase II (9 r8 corpora x 5 papers, first output)\n"]
    for v in VARIANTS:
        R = defaultdict(lambda: defaultdict(list))
        for m in SRC:
            for run in RUNS:
                Rec = records(m, run, v); su = p2_dir(m, run, v) / "schema_used.json"
                if not su.exists(): continue
                fields = [x["name"] for x in json.load(open(su))]
                for p in papers(run):
                    g = OUT / "judge" / v / m / run / f"{p['paper_id'][:150]}.json"
                    if p["paper_id"] not in Rec or not g.exists(): continue
                    S = slots(Rec[p["paper_id"]], fields, p["text"])
                    try: lab = {int(x["i"]): x["label"] for x in json.loads(json.load(open(g))["response"]).get("labels", [])}
                    except Exception: lab = {}
                    # useful = filled, evidence quote found in the paper text (grounded), judged "specific"; per paper / all fields
                    useful = [i for i, x in enumerate(S) if x["filled"] and x["grounded"] and lab.get(i) == "specific"]
                    R[m][run].append(len(useful) / max(1, len(fields)))
        L += ["\n## Field names and descriptions", "| method | useful-value rate | papers |", "|---|---|---|"]
        for m in SRC:
            u = [st.mean(x) for x in R[m].values() if x]
            if u: L.append(f"| {'MemoSchema (REUSEC_M)' if m == OURS else m} | {st.mean(u):.3f} | {sum(len(x) for x in R[m].values())} |")
        missing = [r for r in RUNS if not R[OURS][r]]
        if missing:
            raise SystemExit(f"[memexp36] no judged extractions of {OURS} for {missing}; run the extract and judge stages")
        L.append(f"\nMemoSchema ({OURS}) - baseline, useful-value rate (paired over corpora, bootstrap stratified by subdomain, 95%):")
        for m in SRC:
            if m == OURS: continue
            D = {r: st.mean(R[OURS][r]) - st.mean(R[m][r]) for r in RUNS if R[OURS][r] and R[m][r]}
            if len(D) < 3: continue
            by = defaultdict(list)
            for r in D: by[r.split("_")[0]].append(r)
            random.seed(0); bs = sorted(st.mean([D[r] for rs in by.values() for r in random.choices(rs, k=len(rs))]) for _ in range(5000))
            L.append(f"- vs {m}: {st.mean(D.values()):+.3f} [{bs[125]:+.3f}, {bs[4874]:+.3f}] ({sum(x > 0 for x in D.values())}/{sum(x < 0 for x in D.values())})")
    (OUT / "report.md").write_text("\n".join(L)); print("\n".join(L))

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("extract", "judge", "report"):
        print(__doc__ if sys.argv[1:2] in (["-h"], ["--help"]) else "usage: python scripts/memexp36.py extract|judge|report")
        sys.exit(0 if sys.argv[1:2] in (["-h"], ["--help"]) else 2)
    stage = sys.argv[1]
    require_inputs()
    if stage in ("extract", "judge"):
        with ThreadPoolExecutor(8) as ex:
            list(ex.map(extract_job if stage == "extract" else judge_job, jobs()))
    else:
        report()
