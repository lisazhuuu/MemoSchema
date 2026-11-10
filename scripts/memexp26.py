"""memexp26: organized/LEGACY construction arms, restructuring, and the no-reuse replay control (27 corpora r7-r9).
Stages (TARGETS = batches, e.g. r7; ARMS default ORIG,ORIG2,ORIG3 = three no-reuse reconstructions):
  build      rebuild final-round canonical candidates per arm (LEGACY must run alone: clustering fix off)
  schemas    anchors + draft (select_final_schema) per arm x anchor sample (LEGACY: 6 anchors, anchors not used as fields)
  merge      coverage-driven restructuring (MERGE prompt) of every non-LEGACY draft -> <arm>_M
Summaries: scripts/paper_tables.py (Table V, via scores/contrast) and scripts/memexp38.py report (replay, via cost).
memexp38.py reuses build/schemas/merge for the content-keyed reuse arm REUSEC."""
from __future__ import annotations
import glob, json, os, random, re, statistics as st, sys, threading, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if __name__ == "__main__" and (len(sys.argv) != 2 or sys.argv[1] not in ('build', 'schemas', 'merge')):                     # usage check before any work
    print(__doc__ if sys.argv[1:2] in (["-h"], ["--help"]) else "usage: python scripts/" + Path(__file__).name + " " + 'build | schemas | merge')
    sys.exit(0 if sys.argv[1:2] in (["-h"], ["--help"]) else 2)
sys.path[:0] = [str(REPO / "Phase1"), str(REPO), str(REPO / "scripts")]
import input_package

OUT = Path(os.environ.get("MEMEXP26_OUT", REPO / "results" / "memexp26"))   # override only for smoke tests
WORKERS = int(os.environ.get("MEMEXP_WORKERS", "8")); CALL_CAP = 100000
BATCHES = ["r3", "r4", "r5", "r6", "r7", "r8", "r9"]
TAG = {"r4": "e2e4", "r5": "e2e5", "r6": "e2e6", "r7": "e2e7", "r8": "e2e8", "r9": "e2e9"}
TARGETS = os.environ.get("TARGETS", "r7").split(",")
SAMPLES = [int(x) for x in os.environ.get("SAMPLES", "1,2,3,4,5,6").split(",")]
OR = ["ORIG", "ORIG2", "ORIG3"]
TIER = {"COF": "Covalent Organic Frameworks (COFs)", "MOF": "Metal-organic frameworks (MOFs)", "ZIF": "Zeolitic imidazolate frameworks (ZIFs)"}

# Coverage-driven restructuring prompt (paper Table I).
MERGE = """Below is a draft schema (8 fields) induced from papers on "{lt}", and the best-supported candidate fields of the
same corpus (name: description (number of papers)). Rewrite the draft into exactly 8 fields that TOGETHER cover as many
different kinds of information reported in these papers as possible:
- merge fields that are near-duplicates or narrow sub-attributes of the same kind of information into ONE umbrella field;
- use the freed slots for well-supported kinds of information that the draft does not cover;
- every field's description must list the kinds of values it covers.
Use only kinds of information that appear in the draft or the candidate list. For every output field list the input
field/candidate names it covers in "merged_from".
Return ONLY JSON: {{"fields": [{{"name": "snake_case", "description": "...", "merged_from": ["..."]}}]}}

DRAFT SCHEMA:
{draft}

CANDIDATE FIELDS:
{cands}"""

# LLM call with cap and full logging (calls saved under OUT/calls)
_lock = threading.Lock()
_calls = {"n": 0, "prompt_chars": 0, "response_chars": 0, "failures": 0}

def chat(tag, prompt):
    from schema_induction import llm_chat
    msgs = [{"role": "system", "content": "Return ONLY valid JSON. No markdown."},
            {"role": "user", "content": prompt}]
    for attempt in range(2):
        with _lock:
            if _calls["n"] >= CALL_CAP:
                raise RuntimeError(f"call cap {CALL_CAP} reached")
            _calls["n"] += 1; _calls["prompt_chars"] += len(prompt)
        try:
            raw = llm_chat(msgs, force_json=True)
            with _lock:
                _calls["response_chars"] += len(raw)
            f = OUT / "calls" / f"{tag}.json"; f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps({"prompt": prompt, "response": raw}, ensure_ascii=False, indent=1))
            return json.loads(raw)
        except Exception as e:
            with _lock:
                _calls["failures"] += 1
            print(f"[memexp26] {tag} failed ({attempt + 1}/2): {type(e).__name__}: {e}", flush=True)
            time.sleep(15)
    return None

def save_cost(stage):
    f = OUT / "cost.json"
    with _lock:
        c = json.loads(f.read_text()) if f.exists() else {}
        c[stage] = dict(_calls, est_tokens=(_calls["prompt_chars"] + _calls["response_chars"]) // 4)
        f.write_text(json.dumps(c, indent=2))

def pmap(fn, jobs):
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for f in [ex.submit(fn, j) for j in jobs]:
            try:
                f.result()
            except Exception as e:
                print(f"[memexp26] job failed: {type(e).__name__}: {e}", flush=True)

# Run directories and candidate rebuild with optional cross-run reuse
def run_dir(b, run):
    input_package.check("replay", "memexp26")      # minimal input package: complete, in scope, quotes restored
    hits = sorted(glob.glob(str(REPO / f"results/e2e/e2e_v3/{b}/chem/{run}/iterative_phase1_chem_*")))
    return Path(hits[-1]) if hits else None

def last_tier(rd):
    """The run's own last_tier, as recorded in its extraction prompt."""
    for f in sorted(rd.glob("round0*/schema_from_cumulative_corpus/extraction_prompt.txt")):
        m = re.search(r'information about "([^"]+)"', f.read_text(encoding="utf-8"))
        if m:
            return m.group(1)
    raise RuntimeError(f"cannot recover last_tier for {rd}")

def rows(rd):
    return [json.loads(l) for l in open(rd / "schema_state" / "paper_level_field_proposals.jsonl") if l.strip()]

def mkey(members):
    """Reuse key of a cluster-consolidation result (memexp38 replaces it by a content key)."""
    return frozenset((m.get("paper_id", ""), m.get("name", "")) for m in members)

def caches(target_batch, run):
    """paper cache (subdomain-level) and merge cache from batches before target_batch + earlier rounds of this run."""
    sub = run.split("_")[0]; P, M = {}, {}
    for b in BATCHES[: BATCHES.index(target_batch)]:
        for r in runs_of(b):
            if r.split("_")[0] != sub:
                continue
            rd = run_dir(b, r)
            for row in rows(rd):
                P.setdefault(row["paper_id"], row)
            for f in sorted(glob.glob(str(rd / "round0*/schema_from_cumulative_corpus/merged_field_clusters.json"))):
                for cl in json.load(open(f)):
                    M.setdefault(mkey(cl["members"]), cl["llm_merged"])
    rd = run_dir(target_batch, run)
    for r in ("round01", "round02"):
        f = rd / r / "schema_from_cumulative_corpus" / "merged_field_clusters.json"
        if f.exists():
            for cl in json.load(open(f)):
                M.setdefault(mkey(cl["members"]), cl["llm_merged"])
    return P, M

_hook_lock = threading.Lock()
_CUR = {"state": None}          # one build at a time per process; merges inside it run in threads

def install_merge_hook():
    import schema_lib.schema_cluster_merge as scm
    orig = scm._chunked_llm_merge
    def hooked(members, domain, cid):
        s = _CUR["state"]
        if s is not None and s["cache"] is not None:
            k = mkey(members)
            if k in s["cache"]:
                with _hook_lock:
                    s["hits"] += 1
                return s["cache"][k]
        if s is not None:
            with _hook_lock:
                s["calls"] += 1
        return orig(members, domain=domain, cid=cid)
    scm._chunked_llm_merge = hooked

def build(job):
    b, run, arm = job
    od = OUT / "canon" / arm / TAG[b] / run
    if (od / "canonical_schema_candidates.json").exists():
        return
    from schema_lib.schema_cluster_merge import build_canonical_candidates
    od.mkdir(parents=True, exist_ok=True)
    rd = run_dir(b, run); R = rows(rd); P, M = caches(b, run) if arm.startswith("REUSE") else ({}, None)   # REUSEC (memexp38)
    replaced = 0
    if arm.startswith("REUSE"):
        new = []
        for r in R:
            if r["paper_id"] in P:
                new.append(P[r["paper_id"]]); replaced += 1
            else:
                new.append(r)
        R = new
    with open(od / "paper_level_field_proposals.jsonl", "w") as f:
        for r in R:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    state = {"cache": M, "hits": 0, "calls": 0}
    _CUR["state"] = state
    canon, stats = build_canonical_candidates(paper_level_jsonl=od / "paper_level_field_proposals.jsonl", out_dir=od, domain="chem")
    stats.update({"papers": len(R), "papers_reused": replaced, "merge_llm_calls": state["calls"], "merge_cache_hits": state["hits"]})
    (od / "build_stats.json").write_text(json.dumps(stats, indent=1))
    print(f"[build] {arm} {TAG[b]} {run}: papers reused {replaced}/{len(R)}, merges {state['calls']} LLM / {state['hits']} cached", flush=True)

def kept_of(canon):
    """Eligible candidates (>= 2 papers, not paper-specific), deduplicated by name, by support."""
    seen, out = set(), []
    for c in sorted(canon, key=lambda c: -int(c.get("support_count", 0))):
        if (c.get("scope") or "global") == "paper_specific" or int(c.get("support_count", 0)) < 2 or c["name"] in seen:
            continue
        seen.add(c["name"]); out.append(c)
    return out

def done(b, run):
    rd = run_dir(b, run)
    return rd is not None and (rd / "final_schema_from_cumulative_corpus" / "final_schema.json").exists()

def runs_of(b):
    input_package.check("replay", "memexp26")
    root = REPO / f"results/e2e/e2e_v3/{b}/chem"
    return sorted(p.name for p in root.iterdir() if done(b, p.name)) if root.exists() else []

def legacy():
    return os.environ.get("MEMOSCHEMA_FIX_CLUSTER") != "1"

def schema_job(job):
    b, run, arm, smp = job
    od = OUT / "runs" / TAG[b] / f"s{smp}" / arm / "chem" / run
    if (od / "final_schema.json").exists():
        return
    from schema_lib.subdomain_discovery import discover_subdomain_anchors
    from schema_lib.schema_cluster_merge import select_final_schema
    from schema_induction import llm_chat
    cd = OUT / "canon" / arm / TAG[b] / run; lt = last_tier(run_dir(b, run))
    af = OUT / "anchors" / arm / TAG[b] / f"s{smp}" / run / "discovered_anchors.json"; af.parent.mkdir(parents=True, exist_ok=True)
    anchors = json.loads(af.read_text())["anchors"] if af.exists() else discover_subdomain_anchors(
        llm_chat_fn=lambda msgs, force_json=False: llm_chat(msgs, force_json=True),
        paper_level_jsonl=cd / "paper_level_field_proposals.jsonl", domain="chem", last_tier=lt,
        n_anchors=6 if arm == "LEGACY" else 8, out_path=af)
    od.mkdir(parents=True, exist_ok=True)
    select_final_schema(canonical_candidates=json.loads((cd / "canonical_schema_candidates.json").read_text()), out_dir=od,
                        domain="chem", last_tier=lt, max_final_fields=8, min_support=2, anchors=anchors)

def merge_fields(name, G, canon, sub):
    """Restructure draft G with the top-25 eligible candidates not already in the draft (paper Eq. 4)."""
    kept = kept_of(canon); by = {c["name"]: c for c in kept}; by.update({x["name"]: x for x in G})
    cands = [c for c in kept if c["name"] not in {x["name"] for x in G}][:25]
    out = chat(name, MERGE.format(lt=TIER[sub],
                  draft="\n".join(f"- {x['name']}: {x.get('description', '')[:160]} ({int(x.get('support_count', 0))})" for x in G),
                  cands="\n".join(f"- {c['name']}: {c.get('description', '')[:160]} ({int(c.get('support_count', 0))})" for c in cands))) or {}
    fields = []
    for f in (out.get("fields") or [])[:8]:
        src = [by[n] for n in f.get("merged_from") or [] if n in by]
        papers = sorted({p for c in src for p in (c.get("supporting_paper_ids") or []) if isinstance(c.get("supporting_paper_ids"), list)})
        aliases = list(dict.fromkeys([c["name"] for c in src] + [a for c in src for a in (c.get("aliases") or []) if isinstance(a, str)]))[:30]
        fields.append({"name": f.get("name"), "description": f.get("description", ""), "value_type": "string", "scope": "subdomain",
                       "aliases": aliases, "merged_from": f.get("merged_from") or [], "supporting_paper_ids": papers,
                       "support_count": max([len(papers)] + [int(c.get("support_count", 0)) for c in src]),
                       "evidence_snippets": [e for c in src for e in (c.get("evidence_snippets") or [])][:6]})
    return fields

def write_merged(od, fields):
    if len(fields) >= 6:
        od.mkdir(parents=True, exist_ok=True); (od / "final_schema.json").write_text(json.dumps(fields, ensure_ascii=False, indent=2))
    save_cost("merge")

def merge_job(job):
    b, run, arm, smp = job
    tag = TAG[b]; od = OUT / f"runs/{tag}/s{smp}/{arm}_M/chem/{run}"; gf = OUT / f"runs/{tag}/s{smp}/{arm}/chem/{run}/final_schema.json"
    if (od / "final_schema.json").exists() or not gf.exists():
        return
    write_merged(od, merge_fields(f"merge_{tag}_{run}_{arm}_s{smp}", json.loads(gf.read_text()),
                                  json.loads((OUT / f"canon/{arm}/{tag}/{run}/canonical_schema_candidates.json").read_text()), run.split("_")[0]))

# Report
def scores(fine=True):
    S = {}
    pat = "runs/e2e*/s*/eval_fine/*/rep*/chem/*/metrics_fine.json" if fine else "runs/e2e*/s*/eval/*/rep*/chem/*/metrics.csv"
    for f in glob.glob(str(OUT / pat)):
        p = Path(f).parts; i = p.index("runs")
        v = json.load(open(f))["fine_f1"] if fine else float(open(f).read().splitlines()[1].split(",")[11])
        S.setdefault((p[i + 1], p[i + 7], p[i + 4], int(p[i + 2][1:])), []).append(v)
    return {k: st.mean(v) for k, v in S.items()}

def contrast(S, runs, A, B, draws=5000):
    val = lambda r, arms, smps: (lambda v: st.mean(v) if v else None)([S[(r[0], r[1], a, s)] for a in arms for s in smps if (r[0], r[1], a, s) in S])
    diff = lambda r, a, b, smps: None if None in (x := val(r, a, smps), y := val(r, b, smps)) else x - y
    D = {r: d for r in runs if (d := diff(r, A, B, SAMPLES)) is not None}
    if not D:
        return None
    by = defaultdict(list)
    for r in D: by[r[1].split("_")[0]].append(r)
    random.seed(0); bs = []
    for _ in range(draws):
        v = [z for rs in by.values() for r in random.choices(rs, k=len(rs))
             if (z := diff(r, random.choices(A, k=len(A)), random.choices(B, k=len(B)), random.choices(SAMPLES, k=len(SAMPLES)))) is not None]
        if v: bs.append(st.mean(v))
    bs.sort()
    return {"n": len(D), "mean": st.mean(D.values()), "lo": bs[int(.025 * len(bs))], "hi": bs[int(.975 * len(bs)) - 1],
            "up": sum(v > 0 for v in D.values()), "down": sum(v < 0 for v in D.values()),
            "sub": {s: round(st.mean(D[r] for r in rs), 3) for s, rs in by.items()}}

def cost():
    C = defaultdict(lambda: defaultdict(list))
    for f in glob.glob(str(OUT / "canon/*/*/*/build_stats.json")):
        arm, tag = f.split("/canon/")[1].split("/")[:2]; s = json.load(open(f))
        C[tag][arm].append({"calls": 2 * (s["papers"] - s.get("papers_reused", 0)) + s["merge_llm_calls"] + 2 * len(SAMPLES),
                            "share": s.get("papers_reused", 0) / max(1, s["papers"])})
    return C

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    stage = sys.argv[1]
    os.environ["MEMOSCHEMA_FIX_ANCHOR_SCHEMA"] = "1"; os.environ["MEMOSCHEMA_FIX_CLUSTER"] = "1"
    ARMS = os.environ.get("ARMS", "ORIG,ORIG2,ORIG3").split(",")
    if "LEGACY" in ARMS:                        # original configuration: separate process, fixes off
        assert ARMS == ["LEGACY"], "LEGACY must run alone"
        for k in ("MEMOSCHEMA_FIX_ANCHOR_SCHEMA", "MEMOSCHEMA_FIX_CLUSTER"):
            os.environ.pop(k)
    only = os.environ.get("ONLY_RUNS", "").split(",")                      # smoke tests only
    jobs = [(b, r, arm) for b in TARGETS for r in runs_of(b) for arm in ARMS if only == [""] or r in only]
    if stage == "build":
        install_merge_hook()
        i, n = (int(x) for x in os.environ.get("SHARD", "0/1").split("/"))
        for j in jobs[i::n]:
            build(j)
    elif stage == "schemas":
        pmap(schema_job, [j + (s,) for j in jobs for s in SAMPLES])
    elif stage == "merge":
        pmap(merge_job, [j + (s,) for j in jobs for s in SAMPLES if j[2] != "LEGACY"])
