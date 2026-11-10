"""Audit of the cross-run memory reuse in the live r10 runs (memexp33) against the content-addressed keys (no LLM calls).
Feeds the audit-adjusted accounting of scripts/memexp39_report.py (paper Fig. 3b). The live runs looked papers up by
paper id and merges by the set of (paper_id, field name). The content-addressed keys (Phase1/iterative_lib/schema.py paper_memo_fingerprint,
Phase1/schema_lib/schema_cluster_merge.py merge_cache_key) also require identical input content. This script checks,
for every reuse that happened, whether it also satisfies the content condition:
  paper  the paper text read by the operator (first 12,000 chars) is identical in the source and the target run
         (exact string comparison where both texts are available; otherwise every evidence quote the source distill
         produced must occur in the target text, with the same check on freshly distilled papers as a control)
  merge  the members seen by the merge prompt are identical (same multiset of name, description, value type, scope,
         aliases, paper id, example values, evidence snippets) in the reused source cluster and the target cluster.
Operator code identity across batches is not observable from the artifacts and is stated separately.
Output: results/memo_audit/report.md, details.json"""
import glob, json, re, statistics as st, sys
from collections import defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]; OUT = REPO / "results/memo_audit"; E = REPO / "results/e2e/e2e_v3"
if __name__ == "__main__" and (sys.argv[1:]):                     # usage check before any work
    print(__doc__ if sys.argv[1:2] in (["-h"], ["--help"]) else "usage: python scripts/" + Path(__file__).name + " " + "(no arguments)")
    sys.exit(0 if sys.argv[1:2] in (["-h"], ["--help"]) else 2)
sys.path[:0] = [str(REPO / "Phase1")]
import schema_lib.schema_cluster_merge as scm
from queries_lib.paper_model import pdf_to_text
BATCHES = ["r3", "r4", "r5", "r6", "r7", "r8", "r9", "r10"]
norm = lambda s: re.sub(r"\s+", " ", (s or "")).strip().lower()

def run_dir(b, run):
    import input_package; input_package.check("replay", "memo_audit")   # minimal input package: out of scope, stop
    d = sorted(glob.glob(str(E / f"{b}/chem/{run}/iterative_phase1_chem_*")))
    return Path(d[-1]) if d else None

def runs(b): import input_package; input_package.check("replay", "memo_audit"); return sorted(Path(p).parts[-2] for p in glob.glob(str(E / f"{b}/chem/*/iterative_phase1_chem_*")))

def debug_dirs(rd): return sorted(rd.glob("round0*/schema_from_cumulative_corpus/paper_level_debug"))

def quotes(rd, pid):
    for d in debug_dirs(rd):
        f = d / f"{pid}.distill.raw.json"
        if f.exists():
            try: x = json.loads(f.read_text())
            except Exception: return None
            return [q for kp in x.get("key_points", []) for q in (kp.get("quotes") or []) if isinstance(q, str) and q.strip()]
    return None

INPUT_TEXT = {}
def input_texts():
    if not INPUT_TEXT:
        # The r7-r9 input sets only, as when the audit was run (r10's own frozen texts did not exist yet and must not stand
        # in for a source). A paper id found in several input sets takes the text of its earliest set, in sorted path
        # order (r7 < r8 < r9, then subdomain and corpus); 3 paper ids have two different r7-r9 texts. The r10 PDFs must
        # be read with PyMuPDF 1.26.4, the version the r7-r9 texts were extracted with (README, environments).
        for f in sorted(glob.glob(str(REPO / "results/evaporate/inputs/r[789]/*/papers.jsonl"))):
            for l in open(f):
                x = json.loads(l); INPUT_TEXT.setdefault(x["paper_id"], x["text"])
    return INPUT_TEXT

def quote_rate(qs, text):
    t = norm(text)
    return sum(norm(q) in t for q in qs) / len(qs) if qs else None

# Paper layer
def audit_paper_r10():
    T = input_texts(); out = []; control = []
    for run in runs("r10"):
        rd = run_dir("r10", run); sub = run.split("_")[0]
        xdirs = [Path(d) for d in sorted(glob.glob(str(E / f"r[3-9]/chem/{sub}_*/iterative_phase1_chem_*/schema_state/distill_cache")))]
        distilled_here = {f.name[: -len(".distill.raw.json")] for d in debug_dirs(rd) for f in d.glob("*.distill.raw.json")}
        for f in sorted((rd / "schema_state/distill_cache").glob("*.json")):
            pid = f.stem; pdf = rd / "cumulative_corpus" / f"{pid}.pdf"
            if not pdf.exists(): pdf = next(rd.rglob(f"{pid}.pdf"), None)
            if pdf is None:                     # every reused or distilled r10 paper is audited; no silent skip
                raise SystemExit(f"[memo_audit] missing PDF of r10 {run} paper {pid}; the audit needs the r10 corpus PDFs")
            text = pdf_to_text(pdf)[:12000]
            if pid in distilled_here:
                qs = quotes(rd, pid)
                if qs: control.append(quote_rate(qs, text))
                continue
            src = next((d for d in xdirs if (d / f"{pid}.json").exists()), None)
            if src is None: continue
            srd = src.parents[1]; qs = quotes(srd, pid)
            out.append({"run": run, "paper": pid, "source": str(srd.relative_to(E)),
                        "exact_text_available": pid in T, "text_identical": (T[pid] == text) if pid in T else None,
                        "quote_rate": quote_rate(qs, text) if qs else None,
                        "cache_identical": f.read_bytes() == (src / f"{pid}.json").read_bytes()})
    return out, control

# Merge layer
def cluster_index(batches, sub):
    """v1 key -> list of member-content multisets of all source clusters with that key."""
    I = defaultdict(list)
    for b in batches:
        for r in runs(b):
            if r.split("_")[0] != sub: continue
            for f in sorted(glob.glob(str(run_dir(b, r) / "round0*/schema_from_cumulative_corpus/merged_field_clusters.json"))):
                for cl in json.load(open(f)):
                    if scm.support_count(cl["members"]) >= 2:
                        I[scm.merge_cache_key_v1(cl["members"], "chem")].append(scm.merge_member_content(cl["members"]))
    return I

RUN_TZ = ZoneInfo("America/Chicago")   # run directories are named with the local time of the machine that ran them
def run_start(rd):
    """Start time of a run from its directory name, read in the time zone the runs were made in (explicit, so the
    comparison with store mtimes does not depend on the time zone of the machine running the audit)."""
    return datetime.strptime(re.search(r"(\d{8}_\d{6})", rd.name).group(1), "%Y%m%d_%H%M%S").replace(tzinfo=RUN_TZ).timestamp()

def audit_merge_r10():
    out = []; idx = {}
    for run in runs("r10"):
        sub = run.split("_")[0]; rd = run_dir("r10", run)
        if sub not in idx: idx[sub] = cluster_index(BATCHES[:-1], sub)
        start = run_start(rd)
        store = REPO / "results/memory_store/merge"
        if not store.is_dir():                  # hits are decided by store files and their mtimes; never report 0 hits silently
            raise SystemExit("[memo_audit] results/memory_store/merge is missing; the merge audit needs the store entries "
                             "with their original modification times (pending audit component)")
        for f in sorted(rd.glob("round0*/schema_from_cumulative_corpus/merged_field_clusters.json")):
            for cl in json.load(open(f)):
                if scm.support_count(cl["members"]) < 2: continue
                k = scm.merge_cache_key_v1(cl["members"], "chem"); p = store / f"{k}.json"
                if not (p.exists() and p.stat().st_mtime < start): continue        # hit from earlier batches
                c = scm.merge_member_content(cl["members"])
                out.append({"run": run, "content_identical": any(c == s for s in idx[sub].get(k, [])), "source_found": k in idx[sub]})
    return out

def summarise(L, key):
    v = [x[key] for x in L if x.get(key) is not None]
    return f"{sum(v)}/{len(v)} ({sum(v) / max(1, len(v)):.1%})"

def main():
    OUT.mkdir(parents=True, exist_ok=True); R = ["# Memory audit: reuse in the reported experiments vs content-addressed keys\n"]
    pr, ctrl = audit_paper_r10()
    qr = [x["quote_rate"] for x in pr if x["quote_rate"] is not None]
    R += ["## Paper layer, live r10 (memexp33)",
          f"- reused papers: {len(pr)}; cache file copied byte-identical: {summarise(pr, 'cache_identical')}",
          f"- exact text available (paper also in an r7-r9 input set): {sum(x['exact_text_available'] for x in pr)}; text identical: {summarise(pr, 'text_identical')}",
          f"- evidence-quote check: mean share of source quotes found in the r10 text {st.mean(qr):.3f} (n={len(qr)}); "
          f"all quotes found {sum(q == 1 for q in qr)}/{len(qr)}; < 50% found {sum(q < .5 for q in qr)}",
          f"- control (papers distilled in r10, own quotes vs own text): mean {st.mean(ctrl):.3f} (n={len(ctrl)}); all found {sum(q == 1 for q in ctrl)}/{len(ctrl)}"]
    mr = audit_merge_r10()
    R += ["\n## Merge layer, live r10", f"- hits from earlier batches: {len(mr)}; member content identical to a source cluster: {summarise(mr, 'content_identical')}"]
    # Cost under the content-addressed keys: every reuse that fails the content check becomes an LLM call
    c = json.load(open(REPO / "results/cost_compare.json")); ks = [k for k in c if k.startswith("e2e")]
    per = lambda stage: [sum(c[k][stage][i] for k in ks) / max(1, sum(c[k][stage][0] for k in ks)) for i in (1, 2)]
    usd = lambda tin, tout: tin * 2.5e-6 + tout * 15e-6
    pin, pout = per("per_paper"); min_, mout = per("merge")
    paper_call, merge_call = usd(pin, pout), usd(min_, mout)
    calls_per_paper = sum(c[k]["per_paper"][0] for k in ks) / max(1, sum(1 for k in ks))
    n10 = len(runs("r10"))
    bad_paper = sum(1 for x in pr if x["text_identical"] is False)
    bad_merge = sum(1 for x in mr if not x["content_identical"])
    ppc = 2.0                                   # LLM calls per processed paper (distill + proposal)
    extra_calls = (ppc * bad_paper + bad_merge) / n10; extra_usd = (ppc * bad_paper * paper_call + bad_merge * merge_call) / n10
    R += ["\n## Cost under the content-addressed keys (failed checks recomputed as LLM calls)",
          f"- per-call cost from the no-memory runs: per-paper call ${paper_call:.4f}, merge call ${merge_call:.4f}",
          f"- r10: +{extra_calls:.1f} calls and +${extra_usd:.2f} per corpus (papers with differing text: {bad_paper}; "
          f"merges with differing members: {bad_merge})"]
    json.dump({"paper_r10": pr, "control_r10": ctrl, "merge_r10": mr}, open(OUT / "details.json", "w"), indent=1)
    (OUT / "report.md").write_text("\n".join(R)); print("\n".join(R))

if __name__ == "__main__":
    main()
