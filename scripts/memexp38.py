"""memexp38: final-round replay with content-addressed reuse (arm REUSEC -> REUSEC_M, the MemoSchema configuration).
Stages: build | schemas | merge | report (same code paths as memexp26; only the reuse keys change)."""
import json, os, statistics as st, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
if __name__ == "__main__" and (len(sys.argv) != 2 or sys.argv[1] not in ('build', 'schemas', 'merge', 'report')):                     # usage check before any work
    print(__doc__ if sys.argv[1:2] in (["-h"], ["--help"]) else "usage: python scripts/" + Path(__file__).name + " " + 'build | schemas | merge | report')
    sys.exit(0 if sys.argv[1:2] in (["-h"], ["--help"]) else 2)
os.environ.setdefault("MEMEXP26_OUT", str(REPO / "results/memexp38"))
sys.path[:0] = [str(REPO / "scripts"), str(REPO / "Phase1")]
import memexp26 as m26

def _content_key(members):                                               # content key (was: set of (paper_id, name))
    import schema_lib.schema_cluster_merge as scm                        # imported lazily: needs API configuration
    return tuple(scm.merge_member_content(members))
m26.mkey = _content_key

TEXT_BATCHES = ("r7", "r8", "r9", "r10")   # frozen texts existed for these batches in the paper's replay; r3-r6 had none

def _texts(b, run):
    f = REPO / f"results/evaporate/inputs/{b}/{run}/papers.jsonl"
    if not f.exists():
        if b in TEXT_BATCHES:                 # without them the content check would silently be skipped
            raise SystemExit(f"[m38] missing {f.relative_to(REPO)}; prepare it offline from the run's PDFs with\n  python "
                             f"baselines/evaporate/reproduce.py prepare --run-dir <{b} {run} run dir> --out results/evaporate/inputs/{b}/{run}\n"
                             "  (minimal input package: the texts are not included; MemoSchema-inputs/frozen_texts_r7-r10.tsv lists the "
                             "expected SHA-256, checked by tools/inputs_tool.py texts)")
        return {}
    return {json.loads(l)["paper_id"]: json.loads(l)["text"] for l in open(f)}

_orig_caches = m26.caches
def caches(target_batch, run):
    """memexp26.caches with paper rows dropped when source and target texts are both known and differ."""
    sub = run.split("_")[0]; src = {}
    for b in m26.BATCHES[: m26.BATCHES.index(target_batch)]:
        for r in m26.runs_of(b):
            if r.split("_")[0] != sub: continue
            for row in m26.rows(m26.run_dir(b, r)):
                src.setdefault(row["paper_id"], (b, r))
    P, M = _orig_caches(target_batch, run)
    tgt = _texts(target_batch, run); dropped = 0; srct = {}
    for pid, (b, r) in src.items():
        if pid not in P or pid not in tgt: continue
        if (b, r) not in srct: srct[(b, r)] = _texts(b, r)
        if pid in srct[(b, r)] and srct[(b, r)][pid] != tgt[pid]:
            P.pop(pid); dropped += 1
    print(f"[m38] {target_batch} {run}: paper rows dropped for differing text: {dropped}", flush=True)
    return P, M
m26.caches = caches

def report():
    from memexp27 import scores
    import completeness as C
    C.require_scores([("results/memexp38", "REUSEC_M")] + [("results/memexp26", a) for a in ("ORIG_M", "ORIG2_M", "ORIG3_M")], "memexp38")
    C.require_files([f"results/memexp38/canon/REUSEC/{t}/{r}/build_stats.json" for t, r in C.CORP]
                    + [f"results/memexp26/canon/{a}/{t}/{r}/build_stats.json" for a in m26.OR for t, r in C.CORP],
                    "final-round build statistics (call accounting of Fig. 3a)", "memexp38")
    R26, R38 = REPO / "results/memexp26", REPO / "results/memexp38"
    L = ["# memexp38 report: content-keyed reuse (REUSEC_M) vs recomputation (ORIG_M x 3), 27 corpora\n"]
    for fine in (True, False):
        S = {}
        for root in (R26, R38):
            m26.OUT = root; S.update(m26.scores(fine))
        m26.OUT = R26
        runs = sorted({(k[0], k[1]) for k in S if k[0] in ("e2e7", "e2e8", "e2e9")})
        p = m26.contrast(S, runs, ["REUSEC_M"], ["ORIG_M", "ORIG2_M", "ORIG3_M"])
        mean = lambda arms: st.mean(v for k, v in S.items() if k[2] in arms and (k[0], k[1]) in runs)
        nm = "hierarchical F1" if fine else "strict one-to-one F1"
        L.append(f"## {nm}: REUSEC_M {mean(['REUSEC_M']):.3f}, ORIG_M pooled {mean(['ORIG_M', 'ORIG2_M', 'ORIG3_M']):.3f}")
        if p:
            vd = "non-inferior" if p["lo"] > -.02 else "inferior" if p["hi"] < -.02 else "inconclusive"
            L.append(f"- REUSEC_M - ORIG_M: {p['mean']:+.4f} [{p['lo']:+.4f}, {p['hi']:+.4f}] {p['up']}/{p['down']} {p['sub']}"
                     + (f"\n- non-inferiority test (hierarchical F1, margin 0.02: lower 95% bound > -0.02): {vd}" if fine else ""))
    C26, C38 = m26.cost(), None
    m26.OUT = R38; C38 = m26.cost(); m26.OUT = R26
    o = [x["calls"] for t in C26 for a in m26.OR for x in C26[t].get(a, [])]
    c = [x["calls"] for t in C38 for x in C38[t].get("REUSEC", [])]
    if o and c:
        L.append(f"\n## calls per run (final-round rebuild): ORIG {st.mean(o):.0f}, "
                 f"REUSEC {st.mean(c):.0f} (saved {1 - st.mean(c) / st.mean(o):.1%})")
    (REPO / "results/memexp38/report.md").write_text("\n".join(L)); print("\n".join(L))

if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "report":
        report(); sys.exit()
    m26.OUT.mkdir(parents=True, exist_ok=True)
    os.environ["MEMOSCHEMA_FIX_ANCHOR_SCHEMA"] = "1"; os.environ["MEMOSCHEMA_FIX_CLUSTER"] = "1"
    targets = os.environ.get("TARGETS", "r7,r8,r9").split(",")
    only = os.environ.get("ONLY_RUNS", "").split(",")
    jobs = [(b, r, "REUSEC") for b in targets for r in m26.runs_of(b) if only == [""] or r in only]
    if stage == "build":
        m26.install_merge_hook()
        i, n = (int(x) for x in os.environ.get("SHARD", "0/1").split("/"))
        for j in jobs[i::n]:
            m26.build(j)
    elif stage == "schemas":
        m26.pmap(m26.schema_job, [j + (s,) for j in jobs for s in m26.SAMPLES])
    elif stage == "merge":
        m26.pmap(m26.merge_job, [j + (s,) for j in jobs for s in m26.SAMPLES])
