"""Cross-family judge check: re-score one sample per (method, corpus) with a Gemini judge (same prompts, same scripts)
and compare with the project's judge (Azure gpt-5.4).
Stages:  models  (list Gemini model ids visible to the key)
         run     (needs GEMINI_API_KEY in .env and GEMINI_JUDGE_MODEL; 7 methods x 27 corpora x {fine, coarse})
         report
Methods (first output, sample 1; rep 1): MemoSchema REUSEC_M (memexp38), LEGACY, AutoSchema, LLMs4SD, ArxivDIGESTables (topic),
EVAPORATE full, IA_SR (paper Fig. 4)."""
import csv, glob, json, os, random, statistics as st, subprocess, sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]; OUT = REPO / "results/judge_crosscheck"
if __name__ == "__main__" and (len(sys.argv) != 2 or sys.argv[1] not in ('models', 'run', 'report')):                     # usage check before any work
    print(__doc__ if sys.argv[1:2] in (["-h"], ["--help"]) else "usage: python scripts/" + Path(__file__).name + " " + 'models | run | report')
    sys.exit(0 if sys.argv[1:2] in (["-h"], ["--help"]) else 2)
SRC = {"LEGACY": ("results/memexp26", "LEGACY"),
       "AUTOSCHEMA": ("results/autoschema", "AUTOSCHEMA"), "L4S": ("results/memexp27", "L4S"),
       "ADT_topic": ("results/arxivdigestables", "ADT_topic"), "EVAPORATE_FULL8": ("results/evaporate", "EVAPORATE_FULL8"),
       "REUSEC_M": ("results/memexp38", "REUSEC_M"), "IA_SR": ("results/memexp40", "IA_SR")}
OURS = "REUSEC_M"                 # final system (content-addressed memory keys)
SUB = {"COF": "COF", "MOF": "MOF", "ZIF": "ZIF"}

def jobs():
    J = []
    for m, (root, arm) in SRC.items():
        for f in sorted(glob.glob(str(REPO / root / f"runs/e2e*/s1/{arm}/chem/*/final_schema.json"))):
            p = Path(f).parts; tag, run = p[p.index("runs") + 1], p[-2]
            if tag not in ("e2e7", "e2e8", "e2e9"):        # evaluation scope: the 27 fresh corpora only
                continue
            J.append((m, tag, run, f, root, arm))
    # corpus-major order: all seven methods of one corpus before the next
    order = list(SRC)
    return sorted(J, key=lambda j: (j[1], j[2], order.index(j[0])))

def failed(d):
    """A coarse output from the earlier judge code may be a failure placeholder; it is not a finished score."""
    sys.path.insert(0, str(REPO / "scripts")); import completeness as C
    return C.failed_judgement(Path(d).relative_to(REPO) / "metrics.csv")

def run_one(j):
    m, tag, run, schema, root, arm = j
    env = {**os.environ, "JUDGE_PROVIDER": "gemini"}
    od = OUT / "gemini" / m / tag / run
    for kind, script, args in (("fine", "Evaluation/evaluate_fine.py", ["--subdomain", run.split("_")[0]]),
                               ("coarse", "Evaluation/evaluate_against_ontology.py", ["--subdomain", SUB[run.split("_")[0]]])):
        d = od / kind
        done = (d / "metrics_fine.json").exists() if kind == "fine" else (d / "metrics.csv").exists() and not failed(d)
        if done:
            continue
        d.mkdir(parents=True, exist_ok=True)
        import time; time.sleep(float(os.getenv("THROTTLE_S", "12")))      # stay well under the per-minute limit
        r = subprocess.run([sys.executable, script, "--schema", schema, "--out-dir", str(d)] + args, cwd=REPO, env=env,
                           capture_output=True, text=True)
        log = r.stdout[-4000:] + r.stderr[-4000:]
        if "per_model_per_day" in log or "PerDay" in log:
            import shutil; shutil.rmtree(d, ignore_errors=True)
            (OUT / "DAILY_QUOTA_HIT").write_text(time.ctime()); raise SystemExit("daily quota reached")
        if "Error code" in log or "FAILED" in log or r.returncode != 0:
            # an API failure must not be scored as "no match": both judge scripts exit non-zero and write no score
            import shutil; shutil.rmtree(d, ignore_errors=True)
            (od / f"{kind}.FAILED.log").parent.mkdir(parents=True, exist_ok=True); (od / f"{kind}.FAILED.log").write_text(log)
            if "503" in log: time.sleep(600)          # model overloaded: back off instead of burning the daily quota
            continue
        (d / "judge.log").write_text(log)

def labels_fine(path):
    return {r["concept"]: r["strength"] for r in json.load(open(path))} if path and Path(path).exists() else {}

def labels_coarse(d):
    out = {}
    for f in glob.glob(str(Path(d) / "alignments/*.csv")):
        for r in csv.DictReader(open(f)):
            if r.get("reference_name"):
                out[r["reference_name"]] = r.get("strength", "")
    return out

def report():
    sys.path.insert(0, str(REPO / "scripts"))
    import completeness as C
    C.require_files([p for m, (root, arm) in SRC.items() for t, r in C.CORP for p in (
        f"results/judge_crosscheck/gemini/{m}/{t}/{r}/fine/alignment_fine.json", f"results/judge_crosscheck/gemini/{m}/{t}/{r}/fine/metrics_fine.json",
        f"results/judge_crosscheck/gemini/{m}/{t}/{r}/coarse/metrics.csv", f"results/judge_crosscheck/gemini/{m}/{t}/{r}/coarse/alignments/{r[:3]}.csv",
        f"{root}/runs/{t}/s1/eval_fine/{arm}/rep1/chem/{r}/alignment_fine.json", f"{root}/runs/{t}/s1/eval_fine/{arm}/rep1/chem/{r}/metrics_fine.json",
        f"{root}/runs/{t}/s1/eval/{arm}/rep1/chem/{r}/metrics.csv", f"{root}/runs/{t}/s1/eval/{arm}/rep1/chem/{r}/alignments/{r[:3]}.csv")],
        "Gemini and project-judge outputs (7 methods x 27 corpora, first output)", "judge_crosscheck", check_valid=False)
    C.require_valid([f"{root}/runs/{t}/s1/eval/{arm}/rep1/chem/{r}/metrics.csv" for m, (root, arm) in SRC.items() for t, r in C.CORP],
                    "judge_crosscheck")                                     # project-judge scores: no failure placeholder
    failed_g = [f"{m}/{t}/{r}" for m in SRC for t, r in C.CORP if C.failed_judgement(f"results/judge_crosscheck/gemini/{m}/{t}/{r}/coarse/metrics.csv")]
    L = ["# Cross-family judge check (Gemini vs project judge gpt-5.4), sample 1 per method x corpus\n"]
    if failed_g:                     # Gemini strict outputs whose API calls all failed are included as recorded (F1 = 0)
        L.append(f"Note: {len(failed_g)} Gemini strict (coarse) outputs are API-failure records (judge_call_failed) and enter "
                 "the coarse results with F1 = 0, as in the reported analysis.\n")
    for kind in ("fine", "coarse"):
        agree = tot = 0; conf = defaultdict(int); A = defaultdict(dict); G = defaultdict(dict)
        for m, (root, arm) in SRC.items():
            for gd in glob.glob(str(OUT / f"gemini/{m}/e2e[789]/*/{kind}")):
                p = Path(gd).parts; tag, run = p[-3], p[-2]
                ad = REPO / root / f"runs/{tag}/s1/{'eval_fine' if kind == 'fine' else 'eval'}/{arm}/rep1/chem/{run}"
                if kind == "fine":
                    ga, aa = labels_fine(Path(gd) / "alignment_fine.json"), labels_fine(ad / "alignment_fine.json")
                    gm, am = Path(gd) / "metrics_fine.json", ad / "metrics_fine.json"
                    if gm.exists(): G[m][(tag, run)] = json.load(open(gm))["fine_f1"]
                    if am.exists(): A[m][(tag, run)] = json.load(open(am))["fine_f1"]
                else:
                    ga, aa = labels_coarse(gd), labels_coarse(ad)
                    gm, am = Path(gd) / "metrics.csv", ad / "metrics.csv"
                    f1 = lambda f: float(open(f).read().splitlines()[1].split(",")[11])
                    if gm.exists(): G[m][(tag, run)] = f1(gm)
                    if am.exists(): A[m][(tag, run)] = f1(am)
                for c in set(ga) & set(aa):
                    tot += 1; agree += ga[c] == aa[c]; conf[(aa[c], ga[c])] += 1
        L.append(f"\n## {kind}: label agreement {agree}/{tot} = {agree / max(1, tot):.1%}")
        L.append("project -> gemini: " + ", ".join(f"{a}->{g}: {n}" for (a, g), n in sorted(conf.items())))
        L.append("\n| method | project judge F1 | Gemini judge F1 | n |\n|---|---|---|---|")
        for m in SRC:
            ks = sorted(set(A[m]) & set(G[m]))
            if ks:
                L.append(f"| {m} | {st.mean(A[m][k] for k in ks):.3f} | {st.mean(G[m][k] for k in ks):.3f} | {len(ks)} |")
        L.append(f"\n{OURS} - baseline under each judge, on the SAME corpora: those with recorded {OURS} and baseline scores "
                 "under both judges (paired; bootstrap 95%, 5,000 resamples of corpora stratified by subdomain):")
        L.append("\n| baseline | n | project judge | Gemini judge |\n|---|---|---|---|")
        for m in SRC:
            if m == OURS:
                continue
            ks = sorted(set(A[OURS]) & set(A[m]) & set(G[OURS]) & set(G[m]))
            if len(ks) < 3:
                continue
            by = defaultdict(list)
            for k in ks: by[k[1].split("_")[0]].append(k)          # k = (tag, run); stratify by subdomain
            cells = []
            for S in (A, G):
                D = {k: S[OURS][k] - S[m][k] for k in ks}; random.seed(0)
                bs = sorted(st.mean([D[k] for g in by.values() for k in random.choices(g, k=len(g))]) for _ in range(5000))
                d = list(D.values())
                cells.append(f"{st.mean(d):+.3f} [{bs[125]:+.3f}, {bs[4874]:+.3f}] ({sum(x > 0 for x in d)}/{sum(x < 0 for x in d)})")
            L.append(f"| {m} | {len(ks)} ({', '.join(f'{s} {len(by[s])}' for s in sorted(by))}) | {cells[0]} | {cells[1]} |")
    (OUT / "report.md").write_text("\n".join(L)); print("\n".join(L))

if __name__ == "__main__":
    from dotenv import load_dotenv; load_dotenv(REPO / ".env", override=False)
    stage = sys.argv[1]
    if stage == "models":
        from openai import OpenAI
        c = OpenAI(api_key=os.environ["GEMINI_API_KEY"], base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
        print("\n".join(sorted(m.id for m in c.models.list() if "gemini" in m.id)))
    elif stage == "run":
        J = jobs(); print(len(J), "schemas", flush=True)
        (OUT / "DAILY_QUOTA_HIT").unlink(missing_ok=True)
        def missing(j):
            od = OUT / "gemini" / j[0] / j[1] / j[2]
            return not ((od / "fine" / "metrics_fine.json").exists() and (od / "coarse" / "metrics.csv").exists()
                        and not failed(od / "coarse"))
        for _ in range(3):              # failed (503) jobs are retried in later passes; quota stop raises SystemExit
            todo = [j for j in J if missing(j)]
            if not todo: break
            print(len(todo), "schemas still missing", flush=True)
            for j in todo:              # sequential + throttled (Tier-1 Pro: 250 requests/model/day)
                run_one(j)
        if any(missing(j) for j in J):
            raise SystemExit("incomplete after 3 passes; resume next day")
        (OUT / "ALL_SCHEMAS_DONE").write_text("ok")
        report()
    else:
        report()
