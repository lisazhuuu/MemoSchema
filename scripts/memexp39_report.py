"""memexp39 report: ours (r10 live, content-key adjusted) vs EVAPORATE on the same 9 r10 corpora."""
import json, random, statistics as st
import tiktoken
from collections import defaultdict
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
import sys
if __name__ == "__main__" and (sys.argv[1:]):                     # usage check before any work
    print(__doc__ if sys.argv[1:2] in (["-h"], ["--help"]) else "usage: python scripts/" + Path(__file__).name + " " + "(no arguments)")
    sys.exit(0 if sys.argv[1:2] in (["-h"], ["--help"]) else 2)
RUNS = [f"{s}_{k}" for s in ("COF", "MOF", "ZIF") for k in ("01", "02", "03")]
sys.path.insert(0, str(REPO / "scripts")); import completeness as C       # 9 r10 corpora, both pipelines
C.require_files(["results/memo_audit/details.json", "results/cost_compare.json"]
                + [p for r in RUNS for p in (f"results/memexp33/usage/{r}.jsonl", f"results/memexp33/calls/merge_live_e2e10_{r}.json",
                   f"results/evaporate/runs/e2e10/s1/EVAPORATE8/chem/{r}/cost.json", f"results/evaporate_full/e2e10/{r}/cost_total.json")],
                "r10 accounting inputs", "memexp39_report")
usd = lambda i, o: i * 2.5e-6 + o * 15e-6
A = json.load(open(REPO / "results/memo_audit/details.json"))
c = json.load(open(REPO / "results/cost_compare.json")); ks = [k for k in c if k.startswith("e2e")]
per = lambda s: [sum(c[k][s][i] for k in ks) / max(1, sum(c[k][s][0] for k in ks)) for i in (1, 2)]
pi, po = per("per_paper"); mi, mo = per("merge")
# Token estimate for the unlogged restructuring call: tiktoken o200k_base (verified with tiktoken 0.9.0).
# The call sends two chat messages (system SYSTEM_MSG + user prompt, scripts/memexp26.py chat); prompt tokens are
# the encoded message contents plus 3 per message plus 3 reply-priming tokens. On saved EVAPORATE calls with API
# usage, this formula matches the reported prompt_tokens (ratio 1.000) and gives output tokens within ~2% (median .978).
ENC = tiktoken.get_encoding("o200k_base")
SYSTEM_MSG = "Return ONLY valid JSON. No markdown."
TOKENS_PER_MESSAGE, REPLY_PRIMING = 3, 3
def restructuring_prompt_tokens(prompt):
    return sum(len(ENC.encode(m)) + TOKENS_PER_MESSAGE for m in (SYSTEM_MSG, prompt)) + REPLY_PRIMING
ours, ev = {}, {}
for run in RUNS:
    L = [json.loads(l) for l in open(REPO / f"results/memexp33/usage/{run}.jsonl")]
    S = [x for x in L if x["module"] == "schema_induction"]
    calls = len(S); tin = sum(x["usage"]["prompt_tokens"] for x in S); tout = sum(x["usage"]["completion_tokens"] for x in S)
    lat = sum((x["usage"].get("latency_checkpoint") or {}).get("service_ttlt_ms", 0) for x in S) / 3.6e6
    bm = sum(1 for x in A["merge_r10"] if x["run"] == run and not x["content_identical"])
    bp = sum(1 for x in A["paper_r10"] if x["run"] == run and x["text_identical"] is False)
    extra_calls = bm + 2 * bp; tin += bm * mi + 2 * bp * pi; tout += bm * mo + 2 * bp * po
    lat += extra_calls * (lat / max(1, calls))
    # Restructuring was recorded separately (prompt/response text only, no usage counters); estimate its tokens.
    restructuring = json.load(open(REPO / f"results/memexp33/calls/merge_live_e2e10_{run}.json"))
    json.loads(restructuring["response"])  # require a successful saved response
    tin += restructuring_prompt_tokens(restructuring["prompt"])
    tout += len(ENC.encode(restructuring["response"]))
    extra_calls += 1
    ours[run] = {"calls": calls + extra_calls, "usd": usd(tin, tout), "hours": lat, "in": tin, "out": tout}
    calls = tin = tout = 0; lat = 0.0
    for f in [REPO / f"results/evaporate/runs/e2e10/s1/EVAPORATE8/chem/{run}/cost.json"]:
        x = json.load(open(f)); calls += x["successful_logical_calls"]; tin += x["successful_response_tokens"]["prompt_tokens"]; tout += x["successful_response_tokens"]["completion_tokens"]
    x = json.load(open(REPO / f"results/evaporate_full/e2e10/{run}/cost_total.json"))["totals"]
    calls += x["successful_logical_calls"]; tin += x["successful_response_tokens"]["prompt_tokens"]; tout += x["successful_response_tokens"]["completion_tokens"]
    for d in (REPO / f"results/evaporate/runs/e2e10/s1/EVAPORATE8/chem/{run}/calls", REPO / f"results/evaporate_full/e2e10/{run}/calls"):
        for f in d.glob("*.json"):
            try: lat += (json.load(open(f)).get("elapsed_seconds") or 0) / 3600
            except Exception: pass
    ev[run] = {"calls": calls, "usd": usd(tin, tout), "hours": lat, "in": tin, "out": tout}
def boot(D):
    by = defaultdict(list)
    for r in D: by[r[:3]].append(r)
    random.seed(0); bs = sorted(st.mean([D[r] for rs in by.values() for r in random.choices(rs, k=len(rs))]) for _ in range(5000))
    return st.mean(D.values()), bs[125], bs[4874]
L = ["# memexp39 report: ours vs EVAPORATE on the same 9 r10 corpora (corpus given -> schema)\n",
     "| corpus | ours calls | EVAPORATE calls | ours $ | EVAPORATE $ |", "|---|---|---|---|---|"]
for r in RUNS:
    L.append(f"| {r} | {ours[r]['calls']:.0f} | {ev[r]['calls']} | {ours[r]['usd']:.2f} | {ev[r]['usd']:.2f} |")
L.append(f"| mean | {st.mean(o['calls'] for o in ours.values()):.0f} | {st.mean(e['calls'] for e in ev.values()):.0f} | {st.mean(o['usd'] for o in ours.values()):.2f} | "
         f"{st.mean(e['usd'] for e in ev.values()):.2f} |")
L.append("\nPaired ratio ours / EVAPORATE (mean over corpora, bootstrap stratified by subdomain, 95%):")
for k in ("calls", "usd"):
    m, lo, hi = boot({r: ours[r][k] / ev[r][k] for r in RUNS})
    L.append(f"- {k}: {m:.3f} [{lo:.3f}, {hi:.3f}] -> reduction {1 - m:.1%} [{1 - hi:.1%}, {1 - lo:.1%}]")
L.append("\nAccounting: logged schema-induction calls use API-reported token usage. Content-key adjustment (as before): "
         "M3 hits whose member content differed and M2 hits whose paper text differed are counted as fresh calls "
         "(1 call per merge, 2 per paper), at the per-call mean tokens of results/cost_compare.json. "
         "Accounting correction: includes one separately recorded restructuring call per corpus; no API usage was recorded "
         "for it, so its tokens are estimated from the saved prompt/response with tiktoken o200k_base (verified with "
         "tiktoken 0.9.0), prompt = encoded system + user contents + 3 per message + 3 reply priming. "
         "Latency excludes this unlogged call and is therefore not reported.")
(REPO / "results/memexp39").mkdir(parents=True, exist_ok=True)
(REPO / "results/memexp39/report.md").write_text("\n".join(L)); print("\n".join(L))
