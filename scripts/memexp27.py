"""memexp27: LLMs4SchemaDiscovery baseline (automated adaptation) on the 27 corpora, 3 repetitions per corpus.
Stages: jobs <batch>  (print one L4S command per run x rep, for xargs) | collect <batch>.
scores() is shared by the report scripts; the Table IV comparison is in scripts/paper_tables.py."""
import glob, json, os, shutil, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "scripts")]
OUT = REPO / "results" / "memexp27"
TAG = {"r7": "e2e7", "r8": "e2e8", "r9": "e2e9"}
TIER = {"COF": "Covalent Organic Frameworks (COFs)", "MOF": "Metal-organic frameworks (MOFs)", "ZIF": "Zeolitic imidazolate frameworks (ZIFs)"}
REPS = [1, 2, 3]

def run_dir(b, run):
    import input_package; input_package.check("replay", "memexp27")   # minimal input package: out of scope, stop
    hits = sorted(glob.glob(str(REPO / f"results/e2e/e2e_v3/{b}/chem/{run}/iterative_phase1_chem_*")))
    return Path(hits[-1]) if hits and (Path(hits[-1]) / "final_schema_from_cumulative_corpus/final_schema.json").exists() else None

def runs_of(b):
    return sorted(p.name for p in (REPO / f"results/e2e/e2e_v3/{b}/chem").iterdir() if run_dir(b, p.name))

def raw_dir(b, run, j):
    return OUT / "l4s_raw" / TAG[b] / f"rep{j}" / run

def jobs(b):
    for run in runs_of(b):
        s, k = run.split("_")
        for j in REPS:
            od = raw_dir(b, run, j)
            if (od / "final_schema_from_cumulative_corpus/final_schema.json").exists():
                continue
            print(f'--domain chem --subdomain {s} --last-tier "{TIER[s]}" --seed-dir seeds/seed_chem/{run} '
                  f'--stage3-corpus-dir {run_dir(b, run) / "cumulative_corpus"} --output-root {od}')

def collect(b):
    for run in runs_of(b):
        for j in REPS:
            f = raw_dir(b, run, j) / "final_schema_from_cumulative_corpus/final_schema.json"
            if f.exists():
                od = OUT / "runs" / TAG[b] / f"s{j}" / "L4S" / "chem" / run; od.mkdir(parents=True, exist_ok=True)
                shutil.copy(f, od / "final_schema.json")

def scores(root, fine):
    S = {}
    pat = "runs/e2e*/s*/eval_fine/*/rep*/chem/*/metrics_fine.json" if fine else "runs/e2e*/s*/eval/*/rep*/chem/*/metrics.csv"
    for f in glob.glob(str(root / pat)):
        p = Path(f).parts; i = p.index("runs")
        v = json.load(open(f))["fine_f1"] if fine else float(open(f).read().splitlines()[1].split(",")[11])
        S.setdefault((p[i + 1], p[i + 7], p[i + 4]), []).append(v)        # (tag, run, arm) -> all samples x reps
    return S

if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("jobs", "collect"):
        print(__doc__ if sys.argv[1:2] in (["-h"], ["--help"]) else "usage: python scripts/memexp27.py jobs|collect <batch>")
        sys.exit(0 if sys.argv[1:2] in (["-h"], ["--help"]) else 2)
    OUT.mkdir(parents=True, exist_ok=True)
    {"jobs": lambda: jobs(sys.argv[2]), "collect": lambda: collect(sys.argv[2])}[sys.argv[1]]()
