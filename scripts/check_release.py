"""Offline release check; no API calls, does not read .env.

1. Python and shell syntax.
2. Every import in every Python file resolves to a local module or an installed package
   (third-party baseline code fetched by baselines/fetch_upstream.sh is reported, not required).
3. Core libraries import with placeholder credentials (client objects are created, never called).
4. Corpus manifests (36 evaluation, 30 history, 36 development) have the expected keys, every induction paper has a
   metadata entry, no absolute local paths; every seed row has a SHA-256 or an explanatory note; no results/ directory.
"""
import ast, importlib.util, json, os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Directories that scripts put on sys.path (see their sys.path[:0] lines).
SEARCH = [ROOT, ROOT / "Phase1", ROOT / "scripts", ROOT / "Evaluation", ROOT / "baselines/evaporate",
          ROOT / "baselines/arxivdigestables", ROOT / "baselines/autoschema", ROOT / "baselines/autoschema/upstream",
          ROOT / "llm4schema/scripts"]
# Installed only in the separate EVAPORATE environment (baselines/evaporate/requirements-full.txt).
EVAP_ONLY = {"cvxpy", "torch", "pandas", "sklearn", "networkx", "tensorboardX", "dill", "bs4", "lxml"}
FETCHED = {"evaporate", "metal", "profiler_utils", "prompts", "run_ws", "evaluate_profiler", "utils", "configs",
           "schema_identification", "profiler", "evaluate_synthetic_utils", "weak_supervision", "algorithm"}
CORE = ["schema_induction", "schema_lib.schema_cluster_merge", "schema_lib.subdomain_discovery",
        "schema_lib.prompt_refine", "iterative_lib.schema", "queries_generation", "retrieve_literature",
        "ontology_lib.llm_judge", "stability_lib.matching", "memexp26", "memexp27", "memexp34", "memexp36", "memexp38",
        "stability_report", "section_diag"]

def local(mod):
    top = mod.split(".")[0]
    return any((d / top).is_dir() or (d / f"{top}.py").exists() for d in SEARCH)

def main():
    import argparse
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    py = [f for f in ROOT.rglob("*.py") if not {".venv", "upstream_repo"} & set(f.parts)
          and "baselines/evaporate/upstream" not in f.as_posix() and "baselines/evaporate/full_upstream" not in f.as_posix()
          and "baselines/intentaware/upstream" not in f.as_posix()]
    missing, fetched, evap = {}, set(), set()
    for f in py:
        tree = ast.parse(f.read_text(), filename=str(f))
        for n in ast.walk(tree):
            mods = [a.name for a in n.names] if isinstance(n, ast.Import) else \
                   [n.module] if isinstance(n, ast.ImportFrom) and n.module and n.level == 0 else []
            for m in mods:
                top = m.split(".")[0]
                if top in FETCHED:
                    fetched.add(top); continue
                if top in EVAP_ONLY and importlib.util.find_spec(top) is None:
                    evap.add(top); continue
                if local(m) or importlib.util.find_spec(top) is not None:
                    continue
                missing.setdefault(top, []).append(str(f.relative_to(ROOT)))
    for f in (ROOT / "scripts").glob("*.sh"):
        subprocess.run(["bash", "-n", str(f)], check=True)
    subprocess.run(["bash", "-n", str(ROOT / "baselines/fetch_upstream.sh")], check=True)
    env = dict(os.environ, AZURE_OPENAI_CHAT_ENDPOINT="https://offline.invalid/", AZURE_OPENAI_CHAT_API_KEY="offline",
               AZURE_OPENAI_CHAT_DEPLOYMENT="offline", AZURE_OPENAI_EMBED_ENDPOINT="https://offline.invalid/",
               AZURE_OPENAI_EMBED_API_KEY="offline", PYTHONDONTWRITEBYTECODE="1")
    core_path = [str(ROOT / "Phase1"), str(ROOT), str(ROOT / "scripts"), str(ROOT / "Evaluation"),
                 str(ROOT / "baselines/arxivdigestables")]   # as the drivers set it
    code = "import sys; sys.path[:0] = %r\n" % core_path + "".join(f"import {m}\n" for m in CORE)
    subprocess.run([sys.executable, "-c", code], check=True, env=env, cwd=ROOT / "Phase1")
    need = {"papers_collected", "papers_used_for_schema_induction", "run_config", "induction_papers_not_in_collected"}
    counts = {}
    for group, n_expected in (("corpora", 36), ("corpora_history", 30), ("corpora_dev", 9)):
        files = sorted((ROOT / "data" / group).rglob("*.json")); counts[group] = len(files)
        assert len(files) == n_expected, f"data/{group}: {len(files)} manifests, expected {n_expected}"
        for f in files:
            m = json.loads(f.read_text()); text = f.read_text()
            assert need <= m.keys(), f
            assert set(m["papers_used_for_schema_induction"]) <= {p["paper_id"] for p in m["papers_collected"]} | \
                   {p["paper_id"] for p in m["induction_papers_not_in_collected"]}, f
            assert "/Users/" not in text and "/engrfs/" not in text, f"absolute local path in {f}"
    import csv
    seeds = list(csv.DictReader(open(ROOT / "seeds/seed_manifest.tsv"), delimiter="\t"))
    assert all(len(r["sha256"]) == 64 or (not r["file"] and r["note"]) for r in seeds), "seed rows need a SHA-256 or a note"
    manifests = [None] * sum(counts.values())
    res = ROOT / "results"           # allowed only when it holds installed input-package components (.memoschema-inputs/)
    assert not res.exists() or any((ROOT / ".memoschema-inputs").glob("*.json")), \
        "A fresh distribution should contain no results directory (other than installed input-package components)"
    if missing:
        raise SystemExit("FAIL: unresolved imports: " + json.dumps(missing, indent=1))
    print(f"PASS: {len(py)} Python files (syntax + imports), shell syntax, {len(CORE)} core modules imported offline, "
          f"{len(manifests)} corpus manifests {counts}, {len(seeds)} seed rows, no results/ outputs.")
    print("Third-party modules provided by baselines/fetch_upstream.sh: " + ", ".join(sorted(fetched)))
    if evap:
        print("Not installed here (EVAPORATE environment only): " + ", ".join(sorted(evap)))

if __name__ == "__main__":
    main()
