#!/usr/bin/env python3
"""EVAPORATE-CODE+ end-to-end (OpenIE) attribute flow on frozen MemoSchema inputs.

Follows the author's run_profiler.run_experiment for --do_end_to_end with the author's run.sh settings
(num_attr_to_cascade 50, num_top_k_scripts 10, train_size 10, combiner_mode ws, use_dynamic_backoff), at the pinned
commit, using the author's functions (full_adapter.load_full). Schema identification is the adapter's stage-1 output
(reproduce.py; same author functions, all matched papers). Per attribute: direct LM extraction on the sample,
function synthesis, scoring, application of the selected functions to every paper, weak-supervision combination,
the author's final attribute check; then determine_attributes_to_remove. The final schema is the first --top-k
remaining attributes in the author's ranking.

Adaptations (all recorded in run_manifest.json): generated functions run in a network-less, read-only container
(never on the host); the API transport is the adapter's ChatTransport (audit log, resume cache); each author output
budget is kept (optional --reasoning-allowance, default 0); reference files are never read; weak supervision uses
the adapter's reference-free label model (full_adapter.make_ws: no Y_dev, uniform prior over 5 candidate slots).
"""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import reproduce as r
from full_adapter import load_full, load_functions, make_ws, verify_full_source

IMAGE = "evaporate-functions:pinned"

def check_environment(docker="docker", image=IMAGE):
    out = subprocess.run([docker, "image", "inspect", image, "--format", "{{.Id}}"], capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"Sandbox image {image} missing: docker build -f Dockerfile.functions -t {image} .")
    return out.stdout.strip()

class IsolatedFunctions:
    """Runs one generated function over a set of papers inside the sandbox; cached by content."""
    def __init__(self, workdir, docker, image_id, timeout=600):
        self.cache = Path(workdir) / "function_runs"; self.cache.mkdir(parents=True, exist_ok=True)
        self.docker, self.image, self.timeout = docker, image_id, timeout
    def __call__(self, contents, papers, code, field, **kw):
        payload = {"contents": {p: contents[p] for p in papers}, "papers": list(papers), "function": code, "attribute": field}
        key = hashlib.sha256(r.canonical(payload).encode()).hexdigest()
        f = self.cache / f"{key}.json"
        if f.exists():
            d = json.loads(f.read_text()); return d["values"], d["errors"]
        cmd = [self.docker, "run", "--rm", "-i", "--network", "none", "--read-only", "--tmpfs", "/tmp:size=64m",
               "--memory", "1g", "--cpus", "1", "--pids-limit", "64", "--cap-drop", "ALL",
               "--security-opt", "no-new-privileges", self.image]
        try:
            out = subprocess.run(cmd, input=json.dumps(payload), capture_output=True, text=True, timeout=self.timeout)
            d = json.loads(out.stdout) if out.returncode == 0 and out.stdout.strip() else {"values": {p: [] for p in papers}, "errors": len(papers)}
        except (subprocess.TimeoutExpired, json.JSONDecodeError):
            d = {"values": {p: [] for p in papers}, "errors": len(papers)}
        f.write_text(json.dumps(d))
        return d["values"], d["errors"]

def author_transport(transport, allowance=0):
    """Author code passes max_toks per call; the adapter transport has one config budget."""
    def call(step, max_toks=500, **kw):
        old = transport.config["max_tokens"]
        transport.config["max_tokens"] = int(max_toks) + allowance
        try:
            return transport(step, max_toks=max_toks, **kw)
        finally:
            transport.config["max_tokens"] = old
    return call

def extraction_options(args, root, model):
    return SimpleNamespace(generative_index_path=str(root), data_lake="memoschema", topic=args.topic,
                           gold_extractions_file=None, EXTRACTION_MODELS=[model], GOLD_KEY=model, use_qa_model=False,
                           overwrite_cache=False, combiner_mode=args.combiner, extraction_fraction_thresh=0.9,
                           use_abstension=True, num_top_k_scripts=args.functions, do_end_to_end=True)

def ranked_attributes(schema_dir, n):
    """Author: Counter(identified_schema).most_common(n) sorted by (count, order of addition)."""
    fields = json.loads((schema_dir / "identified_schema.json").read_text())
    order_list = json.loads((schema_dir / "order_of_addition.json").read_text())
    order = {item: (len(order_list) - i) for i, item in enumerate(order_list)}
    ctr = Counter(fields)
    return [a.lower() for a, _ in sorted(ctr.most_common(n), key=lambda x: (x[1], order.get(x[0], 0)), reverse=True)]

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--papers", type=Path, required=True); p.add_argument("--schema-dir", type=Path, required=True)
    p.add_argument("--topic", required=True); p.add_argument("--out", type=Path, required=True)
    p.add_argument("--num-attr-to-cascade", type=int, default=50); p.add_argument("--functions", type=int, default=10)
    p.add_argument("--train-size", type=int, default=10); p.add_argument("--combiner", default="ws")
    p.add_argument("--chunk-size", type=int, default=5000)
    p.add_argument("--reasoning-allowance", type=int, default=0,
                   help="extra output tokens per call; 0 = author budgets (checked: the deployment uses 0 reasoning tokens on short answers)")
    p.add_argument("--docker", default="docker")
    r.shared(p)   # --top-k (8), --max-calls, provider/deployment/env/temperature/stop/token options of the adapter
    a = p.parse_args()
    image_id = check_environment(a.docker)
    verify_full_source()
    root = a.out / "index"; root.mkdir(parents=True, exist_ok=True)
    status = a.out / "status.json"
    if status.exists() and json.loads(status.read_text()).get("status") == "complete":
        print("already complete"); return
    records = r.read_records(a.papers) if hasattr(r, "read_records") else [json.loads(l) for l in open(a.papers)]
    # Author data model: documents are files and their paths are the keys (sample_scripts opens them).
    docs = a.out / "docs"; docs.mkdir(parents=True, exist_ok=True)
    pid_of, contents = {}, {}
    for x in records:
        f = docs / (hashlib.sha256(x["paper_id"].encode()).hexdigest()[:16] + ".txt")
        f.write_text(x["text"], encoding="utf-8"); pid_of[str(f)] = x["paper_id"]; contents[str(f)] = x["text"]
    config = r.settings(a)
    if not a.execute:
        print(json.dumps({"dry_run": True, "papers": len(contents), "attributes": len(ranked_attributes(a.schema_dir, a.num_attr_to_cascade))})); return
    from dotenv import load_dotenv
    load_dotenv(a.env_file, override=False)
    config = r.settings(a)
    transport = r.ChatTransport(a.out, config, a.max_calls)
    model = config["deployment"]
    import html, re, string, unicodedata, math, pickle
    from collections import defaultdict
    from bs4 import BeautifulSoup
    clean_ns = {"html": html, "re": re, "string": string, "unicodedata": unicodedata, "math": math, "pickle": pickle,
                "json": json, "os": os, "Counter": Counter, "defaultdict": defaultdict,
                "BeautifulSoup": BeautifulSoup}   # module-level imports of evaluate_synthetic.py
    load_functions("evaluate_synthetic.py", {"clean_comparison"}, clean_ns)
    algo = load_full(author_transport(transport, a.reasoning_allowance), IsolatedFunctions(a.out, a.docker, image_id),
                     make_ws(clean_ns["clean_comparison"]))
    file2chunks = {pid: algo.get_txt_parse(text, chunk_size=a.chunk_size, mode="train")[1] for pid, text in contents.items()}
    files = sorted(contents)
    sample_files = algo.sample_scripts(files, train_size=a.train_size)
    attributes = ranked_attributes(a.schema_dir, a.num_attr_to_cascade)
    opts = extraction_options(a, root, model)
    r.save(a.out / "run_manifest.json", {"method": "EVAPORATE-CODE+ end-to-end attribute flow (author run.sh settings)",
        "author_commit": "59eda5d34415f71ffb3a72bb902e1505d1f59e83", "full_source": verify_full_source(),
        "sandbox_image": image_id, "papers": len(files), "sample_files": [pid_of[f] for f in sample_files], "attributes_ranked": attributes,
        "settings": {k: getattr(a, k.replace("-", "_")) for k in ("num_attr_to_cascade", "functions", "train_size", "combiner",
                     "chunk_size", "top_k", "reasoning_allowance")}, "transport": config})
    progress = []
    for i, attr in enumerate(attributes):
        _, success = algo.run_profiler("full", opts, file2chunks, contents, sample_files, files, {model: None}, attr, opts)
        progress.append({"attribute": attr, "success": int(success)})
        r.save(a.out / "progress.json", progress)
        print(f"[attr {i + 1}/{len(attributes)}] {attr}: {'kept' if success else 'dropped'}", flush=True)
    removed, _, _ = algo.determine_attributes_to_remove(attributes, opts, "full", a.num_attr_to_cascade)
    fa = algo.get_file_attribute
    kept = [x for x in attributes if (root / f"full_{fa(x)}_file2metadata.json").exists() and x not in removed]
    table = {x: json.loads((root / f"full_{fa(x)}_file2metadata.json").read_text()) for x in kept}
    final = []
    for x in kept[: a.top_k]:
        vals = [v for v in table[x].values() if v][:3]
        final.append({"name": x, "description": "", "value_type": "", "aliases": [],
                      "support_count": sum(bool(v) for v in table[x].values()), "example_values": vals})
    r.save(a.out / "final_schema.json", final)
    r.save(a.out / "final_schema_all_kept.json", [{"name": x} for x in kept])
    r.save(a.out / "removed_attributes.json", {"dedup_removed": removed,
           "dropped_in_profiler": [g["attribute"] for g in progress if not g["success"]]})
    r.save(a.out / "table.json", {pid_of[f]: {x: table[x].get(f, "") for x in kept} for f in files})
    r.save(a.out / "cost_total.json", {"totals": transport.cost()})
    r.save(status, {"status": "complete", "kept": len(kept), "final": [x["name"] for x in final]})
    print("COMPLETE", [x["name"] for x in final], flush=True)

if __name__ == "__main__":
    main()
