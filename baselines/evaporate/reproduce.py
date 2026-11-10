#!/usr/bin/env python3
"""EVAPORATE schema discovery on frozen MemoSchema inputs (no extraction/MERGE).

prepare and dry-run never initialize an API client. Actual generation requires
--execute. Reference answers are never loaded by this program.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

from upstream_adapter import load_algorithm, ranked_names, verify_source

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
VERSION = "1"
TOPICS = {
    "COF": "Covalent Organic Frameworks (COFs)",
    "MOF": "Metal-organic frameworks (MOFs)",
    "ZIF": "Zeolitic imidazolate frameworks (ZIFs)",
}


def digest(value):
    data = value if isinstance(value, bytes) else str(value).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def canonical(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


@contextmanager
def exclusive(out):
    import fcntl
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Another driver owns {out}; do not launch a duplicate") from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def source_papers(run_dir):
    """Read identities ONLY from proposals. Never use baseline field proposals."""
    run_dir = Path(run_dir).resolve()
    rows_path = run_dir / "schema_state/paper_level_field_proposals.jsonl"
    if not rows_path.is_file():
        raise ValueError(f"Missing cumulative paper identity list: {rows_path}")
    identities = {}
    for line in rows_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        pid, source = row["paper_id"], row["source_file"]
        if not isinstance(pid, str) or not isinstance(source, str) or not pid:
            raise ValueError("Invalid paper identity")
        if pid in identities and identities[pid] != Path(source).name:
            raise ValueError(f"Conflicting identity {pid}")
        identities[pid] = Path(source).name
    if not identities:
        raise ValueError("No paper identities found")
    directories = [run_dir / "cumulative_corpus"] + sorted(run_dir.glob("round*/downloads"))
    papers = []
    for pid, filename in sorted(identities.items()):
        matches = [d / filename for d in directories if (d / filename).is_file()]
        if not matches:
            raise ValueError(f"PDF missing for {pid}; expected {filename} inside {run_dir}")
        hashes = {digest(p.read_bytes()) for p in matches}
        if len(hashes) > 1:
            raise ValueError(f"Different PDF contents for {pid}; resolve ambiguity first")
        papers.append((pid, matches[0]))
    return papers, {"kind": "memoschema_cumulative_paper_ids", "run_dir": str(run_dir),
                    "identities_sha256": digest(canonical(identities))}


def papers_from_list(filename):
    filename = Path(filename).resolve()
    papers, seen = [], set()
    for line in filename.read_text().splitlines():
        if not line.strip():
            continue
        path = Path(line.strip())
        if not path.is_absolute():
            path = filename.parent / path
        path = path.resolve()
        if path.stem in seen:
            raise ValueError(f"Duplicate paper_id in list: {path.stem}")
        seen.add(path.stem)
        if not path.is_file():
            raise ValueError(f"Missing input: {path}")
        papers.append((path.stem, path))
    if not papers:
        raise ValueError("Empty paper list")
    return papers, {"kind": "explicit_paper_list", "path": str(filename)}


def extract_text(path):
    if path.suffix.lower() == ".pdf":
        import fitz
        # Same page extraction and joining as Phase1/queries_lib/paper_model.py.
        with fitz.open(path) as doc:
            return "\n".join(page.get_text() for page in doc)
    if path.suffix.lower() == ".txt":
        return path.read_text(encoding="utf-8")
    raise ValueError(f"Only PDF or raw text inputs are supported: {path}")


def prepare(run_dir, paper_list, out, max_chars):
    if max_chars < 0:
        raise ValueError("max-chars must be nonnegative; 0 means full text")
    papers, source = source_papers(run_dir) if run_dir else papers_from_list(paper_list)
    records = []
    for pid, path in papers:
        raw = extract_text(path)
        text = raw[:max_chars] if max_chars else raw
        if not text.strip():
            raise ValueError(f"Empty parsed text for {pid}; do not silently drop matched papers")
        records.append({"paper_id": pid, "source_file": path.name, "source_path": str(path),
                        "source_sha256": digest(path.read_bytes()), "raw_text_sha256": digest(raw),
                        "text_sha256": digest(text), "raw_chars": len(raw), "text": text})
    payload = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
    out = Path(out)
    with exclusive(out):
        target = out / "papers.jsonl"
        if target.exists() and target.read_text() != payload:
            raise ValueError(f"Input changed; choose a new directory instead of overwriting {out}")
        target.write_text(payload, encoding="utf-8")
        save(out / "input_manifest.json", {
            "source": source, "papers": len(records), "max_chars_per_paper": max_chars,
            "papers_sha256": digest(payload), "parser": "PyMuPDF page.get_text, newline joined",
            "input": "raw paper text, never MemoSchema distill or field proposals",
        })
    print(f"Prepared {len(records)} papers: {target}", flush=True)
    return target


def read_records(filename):
    records, seen = [], set()
    for line in Path(filename).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        pid, text = row.get("paper_id"), row.get("text")
        if not isinstance(pid, str) or not pid or pid in seen:
            raise ValueError(f"Missing/duplicate paper_id: {pid}")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Missing raw text: {pid}")
        if row.get("text_sha256", digest(text)) != digest(text):
            raise ValueError(f"Text hash mismatch: {pid}")
        seen.add(pid)
        records.append(row)
    if not records:
        raise ValueError("Empty corpus")
    return records


def settings(args):
    """Do not read credentials or instantiate a client until --execute."""
    deployment = args.deployment or os.environ.get(
        "AZURE_OPENAI_CHAT_DEPLOYMENT" if args.provider == "azure" else "OPENAI_MODEL", ""
    )
    endpoint = os.environ.get("AZURE_OPENAI_CHAT_ENDPOINT" if args.provider == "azure" else "OPENAI_BASE_URL", "")
    return {"provider": args.provider, "deployment": deployment,
            "endpoint_sha256": digest(endpoint),
            "api_version": os.environ.get("AZURE_OPENAI_CHAT_API_VERSION", "2024-12-01-preview"),
            "temperature": None if args.temperature == "omit" else float(args.temperature),
            "max_tokens": args.max_tokens, "token_parameter": args.token_parameter,
            "send_stop": not args.omit_stop, "timeout": args.timeout, "max_attempts": args.max_attempts}


class ChatTransport:
    """Per-request resume cache, durable audit logs, bounded retries, no hidden SDK retries."""
    def __init__(self, out, config, max_calls, client=None):
        self.out, self.config, self.max_calls = out, config, max_calls
        self.client = client
        self.index = 0
        self.new_calls = 0

    def get_client(self):
        if self.client is None:
            from openai import AzureOpenAI, OpenAI
            c = self.config
            if not c["deployment"]:
                raise ValueError("Set AZURE_OPENAI_CHAT_DEPLOYMENT / OPENAI_MODEL or --deployment")
            if c["provider"] == "azure":
                self.client = AzureOpenAI(
                    azure_endpoint=os.environ["AZURE_OPENAI_CHAT_ENDPOINT"],
                    api_key=os.environ["AZURE_OPENAI_CHAT_API_KEY"], api_version=c["api_version"],
                    max_retries=0, timeout=c["timeout"],
                )
            else:
                self.client = OpenAI(max_retries=0, timeout=c["timeout"])
        return self.client

    def __call__(self, step, max_toks=500, **kwargs):
        self.index += 1
        if self.index > self.max_calls:
            raise RuntimeError("Logical call limit exceeded")
        c = self.config
        request = {"model": c["deployment"], "messages": [{"role": "user", "content": step.prompt}],
                   c["token_parameter"]: c["max_tokens"]}
        if c["temperature"] is not None:
            request["temperature"] = c["temperature"]
        if c["send_stop"]:
            request["stop"] = "---"
        key = digest(canonical({"config": c, "request": request}))
        path = self.out / "calls" / f"{self.index:06d}.json"
        if path.exists():
            item = json.loads(path.read_text())
            if item["request_sha256"] != key:
                raise ValueError("Resume request changed; choose a fresh output directory")
            return item["response"], item["usage"].get("prompt_tokens", 0)
        for attempt in range(1, c["max_attempts"] + 1):
            start = time.monotonic()
            try:
                self.new_calls += 1
                response = self.get_client().chat.completions.create(**request)
                choice = response.choices[0]
                raw = choice.message.content
                usage = response.usage.model_dump() if response.usage else {}
                if raw is None or not raw.strip() or choice.finish_reason == "content_filter":
                    raise ValueError("Empty or blocked API response; not a valid no-fields answer")
            except Exception as exc:
                status = getattr(exc, "status_code", None)
                # Avoid persisting exception bodies, which may contain endpoint or credential data.
                save(self.out / "attempts" / f"{self.index:06d}_{attempt}_{time.time_ns()}.json", {
                    "call_index": self.index, "error_type": type(exc).__name__,
                    "http_status": status, "elapsed_seconds": time.monotonic() - start,
                    "usage_unknown": True,
                })
                retryable = status in (408, 409, 429) or (status is not None and status >= 500)
                retryable |= type(exc).__name__ in ("APITimeoutError", "APIConnectionError")
                if not retryable or attempt == c["max_attempts"]:
                    raise RuntimeError(f"API failed ({type(exc).__name__}, status={status}); resume after fixing") from None
                time.sleep(min(60, 2 ** attempt) + random.random())
                continue
            # Match author's local stop cleanup even if --omit-stop is needed by a model.
            text = raw.split("---")[0].strip()
            save(path, {"request_sha256": key, "request": request, "raw_response": raw,
                        "response": text, "usage": usage, "response_model": response.model,
                        "finish_reason": choice.finish_reason, "attempts_this_execution": attempt,
                        "elapsed_seconds": time.monotonic() - start})
            print(f"[call {self.index}] completed ({choice.finish_reason})", flush=True)
            return text, usage.get("prompt_tokens", 0)
        raise AssertionError("unreachable")

    def cost(self):
        entries = [json.loads(p.read_text()) for p in sorted((self.out / "calls").glob("*.json"))]
        usage = Counter()
        for row in entries:
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                usage[key] += row["usage"].get(key, 0) or 0
        failures = list((self.out / "attempts").glob("*.json"))
        return {"successful_logical_calls": len(entries), "failed_attempts": len(failures),
                "successful_response_tokens": dict(usage), "failed_attempt_tokens_unknown": bool(failures),
                "truncated_responses": sum(r["finish_reason"] == "length" for r in entries),
                "api_attempts_this_invocation": self.new_calls,
                "note": "Retries with unknown usage are not counted as free. No judge calls included."}


def schema_fields(names, counts, values):
    # No generated definitions/aliases: the author method outputs attribute names + examples.
    return [{"name": n, "description": "", "value_type": "", "aliases": [],
             "support_count": int(counts[n]), "example_values": values[n][:3]}
            for n in names]


def run(args):
    if args.execute:
        from dotenv import load_dotenv
        load_dotenv(args.env_file, override=False)
    config = settings(args)
    records = read_records(args.papers)
    all_ids = [r["paper_id"] for r in records]
    if args.sample_size < 0 or args.top_k < 1 or args.chunk_chars < 1:
        raise ValueError("Invalid sample size, top-k or chunk size")
    ids = (random.Random(args.sample_seed).sample(all_ids, args.sample_size)
           if 0 < args.sample_size <= len(all_ids) else all_ids)
    algo = load_algorithm(lambda *a, **k: (_ for _ in ()).throw(RuntimeError("dry-run called API")))
    chunks = {r["paper_id"]: algo.get_txt_parse(r["text"], chunk_size=args.chunk_chars)[1] for r in records}
    maximum = sum(len(chunks[pid]) for pid in ids) + 6  # two rerank passes x three examples
    plan = {"available_papers": len(records), "sampled_papers": len(ids),
            "max_logical_calls": maximum, "max_attempts_per_call": args.max_attempts,
            "sample_size": args.sample_size, "top_k": args.top_k,
            "topic": args.topic, "output": str(args.out)}
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if not args.execute:
        return
    if maximum > args.max_calls:
        raise ValueError(f"Preflight call bound {maximum} exceeds --max-calls {args.max_calls}")
    if not config["deployment"]:
        raise ValueError("Missing deployment/model, no API calls made")
    out = Path(args.out)
    fingerprint = {
        "adapter_version": VERSION,
        "adapter_sha256": digest(Path(__file__).read_bytes() + (HERE / "upstream_adapter.py").read_bytes()),
        "upstream": verify_source(), "api": config,
        "topic": args.topic, "chunk_chars": args.chunk_chars,
        "dynamic_backoff": True, "top_k": args.top_k, "sample_seed": args.sample_seed,
        "sample_size": args.sample_size, "selected_paper_ids": ids,
        "input_texts": [{"paper_id": r["paper_id"], "text_sha256": digest(r["text"])} for r in records],
    }
    with exclusive(out):
        manifest = out / "run_manifest.json"
        if manifest.exists() and json.loads(manifest.read_text()) != fingerprint:
            raise ValueError("Configuration/input changed. Use a new output directory.")
        save(manifest, fingerprint)
        transport = ChatTransport(out, config, args.max_calls)
        algo = load_algorithm(transport)
        save(out / "status.json", {"status": "running"})
        try:
            values, counts, _ = algo.directly_extract_from_chunks_w_value(
                chunks, ids, None, topic=args.topic, use_dynamic_backoff=True,
            )
            save(out / "field_values.json", dict(values))
            save(out / "field_counts.json", dict(counts))
            scores, _, order = algo.rerank_metadata(counts, values, args.topic, None, False)
            all_names = ranked_names(scores, order, 0)
            save(out / "identified_schema.json", dict(scores))
            save(out / "order_of_addition.json", order)
            save(out / "ranked_candidates.json", [
                {"name": n, "rank": i + 1, "score": scores[n],
                 "support_count": counts[n], "example_values": values[n][:3]}
                for i, n in enumerate(all_names)
            ])
            for k, filename in [(args.top_k, "final_schema.json"), (35, "final_schema_top35.json"),
                                (0, "final_schema_all_candidates.json")]:
                save(out / filename, schema_fields(ranked_names(scores, order, k), counts, values))
            save(out / "status.json", {"status": "complete", "fields": min(args.top_k, len(scores)),
                                       "empty_schema": not bool(scores), "time": datetime.now(timezone.utc).isoformat()})
        except BaseException as exc:
            save(out / "status.json", {"status": "failed", "error_type": type(exc).__name__})
            raise
        finally:
            save(out / "cost.json", transport.cost())
    print(f"COMPLETE: {out / 'final_schema.json'}", flush=True)


def shared(parser):
    parser.add_argument("--sample-size", type=int, default=0, help="0: all matched papers; author CLI default: 10")
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=8, help="matched field budget; author CLI default: 35")
    parser.add_argument("--chunk-chars", type=int, default=3000)
    parser.add_argument("--provider", choices=["azure", "openai"], default="azure")
    parser.add_argument("--deployment", default="")
    parser.add_argument("--env-file", type=Path, default=REPO / ".env")
    parser.add_argument("--temperature", default="0", help="number, or omit for unsupported models")
    parser.add_argument("--max-tokens", type=int, default=500, help="author schema-call output budget")
    parser.add_argument("--token-parameter", choices=["max_tokens", "max_completion_tokens"], default="max_completion_tokens")
    parser.add_argument("--omit-stop", action="store_true", help="explicit API compatibility deviation; local stop preserved")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--max-attempts", type=int, default=6)
    parser.add_argument("--max-calls", type=int, default=1000, help="preflight limit per run, excluding bounded retries")
    parser.add_argument("--execute", action="store_true", help="allow paid API generation (otherwise dry-run)")


def batch(args):
    """Discover only completed runs; no references, shell interpolation, or automatic judge."""
    root = Path(args.project_root).resolve()
    jobs = []
    for batch_id in args.batches.split(","):
        batch_id = batch_id.strip()
        if batch_id not in {f"r{i}" for i in range(1, 100)}:
            raise ValueError(f"Invalid batch {batch_id}")
        for run_id in args.runs.split(","):
            run_id = run_id.strip()
            if run_id not in {f"{s}_{i:02d}" for s in TOPICS for i in (1, 2, 3)}:
                raise ValueError(f"Invalid chemistry run {run_id}")
            candidates = sorted((root / f"results/e2e/e2e_v3/{batch_id}/chem/{run_id}").glob("iterative_phase1_chem_*"))
            completed = [p for p in candidates if (p / "final_schema_from_cumulative_corpus/final_schema.json").is_file()]
            if len(completed) != 1:
                raise ValueError(f"Expected one completed source run for {batch_id}/{run_id}, found {len(completed)}")
            jobs.append((batch_id, run_id, completed[0]))
    print(f"Batch plan: {len(jobs)} corpora x {args.replicates} replicates", flush=True)
    for batch_id, run_id, rd in jobs:
        prepared = prepare(rd, None, args.output_root / "inputs" / batch_id / run_id, args.max_chars)
        for rep in range(1, args.replicates + 1):
            opts = argparse.Namespace(**vars(args))
            opts.papers, opts.topic = prepared, TOPICS[run_id.split("_")[0]]
            opts.out = args.output_root / "runs" / f"e2e{batch_id[1:]}" / f"s{rep}" / f"EVAPORATE{args.top_k}" / "chem" / run_id
            run(opts)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="freeze matching raw texts, no API")
    source = prep.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir", type=Path)
    source.add_argument("--paper-list", type=Path, help="one PDF/raw text path per line")
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--max-chars", type=int, default=12000)
    gen = sub.add_parser("run", help="dry-run by default")
    gen.add_argument("--papers", type=Path, required=True)
    gen.add_argument("--topic", required=True)
    gen.add_argument("--out", type=Path, required=True)
    shared(gen)
    many = sub.add_parser("batch", help="prepare and discover schemas for selected chemistry runs")
    many.add_argument("--project-root", type=Path, default=REPO)
    many.add_argument("--batches", default="r9")
    many.add_argument("--runs", default="COF_01", help="comma-separated IDs, explicit scope")
    many.add_argument("--replicates", type=int, default=1)
    many.add_argument("--output-root", type=Path, default=REPO / "results/evaporate")
    many.add_argument("--max-chars", type=int, default=12000)
    shared(many)
    args = p.parse_args()
    if args.command == "prepare":
        prepare(args.run_dir, args.paper_list, args.out, args.max_chars)
    else:
        if args.max_attempts < 1 or args.max_tokens < 1 or args.max_calls < 1:
            p.error("Call/token/attempt limits must be positive")
        if args.command == "batch" and args.replicates < 1:
            p.error("replicates must be positive")
        (batch if args.command == "batch" else run)(args)


if __name__ == "__main__":
    main()
