"""Execute pinned author schema-discovery functions, without Manifest/code synthesis.

Only the API transport, fail-fast error handling, and one-document stdev guard
are adapted. The prompts, text chunker, field filters, and reranking are upstream.
"""
from __future__ import annotations

import ast
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import statistics
from types import SimpleNamespace

UPSTREAM = Path(__file__).parent / "upstream"
COMMIT = "59eda5d34415f71ffb3a72bb902e1505d1f59e83"
FUNCTIONS = {
    "directly_extract_from_chunks_w_value", "get_metadata_string_w_value",
    "rerank", "rerank_metadata",
}


def verify_source():
    provenance = json.loads((UPSTREAM / "PROVENANCE.json").read_text())
    if provenance["commit"] != COMMIT:
        raise ValueError("Unexpected upstream commit")
    for name, expected in provenance["files"].items():
        actual = hashlib.sha256((UPSTREAM / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Upstream source changed: {name}")
    return provenance


class FailFastTransport(ast.NodeTransformer):
    def visit_Try(self, node):
        self.generic_visit(node)
        calls_transport = any(
            isinstance(x, ast.Call) and isinstance(x.func, ast.Name)
            and x.func.id == "apply_prompt"
            for stmt in node.body for x in ast.walk(stmt)
        )
        if calls_transport:
            for handler in node.handlers:
                handler.body = [ast.Raise()]
        return node


def load_algorithm(apply_prompt, *, fail_fast=True):
    verify_source()
    ns = {
        "Counter": Counter, "defaultdict": defaultdict,
        "random": random.Random(0),
        "statistics": SimpleNamespace(
            mean=statistics.mean,
            stdev=lambda xs: statistics.stdev(xs) if len(xs) > 1 else 0.0,
        ),
        "tqdm": lambda iterable, **kw: iterable,
        "Step": lambda prompt: SimpleNamespace(prompt=prompt),
        "apply_prompt": apply_prompt,
    }
    selected = []
    for filename, names in [
        ("prompts.py", {"SCHEMA_ID_PROMPTS"}),
        ("profiler_utils.py", {"get_txt_parse"}),
        ("schema_identification.py", FUNCTIONS),
    ]:
        tree = ast.parse((UPSTREAM / filename).read_text())
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in names:
                selected.append(node)
            elif isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id in names for t in node.targets
            ):
                selected.append(node)
    module = ast.Module(body=selected, type_ignores=[])
    if fail_fast:
        module = FailFastTransport().visit(module)
    exec(compile(ast.fix_missing_locations(module), str(UPSTREAM), "exec"), ns)
    return SimpleNamespace(**{k: ns[k] for k in FUNCTIONS | {"get_txt_parse", "SCHEMA_ID_PROMPTS"}})


def ranked_names(scores, order, top_k):
    """Author run_profiler.identify_attributes ordering, including cutoff ties."""
    priority = {name: len(order) - i for i, name in enumerate(order)}
    selected = Counter(scores).most_common(top_k if top_k else None)
    return [name for name, _ in sorted(
        selected, key=lambda x: (x[1], priority[x[0]]), reverse=True
    )]
