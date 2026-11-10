"""Pinned author extraction stages, with explicit transport/isolation adaptations."""
import ast
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from upstream_adapter import FailFastTransport

SOURCE = Path(__file__).parent / 'full_upstream'


def verify_full_source():
    manifest = json.loads((SOURCE / 'PROVENANCE.json').read_text())
    for name, expected in manifest['files'].items():
        if hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Author source changed: ' + name)
    return manifest


def load_functions(filename, names, namespace):
    tree = ast.parse((SOURCE / filename).read_text())
    body = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    if len(body) != len(names):
        raise ValueError('Missing requested author function')
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), filename, 'exec'), namespace)


def load_full(transport, isolated_functions, ws):
    verify_full_source()
    # Third-party imports remain intact; only author-package imports are wired here.
    ns = {'__name__': 'evaporate_adapter', 'apply_prompt': transport, 'run_ws': ws}
    load_functions('utils.py', {'get_file_attribute'}, ns)
    for filename in ('prompts.py', 'profiler_utils.py', 'evaluate_profiler.py', 'profiler.py'):
        tree = ast.parse((SOURCE / filename).read_text())
        tree.body = [n for n in tree.body if not (
            isinstance(n, ast.ImportFrom) and (n.module or '').startswith('evaporate'))]
        if filename == 'evaluate_profiler.py':
            # Upstream shadows the list `golds` with one string while computing
            # coverage. Later text_f1 then compares predictions to characters.
            # Rename only this inner loop's binding, preserving the full list.
            class RenameCoverageValue(ast.NodeTransformer):
                def visit_Name(self, node):
                    if node.id == 'golds':
                        node.id = 'coverage_gold_value'
                    return node
            for node in ast.walk(tree):
                if (isinstance(node, ast.For) and isinstance(node.target, ast.Name)
                        and node.target.id == 'golds' and isinstance(node.iter, ast.Subscript)
                        and isinstance(node.iter.value, ast.Name) and node.iter.value.id == 'key2golds'):
                    rename = RenameCoverageValue()
                    node.target = rename.visit(node.target)
                    node.body = [rename.visit(n) for n in node.body]
        tree = FailFastTransport().visit(tree)
        exec(compile(ast.fix_missing_locations(tree), filename, 'exec'), ns)
    ns['apply_final_profiling_functions'] = isolated_functions
    load_functions('run_profiler.py', {'determine_attributes_to_remove'}, ns)
    return SimpleNamespace(**ns)


def vote_matrix(all_votes, has_abstains, threshold, clean, num_elts=5):
    """Author per-document vote buckets, without reading any gold file.

    Unlike upstream get_data, cardinality is fixed by the vote encoding and
    class balance is uniform; labels never enter training or inference.
    """
    import random
    rng = random.Random(0)
    rows, reverse = [], []
    for values in all_votes.values():
        values = [clean(v) for v in values]
        if has_abstains >= threshold:
            values = [v if v else 'abstain' for v in values]
        buckets = [v for v, _ in Counter(values).most_common(num_elts) if v != 'abstain']
        buckets += [f'dummy{i}' for i in range(num_elts - len(buckets))]
        rng.shuffle(buckets)
        mapping = {v: i for i, v in enumerate(buckets)}
        rows.append([mapping[v] + 1 if v in mapping else 0 for v in values])
        reverse.append(dict(enumerate(buckets, start=1)))
    return rows, reverse


def make_ws(clean, epochs=10000, dependency_epochs=80000):
    def run_ws(all_votes, gold_extractions_file=None, symmetric=True, attribute='',
               has_abstains=1.0, extraction_fraction_thresh=.9):
        import sys
        import collections
        import collections.abc
        import numpy as np
        # Compatibility aliases for pinned 2019 Metal on contemporary Python/NumPy.
        for name, value in [('int', int), ('float', float), ('bool', bool)]:
            if name not in np.__dict__:
                setattr(np, name, value)
        for name in ('Mapping', 'MutableMapping', 'Sequence'):
            if not hasattr(collections, name):
                setattr(collections, name, getattr(collections.abc, name))
        sys.path.insert(0, str(SOURCE))
        from metal.label_model import LabelModel
        import scipy as sp
        import cvxpy as cp
        if gold_extractions_file:
            raise ValueError('Reference files are forbidden in inference')
        rows, reverse = vote_matrix(all_votes, has_abstains, extraction_fraction_thresh, clean)
        if not rows:
            return [], False, []
        votes = np.asarray(rows, dtype=int)
        if len({len(row) for row in rows}) != 1:
            raise ValueError('Unequal function vote counts')
        model = LabelModel(k=5, seed=123)
        kwargs = dict(abstains=bool((votes == 0).any()), symmetric=symmetric,
                      class_balance=[.2] * 5, log_train_every=1000)
        model.train_model(votes, n_epochs=epochs, lr=.00001, **kwargs)
        preds = model.predict(votes)          # author: no-deps predictions are computed first (label_model.score)
        deps_used = False
        # Same dependency learner as author. No gold-driven cardinality or Y_dev.
        helpers = {'np': np, 'sp': sp, 'cp': cp}
        load_functions('weak_supervision/run_ws.py', {
            'get_top_deps_from_inverse_sig', 'learn_structure',
            'learn_structure_multiclass', 'get_min_off_diagonal'}, helpers)
        try:
            if votes.shape[0] >= 2 and votes.shape[1] >= 2:
                unscaled = np.where(votes == 0, 0, votes - 1)
                estimate = helpers['learn_structure_multiclass'](unscaled, 5).mean(axis=0)
                if helpers['get_min_off_diagonal'](estimate) < 1:
                    deps = helpers['get_top_deps_from_inverse_sig'](estimate, 1)
                    model.train_model(votes, n_epochs=dependency_epochs, lr=.000001, deps=deps, **kwargs)
                    preds = model.predict(votes)   # author: inside the same try; any failure keeps no-deps preds
                    deps_used = True
        except Exception as exc:
            print('Dependency learning fallback:', type(exc).__name__, flush=True)
        return [mapping.get(int(p), '') for mapping, p in zip(reverse, preds)], deps_used, []
    return run_ws
