"""Small live M3 cache check. Requires API credentials; --execute enables paid calls."""
import argparse, copy, hashlib, json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        print('Five cases: cold, identical, reordered, changed content, changed chunk configuration. Add --execute for paid calls.')
        return
    out = args.out_dir.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error('Use a new empty output directory.')
    out.mkdir(parents=True, exist_ok=True)
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env', override=False)
    os.environ['MEMOSCHEMA_FIX_CLUSTER'] = '1'
    os.environ['MEMOSCHEMA_MERGE_CACHE_DIR'] = str(out / 'cache')
    sys.path[:0] = [str(ROOT / 'Phase1'), str(ROOT)]
    from schema_lib import schema_cluster_merge as m
    from openai.resources.chat.completions import Completions
    from openai.resources.embeddings import Embeddings
    calls = []
    current = {'case': None}
    def instrument(cls, kind):
        original = cls.create
        def wrapped(self, *a, **kw):
            response = original(self, *a, **kw)
            record = {'case': current['case'], 'kind': kind,
                      'model': getattr(response, 'model', kw.get('model')),
                      'usage': response.usage.model_dump() if response.usage else None}
            calls.append(record)
            with (out / 'api_usage.jsonl').open('a') as f:
                f.write(json.dumps(record) + '\n')
            return response
        cls.create = wrapped
    instrument(Completions, 'chat')
    instrument(Embeddings, 'embedding')
    field = {'name': 'thermal_stability', 'description': 'Thermal decomposition temperature of the framework.',
             'value_type': 'string', 'scope': 'subdomain', 'possible_aliases': [],
             'example_values': ['300 C'], 'evidence_snippets': []}
    rows = [{'paper_id': pid, 'source_file': pid + '.pdf', 'fields': [copy.deepcopy(field)]}
            for pid in ['synthetic_paper_a', 'synthetic_paper_b']]
    changed = copy.deepcopy(rows)
    changed[0]['fields'][0]['example_values'] = ['350 C']
    cases = [('cold', rows, 80, 1), ('identical', rows, 80, 0),
             ('reordered', list(reversed(rows)), 80, 0),
             ('changed_content', changed, 80, 1), ('changed_configuration', rows, 81, 1)]
    results = []
    for name, inputs, threshold, expected in cases:
        current['case'] = name
        sd = out / name
        sd.mkdir()
        fp = sd / 'proposals.jsonl'
        fp.write_text(''.join(json.dumps(r) + '\n' for r in inputs))
        # Simulates a newly started operator with a changed chunk setting.
        m._MERGE_CHUNK_THRESHOLD = threshold
        m._MERGE_OP_VERSION = None
        members = [dict(r['fields'][0], paper_id=r['paper_id'], source_file=r['source_file']) for r in inputs]
        key = m.merge_cache_key(members, 'chem')
        cache = m._merge_cache_path(members, 'chem')
        present = cache.exists()
        before = len(calls)
        canonical, stats = m.build_canonical_candidates(fp, sd, 'chem')
        delta = calls[before:]
        chat_count = sum(c['kind'] == 'chat' for c in delta)
        assert stats['total_coarse_clusters'] == 1, stats
        assert canonical, 'No canonical candidate produced'
        assert cache.exists(), 'Cache record not saved'
        assert chat_count == expected, (name, chat_count, expected)
        assert present == (expected == 0), (name, present)
        result = {'case': name, 'fingerprint': key, 'cache_present_before': present,
                  'chat_calls': chat_count, 'embedding_calls': sum(c['kind'] == 'embedding' for c in delta),
                  'chunk_threshold': threshold, 'cache_sha256': hashlib.sha256(cache.read_bytes()).hexdigest()}
        results.append(result)
        print(json.dumps(result), flush=True)
    assert results[0]['fingerprint'] == results[1]['fingerprint'] == results[2]['fingerprint']
    assert len({results[i]['fingerprint'] for i in [0, 3, 4]}) == 3
    assert results[0]['cache_sha256'] == results[1]['cache_sha256'] == results[2]['cache_sha256']
    source = Path(m.__file__)
    (out / 'verification.json').write_text(json.dumps({'passed': True, 'scope': 'M3 cluster consolidation only; synthetic two-paper fixture',
        'implementation_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'cases': results}, indent=2))
    print('PASS: cache hits, content invalidation, configuration invalidation, and reorder invariance.')

if __name__ == '__main__':
    main()
