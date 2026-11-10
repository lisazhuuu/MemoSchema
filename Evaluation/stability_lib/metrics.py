from __future__ import annotations
from typing import List, Set
from stability_lib.matching import Alignment
from stability_lib.normalize import canonical_key
from stability_lib.schema_io import LoadedSchema

def canonical_key_set(schema: LoadedSchema) -> Set[str]:
    """Set of canonicalized field names."""
    return {canonical_key(f.name) for f in schema.fields if f.name}

def overlap_coefficient(a: LoadedSchema, b: LoadedSchema) -> float:
    keys_a = canonical_key_set(a)
    keys_b = canonical_key_set(b)
    if not keys_a or not keys_b:
        return 0.0
    inter = len(keys_a & keys_b)
    if inter == 0:
        return 0.0
    return inter / min(len(keys_a), len(keys_b))

def soft_jaccard(alignment: Alignment, n_a: int, n_b: int) -> float:
    """Alias-aware Jaccard: |matched| / (n_a + n_b - |matched|)."""
    m = len(alignment.pairs)
    denom = n_a + n_b - m
    if denom <= 0:
        return 0.0
    return m / denom

def mean_pair_score(alignment: Alignment) -> float:
    if not alignment.pairs:
        return 0.0
    return sum(p.score for p in alignment.pairs) / len(alignment.pairs)
