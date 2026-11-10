from __future__ import annotations
from dataclasses import dataclass
from typing import List, Set, Tuple
from stability_lib.config import SOFT_MATCH_MIN_OVERLAP
from stability_lib.normalize import canonical_key, field_term_set, overlap
from stability_lib.schema_io import LoadedSchema, SchemaField

# Signatures
@dataclass
class FieldSignature:
    index: int            # position within the LoadedSchema.fields list
    name: str
    canonical: str        # canonical_key(name)
    terms: Set[str]       # field_term_set(...) — used for soft (overlap) match

def build_signatures(schema: LoadedSchema) -> List[FieldSignature]:
    out: List[FieldSignature] = []
    for i, f in enumerate(schema.fields):
        out.append(
            FieldSignature(
                index=i,
                name=f.name,
                canonical=canonical_key(f.name),
                terms=field_term_set(f),
            )
        )
    return out

# Alignment between two signature lists
@dataclass
class AlignedPair:
    a_index: int
    b_index: int
    a_name: str
    b_name: str
    score: float          # term-set overlap coefficient |A∩B|/min(|A|,|B|)
    hard_match: bool      # canonical_keys are identical

@dataclass
class Alignment:
    pairs: List[AlignedPair]
    unmatched_a: List[int]   # field indices in schema A with no partner
    unmatched_b: List[int]   # field indices in schema B with no partner

def _build_score_matrix(
    sigs_a: List[FieldSignature],
    sigs_b: List[FieldSignature],
) -> List[List[float]]:
    return [[overlap(sa.terms, sb.terms) for sb in sigs_b] for sa in sigs_a]

def _assign_greedy(score_matrix: List[List[float]]) -> List[Tuple[int, int]]:
    """Repeatedly pick the highest-scoring remaining pair."""
    n_a, n_b = len(score_matrix), (len(score_matrix[0]) if score_matrix else 0)
    candidates: List[Tuple[float, int, int]] = []
    for i in range(n_a):
        for j in range(n_b):
            s = score_matrix[i][j]
            if s > 0:
                candidates.append((s, i, j))
    candidates.sort(reverse=True)

    used_a: Set[int] = set()
    used_b: Set[int] = set()
    pairs: List[Tuple[int, int]] = []
    for s, i, j in candidates:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        pairs.append((i, j))
    return pairs

def align_schemas(
    schema_a: LoadedSchema,
    schema_b: LoadedSchema,
    threshold: float = SOFT_MATCH_MIN_OVERLAP,
) -> Alignment:
    sigs_a = build_signatures(schema_a)
    sigs_b = build_signatures(schema_b)
    score = _build_score_matrix(sigs_a, sigs_b)
    assignments = _assign_greedy(score)   # greedy one-to-one, as in the paper

    pairs: List[AlignedPair] = []
    matched_a: Set[int] = set()
    matched_b: Set[int] = set()
    for i, j in assignments:
        s = score[i][j]
        if s < threshold:
            continue
        pairs.append(
            AlignedPair(
                a_index=sigs_a[i].index,
                b_index=sigs_b[j].index,
                a_name=sigs_a[i].name,
                b_name=sigs_b[j].name,
                score=s,
                hard_match=(sigs_a[i].canonical == sigs_b[j].canonical and bool(sigs_a[i].canonical)),
            )
        )
        matched_a.add(i)
        matched_b.add(j)

    unmatched_a = [s.index for k, s in enumerate(sigs_a) if k not in matched_a]
    unmatched_b = [s.index for k, s in enumerate(sigs_b) if k not in matched_b]

    return Alignment(pairs=pairs, unmatched_a=unmatched_a, unmatched_b=unmatched_b)
