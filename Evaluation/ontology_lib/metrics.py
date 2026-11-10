from __future__ import annotations
from dataclasses import dataclass
from typing import List
from ontology_lib.config import PARTIAL_CREDIT_WEIGHT
from ontology_lib.llm_judge import AlignmentRow

@dataclass
class Metrics:
    n_reference: int
    n_induced: int
    n_strong: int
    n_partial: int
    n_none: int
    n_induced_used: int

    # STRICT = partial counts as 0.5
    strict_tp: float
    strict_fp: float
    strict_fn: float
    strict_precision: float
    strict_recall: float
    strict_f1: float

    # LENIENT = partial counts as 1.0
    lenient_tp: float
    lenient_fp: float
    lenient_fn: float
    lenient_precision: float
    lenient_recall: float
    lenient_f1: float

    partial_credit_weight: float

def _f1(p: float, r: float) -> float:
    return (2 * p * r / (p + r)) if (p + r) else 0.0

def compute(rows: List[AlignmentRow], n_induced: int,
            partial_weight: float = PARTIAL_CREDIT_WEIGHT) -> Metrics:
    n_reference = len(rows)
    n_strong = sum(1 for r in rows if r.strength == "strong")
    n_partial = sum(1 for r in rows if r.strength == "partial")
    n_none = sum(1 for r in rows if r.strength == "none")

    used_any = {r.matched_induced for r in rows
                if r.strength != "none" and r.matched_induced}

    fp_count = max(0, n_induced - len(used_any))

    # Strict: a partial match counts `partial_weight` (default 0.5)
    strict_tp = n_strong + partial_weight * n_partial
    strict_fp = fp_count
    strict_fn = n_none + (1 - partial_weight) * n_partial
    strict_p = (strict_tp / (strict_tp + strict_fp)) if (strict_tp + strict_fp) else 0.0
    strict_r = (strict_tp / n_reference) if n_reference else 0.0
    strict_f1 = _f1(strict_p, strict_r)

    # Lenient: strong and partial matches both count fully
    lenient_tp = n_strong + n_partial
    lenient_fp = fp_count
    lenient_fn = n_none
    lenient_p = (lenient_tp / (lenient_tp + lenient_fp)) if (lenient_tp + lenient_fp) else 0.0
    lenient_r = (lenient_tp / n_reference) if n_reference else 0.0
    lenient_f1 = _f1(lenient_p, lenient_r)

    return Metrics(
        n_reference=n_reference,
        n_induced=n_induced,
        n_strong=n_strong,
        n_partial=n_partial,
        n_none=n_none,
        n_induced_used=len(used_any),
        strict_tp=strict_tp,
        strict_fp=strict_fp,
        strict_fn=strict_fn,
        strict_precision=strict_p,
        strict_recall=strict_r,
        strict_f1=strict_f1,
        lenient_tp=lenient_tp,
        lenient_fp=lenient_fp,
        lenient_fn=lenient_fn,
        lenient_precision=lenient_p,
        lenient_recall=lenient_r,
        lenient_f1=lenient_f1,
        partial_credit_weight=partial_weight,
    )
