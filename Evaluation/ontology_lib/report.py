from __future__ import annotations
import csv
from pathlib import Path
from typing import List
from ontology_lib.llm_judge import AlignmentRow
from ontology_lib.metrics import Metrics
from ontology_lib.schema_io import SchemaField

def print_summary(
    subdomain: str,
    references: List[str],
    induced: List[SchemaField],
    rows: List[AlignmentRow],
    m: Metrics,
) -> None:
    print()
    print("=" * 72)
    print(f"  Ontology evaluation — subdomain: {subdomain}")
    print("=" * 72)
    n_matched = m.n_strong + m.n_partial
    print(f"  reference concepts:    {m.n_reference}")
    print(f"  induced fields:        {m.n_induced}")
    print(f"  matched pairs:         {n_matched}   "
          f"(strong {m.n_strong}  +  partial {m.n_partial})")
    print(f"  missed refs (FN):      {m.n_none}")
    print(f"  extra induced (FP):    {m.n_induced - m.n_induced_used}")
    print("  " + "-" * 60)
    print(f"  STRICT   (strong=1, partial={m.partial_credit_weight})    "
          f"P={m.strict_precision:.3f}  R={m.strict_recall:.3f}  F1={m.strict_f1:.3f}")
    print(f"  LENIENT  (strong=1, partial=1)     "
          f"P={m.lenient_precision:.3f}  R={m.lenient_recall:.3f}  F1={m.lenient_f1:.3f}")
    print("=" * 72)

    # Matched pairs: strong first, then partial; nothing else here
    matched = [r for r in rows if r.matched_induced and r.strength != "none"]
    if matched:
        order = {"strong": 0, "partial": 1}
        matched.sort(key=lambda r: (order.get(r.strength, 9), r.ref_name))
        print("\n  Matched pairs:")
        for r in matched:
            tag = "STRONG " if r.strength == "strong" else "partial"
            print(f"    [{tag}]  {r.ref_name:40s}  <->  {r.matched_induced}")

    # FN: reference concepts with no match
    unmatched_refs = [r.ref_name for r in rows if r.strength == "none"]
    if unmatched_refs:
        print("\n  Reference concepts NOT matched (false negatives):")
        for n in unmatched_refs:
            print(f"    - {n}")

    # FP: induced fields nobody picked
    used = {r.matched_induced for r in rows if r.matched_induced and r.strength != "none"}
    unused = [f.name for f in induced if f.name not in used]
    if unused:
        print("\n  Induced fields NOT in reference (false positives):")
        for n in unused:
            print(f"    - {n}")

# CSV writers
def write_alignment_csv(
    path: Path,
    subdomain: str,
    induced: List[SchemaField],
    rows: List[AlignmentRow],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subdomain", "status", "reference_name", "induced_name",
                    "strength", "reasoning"])
        for r in rows:
            if r.matched_induced and r.strength != "none":
                w.writerow([subdomain, "matched", r.ref_name, r.matched_induced,
                            r.strength, r.reasoning])
            else:
                w.writerow([subdomain, "missing_reference", r.ref_name, "",
                            r.strength, r.reasoning])
        used = {r.matched_induced for r in rows
                if r.matched_induced and r.strength != "none"}
        for f_ind in induced:
            if f_ind.name not in used:
                w.writerow([subdomain, "extra_induced", "", f_ind.name, "", ""])

def write_metrics_csv(path: Path, subdomain: str, m: Metrics) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow([
                "subdomain",
                "n_reference", "n_induced",
                "n_strong", "n_partial", "n_none",
                "strict_TP", "strict_FP", "strict_FN",
                "strict_precision", "strict_recall", "strict_f1",
                "lenient_TP", "lenient_FP", "lenient_FN",
                "lenient_precision", "lenient_recall", "lenient_f1",
                "partial_credit_weight",
            ])
        w.writerow([
            subdomain,
            m.n_reference, m.n_induced,
            m.n_strong, m.n_partial, m.n_none,
            f"{m.strict_tp:.2f}", f"{m.strict_fp:.2f}", f"{m.strict_fn:.2f}",
            f"{m.strict_precision:.4f}", f"{m.strict_recall:.4f}", f"{m.strict_f1:.4f}",
            f"{m.lenient_tp:.2f}", f"{m.lenient_fp:.2f}", f"{m.lenient_fn:.2f}",
            f"{m.lenient_precision:.4f}", f"{m.lenient_recall:.4f}", f"{m.lenient_f1:.4f}",
            m.partial_credit_weight,
        ])
