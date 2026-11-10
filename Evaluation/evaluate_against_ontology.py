from __future__ import annotations
import argparse, sys
from pathlib import Path
from typing import List, Tuple
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from ontology_lib.config import DEFAULT_MODEL_DEPLOYMENT, DEFAULT_OUT_DIR
from ontology_lib.llm_judge import (
    JudgeCallFailed,
    build_azure_client,
    call_alignment,
    get_model_deployment,
)
from ontology_lib.metrics import compute
from ontology_lib.references import get_reference, list_subdomains
from ontology_lib.report import (
    print_summary,
    write_alignment_csv,
    write_metrics_csv,
)
from ontology_lib.schema_io import load_schema

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LLM-as-judge schema-vs-ontology evaluator.")
    p.add_argument("--schema", type=Path, default=None,
                   help="Single Phase-1 final_schema.json (paired with --subdomain).")
    p.add_argument("--subdomain", type=str, default=None,
                   help=f"Single subdomain name. One of: {', '.join(list_subdomains())}.")
    p.add_argument("--schemas", nargs="+", type=Path, default=None,
                   help="Batch mode: list of final_schema.json paths.")
    p.add_argument("--subdomains", nargs="+", type=str, default=None,
                   help="Batch mode: subdomain names, paired with --schemas by index.")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR, type=Path,
                   help="Where to write metrics.csv and alignments/.")
    p.add_argument("--model", default=None,
                   help="Override Azure deployment "
                        f"(default env AZURE_OPENAI_CHAT_DEPLOYMENT or '{DEFAULT_MODEL_DEPLOYMENT}').")
    return p.parse_args()

def _validate(args: argparse.Namespace) -> List[Tuple[Path, str]]:
    if args.schemas and args.subdomains:
        if len(args.schemas) != len(args.subdomains):
            print("ERROR: --schemas and --subdomains must have the same length.")
            sys.exit(1)
        return [(s, d) for s, d in zip(args.schemas, args.subdomains)]
    if args.schema and args.subdomain:
        return [(args.schema, args.subdomain)]
    print("ERROR: Provide either (--schema --subdomain) or (--schemas --subdomains).")
    print(f"   Available subdomains: {', '.join(list_subdomains())}")
    sys.exit(1)

def main() -> int:
    args = _parse_args()
    pairs = _validate(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Replace metrics.csv with this run's results.
    metrics_path = args.out_dir / "metrics.csv"
    if metrics_path.exists():
        metrics_path.unlink()
    alignments_dir = args.out_dir / "alignments"

    # Reuse one Azure client for all schemas.
    client = build_azure_client()
    model = args.model or get_model_deployment()
    print(f"INFO: judge model: {model}")

    failed = []                                     # pairs without a valid score: no alignment or metrics row written
    for schema_path, subdomain in pairs:
        if not schema_path.exists():
            print(f"ERROR: schema not found: {schema_path}")
            failed.append(subdomain); continue
        try:
            references = get_reference(subdomain)
        except KeyError as e:
            print(f"ERROR: {e}")
            failed.append(subdomain); continue

        induced = load_schema(schema_path)
        print(f"\nINFO: {subdomain}: {len(references)} reference concepts vs "
              f"{len(induced)} induced fields  ({schema_path.name})")
        try:
            rows = call_alignment(client, model, subdomain, references, induced)
        except JudgeCallFailed as e:
            print(f"ERROR: {e}; no alignment or score written")
            (alignments_dir / f"{subdomain}.csv").unlink(missing_ok=True)    # no stale alignment next to the new metrics.csv
            failed.append(subdomain); continue
        m = compute(rows, n_induced=len(induced))

        print_summary(subdomain, references, induced, rows, m)
        write_alignment_csv(alignments_dir / f"{subdomain}.csv", subdomain, induced, rows)
        write_metrics_csv(metrics_path, subdomain, m)

    print(f"\nINFO: wrote:")
    print(f"   {metrics_path}")
    print(f"   {alignments_dir}/<subdomain>.csv")
    if failed:
        print(f"FAILED: no valid score for {len(failed)} schema(s): {', '.join(failed)}")
        if not metrics_path.exists():
            print("   (no metrics.csv written)")
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
