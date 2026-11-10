#!/usr/bin/env python3
"""AutoSchema Phase II (evidence-grounded extraction, unmodified) on frozen paper texts.
Only change: extract_pdf_text returns the frozen first-12k text of each paper (placeholder .pdf files carry the ids).
Usage: run_phase2.py --schema S.json --papers papers.jsonl --ids id1,id2,... --out DIR [--strip]
--strip keeps only name, description and value_type of every field (equal retrieval inputs for every method)."""
import argparse, json, os, sys, tempfile
from pathlib import Path
HERE = Path(__file__).resolve().parent; REPO = HERE.parents[1]; UP = HERE / "upstream"
sys.path.insert(0, str(HERE))
from run_paired import env_setup

def main():
    p = argparse.ArgumentParser(); p.add_argument("--schema", required=True); p.add_argument("--papers", required=True)
    p.add_argument("--ids", required=True); p.add_argument("--out", type=Path, required=True); p.add_argument("--strip", action="store_true")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    env_setup(a.out)
    texts = {json.loads(l)["paper_id"]: json.loads(l)["text"] for l in open(a.papers)}
    ids = [i for i in a.ids.split("|") if i in texts]
    S = json.load(open(a.schema))
    if a.strip:
        S = [{"name": x.get("name"), "description": x.get("description", "") or "", "value_type": x.get("value_type", "string") or "string"} for x in S]
    sp = a.out / "schema_used.json"; sp.write_text(json.dumps(S, ensure_ascii=False, indent=1))
    pdf_dir = Path(tempfile.mkdtemp())
    for i in ids:
        (pdf_dir / f"{i}.pdf").write_bytes(b"")
    sys.path.insert(0, str(UP)); os.chdir(UP)
    import extract_schema_values as esv
    esv.extract_pdf_text = lambda pdf_path: texts[Path(pdf_path).stem]
    sys.argv = ["extract_schema_values.py", "--pdf_dir", str(pdf_dir), "--schema_path", str(sp), "--out_dir", str(a.out),
                "--max_chars_per_paper", "12000"]
    esv.main()

if __name__ == "__main__":
    main()
