"""Build a corpus manifest from one iterative run directory (no API calls, no PDFs copied).

  python scripts/build_corpus_manifest.py --run-dir <output_root>/iterative_phase1_<domain>_<timestamp> \
      --batch r7 --run ZIF_01 --repo-root <repo the run was started from> --out data/corpora/r7/ZIF_01.json

Fields:
  papers_collected                  every new unique paper per retrieval round (roundNN/new_unique_papers.jsonl, in file
                                    order): paper_id (PDF file stem), title, doi, source, url (from
                                    roundNN/downloaded_metadata.jsonl when the PDF was downloaded through it, else null)
  papers_used_for_schema_induction  sorted paper ids in schema_state/paper_level_field_proposals.jsonl, i.e. the papers
                                    that were actually distilled and proposed fields
  induction_papers_not_in_collected used papers that a round downloaded but did not list as new unique papers, with
                                    title/doi/source/url from roundNN/downloaded_metadata.jsonl (metadata_found=false and
                                    nulls when no record exists)
  run_config                        latest <output_root>/run_config_<timestamp>.json not newer than the run directory
                                    (written by run_memoschema_pipeline.py), absolute repo path replaced by "<repo>";
                                    null if the run was not started through that script
Extra keys can be added with --meta key=value (e.g. role annotations). Missing information stays null; nothing is guessed.
"""
import argparse, json, re
from pathlib import Path

def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if p.exists() else []

def build(rd: Path, batch: str, run: str, repo_root: str | None):
    rounds = sorted((p for p in rd.iterdir() if re.fullmatch(r"round\d+", p.name)), key=lambda p: int(p.name[5:]))
    collected, downloaded = [], {}
    for rdir in rounds:
        dm = {}
        for x in jl(rdir / "downloaded_metadata.jsonl"):
            if x.get("expected_filename"):
                dm[Path(x["expected_filename"]).stem] = x
                downloaded.setdefault(Path(x["expected_filename"]).stem, (int(rdir.name[5:]), x))
        for x in jl(rdir / "new_unique_papers.jsonl"):
            pid = Path(x["pdf_filename"]).stem if x.get("pdf_filename") else None
            collected.append({"round": int(rdir.name[5:]), "paper_id": pid, "title": x.get("title"), "doi": x.get("doi"),
                              "source": x.get("source"), "url": (dm.get(pid) or {}).get("url")})
    used = sorted({x["paper_id"] for x in jl(rd / "schema_state" / "paper_level_field_proposals.jsonl")})
    seen = {c["paper_id"] for c in collected}
    extra = []
    for pid in used:
        if pid in seen: continue
        r, x = downloaded.get(pid, (None, {}))
        extra.append({"round": r, "paper_id": pid, "title": x.get("title"), "doi": x.get("doi"), "source": x.get("source"),
                      "url": x.get("url"), "metadata_found": pid in downloaded})
    # The pipeline writes run_config_<t>.json just before iterative_expand creates iterative_phase1_<domain>_<t'>:
    # take the latest config not newer than the run directory.
    stamp = "_".join(rd.name.rsplit("_", 2)[-2:])
    cfgs = sorted(f for f in rd.parent.glob("run_config_*.json") if f.stem[len("run_config_"):] <= stamp)
    cfg = json.loads(cfgs[-1].read_text()) if cfgs else None
    if cfg is not None and repo_root:
        cfg = json.loads(json.dumps(cfg).replace(json.dumps(repo_root.rstrip("/"))[1:-1], "<repo>"))
    return {"batch": batch, "run": run, "papers_collected": collected, "papers_used_for_schema_induction": used, "run_config": cfg,
            "induction_papers_not_in_collected": extra}

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True); ap.add_argument("--batch", required=True)
    ap.add_argument("--run", required=True); ap.add_argument("--repo-root"); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--meta", action="append", default=[], help="extra top-level key=value (value parsed as JSON if possible)")
    a = ap.parse_args()
    m = build(a.run_dir, a.batch, a.run, a.repo_root)
    for kv in a.meta:
        k, v = kv.split("=", 1)
        try: v = json.loads(v)
        except ValueError: pass
        m[k] = v
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(m, ensure_ascii=False, indent=1) + "\n")

if __name__ == "__main__":
    main()
