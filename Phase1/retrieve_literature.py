
import json, os, re, threading, time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Set
from urllib.parse import urlparse

_DOWNLOAD_HARD_CAP_S = 180

def _timed_download(fn: Callable[..., bool], *args, cap: int = _DOWNLOAD_HARD_CAP_S) -> bool:
    result = [False]

    def _run():
        try:
            result[0] = fn(*args)
        except Exception:
            pass

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=cap)
    if t.is_alive():
        print(f"  [_timed_download] hard cap ({cap}s) hit for {fn.__name__}")
        # Best-effort cleanup: remove a partial file if dest was passed as arg 2
        try:
            dest_candidates = [a for a in args if isinstance(a, (str, Path))]
            if len(dest_candidates) >= 2:
                dest = Path(dest_candidates[1])  # convention: (hit, download_dir, fname)
                # dest_candidates[1] is download_dir; fname is dest_candidates[2]
            # actual dest = download_dir / fname
            dl_dir = Path(dest_candidates[1]) if len(dest_candidates) > 1 else None
            fname  = dest_candidates[2]        if len(dest_candidates) > 2 else None
            if dl_dir and fname:
                p = dl_dir / fname
                if p.exists():
                    p.unlink(missing_ok=True)
        except Exception:
            pass
        return False
    return result[0]

# Hard cap (seconds) on a single source's SEARCH call (not download).
_SEARCH_HARD_CAP_S = 30

def _timed_search(fn: Callable[..., list], *args,
                  cap: int = _SEARCH_HARD_CAP_S, label: str = "") -> list:
    result = [[]]

    def _run():
        try:
            result[0] = fn(*args) or []
        except Exception as e:
            print(f"[_timed_search] {label or fn.__name__} error: {e}")

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=cap)
    if t.is_alive():
        print(f"[retrieve_literature] source search hard cap ({cap}s) hit: "
              f"{label or fn.__name__} — returning []")
        return []
    return result[0]

from retrieve_lib.searches import (
    download_acl_hit,
    download_biorxiv_hit,
    download_openreview_hit,
    download_pubmed_or_pmc_hit,
    filter_query_for_biorxiv,
    search_acl,
    search_arxiv,
    search_biorxiv,
    search_core,
    search_crossref_nature,
    search_openalex,
    search_openreview,
    search_pubmed,
)
from retrieve_lib.utils import (
    dedupe,
    delete_empty_folders,
    delete_invalid_pdfs,
    download_pdf,
    fname_with_tag,
)

def get_record_key(rec: dict) -> str | None:
    if not rec:
        return None
    doi = rec.get("doi")
    if doi:
        return doi.lower().strip()
    rid = rec.get("id")
    if rid:
        return str(rid).lower().strip()
    title = rec.get("title") or ""
    if title:
        return re.sub(r"\s+", " ", title.strip().lower())
    return None

def _load_queries(queries_path: Path) -> List[Dict[str, str]]:
    raw = json.loads(queries_path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        items = raw.get("queries", [])
    elif isinstance(raw, list):
        items = raw
    else:
        raise ValueError(f"Unsupported queries.json format: {type(raw)}")

    out = []
    for q in items:
        if isinstance(q, str):
            out.append({"query_bool": q, "query_nl": q})
        elif isinstance(q, dict):
            qb = q.get("query_bool") or q.get("query_nl") or ""
            qn = q.get("query_nl") or q.get("query_bool") or ""
            out.append({"query_bool": qb, "query_nl": qn})
    if not out:
        raise ValueError(f"No usable queries found in {queries_path}")
    return out

# Search dispatch + query shaping
def _build_query(src: str, q: dict, domain_keyword: str) -> str:
    qb = q.get("query_bool") or q.get("query_nl") or ""
    qn = q.get("query_nl") or qb

    if domain_keyword:
        simplified = f'"{domain_keyword}" AND {qn}'.replace('"', '').replace("'", "")
    else:
        simplified = qn

    if src in ("core", "pubmed"):
        return qn or simplified
    if src == "biorxiv":
        return filter_query_for_biorxiv(qn or simplified)
    # NL-style sources: pass natural-language with optional domain prefix.
    if src in ("openalex", "nature", "openreview", "acl"):
        return simplified
    return qb  # arxiv and anything else (boolean-aware sources)

def _run_search(src: str, query: str, per_source: int, *,
                core_api_key: str | None,
                openalex_mailto: str | None,
                crossref_mailto: str | None,
                arxiv_cats: list,
                domain_spec=None) -> list:
    if src == "arxiv":
        return search_arxiv(query, per_source, arxiv_cats, domain_spec=domain_spec) or []
    if src == "openalex":
        return _timed_search(
            search_openalex, query, per_source,
            openalex_mailto, domain_spec,
            cap=_SEARCH_HARD_CAP_S, label="openalex",
        )
    if src == "nature":
        return search_crossref_nature(query, per_source, mailto=crossref_mailto,
                                      domain_spec=domain_spec) or []
    if src == "biorxiv":
        return search_biorxiv(query, per_source, domain_spec=domain_spec) or []
    if src == "pubmed":
        return search_pubmed(query, per_source, mailto=openalex_mailto,
                             domain_spec=domain_spec) or []
    if src == "core":
        # CORE returns None on rate-limit (vs [] on no results).
        # Wrapped in _timed_search so a slow CORE API cannot stall the pipeline.
        def _core_with_retry():
            for attempt in range(2):
                hits = search_core(query, per_source,
                                   api_key=core_api_key, domain_spec=domain_spec)
                if hits is None:
                    wait = 5 * (attempt + 1)
                    print(f"[CORE] Rate limited. Retry in {wait}s...")
                    time.sleep(wait)
                    continue
                time.sleep(1.0 if hits else 0.8)
                return hits or []
            return []
        return _timed_search(_core_with_retry, cap=_SEARCH_HARD_CAP_S, label="core")
    if src == "openreview":
        return search_openreview(query, per_source, domain_spec=domain_spec) or []
    if src == "acl":
        return search_acl(query, per_source, domain_spec=domain_spec) or []
    print(f"[retrieve_literature] Unknown source: {src}")
    return []

# Ranking
def _dl_score(r: Dict) -> tuple:
    host = (urlparse(r.get("pdf_url") or r.get("url") or "").hostname or "").lower()
    return (
        # domain_score is set by biorxiv/core; legacy ad_score kept for back-compat
        r.get("domain_score", r.get("ad_score", 0)),
        1 if r.get("pdf_url") else 0,
        1 if "arxiv.org" in host else 0,
    )

# DomainSpec.arxiv_bool is a string like "(cat:cs.CL OR cat:cs.AI OR cat:cs.LG)"
_CAT_RE = re.compile(r"cat:([\w\-\.]+)")

def _parse_arxiv_cats(arxiv_bool: str) -> List[str]:
    if not arxiv_bool:
        return []
    return _CAT_RE.findall(arxiv_bool)

# Retrieval for one round: collect, rank, select, download
def run_one_round_search_and_download(
    queries_path: Path,
    round_dir: Path,
    domain_spec,
    core_api_key: str = None,
    results_per_source: int = 7,
    sources=("biorxiv", "openalex", "pubmed", "core"),
    contact_emails: Dict[str, str] = None,
    exclude_keys: Set[str] = None,
):
    if exclude_keys is None:
        exclude_keys = set()

    target_per_round = results_per_source * len(sources)
    candidate_budget = max(target_per_round * 3, 50)

    round_dir.mkdir(parents=True, exist_ok=True)
    download_dir = round_dir / "downloads"
    download_dir.mkdir(parents=True, exist_ok=True)

    queries = _load_queries(queries_path)

    crossref_mailto = (contact_emails or {}).get("CROSSREF_MAILTO") or os.getenv("CROSSREF_MAILTO")
    openalex_mailto = (contact_emails or {}).get("OPENALEX_MAILTO") or os.getenv("OPENALEX_MAILTO") or crossref_mailto

    domain_keyword = domain_spec.tiers[-1] if domain_spec.tiers else ""
    # arxiv_bool is a STRING in DomainSpec (e.g. "(cat:cs.CL OR cat:cs.AI)").
    arxiv_cats = _parse_arxiv_cats(getattr(domain_spec, "arxiv_bool", "") or "")

    # Collect candidates
    candidate_map: Dict[str, Dict] = {}

    for q in queries:
        if len(candidate_map) >= candidate_budget:
            break
        for src in sources:
            if len(candidate_map) >= candidate_budget:
                break
            need = candidate_budget - len(candidate_map)
            per_source = min(results_per_source, need)
            query = _build_query(src, q, domain_keyword)

            try:
                hits = _run_search(
                    src, query, per_source,
                    core_api_key=core_api_key,
                    openalex_mailto=openalex_mailto,
                    crossref_mailto=crossref_mailto,
                    arxiv_cats=arxiv_cats,
                    domain_spec=domain_spec,
                )
            except Exception as e:
                print(f"[ERROR] {src} search failed for '{query[:50]}...': {e}")
                continue

            if not hits:
                if src == "pubmed":
                    print(f"[pubmed] No results for: {query}")
                continue

            for h in hits:
                key = get_record_key(h)
                if not key or key in exclude_keys or key in candidate_map:
                    continue
                candidate_map[key] = h

    # Rank + select
    all_hits = sorted(dedupe(list(candidate_map.values())), key=_dl_score, reverse=True)
    for rec in all_hits:
        rec["expected_filename"] = fname_with_tag(rec)

    # search_results.jsonl: top target_per_round candidates as a metadata snapshot
    selected_for_metadata = all_hits[:target_per_round]
    search_results_path = round_dir / "search_results.jsonl"
    with open(search_results_path, "w", encoding="utf-8") as f:
        for r in selected_for_metadata:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    downloaded_paths: List[Path] = []
    downloaded_per_source: Dict[str, int] = {}
    attempted_keys: Set[str] = set()

    def _record_success(src_lower: str, path: Path) -> None:
        downloaded_paths.append(path)
        downloaded_per_source[src_lower] = downloaded_per_source.get(src_lower, 0) + 1

    def _try_download(rec: Dict) -> None:
        source = (rec.get("source") or "").lower()
        fname = fname_with_tag(rec)
        dest = download_dir / fname

        if dest.exists() and dest.stat().st_size > 1000:
            if dest not in downloaded_paths:
                _record_success(source, dest)
            return

        # PubMed/PMC: Europe PMC + scrape, no browser needed
        if source == "pubmed":
            if _timed_download(download_pubmed_or_pmc_hit, rec, download_dir, fname):
                if dest.exists() and dest.stat().st_size > 1000:
                    _record_success(source, dest)
            return

        # bioRxiv: direct biorxiv -> Europe PMC preprint fallback (Cloudflare-safe)
        if source == "biorxiv":
            if _timed_download(download_biorxiv_hit, rec, download_dir, fname):
                if dest.exists() and dest.stat().st_size > 1000:
                    _record_success(source, dest)
            return

        # OpenReview: direct PDF (no auth, no anti-bot)
        if source == "openreview":
            if _timed_download(download_openreview_hit, rec, download_dir, fname):
                if dest.exists() and dest.stat().st_size > 1000:
                    _record_success(source, dest)
            return

        # ACL Anthology: direct PDF (no auth, no anti-bot)
        if source == "acl":
            if _timed_download(download_acl_hit, rec, download_dir, fname):
                if dest.exists() and dest.stat().st_size > 1000:
                    _record_success(source, dest)
            return

        # Pre-existing local file
        if rec.get("local_path") and os.path.exists(rec["local_path"]):
            _record_success(source, Path(rec["local_path"]))
            return

        # Generic direct PDF
        pdf_guess = rec.get("pdf_url")
        if pdf_guess and download_pdf(pdf_guess, dest):
            if dest.exists() and dest.stat().st_size > 1000:
                _record_success(source, dest)
                return

    def _record_key(rec: Dict) -> str:
        return get_record_key(rec) or f"_obj_{id(rec)}"

    # Pass 1: per-source cap
    for rec in all_hits:
        if len(downloaded_paths) >= target_per_round:
            break
        source = (rec.get("source") or "").lower()
        if downloaded_per_source.get(source, 0) >= results_per_source:
            continue  # cap hit; defer this record to pass 2
        key = _record_key(rec)
        if key in attempted_keys:
            continue
        attempted_keys.add(key)
        _try_download(rec)

    pass1_per_source = dict(downloaded_per_source)
    print(f"[retrieve_literature] Pass 1 (capped) per-source: {pass1_per_source} "
          f"-> {len(downloaded_paths)}/{target_per_round}")

    # Pass 2: overflow (no cap) to backfill empty slots
    if len(downloaded_paths) < target_per_round:
        for rec in all_hits:
            if len(downloaded_paths) >= target_per_round:
                break
            key = _record_key(rec)
            if key in attempted_keys:
                continue  # already tried in pass 1 (succeeded or failed)
            attempted_keys.add(key)
            _try_download(rec)
        overflow = {s: downloaded_per_source[s] - pass1_per_source.get(s, 0)
                    for s in downloaded_per_source
                    if downloaded_per_source[s] - pass1_per_source.get(s, 0) > 0}
        if overflow:
            print(f"[retrieve_literature] Pass 2 (overflow) added: {overflow}")

    print(f"[retrieve_literature] Final per-source downloads: {dict(downloaded_per_source)}")

    # Cleanup + final manifest
    print(f"[Debug] Before cleanup: {[f.name for f in download_dir.glob('*.pdf')]}")
    delete_invalid_pdfs(download_dir)
    delete_empty_folders(download_dir)

    # Only count files this round actually picked up (cached or freshly fetched).
    final_pdf_paths = [p for p in downloaded_paths
                       if p.exists() and p.stat().st_size > 1000]
    final_names = {p.name for p in final_pdf_paths}

    # Walk the full ranked pool so any candidate that actually got a PDF
    # on disk ends up in the manifest, but dedupe by filename — multiple
    # candidate records (different OpenReview IDs, ACL+OpenReview duplicates)
    # often map to the same physical file.
    downloaded_metadata = []
    seen_filenames: Set[str] = set()
    for rec in all_hits:
        fname = rec.get("expected_filename")
        if fname and fname in final_names and fname not in seen_filenames:
            seen_filenames.add(fname)
            rec_out = dict(rec)
            rec_out["local_pdf_path"] = str(download_dir / fname)
            downloaded_metadata.append(rec_out)

    downloaded_metadata_path = round_dir / "downloaded_metadata.jsonl"
    with open(downloaded_metadata_path, "w", encoding="utf-8") as f:
        for r in downloaded_metadata:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return {
        "download_dir": download_dir,
        "selected_pdf_paths": final_pdf_paths,
        "search_results_path": search_results_path,
        "downloaded_metadata_path": downloaded_metadata_path,
    }

# Command line
if __name__ == "__main__":
    import argparse
    from dataclasses import replace
    from queries_lib.domain import DOMAINS, get_spec

    parser = argparse.ArgumentParser(description="Run one expansion round.")
    parser.add_argument("--queries", required=True,
                        help="Path to queries.json for this round")
    parser.add_argument("--output", default=None,
                        help="Round directory (default: runs/<domain>/test_expand_<ts>)")
    parser.add_argument("--domain", required=True, choices=list(DOMAINS.keys()),
                        help="Active domain (must match an entry in queries_lib.domain.DOMAINS)")
    parser.add_argument(
        "--last_tier",
        required=True,
        help="Most-specific subdomain term (e.g. 'Neuroinflammation microglia'). "
             "Appended to domain_spec.tiers at runtime so this script and "
             "queries_lib/domain.py do not need to agree on a hardcoded last tier.",
    )
    parser.add_argument("--results_per_source", type=int, default=20)
    parser.add_argument("--sources", required=True,
                        help="Comma-separated subset of: "
                             "arxiv,openalex,nature,pubmed,biorxiv,core,openreview,acl. "
                             "No default — pick the sources that fit your domain.")
    parser.add_argument(
        "--crossref_email",
        default=os.getenv("CROSSREF_MAILTO"),
    )
    parser.add_argument(
        "--openalex_email",
        default=os.getenv("OPENALEX_MAILTO"),
    )
    parser.add_argument("--refresh_downloads", action="store_true",
                        help="If set, clears the round's downloads/ folder before fetching")
    parser.add_argument("--core_api_key", default=os.getenv("CORE_API_KEY"))
    args = parser.parse_args()

    queries_path = Path(args.queries)
    if not queries_path.exists():
        raise FileNotFoundError(f"queries.json not found: {queries_path}")

    BASE_DIR = Path(__file__).resolve().parent.parent
    if args.output is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        round_dir = BASE_DIR / "runs" / args.domain / f"test_expand_{ts}"
    else:
        round_dir = Path(args.output)
    round_dir.mkdir(parents=True, exist_ok=True)
    print(f"[retrieve_literature] Using round dir: {round_dir}")

    domain_spec = get_spec(args.domain)
    # Append CLI-supplied last_tier so domain_spec.tiers[-1] is correct
    # for both `_build_query` and the source-side filters.
    domain_spec = replace(domain_spec, tiers=list(domain_spec.tiers) + [args.last_tier])

    if args.refresh_downloads:
        dl_dir = round_dir / "downloads"
        if dl_dir.exists():
            for p in dl_dir.glob("*"):
                try:
                    p.unlink()
                except IsADirectoryError:
                    pass

    sources = tuple(s.strip() for s in args.sources.split(",") if s.strip())
    contact_emails = {}
    if args.crossref_email:
        contact_emails["CROSSREF_MAILTO"] = args.crossref_email
    if args.openalex_email:
        contact_emails["OPENALEX_MAILTO"] = args.openalex_email

    result = run_one_round_search_and_download(
        queries_path=queries_path,
        round_dir=round_dir,
        domain_spec=domain_spec,
        results_per_source=args.results_per_source,
        sources=sources,
        contact_emails=contact_emails or None,
        core_api_key=args.core_api_key,
    )

    print("\n[retrieve_literature] Completed this round.")
    print(f" - Metadata: {result['search_results_path']}")
    print(f" - Downloads dir: {result['download_dir']}")
    pdfs = list(result["selected_pdf_paths"])
    print(f" - Downloaded PDFs: {len(pdfs)}")
    for p in pdfs[:3]:
        print(f"   • {p.name}")
    if len(pdfs) > 3:
        print(f"   … (+{len(pdfs)-3} more)")
