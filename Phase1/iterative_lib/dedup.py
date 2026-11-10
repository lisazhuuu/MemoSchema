import shutil, json, re
from pathlib import Path
# Imports
import retrieve_literature
from retrieve_lib.utils import fname_with_tag
from retrieve_literature import get_record_key

# Title-based fuzzy deduplication
def normalize_title(title: str | None) -> str:
    if not title:
        return ""
    return re.sub(r"\s+", " ", str(title).lower()).strip()

def collect_round_titles(round_dir: Path) -> set[str]:
    titles: set[str] = set()
    p = round_dir / "search_results.jsonl"
    if not p.exists():
        return titles
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = normalize_title(rec.get("title"))
            if t:
                titles.add(t)
    return titles

def prune_by_history_title(
    round_idx: int,
    round_dir: Path,
    download_dir: Path,
    history_titles: set[str],
) -> int:
    if not history_titles:
        return 0

    search_results_path = round_dir / "search_results.jsonl"
    if not search_results_path.exists():
        return 0

    kept_records: list[dict] = []
    removed_count = 0
    with open(search_results_path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                continue

            t = normalize_title(rec.get("title"))
            if t and t in history_titles:
                # Title-collision with history -> drop the PDF
                try:
                    fname = fname_with_tag(rec)
                except Exception:
                    fname = None
                if fname:
                    pdf_path = download_dir / fname
                    if pdf_path.exists():
                        try:
                            pdf_path.unlink()
                            removed_count += 1
                        except OSError:
                            pass
                continue
            kept_records.append(rec)

    if removed_count > 0:
        with open(search_results_path, "w", encoding="utf-8") as f:
            for rec in kept_records:
                f.write(json.dumps(rec) + "\n")
        print(
            f"[dedup-title] round {round_idx}: removed {removed_count} PDF(s) "
            f"by fuzzy-title match against history "
            f"(history_titles_size={len(history_titles)})."
        )

    return removed_count

# Load all paper keys from a round's search_results.jsonl & return records
def load_round_paper_keys(round_dir: Path, download_dir: Path) -> tuple[set[str], list[dict]]:
    downloaded_meta_path = round_dir / "downloaded_metadata.jsonl"
    search_results_path = round_dir / "search_results.jsonl"
    if downloaded_meta_path.exists():
        source_path = downloaded_meta_path
    elif search_results_path.exists():
        source_path = search_results_path
    else:
        raise FileNotFoundError(
            f"Neither downloaded_metadata.jsonl nor search_results.jsonl found under {round_dir}"
        )

    keys: set[str] = set()
    records: list[dict] = []

    with open(source_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            fname = fname_with_tag(rec)
            pdf_path = download_dir / fname
            if not pdf_path.exists():
                # Skip when PDF file is missing
                continue
            key = get_record_key(rec)
            if key is None:
                continue

            keys.add(key)
            records.append(rec)

    return keys, records

# Delete overlapping papers that exceed the overlap threshold
def dedup_against_history(
    round_idx: int,
    round_dir: Path,
    download_dir: Path,
    history_keys: set[str],
    overlap_threshold: float = 0.5,
) -> tuple[set[str], int]:
    current_keys, _ = load_round_paper_keys(round_dir, download_dir)

    if not current_keys:
        print(f"[orchestrator] round {round_idx}: no downloaded PDFs to dedup.")
        return set(), 0

    overlap_keys = current_keys & history_keys
    unique_keys = current_keys - history_keys

    n = len(current_keys)
    o = len(overlap_keys)
    y = len(unique_keys)

    overlap_ratio = o / n if n > 0 else 0.0

    print(
        f"[orchestrator] round {round_idx}: {n} papers total, "
        f"{o} overlap vs history, {y} unique (overlap ratio={overlap_ratio:.3f})"
    )

    # Retain all if within threshold
    if overlap_ratio <= overlap_threshold or overlap_threshold >= 1.0:
        print(
            f"[orchestrator] round {round_idx}: overlap <= threshold "
            f"({overlap_ratio:.3f} <= {overlap_threshold:.3f}), keep all."
        )
        return current_keys, 0

    # Remove excess overlapping papers.
    t = overlap_threshold
    if y == 0:
        # All papers overlap.
        print(
            f"[orchestrator] round {round_idx}: no unique papers vs history, "
            "cannot reduce overlap ratio without deleting everything; "
            "keeping all for this round."
        )
        return current_keys, 0

    max_overlap_kept_float = (t / (1.0 - t)) * y
    max_overlap_kept = int(max_overlap_kept_float)
    max_overlap_kept = min(max_overlap_kept, o)
    if max_overlap_kept < 0:
        max_overlap_kept = 0

    # Select which overlapping keys to keep (by sorted order)
    if max_overlap_kept == 0:
        overlap_keep = set()
    else:
        sorted_overlap = sorted(overlap_keys)
        overlap_keep = set(sorted_overlap[:max_overlap_kept])

    overlaps_to_remove = overlap_keys - overlap_keep

    print(
        f"[orchestrator] round {round_idx}: overlap_ratio {overlap_ratio:.3f} > "
        f"threshold {overlap_threshold:.3f}, "
        f"keeping {len(overlap_keep)} overlapping papers, "
        f"removing {len(overlaps_to_remove)} overlapping papers."
    )

    # Re-scan search_results.jsonl, delete PDFs to remove, rewrite file
    search_results_path = round_dir / "search_results.jsonl"
    kept_records: list[dict] = []
    removed_pdf_count = 0

    with open(search_results_path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                continue

            fname = fname_with_tag(rec)
            pdf_path = download_dir / fname
            if not pdf_path.exists():
                continue

            key = get_record_key(rec)
            if key is None:
                continue

            if key in overlaps_to_remove:
                try:
                    pdf_path.unlink()
                    removed_pdf_count += 1
                except OSError:
                    pass
            else:
                kept_records.append(rec)

    # Rewrite search_results.jsonl with kept records
    with open(search_results_path, "w", encoding="utf-8") as f:
        for rec in kept_records:
            f.write(json.dumps(rec) + "\n")

    print(
        f"[orchestrator] round {round_idx}: removed {removed_pdf_count} PDFs; "
        f"{len(kept_records)} records remain in search_results.jsonl."
    )

    remaining_keys = unique_keys | overlap_keep

    final_overlap = len(remaining_keys & history_keys)
    final_total = len(remaining_keys)
    if final_total > 0:
        final_ratio = final_overlap / final_total
        print(
            f"[orchestrator] round {round_idx}: after partial dedup vs history, "
            f"{final_total} papers, overlap={final_overlap}, "
            f"overlap ratio={final_ratio:.3f}"
        )

    return remaining_keys, removed_pdf_count

# Top-off to acquire additional new unique PDFs & return new keys
def top_off_with_new_pdfs(
    round_idx: int,
    round_dir: Path,
    download_dir: Path,
    queries_path: Path,
    domain: str,
    sources: str,
    crossref_email: str | None,
    openalex_email: str | None,
    already_seen_keys: set[str],
    needed_new: int,
    base_results_per_source: int,
    core_api_key: str | None = None,
    max_attempts: int = 2,
) -> set[str]:
    if needed_new <= 0:
        return set()

    import os
    from dataclasses import replace
    from queries_lib.domain import DOMAINS
    domain_spec = DOMAINS[domain]
    # Mirror iterative_expand.RuntimeModules: append the CLI last_tier
    # (set by iterative_expand into the LAST_TIER env var) so this top-off
    # search uses the same SPEC.tiers as the main retrieve call.
    last_tier_env = os.environ.get("LAST_TIER")
    if last_tier_env and (not domain_spec.tiers or domain_spec.tiers[-1] != last_tier_env):
        domain_spec = replace(domain_spec, tiers=list(domain_spec.tiers) + [last_tier_env])

    contact_emails = {}
    if crossref_email:
        contact_emails["CROSSREF_MAILTO"] = crossref_email
    if openalex_email:
        contact_emails["OPENALEX_MAILTO"] = openalex_email

    need_left = needed_new
    new_keys: set[str] = set()

    main_search_results_path = round_dir / "search_results.jsonl"
    topoff_results_per_source = max(base_results_per_source, needed_new)
    sources_tuple = tuple(s.strip() for s in sources.split(",") if s.strip())

    for attempt in range(1, max_attempts + 1):
        if need_left <= 0:
            break

        topoff_dir = round_dir / f"topoff_{attempt}"
        topoff_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"[orchestrator] round {round_idx}: top-off attempt {attempt}, "
            f"need {need_left} more unique PDFs (results_per_source={topoff_results_per_source})"
        )

        # Re-run search & download for top-off
        result_extra = retrieve_literature.run_one_round_search_and_download(
            queries_path=queries_path,
            round_dir=topoff_dir,
            domain_spec=domain_spec,
            results_per_source=topoff_results_per_source,
            sources=sources_tuple,
            contact_emails=contact_emails or None,
            core_api_key=core_api_key,
        )

        extra_download_dir = result_extra["download_dir"]
        extra_keys, extra_records = load_round_paper_keys(topoff_dir, extra_download_dir)

        print(
            f"[orchestrator] round {round_idx}: top-off attempt {attempt} "
            f"downloaded {len(extra_keys)} PDFs."
        )

        picked_records: list[dict] = []

        for rec in extra_records:
            key = get_record_key(rec)
            if key is None:
                continue
            if key in already_seen_keys or key in new_keys:
                continue
            fname = fname_with_tag(rec)
            src_pdf = extra_download_dir / fname
            if not src_pdf.exists():
                continue
            # Move PDF to main download directory
            dst_pdf = download_dir / fname
            if not dst_pdf.exists():
                shutil.move(src_pdf, dst_pdf)

            picked_records.append(rec)
            new_keys.add(key)
            need_left -= 1

            if need_left <= 0:
                break

        if picked_records:
            with open(main_search_results_path, "a", encoding="utf-8") as f:
                for rec in picked_records:
                    f.write(json.dumps(rec) + "\n")

        print(
            f"[orchestrator] round {round_idx}: top-off attempt {attempt} "
            f"added {len(picked_records)} new unique PDFs "
            f"(need_left={need_left})."
        )

    if need_left > 0:
        print(
            f"[orchestrator] round {round_idx}: top-off finished but still "
            f"{need_left} PDFs short of target (APIs may not have more unique results)."
        )

    print(f"[dedup] round {round_idx}: Cleaning up temporary topoff directories.")
    for attempt in range(1, max_attempts + 1):
        topoff_dir = round_dir / f"topoff_{attempt}"
        if topoff_dir.exists():
            try:
                shutil.rmtree(topoff_dir)
            except OSError as e:
                print(f"[dedup] WARNING: Failed to remove topoff dir {topoff_dir}: {e}")
                
    return new_keys