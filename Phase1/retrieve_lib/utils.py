import re, os, socket, threading
from pathlib import Path
from typing import List, Dict, Set
from urllib.parse import urlparse
from pathlib import Path
from datetime import datetime

from retrieve_lib.preprocessing import norm_title

# Remove duplicate records based on DOI, ID, or normalized title
def dedupe(records: List[Dict]) -> List[Dict]:
    # Empty input
    seen: Set[str] = set(); out = []
    
    # Keep the first record for each key.
    for r in records:
        key = r.get("doi") or r.get("id") or norm_title(r.get("title",""))
        if key and key not in seen:
            seen.add(key); out.append(r)
            
    return out

# Derive a filename suffix from the source URL or DOI.
def site_tag(rec: dict, collapse_springer_nature: bool = True) -> str:
    # Collapse all Springer Nature umbrella sites into "nature"
    u   = rec.get("pdf_url") or rec.get("url") or ""
    host = (urlparse(u).hostname or "").lower()
    doi  = (rec.get("doi") or "").lower()

    if collapse_springer_nature:
        # DOI prefixes under Springer Nature umbrella
        if doi.startswith(("10.1038/", "10.1007/", "10.1186/")):
            return "nature"
        # Common hosts under Springer Nature umbrella
        if any(h in host for h in [
            "nature.com",
            "springernature.com",
            "link.springer.com",
            "biomedcentral.com",
            "bmc.com"
        ]):
            return "nature"

    # Always use source for OpenAlex records
    if rec.get("source") == "openalex":
        return "openalex"
    
    # Specific known sources by URL/DOI
    if "arxiv.org" in host: return "arxiv"

    # Otherwise, use DOI prefix if available such as 10.1038
    if host and host != "doi.org": # When host is not doi.org
        parts = host.split(".") # Split hostname by dots
        return parts[-2] if len(parts) >= 2 else host # Return second last part or full host
    return (rec.get("source") or "unknown").lower() # Fallback to source or unknown

# Filename generator using site tag
def fname_with_tag(rec: dict) -> str:
    # Use the title or ID as the filename base.
    base = re.sub(r"[^a-zA-Z0-9_.-]+", "_", (rec.get("title") or rec.get("id") or "paper")).strip("_")
    tag = site_tag(rec) # Get site tag
    suffix = f"__{tag}.pdf" # Suffix with site tag
    return (base[: max(1, 120 - len(suffix))] + suffix) # Limit base length to fit within 120 chars

# Download PDF from URL
_PDF_CONNECT_TIMEOUT = 15   # seconds to establish TCP connection
_PDF_READ_TIMEOUT    = 30   # seconds of silence before aborting mid-stream
_PDF_TOTAL_TIMEOUT   = 180  # hard wall-clock cap via a daemon thread

def download_pdf(url: str, dest: Path) -> bool:
    import requests

    headers = {
        "User-Agent": f"paper-pipeline/0.1 ({os.getenv('OPENALEX_MAILTO') or 'you@school.edu'})",
        "Referer": "https://www.ncbi.nlm.nih.gov/"
    }

    result = [False]

    def _do_download():
        try:
            with requests.get(
                url, stream=True,
                timeout=(_PDF_CONNECT_TIMEOUT, _PDF_READ_TIMEOUT),
                allow_redirects=True,
                headers=headers,
            ) as resp:
                resp.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in resp.iter_content(1024 * 64):
                        f.write(chunk)
            result[0] = True
        except Exception:
            pass

    t = threading.Thread(target=_do_download, daemon=True)
    t.start()
    t.join(timeout=_PDF_TOTAL_TIMEOUT)
    if t.is_alive():
        # Thread is still blocked (e.g. Cloudflare slow-drip).  The daemon
        # flag means it won't prevent process exit, but we clean up the
        # partial file so downstream validation doesn't accept a truncated PDF.
        print(f"  [download_pdf] hard timeout ({_PDF_TOTAL_TIMEOUT}s) hit: {url}")
        try:
            dest.unlink(missing_ok=True)
        except Exception:
            pass
        return False
    return result[0]
    
# Delete empty folders recursively
def delete_empty_folders(path: Path):
    if not path.is_dir():
        return
    if not any(path.iterdir()):
        path.rmdir()
        print(f"Deleted empty folder: {path}")
    else:
        return

# Check if a PDF file is valid by checking its header and footer
def _is_valid_pdf(p: Path) -> bool:
    try:
        size = p.stat().st_size
        with open(p, "rb") as f:
            # Check PDF header
            head = f.read(5)
            if size > 2048:
                f.seek(size - 2048)
            # Check PDF footer
            tail = f.read()
        return head.startswith(b"%PDF-") and (b"%%EOF" in tail)
    except Exception:
        return False

# Delete invalid PDF files in a given path (file or directory)
def delete_invalid_pdfs(path: str | Path):
    p = Path(path)
    deleted = []
    try:
        if p.is_file():
            if not _is_valid_pdf(p):
                p.unlink()
                deleted.append(p.name)
        elif p.is_dir():
            for f in p.glob("*.pdf"):
                if not _is_valid_pdf(f):
                    f.unlink()
                    deleted.append(f.name)
    except Exception as e:
        print(f"[delete_invalid_pdfs] error: {e}")

    print(f"deleted {len(deleted)} bad pdfs")
    return deleted

def make_unique_run_dir(base: Path, prefix: str = "round") -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base / f"{prefix}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir