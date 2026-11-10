from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import tempfile, re

# Extract PDF text and save it to a text file.
def pdf_to_text_path(pdf_path: Path) -> str:
    import fitz 
    
    text_dir = Path(tempfile.gettempdir()) / "autoschema_texts"
    text_dir.mkdir(exist_ok=True)
    doc = fitz.open(pdf_path)
    chunks = [p.get_text() for p in doc]
    # Save to text file
    outp = text_dir / (pdf_path.stem + ".txt")
    outp.write_text("\n".join(chunks), encoding="utf-8")
    
    return str(outp)

def pdf_to_text(pdf_path: Path) -> str:
    import fitz

    text_dir = Path(tempfile.gettempdir()) / "autoschema_texts"
    text_dir.mkdir(exist_ok=True)

    doc = fitz.open(pdf_path)
    chunks = [p.get_text() for p in doc]
    text = "\n".join(chunks)

    # Save to text file (for debugging / caching)
    outp = text_dir / (pdf_path.stem + ".txt")
    outp.write_text(text, encoding="utf-8")

    # Return the text itself.
    return text

def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def get_sentence_overlap(text: str, max_chars: int = 400) -> str:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if not sentences:
        return ""

    selected = []
    total = 0

    for sent in reversed(sentences):
        if not sent:
            continue
        if total + len(sent) > max_chars and selected:
            break
        selected.append(sent)
        total += len(sent) + 1

    return " ".join(reversed(selected)).strip()

def chunk_text(text: str, chunk_size: int = 4500, overlap: int = 400) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []

    # split into paragraph blocks first
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    chunks = []
    current = ""

    for para in paragraphs:
        # split overly long paragraphs at sentence boundaries
        if len(para) > chunk_size:
            sentences = re.split(r"(?<=[.!?])\s+", para)
        else:
            sentences = [para]

        for unit in sentences:
            if not unit:
                continue

            if len(current) + len(unit) + 2 <= chunk_size:
                current = f"{current}\n\n{unit}".strip()
            else:
                if current:
                    chunks.append(current)

                # overlap: keep some context from the end of the previous chunk
                if overlap > 0 and chunks:
                    tail = get_sentence_overlap(chunks[-1], max_chars=overlap)
                    current = f"{tail}\n\n{unit}".strip() if tail else unit
                else:
                    current = unit

    if current:
        chunks.append(current)

    return chunks

# Paper metadata
@dataclass
class Paper: 
    paper_id: str
    title: Optional[str] = None
    authors: Optional[List[str]] = None
    year: Optional[int] = None
    venue: Optional[str] = None # Where it was published
    doi: Optional[str] = None # Digital Object Identifier (if available)
    source: str = "seed"
    raw_text_path: Optional[str] = None
    extracted: Dict[str, Any] = None # A dictionary of structured data extracted from the paper