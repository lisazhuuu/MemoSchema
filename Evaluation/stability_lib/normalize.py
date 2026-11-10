from __future__ import annotations
import re
from typing import Iterable, List, Set
from stability_lib.config import NOISE_TOKENS, STOP_WORDS
from stability_lib.schema_io import SchemaField

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")

# Token stemming and normalization
_INVARIANT_WORDS = {
    # Words ending in -s that are singular in scientific writing — the naive
    # rule below would mangle them ("status" -> "statu", "species" -> "specie",
    # "analysis" -> "analysi"). Keep them as-is.
    "species", "status", "basis", "analysis", "axis", "crisis", "thesis",
    "kinesis", "atlas", "abscess", "process", "access", "address", "compass",
    "stress", "loss", "mass", "class", "boss",
}

def _simple_stem(word: str) -> str:
    w = word.lower()
    if len(w) <= 4:
        return w
    if w in _INVARIANT_WORDS:
        return w
    # cities -> city ; companies -> company
    if w.endswith("ies"):
        return w[:-3] + "y"
    # foxes -> fox ; quizzes -> quiz ; matches -> match ; brushes -> brush
    if w.endswith(("xes", "zes", "ches", "shes")):
        return w[:-2]
    # Trailing 's' that is part of the stem (e.g. status, axis, atlas, mass).
    # Returning early protects these from the generic 'drop s' rule below.
    if w.endswith(("us", "is", "as", "os", "ss")):
        return w
    if w.endswith("s"):
        return w[:-1]
    return w

def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall((text or "").lower())

# Canonical key (used for HARD matching)
def canonical_key(name: str) -> str:
    toks = _tokenize(name)
    toks = [_simple_stem(t) for t in toks if t]
    toks = [t for t in toks if t not in STOP_WORDS]
    # strip noise tokens, but never reduce to empty
    content = [t for t in toks if t not in NOISE_TOKENS]
    if not content:
        content = toks
    content = sorted(set(content))
    return "_".join(content)

# Term set (used for SOFT matching)
def _all_surface_forms(field: SchemaField) -> List[str]:
    return [field.name] + list(field.aliases) + list(field.member_field_names)

def field_term_set(field: SchemaField) -> Set[str]:
    bag: Set[str] = set()
    for surf in _all_surface_forms(field):
        if not surf:
            continue
        ck = canonical_key(surf)
        if ck:
            bag.add(ck)
        # also throw in individual content tokens for fuzzier overlap
        for t in _tokenize(surf):
            s = _simple_stem(t)
            if s and s not in STOP_WORDS and s not in NOISE_TOKENS:
                bag.add(s)
    return bag

def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / len(a | b)

def overlap(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / min(len(a), len(b))
