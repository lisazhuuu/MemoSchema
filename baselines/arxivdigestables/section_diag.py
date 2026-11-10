"""Section-heading detection used by the ArxivDIGESTables adapter (reproduce.py: abstract_of) to locate the abstract.

Verbatim copy of NUM, ALIASES, PATS, HEADLIKE and sections() from the project's scripts/section_diag.py
(source SHA-256 8d8dbb620b5f35c326e41625c968b82faab6e3a1e491a57ed4ed625826fb396a); that file's corpus diagnostic (main) is not part of the paper and is omitted.
"""
from __future__ import annotations
import re

NUM = r"^(?:\d{1,2}(?:\.\d{1,2})*\.?|[IVX]{1,4}\.|[A-H]\.)?\s*"
ALIASES = {
    "abstract": r"abstract|a\s?b\s?s\s?t\s?r\s?a\s?c\s?t|summary|synopsis",
    "intro": r"introduction|background|motivation",
    "experimental": (r"experimental(?: section| details| procedures?| methods?| part)?|materials? and methods|methods?"
                     r"|methodology|materials?(?: and (?:reagents|chemicals|instrumentation|characteri[sz]ation))?"
                     r"|chemicals(?: and materials)?|reagents|synthes[ie]s(?: of .{0,60})?|preparation(?: of .{0,60})?"
                     r"|sample preparation|characteri[sz]ations?(?: methods?| techniques?)?|instrumentation|measurements?"
                     r"|computational (?:details|methods?)|simulation (?:details|methods?)|experimental and computational methods"),
    "results": r"results(?: and discussions?| & discussions?)?|discussion|results and analysis|findings",
    "conclusion": r"conclusions?|concluding remarks|summary and (?:outlook|conclusions?)|outlook|perspectives?|future (?:directions|outlook)",
    "backmatter": (r"references|bibliography|notes and references|acknowledge?ments?|author contributions?|conflicts? of interest"
                   r"|competing interests?|data availability|supporting information|associated content|funding"),
}
PATS = {k: re.compile(NUM + r"(?:" + v + r")\s*[:.]?\s*$", re.I) for k, v in ALIASES.items()}
HEADLIKE = re.compile(NUM + r"[A-Z][A-Za-z\-\s,&/()]{2,60}$")

def sections(text):
    hits, unmatched, pos = [], [], 0
    for line in text.split("\n"):
        s = line.strip()
        if 2 < len(s) <= 80:
            for k, p in PATS.items():
                if p.match(s):
                    hits.append((k, pos)); break
            else:
                if HEADLIKE.match(s) and len(s.split()) <= 8 and not s.endswith("."):
                    unmatched.append(re.sub(NUM, "", s).lower())
        pos += len(line) + 1
    first = {}
    for k, p in hits:
        if k not in first and not (k == "experimental" and p < 300):    # skip title-page noise
            first[k] = p
    return first, unmatched
