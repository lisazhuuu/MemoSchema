import json, re
from pathlib import Path
from typing import List, Dict

# Normalize title for deduplication
def norm_title(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").strip().lower())

# Load and normalize search prompt from queries.json from domain-specific output dir
def load_queries(path: str) -> List[Dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    fixed = [] # Store cleaned-up queries
    for q in data:
        qb = q.get("query_bool") or q.get("query") or ""
        qn = q.get("query_nl") or qb 
        # Store all parts together
        fixed.append({"query_bool": qb, "query_nl": qn, "why": q.get("why","")})
        
    return fixed
