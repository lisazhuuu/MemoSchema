import math
from typing import List, Dict, Any

KEEP_RATIO = 0.65  # fixed, not user-configurable

def apply_keep_floor(scored_queries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not scored_queries:
        return []

    min_keep = max(1, math.ceil(KEEP_RATIO * len(scored_queries)))
    keep = list(scored_queries)

    # only remove bad queries, starting from the worst
    bad_indices = [i for i, q in enumerate(keep) if q.get("is_bad", False)]

    for i in reversed(bad_indices):
        if len(keep) <= min_keep:
            break
        del keep[i]

    return keep