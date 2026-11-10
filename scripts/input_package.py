"""Guard for the separately distributed input packages (MemoSchema-inputs, MemoSchema-inputs-pending; README "Obtain inputs").

`inputs_tool.py install` records every installed component in .memoschema-inputs/<component>.json (files, sizes, the
run-directory root it fills, the scripts it serves, whether its evidence quotes are redacted). Packaged run directories
under Ablation/ (introduction diagnostic) and results/e2e/e2e_v3/ (history replay, cost and audit inputs) hold only the
files these scripts read, so other scripts must not use them, and redacted quotes must be restored first (selection
ranks candidates partly on quote text; the merge prompt shows quotes). Without installed components (complete run
directories built by the pipeline) nothing is checked here."""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REGISTRY = REPO / ".memoschema-inputs"
_done = set()

def components(root=None):
    out = [json.loads(f.read_text()) for f in sorted(REGISTRY.glob("*.json"))] if REGISTRY.exists() else []
    return [c for c in out if root is None or c.get("root") == root]

def check(root: str, user: str) -> None:
    """Stop if `root` ('dev' or 'replay') holds packaged run directories and `user` is outside their scope, files are
    missing or changed, or quotes are still redacted."""
    if (root, user) in _done:
        return
    C = components(root)
    if not C:
        _done.add((root, user)); return
    scope = sorted({s for c in C for s in c["scope"]})
    if user not in scope:
        raise SystemExit(f"[inputs] the {root} run directories come from the input package (scripts served: {', '.join(scope)}); "
                         f"{user} needs complete run directories or a further package component (MemoSchema-inputs README)")
    miss = [f for c in C for f, size in c["files"].items() if not (REPO / f).exists() or (REPO / f).stat().st_size != size]
    if miss:
        raise SystemExit(f"[inputs] {len(miss)} packaged files missing or changed, e.g. {miss[0]}; reinstall the package")
    red = [c["component"] for c in C if c["evidence"] == "redacted"]
    if red:
        raise SystemExit(f"[inputs] evidence quotes are redacted in {', '.join(red)}; restore them first "
                         "(inputs_tool.py restore with the pending quotes table)")
    _done.add((root, user))
