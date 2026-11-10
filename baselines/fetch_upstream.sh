#!/usr/bin/env bash
# Fetch the third-party baseline code at the exact commits used in the paper and verify file hashes.
# The authors' code is not redistributed in this repository. Usage: baselines/fetch_upstream.sh [target_baselines_dir]
set -euo pipefail
B="${1:-$(cd "$(dirname "$0")" && pwd)}"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT

clone() {  # clone <url> <commit> <dir>
  git clone -q "$1" "$3"; git -C "$3" checkout -q "$2"
}

# ArxivDIGESTables (Newman et al., EMNLP 2024)
if [ ! -d "$B/arxivdigestables/upstream_repo" ]; then
  clone https://github.com/bnewm0609/arxivDIGESTables 018d8f5602f8b9235ddee27b868b11c2cab9d4d5 "$B/arxivdigestables/upstream_repo"
fi

# Intent-aware schema generation and refinement (Padmakumar et al., Findings of EMNLP 2025)
if [ ! -d "$B/intentaware/upstream" ]; then
  clone https://github.com/vishakhpk/arxivdigestables-with-intent "$(cat "$B/intentaware/UPSTREAM_COMMIT.txt")" "$B/intentaware/upstream"
fi

# EVAPORATE (Arora et al., PVLDB 2023): selected files of the author repository and its metal-evap submodule
clone https://github.com/HazyResearch/evaporate 59eda5d34415f71ffb3a72bb902e1505d1f59e83 "$TMP/evaporate"
clone https://github.com/simran-arora/metal-evap 028da57cf81e369c6cc7c8495a66b51ddb8f872f "$TMP/metal-evap"
python3 - "$B/evaporate" "$TMP" <<'PY'
import hashlib, json, shutil, sys
from pathlib import Path
dst, tmp = Path(sys.argv[1]), Path(sys.argv[2])
src_evap, src_metal = tmp / "evaporate" / "evaporate", tmp / "metal-evap"
def place(target_dir, provenance):
    files = json.load(open(target_dir / "PROVENANCE.json"))["files"]
    for rel, digest in files.items():
        if rel == "METAL_LICENSE": src = src_metal / "LICENSE"
        elif rel.startswith("metal/"): src = src_metal / rel
        else: src = src_evap / rel
        out = target_dir / rel; out.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(src, out)
        got = hashlib.sha256(out.read_bytes()).hexdigest()
        if got != digest: raise SystemExit(f"hash mismatch for {target_dir.name}/{rel}")
    print(f"{target_dir.name}: {len(files)} files verified")
place(dst / "upstream", "PROVENANCE.json")
place(dst / "full_upstream", "PROVENANCE.json")
PY
echo "upstream code ready"
