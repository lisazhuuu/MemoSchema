from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

@dataclass
class SchemaField:
    name: str
    description: str = ""
    value_type: str = ""
    aliases: List[str] = field(default_factory=list)
    member_field_names: List[str] = field(default_factory=list)

def load_schema(schema_path: Path) -> List[SchemaField]:
    obj = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    raw = obj if isinstance(obj, list) else obj.get("fields", [])
    out: List[SchemaField] = []
    for f in raw:
        if not isinstance(f, dict) or "name" not in f:
            continue
        out.append(
            SchemaField(
                name=str(f["name"]),
                description=str(f.get("description", "") or ""),
                value_type=str(f.get("value_type", "") or ""),
                aliases=[str(a) for a in (f.get("aliases") or []) if a],
                member_field_names=[
                    str(m) for m in (f.get("member_field_names") or []) if m
                ],
            )
        )
    return out
