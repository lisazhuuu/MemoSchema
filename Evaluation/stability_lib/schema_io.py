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
    scope: str = ""
    aliases: List[str] = field(default_factory=list)
    member_field_names: List[str] = field(default_factory=list)
    support_count: int = 0

@dataclass
class LoadedSchema:
    label: str            # e.g. "run01"  (used in CSV column headers + filenames)
    path: Path
    fields: List[SchemaField]

def load_schema_file(path: Path, label: str | None = None) -> LoadedSchema:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(obj, list):
        raw_fields = obj
    elif isinstance(obj, dict) and "fields" in obj:
        raw_fields = obj["fields"]
    else:
        raise ValueError(f"Unsupported schema format in {path}")

    parsed: List[SchemaField] = []
    for f in raw_fields:
        if not isinstance(f, dict) or "name" not in f:
            continue
        parsed.append(
            SchemaField(
                name=str(f["name"]),
                description=str(f.get("description", "") or ""),
                value_type=str(f.get("value_type", "") or ""),
                scope=str(f.get("scope", "") or ""),
                aliases=[str(a) for a in (f.get("aliases") or []) if a],
                member_field_names=[
                    str(m) for m in (f.get("member_field_names") or []) if m
                ],
                support_count=int(f.get("support_count") or 0),
            )
        )

    return LoadedSchema(
        label=label or Path(path).stem,
        path=Path(path),
        fields=parsed,
    )

def load_all(paths: List[Path], labels: List[str] | None = None) -> List[LoadedSchema]:
    """Convenience: load several schema files, with optional pre-set labels."""
    if labels and len(labels) != len(paths):
        raise ValueError("--labels must have the same length as --schemas")
    out: List[LoadedSchema] = []
    for i, p in enumerate(paths):
        lab = labels[i] if labels else Path(p).parent.name or Path(p).stem
        out.append(load_schema_file(Path(p), label=lab))
    return out
