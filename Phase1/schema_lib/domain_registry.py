from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

@dataclass(frozen=True)
class DomainSpec:
    name: str
    task_noun: str
    extractor_role: str
    schema_task_description: str
    avoid_fields: List[str]

    # Surface-level text normalizations (e.g. "Aβ" -> "abeta") used for
    # similarity scoring only. Typographical, not content rules.
    text_normalizations: List[Tuple[str, str]] = field(default_factory=list)

    default_last_tier: str = "target system"

DOMAIN_REGISTRY: Dict[str, DomainSpec] = {
    "ad": DomainSpec(
        name="ad",
        task_noun="structured Alzheimer's disease study database",
        extractor_role="high-precision Alzheimer's disease literature extractor",
        schema_task_description=(
            "Design a reusable schema for extracting key study variables from "
            "Alzheimer's disease papers."
        ),
        avoid_fields=[
            "generic bibliographic metadata",
            "fields too narrow to a single paper",
            "fields not reusable across many papers in the same subdomain",
        ],
        text_normalizations=[
            ("aβ", "abeta"),
            ("β-amyloid", "abeta amyloid"),
            ("amyloid-beta", "amyloid beta"),
            ("amyloid β", "amyloid beta"),
            ("p-tau", "ptau tau"),
            ("phospho-tau", "ptau tau"),
        ],
        default_last_tier="Alzheimer's disease",
    ),

    "chem": DomainSpec(
        name="chem",
        task_noun="structured materials chemistry database",
        extractor_role="high-precision materials chemistry literature extractor",
        schema_task_description=(
            "Design a reusable schema for extracting key variables from materials "
            "chemistry papers, covering material identity / composition, building "
            "blocks or precursors, synthesis or fabrication conditions, structure "
            "and morphology, characterization methods, properties and performance "
            "metrics, and applications."
        ),
        avoid_fields=[
            "generic bibliographic metadata",
            "fields too specific to only one paper",
            "fields not repeatedly reported across the corpus",
        ],
        text_normalizations=[
            # Hyphen / en-dash unification for common compound terms
            ("metal–organic", "metal organic"),
            ("metal-organic", "metal organic"),
            ("covalent–organic", "covalent organic"),
            ("covalent-organic", "covalent organic"),
            ("zeolitic–imidazolate", "zeolitic imidazolate"),
            ("zeolitic-imidazolate", "zeolitic imidazolate"),
            # Acronym expansion so similarity scoring matches across abbreviated
            # / spelled-out variants of the same porous-material family
            ("mof", "mof metal organic framework"),
            ("cof", "cof covalent organic framework"),
            ("zif", "zif zeolitic imidazolate framework"),
        ],
        default_last_tier="porous materials",
    ),

    "cs": DomainSpec(
        name="cs",
        task_noun="structured natural language processing and large language models study database",
        extractor_role=(
            "high-precision natural language processing and large language models "
            "literature extractor"
        ),
        schema_task_description=(
            "Design a reusable schema for extracting key variables from natural "
            "language processing and large language models papers, covering "
            "methods and techniques, models / datasets / benchmarks, prompting "
            "and training strategies, evaluation protocols, robustness and "
            "failure modes, and empirical findings."
        ),
        avoid_fields=[
            "generic bibliographic metadata",
            "fields too narrow to a single paper",
            "fields not reusable across many papers in the same subdomain",
        ],
        text_normalizations=[
            # Hyphen / en-dash unification only — no acronym expansion to
            # avoid substring collisions with normal English (e.g. "cot" in
            # "scotch", "rag" in "fragment").
            ("chain–of–thought", "chain of thought"),
            ("chain-of-thought", "chain of thought"),
            ("step–by–step", "step by step"),
            ("step-by-step", "step by step"),
            ("in–context", "in context"),
            ("in-context", "in context"),
            ("retrieval–augmented", "retrieval augmented"),
            ("retrieval-augmented", "retrieval augmented"),
            ("few–shot", "few shot"),
            ("few-shot", "few shot"),
            ("zero–shot", "zero shot"),
            ("zero-shot", "zero shot"),
        ],
        default_last_tier="Large language models",
    ),
}

def get_domain_spec(domain: str) -> DomainSpec:
    if domain not in DOMAIN_REGISTRY:
        valid = ", ".join(sorted(DOMAIN_REGISTRY))
        raise ValueError(f"Unknown domain '{domain}'. Valid domains: {valid}")
    return DOMAIN_REGISTRY[domain]
