import os, argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class DomainSpec:
    tiers: List[str]
    arxiv_bool: str
    system_role: str
    chunk_extract_prompt: str = ""
    paper_merge_prompt: str = ""
    extract_prompt: str = ""
    prompt_gen_general: str = ""
    prompt_gen_domain: str = ""

cs_tiers = [
    "Computer science",
    "Artificial intelligence",
    "Natural language processing",
    "Large language models",
]

chem_tiers = [
    "Physical sciences",
    "Chemistry",
    "Materials chemistry",
    "Porous materials",
]

ad_tiers = [
    "Life sciences",
    "Neuroscience",
    "Neurodegenerative diseases",
    "Alzheimer's disease",
]

DOMAINS: Dict[str, DomainSpec] = {
    "cs": DomainSpec(
        tiers=cs_tiers,
        arxiv_bool="(cat:cs.CL OR cat:cs.AI OR cat:cs.LG)",
        system_role=(
            "expert in natural language processing and large language models, "
            "especially LLM prompting, fine-tuning, agents, and evaluation research."
        ),
        chunk_extract_prompt="""
            Return ONLY valid JSON. Use null or [] if unknown. No prose/backticks.

            You are reading ONE CHUNK of a seed paper for retrieval-oriented summarization.
            Focus on what will help generate good search queries for the LAST tier topic.

            Extract:
            - title
            - core_subtopic
            - key_mechanisms_or_entities[]
            - important_proteins_genes_pathways[]
              (in this domain: models, datasets, or benchmarks the paper centers on)
            - model_system_or_study_type[]
            - main_experimental_setting_or_evidence_type[]
            - likely_retrieval_keywords[]
            - off_topic_or_should_avoid[]
            - important_terms_from_this_chunk[]
            - chunk_summary

            Pull every value from this chunk. Do not invent named methods, models,
            or benchmarks that are not mentioned in the text.

            Text:
            ```{paper_text}```
        """,
        paper_merge_prompt="""
            Return ONLY valid JSON. Use null or [] if unknown. No prose/backticks.

            Below are chunk-level summaries from one paper.
            Merge them into ONE paper-level retrieval-oriented structured summary
            for query generation.

            Extract:
            - title
            - authors
            - year
            - venue
            - doi_or_id
            - url
            - core_subtopic
            - key_mechanisms_or_entities[]
            - important_proteins_genes_pathways[]
              (in this domain: models, datasets, or benchmarks)
            - model_system_or_study_type[]
            - main_experimental_setting_or_evidence_type[]
            - likely_retrieval_keywords[]
            - off_topic_or_should_avoid[]
            - concise_paper_summary
            - why_this_seed_is_useful_for_retrieval

            Use only what appears in the chunk summaries. Do not introduce new
            named methods, models, or benchmarks.

            Chunk summaries:
            ```{chunk_summaries}```
        """,
        prompt_gen_general=f"""
            Return ONLY a JSON object: {{"queries":[...]}} with 10 items.
            Each item has: "query_bool", "query_nl", "synonyms" (<=3), "why" (<=12 words).

            Scope: NATURAL LANGUAGE PROCESSING / LARGE LANGUAGE MODELS.
            Focus the queries on the SAME research topic represented by the LAST
            element of this domain's tiers list, as reflected in the seed papers.

            Across the 10 queries, cover different aspects of this last-tier focus, such as:
                - methods, techniques, or algorithmic approaches involved
                - models, datasets, or benchmarks commonly used
                - prompting strategies, training procedures, or evaluation protocols
                - failure modes, robustness, or safety considerations
                - reproducibility, cost, latency, and efficiency aspects
                - empirical evidence, ablations, or scaling trends
                - any other important themes that appear in the seed papers

            Boolean query guidance:
                - Prefer arXiv cats in query_bool: (cat:cs.CL OR cat:cs.AI OR cat:cs.LG)
                - Use ti: for titles and abs: for abstracts; combine terms with AND/OR
                - Use terminology that is grounded in the seed papers, not pre-listed examples

            Be concise.
        """,
        prompt_gen_domain=f"""
            Return ONLY a JSON object: {{"queries":[...]}} with 8 items.

            Each item must contain:
                - "tiers": exactly {cs_tiers}
                - "query_bool": Boolean search query for arXiv or similar
                - "query_nl": 1-2 sentence natural language description
                - "synonyms": up to 3 related terms
                - "why": purpose in <= 12 words

            Scope: NATURAL LANGUAGE PROCESSING / LARGE LANGUAGE MODELS.

            Use the LAST item in the tiers list as the main research focus.
            All queries must clearly target this last-tier topic as reflected
            by the seed papers.

            Cover different aspects of this last-tier topic across the 8 queries, such as:
                - methods, techniques, or algorithmic approaches involved
                - models, datasets, or benchmarks commonly used in this area
                - prompting strategies, training procedures, or evaluation protocols
                - failure modes, robustness, or safety considerations
                - reproducibility, cost, latency, and efficiency aspects
                - empirical evidence, ablations, or scaling trends
                - any other important themes appearing in the seed papers

            Boolean query guidance:
                - Prefer arXiv categories from this domain: (cat:cs.CL OR cat:cs.AI OR cat:cs.LG)
                - Use ti: for titles and abs: for abstracts; combine with AND/OR
                - Use terminology grounded in the seed papers, not pre-listed examples

            Use these EXACT tiers. Do NOT create new tiers or modify text.
            All queries MUST be about the last tier: "{cs_tiers[-1]}".

            Be concise.
        """,
    ),

    "chem": DomainSpec(
        tiers=chem_tiers,
        arxiv_bool="(cat:cond-mat.mtrl-sci OR cat:physics.chem-ph)",
        system_role=(
            "expert in materials chemistry, especially porous-framework materials "
            "(MOFs, COFs, ZIFs) — synthesis, structure, characterization, properties, "
            "and applications."
        ),
        chunk_extract_prompt="""
            Return ONLY valid JSON. Use null or [] if unknown. No prose/backticks.

            You are reading ONE CHUNK of a seed paper for retrieval-oriented summarization.
            Focus on what will help generate good search queries for the LAST tier topic.

            Extract:
            - title
            - core_subtopic
            - key_mechanisms_or_entities[]
            - important_proteins_genes_pathways[]
              (in this domain: compounds, linkers, metal nodes, ligands, monomers,
               or precursors the paper centers on)
            - model_system_or_study_type[]
            - main_experimental_setting_or_evidence_type[]
            - likely_retrieval_keywords[]
            - off_topic_or_should_avoid[]
            - important_terms_from_this_chunk[]
            - chunk_summary

            Pull every value from this chunk. Do not invent named compounds,
            materials, or characterization methods that are not in the text.

            Text:
            ```{paper_text}```
        """,
        paper_merge_prompt="""
            Return ONLY valid JSON. Use null or [] if unknown. No prose/backticks.

            Below are chunk-level summaries from one paper.
            Merge them into ONE paper-level retrieval-oriented structured summary
            for query generation.

            Extract:
            - title
            - authors
            - year
            - venue
            - doi_or_id
            - url
            - core_subtopic
            - key_mechanisms_or_entities[]
            - important_proteins_genes_pathways[]
              (in this domain: compounds, linkers, metal nodes, ligands, precursors)
            - model_system_or_study_type[]
            - main_experimental_setting_or_evidence_type[]
            - likely_retrieval_keywords[]
            - off_topic_or_should_avoid[]
            - concise_paper_summary
            - why_this_seed_is_useful_for_retrieval

            Use only what appears in the chunk summaries. Do not introduce new
            named compounds, materials, or characterization methods.

            Chunk summaries:
            ```{chunk_summaries}```
        """,
        prompt_gen_general=f"""
            Return ONLY a JSON object: {{"queries":[...]}} with 10 items.
            Each item has: "query_bool", "query_nl", "synonyms" (<=3), "why" (<=12 words).

            Scope: CHEMISTRY / MATERIALS. Focus the queries on the SAME research topic
            represented by the LAST element of this domain's tiers list, as reflected
            in the seed papers.

            Across the 10 queries, cover different aspects of this last-tier focus, such as:
                - how it is synthesized or prepared
                - building blocks, linkers, metal nodes, or precursors used
                - how its structure and composition are characterized
                - how its stability or degradation is evaluated
                - how its physical / chemical properties or performance are measured
                - how it is applied or used in different contexts
                - any other important themes that appear in the seed papers

            Boolean query guidance:
                - Prefer arXiv cats in query_bool: (cat:cond-mat.mtrl-sci OR cat:physics.chem-ph)
                - Use ti: for titles and abs: for abstracts; combine terms with AND/OR
                - Use terminology grounded in the seed papers, not pre-listed examples

            Be concise.
        """,
        prompt_gen_domain=f"""
            Return ONLY a JSON object: {{"queries":[...]}} with 8 items.

            Each item must contain:
                - "tiers": exactly {chem_tiers}
                - "query_bool": Boolean search query for arXiv or similar
                - "query_nl": 1-2 sentence natural language description
                - "synonyms": up to 3 related terms
                - "why": purpose in <= 12 words

            Scope: CHEMISTRY / MATERIALS.

            Use the LAST item in the tiers list as the main research focus.
            All queries must clearly target this last-tier topic as reflected
            by the seed papers.

            Cover different aspects of this last-tier topic across the 8 queries, such as:
                - how it is synthesized or prepared
                - building blocks, linkers, metal nodes, or precursors used
                - how its structure or composition is characterized
                - how its stability or degradation is evaluated
                - how its physical / chemical properties or performance are measured
                - how it is applied in different contexts
                - any other important themes in the seed papers

            Boolean query guidance:
                - Prefer arXiv categories from this domain: (cat:cond-mat.mtrl-sci OR cat:physics.chem-ph)
                - Use ti: for titles and abs: for abstracts; combine with AND/OR
                - Use terminology grounded in the seed papers, not pre-listed examples

            Use these EXACT tiers. Do NOT create new tiers or modify text.
            All queries MUST be about the last tier: "{chem_tiers[-1]}".

            Be concise.
        """,
    ),
    
    "ad": DomainSpec(
        tiers=ad_tiers,
        arxiv_bool="(cat:q-bio.NC OR cat:q-bio.QM OR cat:q-bio.BM)",
        system_role=(
            "expert in neuroscience and neurodegenerative diseases, "
            "especially Alzheimer's disease research."
        ),
        chunk_extract_prompt="""
            Return ONLY valid JSON. Use null or [] if unknown. No prose/backticks.

            You are reading ONE CHUNK of a seed paper for retrieval-oriented summarization.
            Focus on what will help generate good search queries for the LAST tier topic.

            Extract:
            - title
            - core_subtopic
            - key_mechanisms_or_entities[]
            - important_proteins_genes_pathways[]
            - model_system_or_study_type[]
            - main_experimental_setting_or_evidence_type[]
            - likely_retrieval_keywords[]
            - off_topic_or_should_avoid[]
            - important_terms_from_this_chunk[]
            - chunk_summary

            Pull every value from this chunk. Do not invent named entities
            that are not in the text.

            Text:
            ```{paper_text}```
        """,
        paper_merge_prompt="""
            Return ONLY valid JSON. Use null or [] if unknown. No prose/backticks.

            Below are chunk-level summaries from one paper.
            Merge them into ONE paper-level retrieval-oriented structured summary for query generation.

            Extract:
            - title
            - authors
            - year
            - venue
            - doi_or_id
            - url
            - core_subtopic
            - key_mechanisms_or_entities[]
            - important_proteins_genes_pathways[]
            - model_system_or_study_type[]
            - main_experimental_setting_or_evidence_type[]
            - likely_retrieval_keywords[]
            - off_topic_or_should_avoid[]
            - concise_paper_summary
            - why_this_seed_is_useful_for_retrieval

            Use only what appears in the chunk summaries.

            Chunk summaries:
            ```{chunk_summaries}```
        """,
        prompt_gen_general=f"""
            Return ONLY a JSON object: {{"queries":[...]}} with 10 items.
            Each item has: "query_bool", "query_nl", "synonyms" (<=3), "why" (<=12 words).

            Scope: NEUROSCIENCE / ALZHEIMER'S DISEASE.
            Focus the queries on the SAME research topic represented by the LAST element
            of this domain's tiers list, as reflected in the seed papers.

            Across the 10 queries, cover different aspects of this last-tier focus, such as:
                - disease mechanisms and pathological processes
                - molecular and cellular pathways
                - biomarkers and measurable indicators
                - genetic and environmental risk factors
                - affected brain regions and cell types
                - progression, diagnosis, and cognitive decline
                - therapeutic or intervention relevance
                - important experimental or clinical evidence

            Boolean query guidance:
                - Prefer arXiv cats in query_bool: (cat:q-bio.NC OR cat:q-bio.QM OR cat:q-bio.BM)
                - Use ti: for titles and abs: for abstracts; combine terms with AND/OR
                - Use terminology grounded in the seed papers, not pre-listed examples

            Be concise.
        """,
        prompt_gen_domain=f"""
            Return ONLY a JSON object: {{"queries":[...]}} with 8 items.

            Each item must contain:
                - "tiers": exactly {ad_tiers}
                - "query_bool": Boolean search query for arXiv or similar
                - "query_nl": 1-2 sentence natural language description
                - "synonyms": up to 3 related terms
                - "why": purpose in <= 12 words

            Scope: NEUROSCIENCE / ALZHEIMER'S DISEASE.

            Use the LAST item in the tiers list as the main research focus.
            All queries must clearly target this last tier topic as reflected by the seed papers.

            Cover different aspects of this last tier topic across the 8 queries, such as:
                - biological mechanisms or disease processes involved
                - molecular or cellular pathways related to the topic
                - genetic or environmental risk factors
                - biomarkers or measurable indicators associated with the topic
                - affected brain regions, cell types, or neural systems
                - experimental or clinical evidence used to study the topic
                - links to disease progression or cognitive decline
                - any other important themes appearing in the seed papers

            Boolean query guidance:
                - Prefer arXiv categories from this domain: (cat:q-bio.NC OR cat:q-bio.QM OR cat:q-bio.BM)
                - Use ti: for titles and abs: for abstracts; combine with AND/OR
                - Use terminology grounded in the seed papers, not pre-listed examples

            Use these EXACT tiers. Do NOT create new tiers or modify text.
            All queries MUST be about the last tier: "{ad_tiers[-1]}".

            Be concise.
        """,
    ),
}

def get_spec(domain: str = "ad") -> DomainSpec:
    if domain not in DOMAINS:
        valid = ", ".join(sorted(DOMAINS.keys()))
        raise ValueError(f"Unknown domain '{domain}'. Valid domains: {valid}")
    return DOMAINS[domain]
