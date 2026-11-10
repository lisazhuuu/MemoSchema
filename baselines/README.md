# Baselines

All baselines run on the corpus instances of the corresponding MemoSchema runs, use the same GPT-5.4 deployment, and
target eight fields. AutoSchema, EVAPORATE, ArxivDIGESTables and IA_SR use the recorded list of papers each MemoSchema
run used for schema induction, with each method's own sampling and text-input settings (first 12,000 characters of
each PDF unless stated otherwise). LLMs4SchemaDiscovery samples at most 40 papers from the run's `cumulative_corpus/`.
Reference schemas are never given to any method; they are used only by the evaluator.

AutoSchema's upstream code is included (`autoschema/upstream/`, MIT, our own prior artifact; files listed with SHA-256
in `autoschema/UPSTREAM_SHA256.txt`). The upstream code of EVAPORATE, ArxivDIGESTables and intent-aware
self-refinement is not included; `fetch_upstream.sh` clones each repository at the commit used in the paper and
verifies every file against the recorded SHA-256 hashes. LLMs4SchemaDiscovery is a re-implementation
(`../llm4schema/`), so there is no upstream code to fetch.

```bash
bash baselines/fetch_upstream.sh
```

| Baseline | Code | Pinned version | Our driver |
|---|---|---|---|
| AutoSchema (KDD 2026) | `autoschema/upstream` (our own prior artifact, MIT, included) | `autoschema/UPSTREAM_SHA256.txt` | `autoschema/run_paired.py`, `scripts/run_memexp32.sh` |
| EVAPORATE (PVLDB 2023) | github.com/HazyResearch/evaporate + metal-evap | `59eda5d`, metal-evap `028da57` (`evaporate/*/PROVENANCE.json`) | `evaporate/reproduce.py` (schema identification), `evaporate/full_pipeline.py` (full flow), `scripts/run_memexp30.sh`, `scripts/run_memexp39.sh` (r10) |
| ArxivDIGESTables (EMNLP 2024) | github.com/bnewm0609/arxivDIGESTables | `018d8f5` | `arxivdigestables/reproduce.py`, `scripts/run_memexp29.sh` |
| LLMs4SchemaDiscovery (ESWC 2025) | re-implementation of the automated workflow | — | `../llm4schema/scripts/run_llms4schema_baseline.py`, `scripts/run_memexp27.sh` |
| Intent-aware self-refinement (Findings of EMNLP 2025) | github.com/vishakhpk/arxivdigestables-with-intent | `intentaware/UPSTREAM_COMMIT.txt` | `scripts/memexp40.py` |

## Adaptations (disclosed in the paper)

**AutoSchema.** The unmodified Phase-I induction is run round by round on the papers of the corresponding MemoSchema
run; the only change replaces its PDF loader with the frozen first-12k-character texts (same PyMuPDF parser).
`run_phase2.py` runs the unmodified AutoSchema Phase-II extractor for the downstream-utility evaluation.

**EVAPORATE.** Author prompts, chunking, candidate filtering, LLM re-ranking, function generation, validation,
attribute checks and deduplication are taken from the pinned code. Changes:
1. The completion API is replaced by the project's chat deployment; the authors' output budgets (10/100/500 tokens)
   are kept. The deployment rejects `stop` and `temperature`, so runs use `--omit-stop --temperature omit`.
2. Schema identification uses all papers of the run (`--sample-size 0`) instead of the authors' default sample of 10.
3. Eight fields are retained from the authors' final ranking (`--top-k 8`).
4. Generated extraction functions run in an isolated container (`Dockerfile.functions`: no network, read-only, no
   credentials).
5. Weak supervision does not read reference extractions: the authors' `run_ws.get_data` uses gold labels for class
   priors; we use the authors' candidate encoding with a uniform prior and no `Y_dev`.
6. A local-variable shadowing bug in the authors' `evaluate_profiler.evaluate` is avoided by renaming the inner loop
   variable in the adapter; the pinned source files are not modified.
EVAPORATE is run once per corpus because of the cost of the full extraction flow.

EVAPORATE environment and order of runs (r7–r9; `scripts/run_memexp39.sh` does both steps for r10):

```bash
python3.11 -m venv baselines/evaporate/.venv-full
baselines/evaporate/.venv-full/bin/pip install -r baselines/evaporate/requirements-full.txt
docker build -f baselines/evaporate/Dockerfile.functions -t evaporate-functions:pinned baselines/evaporate
baselines/evaporate/.venv-full/bin/python baselines/evaporate/reproduce.py batch --batches r7,r8,r9 \
  --runs COF_01,COF_02,COF_03,MOF_01,MOF_02,MOF_03,ZIF_01,ZIF_02,ZIF_03 --omit-stop --temperature omit --execute
scripts/run_memexp30.sh    # full flow on the stage-1 outputs, then judges
```

**ArxivDIGESTables.** Only the "topic" setting is used: the authors' "caption" prompt with a topic-based caption ("Comparison of research papers
on <subdomain>") and titles plus abstracts as input. At the pinned commit the authors' helper returns only the last
paper of the list; the adapter concatenates all papers, as described in the paper. Titles come from the paper ids; abstracts
are cut from the full PyMuPDF page text (EVAPORATE's `extract_text`), starting at the "Abstract" heading found by
`arxivdigestables/section_diag.py` (project code), up to the next Introduction heading or 2,000 characters; without a
heading, the first 2,000 characters are used.

**LLMs4SchemaDiscovery.** Automated workflow without expert feedback: specification -> initial schema, refinement with
the seed papers, then refinement with 40 sampled corpus papers in batches of five; at most eight final fields.

**Intent-aware self-refinement (IA_SR, SR_DRAFT).** Author prompts are used verbatim except for one added
sentence requiring exactly eight attributes. Intents are fixed topic questions per subdomain (the authors' intent
synthesis requires the reference table and is therefore not used). IA_SR applies one critique-and-revision cycle to
each initially generated schema. SR_DRAFT applies the same cycle to MemoSchema's draft with the same candidate shortlist that
restructuring receives (an adaptation, not the authors' method).
