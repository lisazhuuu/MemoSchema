# MemoSchema — reproduction code

Code, prompts, configuration, reference inventories and input manifests for *MemoSchema: Granularity-Aware Schema Induction with Cross-Run Memory for Scientific Literature*. No PDFs, model outputs, judge scores, logs or caches are included; drivers write to `results/` (created on first run). The offline checks show that the code is consistent and importable — they are **not** a reproduction of the experiments.

| Path | Contents |
|---|---|
| `Phase1/` | Method: corpus construction (seed summaries M1, queries, retrieval), paper proposals (M2), candidate clustering/consolidation (M3), anchors, draft selection |
| `scripts/memexp26.py` | Restructuring prompt `MERGE` (paper Table I); construction arms (ORIG ×3 no-reuse, LEGACY) |
| `scripts/memexp38.py` | Final MemoSchema configuration: content-keyed reuse, arm `REUSEC_M` |
| `scripts/` | Experiment drivers (`run_*.sh`) and summaries; full mapping in `EXPERIMENTS.md` |
| `Evaluation/` | Judges and reference inventories (hierarchical F1, strict one-to-one F1); stability metric |
| `baselines/`, `llm4schema/` | Baseline adapters (`baselines/README.md`) |
| `data/corpora/{r7,r8,r9,r10}/` | 36 evaluation corpus manifests |
| `data/corpora_history/{r3,r4,r5,r6}/` | 30 manifests of the cross-run history used by the reuse replay |
| `data/corpora_dev/<domain>/` | 9 manifests of the development corpora of the introduction diagnostic (seed set 02, see below) |
| `envs/` | Confirmed versions for text extraction and the memory audit; the tested environment (`envs/README.md`) |
| `AVAILABILITY.md` | Short code and data availability statement |
| `seeds/seed_manifest.tsv` | Seed PDFs of all relevant seed sets: file, SHA-256, title, DOI/arXiv id, link, runs that used them |

Each corpus manifest lists every collected paper (round, id, title, DOI, source, URL), the papers actually used for schema induction (`papers_used_for_schema_induction`, with metadata for those not among the collected papers in `induction_papers_not_in_collected`) and the run configuration. Missing values are null; nothing is inferred. `scripts/build_corpus_manifest.py` writes a manifest from any run directory; it reproduces the 36 evaluation manifests exactly from their original runs.

## 1. Install (offline except the fetch)

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # version constraints for text extraction and the audit: envs/README.md
cp .env.example .env                       # Azure OpenAI chat (GPT-5.4) + embedding (text-embedding-3-large); Gemini for Fig. 4
python scripts/check_release.py            # offline: syntax, imports, manifests
bash baselines/fetch_upstream.sh           # network, no API: pinned EVAPORATE / ArxivDIGESTables / intent-aware code
```

EVAPORATE's full flow also needs its own environment and a Docker image (`baselines/README.md`).

## 2. Obtain inputs

- **Seeds** — put the PDFs of `seeds/seed_manifest.tsv` under `seeds/seed_<domain>/<SUB_SET>/` (e.g. `seeds/seed_chem/ZIF_01/`) and check them with `shasum -a 256 seeds/seed_*/*/*.pdf` against the `sha256` column. Titles in the table are those recorded by the pipeline's seed summarizer (an LLM) unless `title_source` says otherwise; `title_check` records the check against the PDF with that SHA-256: the 45 chem rows were compared with the formal title on the article's first page (43 identical; 2 identical in wording, but printed with a long dash where the table has a hyphen); the 25 ad/cs rows with a PDF were found in its text; the 5 `seed_ad/tau` rows could not be checked (PDFs not available). DOIs/arXiv ids come only from the PDF's own metadata or arXiv stamp and are blank otherwise.
- **Corpora** — the manifests list the papers; PDFs are not redistributed. Retrieval (`scripts/run_e2e_local.sh`, `scripts/run_history.sh`) is time-dependent and builds *new* collections; it does not recover the frozen ones.
- **Input packages (distributed separately; download location to be added on publication)** — `MemoSchema-inputs/` (public: model-generated run records of the 9 development runs and the r3–r9 history, and the judged outputs and records that the summaries and re-runs read; paper text replaced by hashes) and `MemoSchema-inputs-pending/` (paper texts, quotes, abstracts and extraction records whose licence is not established; not published). `MemoSchema-inputs/README.md` maps every paper experiment to its command, its inputs and the package that holds them, gives the install / restore / check commands, and separates (a) what the public package recomputes offline: the summaries of Tables IV–V, VI (stability), Figs. 3a/3b, the introduction diagnostic, the Gemini and the name-only checks; (b) what can be re-run after obtaining the papers (texts checked against published hashes); (c) what cannot be run under the original setting without the pending material: the downstream-extraction summary, the development-schema regeneration, the replay, MemoSchema and ablation re-runs, `cost_compare` and `memo_audit`. While packaged run directories are installed, scripts that need complete run directories stop with a message, and corpus construction into `results/e2e/e2e_v3` is refused.
- **Paper texts** for the baselines, downstream extraction and the reuse content checks are frozen from a run directory (offline): `python baselines/evaporate/reproduce.py prepare --run-dir <run_dir> --out results/evaporate/inputs/<batch>/<SUB_SET>` (also done by `reproduce.py batch`).

## 3. Run: what calls paid APIs and what does not

| Paid (LLM / embedding / judge API) | Offline (no API calls) |
|---|---|
| corpus construction: `run_e2e_local.sh`, `run_history.sh`, `run_memoschema_pipeline.py` | `check_release.py`, `build_corpus_manifest.py`, `reproduce.py prepare` |
| `run_fixed_corpus.py --execute`, `check_memory_reuse.py --execute` | `run_fixed_corpus.py` / `check_memory_reuse.py` without `--execute` (validation only) |
| construction drivers `run_memexp26.sh`, `run_memexp37.sh`, `run_memexp38.sh` (build / schemas / merge stages) | `prefill_memory.py`, `cost_compare.py`, `memo_audit.py` (replay saved records; need `.env` values present for imports) |
| baselines `run_memexp27/29/30/32/39/40.sh` | `pool_coverage.py`, `dev_baseline_schemas.py` (lexical selection only) |
| judges `eval_fine.sh`, `eval_replay.sh`, `run_judge_crosscheck.sh` | summaries: `summarize_fixed_corpus.py`, `paper_tables.py`, `memexp38.py report`, `memexp39_report.py`, `slot_analysis.py`, `stability_report.py`, `pool_diagnosis_report.py`, `memexp34.py build/report`, `memexp36.py report`, `judge_crosscheck.py report`, `memexp40.py report`, `memexp27.py collect` |
| `memexp33_merge.py`, `memexp36.py extract/judge` | |

Method on a fixed PDF set (every PDF in the directory, first 12,000 characters):

```bash
python scripts/run_fixed_corpus.py --pdf-dir PDFS/ --subdomain ZIF --out-dir results/new/ZIF_01            # validate only
python scripts/run_fixed_corpus.py --pdf-dir PDFS/ --subdomain ZIF --out-dir results/new/ZIF_01 --execute  # paid
python scripts/summarize_fixed_corpus.py results/new/ZIF_01                                                # offline
```

Paper experiments: `EXPERIMENTS.md` gives the order, directory layout and summary script of each table and figure. Fresh LLM calls give numbers that can differ from the paper.

## 4. What can and cannot be recovered

- **Evaluation corpora r7–r10, history r3–r6** — collections are fully described by manifests and can be rebuilt only by obtaining the listed PDFs; there is no tool that turns a manifest back into a run directory, because a run directory also holds LLM outputs (seed summaries, proposals, consolidations) that differ on every call. `run_history.sh` + `run_memexp26.sh` regenerate a history and corpora with the same settings, i.e. a *new* replay setting. The paper's original proposals and consolidations cannot be regenerated identically; those the replay reads are in the input packages (offline check: all 27 targets reuse the same papers, drop the same rows and get the same merge-reuse table as the original replay). Their evidence quotes are verbatim excerpts from the papers and are in the pending package.
- **Content checks in the replay** compare source and target paper texts only where both were frozen; the original replay had texts for r7–r10 only. Preparing r3–r6 texts as well makes the check stricter than in the paper. The frozen r7–r10 texts are in the pending package; the public package lists their SHA-256 (`frozen_texts_r7-r10.tsv`). If they are withheld, they must be re-prepared from newly obtained PDFs (the r3–r9 PDFs are no longer available) with **PyMuPDF 1.26.4** (r7–r9; the r10 texts used 1.28.2), and the replay equals the paper's only if they match the listed hashes.
- **Environment and platform** — confirmed versions and install commands: `envs/README.md`. `memo_audit.py` reproduces the audit only with PyMuPDF 1.26.4, the version the r7–r9 texts were extracted with (590 of the 592 comparable r10 texts then match their r7–r9 texts; with 1.28.2 only 373). All summaries use a fixed traversal order (sorted corpora, subdomains COF, MOF, ZIF; bootstrap seed 0), so their output does not depend on the file system. `memo_audit.py` takes, for a paper in several r7–r9 input sets, the text of the earliest set (sorted path order), and reads the run start times as America/Chicago local time (the zone the run directories were named in); the merge-store modification times it compares with are restored by `MemoSchema-inputs/tools/inputs_tool.py install`. Compared with the first version of these scripts (directory-order dependent), five reported values change by 0.001 (0.1 percentage point for Fig. 3b); see `EXPERIMENTS.md`.
- **Development corpora (introduction diagnostic)** — nine runs, seed set 02 of chem COF/MOF/ZIF, ad amyloid/neuroinflammation/tau and cs RLHF/llm/prompt (`scripts/dev_diagnostic.py`, manifests `data/corpora_dev/`). The top-40 candidate pool and the 8-field baseline schema are both selected from the same run of each corpus, placed at `Ablation/<domain>/<SUB>_02/`. The files the diagnostic reads are in the minimal input package (offline check: the 18 generated schemas are byte-identical to those the judge scored); the runs cannot be regenerated identically: they predate the run-config record (`run_config: null`, see `configuration_note`) and were built without memory with the LEGACY clustering. Three runs used a seed folder different from their id (`seed_set_matches_run_id: false`): ZIF_02 used `seed_chem/ZIF_01`, llm_02 used `seed_cs/llm_01`, and tau_02 used `seed_ad/tau`, which is not available (titles only in the seed list).
- **External access** — API quota (Azure OpenAI, Gemini), network for retrieval and `fetch_upstream.sh`, Docker for EVAPORATE.

## License

`baselines/autoschema/upstream/` keeps its MIT license; fetched third-party code keeps its original terms (`baselines/*/PROVENANCE.json`, `UPSTREAM_*`). `baselines/arxivdigestables/section_diag.py` is project code (provenance in its header). No new license is granted for the remaining code.
