# Paper experiments → code

Run from the repository root; everything reads/writes `results/`. Generation and judge steps make paid API calls; the summary scripts and the steps marked *offline* do not (README §3). Scores are averaged within a corpus (samples × judge repetitions), then across corpora. Paired differences use 95% bootstrap intervals over corpora, stratified by subdomain. MemoSchema always refers to the final content-keyed configuration: `results/memexp38/runs/<tag>/s<k>/REUSEC_M/`. "First output" means anchor sample `s1`.

## 0. Common settings

- **Corpora.** 27 main instances: r7–r9 × COF/MOF/ZIF × seed sets 01–03. Nine more r10 instances are used only for the EVAPORATE efficiency comparison. Manifests are in `data/corpora/`.
- **Corpus construction.** `scripts/run_e2e_local.sh` runs `scripts/run_memoschema_pipeline.py --fixes cluster,anchorschema` with 3 rounds, a target of 250 papers, and sources core, openalex and nature. Output goes to `results/e2e/e2e_v3/<batch>/chem/<SUB_SET>/`. `run_memexp26.sh` calls it for r7–r9, and `run_memexp33.sh` for r10.
- **Configurations.** `MEMOSCHEMA_FIX_CLUSTER=1` turns on average-linkage clustering (τc = 0.60). `MEMOSCHEMA_FIX_ANCHOR_SCHEMA=1` turns on anchor-guided drafting (τa = 0.60). With neither set, the code runs LEGACY.
- **Restructuring.** Budget 8, shortlist L = 25, prompt `MERGE` in `scripts/memexp26.py`.
- **Judges.** `Evaluation/evaluate_fine.py` gives hierarchical F1 (detailed inventory). `Evaluation/evaluate_against_ontology.py` gives strict one-to-one F1 (10 core concepts). Batch wrappers are `scripts/eval_fine.sh <root> <reps>` and `scripts/eval_replay.sh <root> <reps>`. Each schema is judged twice; the name-only check judges once.

## 0b. Directory layout and order of the generation steps

```
results/e2e/e2e_v3/<batch>/chem/<SUB>_<SET>/iterative_phase1_chem_<timestamp>/   one iterative run (r3-r10)
    round0N/{new_unique_papers,downloaded_metadata}.jsonl, round0N/schema_from_cumulative_corpus/merged_field_clusters.json
    schema_state/paper_level_field_proposals.jsonl, schema_state/distill_cache/, run_summary.json
results/e2e/e2e_v3/<batch>/chem/<SUB>_<SET>/run_config_<timestamp>.json
results/evaporate/inputs/<batch>/<SUB>_<SET>/papers.jsonl                           frozen 12k-character paper texts
results/memexp26 | memexp38 | ...  /runs/<e2eN>/s<k>/<ARM>/chem/<run>/final_schema.json, eval_fine/, eval/
```

0. With the input packages (`MemoSchema-inputs/README.md`) installed, the corpus construction of steps 1–2 (`run_history.sh`, the E2E part of `run_memexp26.sh`) is replaced by the packaged r3–r9 runs; the construction arms, the replay and `memexp37` read only those files. Step 3 (texts r7–r10) is covered by the pending package, or must be re-prepared (README §4).
1. History r3–r6 (needed by the reuse replay and the r10 live run): `scripts/run_history.sh` (paid; r3 = seed sets 02, r4–r6 = all nine sets, as in `data/corpora_history/`). The replay treats every completed chem run of r3..(target−1) of the same subdomain as history.
2. Evaluation corpora r7–r9 and the construction arms: §1 below.
3. Paper texts, *offline*, for every r7–r10 run (the paper's replay had texts only for these batches):
   `python baselines/evaporate/reproduce.py prepare --run-dir <run_dir> --out results/evaporate/inputs/<batch>/<SUB>_<SET>`
   (EVAPORATE's `reproduce.py batch` does the same before its stage 1). Needed by `memexp38.py` (stops if an r7–r10 file is missing; r3–r6 texts are optional and, if prepared, make the content check stricter than in the paper), `run_paired.py`, `memexp36.py`, `cost_compare.py`, `memo_audit.py`.
4. Reuse replay §1.3, baselines §2, r10 live run §4b.

## 1. Construction arms and MemoSchema (run first)

1. `mkdir -p results/memexp26 && scripts/run_memexp26.sh > results/memexp26/driver.log 2>&1`
   - Builds r7–r9 and the no-reuse arms ORIG/ORIG2/ORIG3 (three reconstructions × six anchor samples).
   - Builds LEGACY.
   - Restructures every non-LEGACY draft into `*_M`, then runs the judges.
   - `run_memexp27.sh` waits on this log.
2. `scripts/run_memexp37.sh` restructures LEGACY (sample 1) into `LEGACY_M`.
3. `scripts/run_memexp38.sh` builds REUSEC with content-addressed reuse from the r3–r9 history, makes six anchor samples, restructures them into `REUSEC_M`, and judges. Requires steps 0b.1 and 0b.3.

## 2. Table IV: schema quality

| Method | Driver | Arm (root) |
|---|---|---|
| MemoSchema | `run_memexp38.sh` | `REUSEC_M` (`results/memexp38`) |
| AutoSchema | `run_memexp32.sh` | `AUTOSCHEMA` (`results/autoschema`) |
| EVAPORATE | stage 1 + `run_memexp30.sh` (`baselines/README.md`) | `EVAPORATE_FULL8` (`results/evaporate`) |
| LLMs4SchemaDiscovery | `run_memexp27.sh` | `L4S` (`results/memexp27`) |
| ArxivDIGESTables | `run_memexp29.sh` | `ADT_topic` (`results/arxivdigestables`) |
| IA_SR | `run_memexp40.sh` (`memexp40.py part1`; after `run_memexp29.sh`, whose `results/arxivdigestables/inputs/` it reads) | `IA_SR` (`results/memexp40`) |

Summary: `python scripts/paper_tables.py`.

## 3. Table V / §VI-C: construction ablations (reuse disabled)

- SR_DRAFT: `run_memexp40.sh` (`memexp40.py part2`). It runs one critique-and-revision cycle on the same draft and shortlist, 3 repetitions, reading the prompts saved under `results/memexp26/calls/`.
- Summary: `python scripts/paper_tables.py`. It keeps the original resampling:
  - organized vs LEGACY, and restructured vs draft: `memexp26.contrast` with 2,000 draws;
  - both drafts restructured: sample 1 with 5,000 draws;
  - SR_DRAFT: 5,000 paired corpus resamples.
- Field-assignment diagnostics: `python scripts/slot_analysis.py`.
- Introduction diagnostic (top-40 candidate pool vs 8-field baseline; nine development corpora, seed set 02, README §4; manifests in `data/corpora_dev/`). Both sides use the same run of each corpus:
  1. place the nine runs at `Ablation/<domain>/<SUB>_02/` (list in `scripts/dev_diagnostic.py`; a missing run stops every step below), e.g. by installing the input packages and restoring the quotes (`MemoSchema-inputs/README.md`; `MemoSchema-inputs/tools/inputs_tool.py check-dev` then compares the outputs of steps 2–3 with the judged schemas);
  2. *offline*: `python scripts/pool_coverage.py` (top-40 candidates per run -> `results/abl_pool_old/top40`);
  3. *offline*: `python scripts/dev_baseline_schemas.py` (LEGACY 8-field schemas -> `results/abl_old/baseline`);
  4. paid judge, as in the paper: pool 1 repetition `scripts/eval_replay.sh results/abl_pool_old 1`, baseline 3 repetitions `scripts/eval_replay.sh results/abl_old 3`;
  5. *offline*: `python scripts/pool_diagnosis_report.py` (lenient recall averaged over repetitions within each corpus, then over the nine corpora).

## 4. Fig. 3: reuse and efficiency

- **(a) Replay.** `python scripts/memexp38.py report` compares REUSEC_M with ORIG_M ×3 on F1, using non-inferiority margin 0.02 on hierarchical F1. It also reports final-round reconstruction calls, computed as 2 × processed papers + consolidation calls + 2 × 6 anchor/restructuring calls.
- **(b) r10 vs EVAPORATE**, in this order:
  1. *offline*: `python scripts/prefill_memory.py` (merge and seed memory from the r3–r9 runs and `seeds/seed_chem/`)
  2. `scripts/run_memexp33.sh`: live r10 runs with M1/M2/M3 memory and usage logs, then restructuring.
  3. `scripts/run_memexp39.sh`: EVAPORATE stage 1 and full flow on r10.
  4. *offline*: `python scripts/cost_compare.py`, which writes `results/cost_compare.json`.
  5. *offline*: `python scripts/memo_audit.py`, which writes `results/memo_audit/details.json` and needs step 4, the r10 PDFs, the merge-store entries with their modification times, and PyMuPDF 1.26.4 (`envs/README.md`).
  6. *offline*: `python scripts/memexp39_report.py`.

  Token estimates for the separately recorded restructuring calls use tiktoken `o200k_base`.

## 5. Table VI / §VI-E: downstream utility and stability (first output)

- **Extraction utility.** `scripts/run_memexp36.sh` runs `memexp36.py extract`, `judge` and `report`.
  - Corpora: the 9 r8 corpora.
  - Papers: 5 per corpus, the first 5 of `results/evaporate/inputs/r8/<run>/papers.jsonl` after a seed-0 shuffle.
  - Extractor: AutoSchema Phase II (`baselines/autoschema/run_phase2.py`), given field names and descriptions.
  - MemoSchema input: `results/memexp38/runs/e2e8/s1/REUSEC_M`. The paper's REUSEC_M extraction records were produced by a separate run (`memexp41` in the development project); the input packages place them at the `results/memexp36/` paths read here, so `memexp36.py report` reproduces Table VI. Likewise the REUSEC_M name-only scores (`memexp42`) are placed under `results/memexp34/`.
- **Stability.** `python scripts/stability_report.py`. Name-only soft Jaccard with greedy one-to-one matching and threshold 0.30, on 108 within-subdomain pairs. The results are descriptive.

## 6. §VI-F: robustness (first output)

- **Cross-family judge.** `scripts/run_judge_crosscheck.sh` (`judge_crosscheck.py run`, then `report`; Gemini). It covers MemoSchema REUSEC_M vs LEGACY, AutoSchema, EVAPORATE, LLMs4SD, ArxivDIGESTables and IA_SR.
- **Name-only evaluation.** `scripts/run_memexp34.sh` (`memexp34.py build`, judges, `report`). It covers MemoSchema REUSEC_M vs LEGACY, AutoSchema, EVAPORATE, LLMs4SD and ArxivDIGESTables.

`memexp34.py`, `memexp36.py` and `stability_report.py` stop with an error if any MemoSchema `REUSEC_M` input is missing. They never fall back to another arm.

## 7. Deterministic order: values that differ from the first version of the paper

The summaries traverse corpora, subdomains (COF, MOF, ZIF) and samples in sorted order with bootstrap seed 0, and
`memo_audit.py` resolves a paper found in several r7–r9 input sets to the earliest set (sorted path order). The first
version of `paper_tables.py` and `memexp40.py report` filled the bootstrap strata in directory-listing order, and
`memo_audit.py` took the first text found in that order (3 papers have two different r7–r9 texts). Recomputed offline
from the same records, without new LLM calls, these values change (methods, samples and statistics unchanged):

| Paper location | Quantity | Previous | Now |
|---|---|---|---|
| §VI-B text (Table IV) | MemoSchema − AutoSchema, strict F1 | 0.078 [0.052, 0.102] | 0.078 [0.053, 0.102] |
| §VI-B text (Table IV) | MemoSchema − ArxivDIGESTables, strict F1 | 0.134 [0.109, 0.159] | 0.134 [0.108, 0.159] |
| §VI-C text (Table V) | restructuring − SR_DRAFT, strict F1 | 0.036 [0.006, 0.066] | 0.036 [0.006, 0.065] |
| §VI-C text (Table V) | SR_DRAFT − input draft, strict F1 | −0.004 [−0.029, 0.020] | −0.004 [−0.030, 0.020] |
| §VI-D text and Fig. 3b | cost reduction vs EVAPORATE | 48.2% [42.8%, 52.8%] | 48.2% [42.9%, 52.8%] (ratio CI upper end 0.572 → 0.571) |

No interval changes sign or crosses zero; every significance statement stays as it is. Other intervals printed by the
scripts change in the same way but are not quoted in the paper. The memory audit now finds 590 instead of 589 of
592 comparable r10 texts identical (one paper's r7–r9 text is taken from another input set); the Fig. 3b means
(410 calls, $6.01; 81.9% [80.5%, 83.1%] fewer calls) are unchanged, the per-corpus calls of COF_02 change from 479 to 477.
