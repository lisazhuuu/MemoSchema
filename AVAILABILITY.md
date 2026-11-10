# Code and data availability

Short statement for submission forms. Details: `README.md`, `EXPERIMENTS.md`, `MemoSchema-inputs/README.md`.

**Code.** All code of the method, the experiment drivers, the evaluation judges and the baseline adapters, with prompts,
configuration, reference inventories (ground truth), corpus and seed manifests (bibliographic records and SHA-256,
no PDFs) and the commands for every reported experiment. Third-party baseline code is fetched from pinned upstream
versions.

**Data.** A separate input package holds the model-generated run records and the judged outputs the experiments
read, with every verbatim excerpt of a paper replaced by its hash. With it, the reported numbers of Tables IV and V,
schema stability, Fig. 3 and the robustness analyses can be recomputed offline without API calls. Material that
contains text of the corpus papers (frozen paper texts, evidence quotes, abstracts, extraction records) is not
included, because its redistribution licence has not been established. Without it, the downstream useful-value rate
cannot be recomputed, and MemoSchema and the ablations cannot be re-run under the original setting. Corpus and
seed PDFs are not redistributed; they have to be obtained from the publishers, and texts prepared from them can be
checked against the published hashes.

**Limits.** New LLM calls give different outputs; re-runs are not expected to reproduce the reported numbers exactly.
Retrieval is time-dependent, so corpora built anew are new collections. Some original material is no longer available
(PDFs of the r3–r9 corpora, one seed set, the run configuration of the nine development runs).

**Access.** Download location: *to be added on publication* (no URL or DOI yet).
