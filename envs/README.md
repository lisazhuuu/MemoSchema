# Environments

Only the versions below are confirmed for the paper's experiments: re-running the step with them reproduces a
recorded output byte for byte. No other version of the original environments was recorded, so none is pinned here.

| Use | Constraint file | Confirmed by |
|---|---|---|
| Re-extracting the r7–r9 frozen texts; `scripts/memo_audit.py` (reads the r10 PDFs); `scripts/memexp39_report.py` token estimate | `constraints-r7-r9-audit.txt` (PyMuPDF 1.26.4, tiktoken 0.9.0) | r7–r9 text SHA-256; recorded audit (paper layer, merge layer); Fig. 3b report |
| Preparing the r10 frozen texts (`baselines/evaporate/reproduce.py prepare`) | `constraints-r10-texts.txt` (PyMuPDF 1.28.2) | r10 text SHA-256 |
| Anything else (summaries, re-runs) | none: `requirements.txt` ranges | the offline summaries give identical reports with either PyMuPDF version (they read no PDF) |

```bash
# Python 3.11 (the tested interpreter: 3.11.9; the interpreter version of the original runs was not recorded)
python3.11 -m venv .venv-audit && .venv-audit/bin/pip install -r requirements.txt -c envs/constraints-r7-r9-audit.txt
python3.11 -m venv .venv-r10   && .venv-r10/bin/pip install -r requirements.txt -c envs/constraints-r10-texts.txt
```

`tiktoken` downloads the `o200k_base` encoding on first use (public, no key) and caches it; later runs are offline.

`test-environment-lock.txt` is a `pip freeze` of the environment the submitted code was tested in (Nov 2026). It is a
test environment, not the original one; its PyMuPDF 1.28.2 does not reproduce the r7–r9 texts or the audit.
