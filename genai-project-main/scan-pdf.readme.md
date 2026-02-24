

## Scan-PDF Quick start

1. unzip → cd into `guard-pdf/`
2. create venv + install:

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e .
```

3. run a scan:

```bash
guard-pdf scan /path/to/file.pdf --rules rules/guardrail_rules.yaml --json
# exit code: 0 ALLOW, 2 BLOCK, 1 error
```

### What’s included

* `pyproject.toml` with console script `guard-pdf`
* `src/guard_pdf/`:

  * `cli.py` – CLI entrypoint (`guard-pdf scan ...`)
  * `scanner.py` – PDF text extraction (pdfminer), regex/heuristics (Luhn, ABA), YAML-driven rules
  * `pdf_utils.py` – blocks risky PDF features (/JavaScript, /Launch, /EmbeddedFile, etc.) using pikepdf
  * `decision.py` – policy that turns hits into **BLOCK/ALLOW** + reasons
  * `rules.py` – loads YAML rulepacks
* `rules/guardrail_rules.yaml` – starter policies you can extend over time
* `tests/` – `pytest` unit tests for detection and decisions
* `README.md` + minimal `LICENSE`

### Extend over time

* Add new detectors to `rules/guardrail_rules.yaml` (regex/keywords) and bump `version/last_updated`.
* If you want extra providers (e.g., Azure keys), append regex entries under `content_rules`.
* Tighten the policy (e.g., “block on any high”) by editing `decision_policy.block_on`.

### Notes

* No secrets are logged: outputs include **redacted** snippets (last 4 chars only).
* PDF parsing is *non-executing* and checks for embedded risky features before text extraction.
* Bias is towards safety: multiple high-severity signals or any critical → **BLOCK**.

If you want, I can add detectors for specific cloud keys you use (Azure, GCP service accounts, GitLab, etc.) or wire this into your existing pre-ingest pipeline.
