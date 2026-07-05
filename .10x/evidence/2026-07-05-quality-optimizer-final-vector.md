Status: recorded
Created: 2026-07-05
Updated: 2026-07-05
Relates-To: .10x/tickets/done/2026-07-04-quality-optimizer-hill-climb.md, .10x/tickets/2026-07-05-ratify-docs-node20-docusaurus-upgrade.md, .10x/tickets/2026-07-05-ratify-type-checker-adoption-scope.md

# Quality optimizer final vector

## What Was Observed

Final current-state validation after commit `230cbef` on branch `codex/quality-optimizer-hill-climb` produced this metric vector:

```text
ruff_lint_errors: 0
ruff_format_drift: 0
basedpyright_errors: 0
pytest_failures: 0
pytest_result: 959 passed, 15 skipped, 2 warnings
coverage_total: 71.66%
codeql_python_security_and_quality_findings: 0
uv_audit_vulnerabilities: 0
osv_vulnerabilities: 5 docs npm vulnerabilities
npm_audit_vulnerabilities: 20 docs advisories
gitleaks_dir_findings: 0
semgrep_default_findings: 33, already triaged in .10x/evidence/2026-07-04-semgrep-triage.md
ty_status: not clean project-wide; adoption scope owned by .10x/tickets/2026-07-05-ratify-type-checker-adoption-scope.md
mypy_status: not clean project-wide; adoption scope owned by .10x/tickets/2026-07-05-ratify-type-checker-adoption-scope.md
```

Major deltas from the quality-optimizer workstream:

```text
CodeQL Python security-and-quality: 76 -> 0 findings
OSV source scan: 38 -> 5 vulnerabilities
uv audit: vulnerable -> clean
npm audit docs: 26 -> 20 advisories
coverage: 71.38% earlier in run -> 71.66% final
```

The remaining OSV/npm findings are all in `docs/package-lock.json` and are owned by `.10x/tickets/2026-07-05-ratify-docs-node20-docusaurus-upgrade.md`.

## Procedure

- Confirmed branch status was clean and tracking remote after the final pushed commit.
- Ran whole-repo Ruff:

```bash
uvx ruff==0.15.17 check .
uvx ruff==0.15.17 format --check .
```

Results:

```text
All checks passed.
247 files already formatted.
```

- Ran basedpyright:

```bash
uv run basedpyright --level error
```

Result: 0 errors, 0 warnings, 0 notes.

- Ran full pytest with coverage:

```bash
COVERAGE_FILE=/tmp/dbt-osmosis-ai-quality/continuation/.coverage-final \
PYTHONPYCACHEPREFIX=/tmp/dbt-osmosis-ai-quality/continuation/pycache-final-coverage \
  uv run coverage run --branch -m pytest -q \
  -o cache_dir=/tmp/dbt-osmosis-ai-quality/continuation/pytest-cache-final-coverage

COVERAGE_FILE=/tmp/dbt-osmosis-ai-quality/continuation/.coverage-final \
  uv run coverage report --show-missing
```

Result: 959 passed, 15 skipped, 2 warnings; total coverage 71.66%.

- Ran final CodeQL after the last type-tool polish:

```bash
codeql database create /tmp/dbt-osmosis-ai-quality/continuation/codeql-db-final-type-after \
  --language=python \
  --source-root . \
  --overwrite

codeql database analyze \
  /tmp/dbt-osmosis-ai-quality/continuation/codeql-db-final-type-after \
  codeql/python-queries:codeql-suites/python-security-and-quality.qls \
  --format=sarif-latest \
  --output=/tmp/dbt-osmosis-ai-quality/continuation/codeql-results-final-type-after.sarif
```

Result: exit 0; SARIF result count 0.

- Ran dependency and secrets checks:

```bash
uv audit --frozen
osv-scanner scan source -r .
npm --prefix docs audit --json
gitleaks dir --no-banner --redact \
  --report-format json \
  --report-path /tmp/dbt-osmosis-ai-quality/continuation/gitleaks-dir-final.json \
  .
```

Results:

```text
uv audit: no known vulnerabilities in 177 packages
OSV: 3 docs npm packages affected by 5 vulnerabilities
npm audit: 20 docs advisories, 1 high and 19 moderate
gitleaks dir: no leaks found
```

- Ran Semgrep default policy:

```bash
uv run --no-sync --with semgrep semgrep scan \
  --config p/default \
  --error \
  --json \
  --output /tmp/dbt-osmosis-ai-quality/continuation/semgrep-final.json \
  .
```

Result: 33 findings, same policy surface previously triaged in `.10x/evidence/2026-07-04-semgrep-triage.md`.

- Ran `ty` and mypy as exploratory non-mutating checks:

```bash
uv run --no-sync --with ty ty check
uv run --no-sync --with mypy mypy .
```

Result: both are not clean project-wide; adoption scope is owned by `.10x/tickets/2026-07-05-ratify-type-checker-adoption-scope.md`.

## What This Supports Or Challenges

This supports the final quality-optimizer closeout:

- Python CodeQL findings were reduced to zero.
- The Python test suite passes under normal pytest and coverage.
- Ruff formatting/linting and basedpyright error-level gates are clean.
- Python package vulnerability audit is clean.
- Current working tree secret scan is clean.
- Remaining dependency/security-policy work is durably owned instead of hidden.

## Limits

Semgrep default policy findings remain because they include policy choices already triaged under `.10x/evidence/2026-07-04-semgrep-triage.md` and open ratification tickets. Docs npm vulnerabilities remain blocked on the Docusaurus/Node 20 upgrade decision. Whole-repo `ty` and mypy are not adopted gates yet and need the dedicated type-checker adoption ticket.
