Status: recorded
Created: 2026-07-05
Updated: 2026-07-05
Relates-To: .10x/tickets/done/2026-07-05-harden-workbench-feed-fetcher.md

# Workbench feed fetcher hardening evidence

## What Was Observed

The exhaustive Semgrep security-audit run reported one finding before this change:

```text
python.lang.security.audit.dynamic-urllib-use-detected.dynamic-urllib-use-detected
src/dbt_osmosis/workbench/app.py:156
```

The finding was on the opt-in external RSS feed fetcher, which used `urllib.request.urlopen(feed_url, timeout=...)`.

## Procedure

The fetcher was changed to parse the feed URL, reject non-`http` and non-`https` schemes, and use `http.client.HTTPConnection` or `http.client.HTTPSConnection` directly. This avoids `urllib.request.urlopen`, which accepts schemes such as `file://`.

Commands run:

```text
uvx ruff==0.15.17 format src/dbt_osmosis/workbench/app.py tests/core/test_workbench_app.py
uvx ruff==0.15.17 check src/dbt_osmosis/workbench/app.py tests/core/test_workbench_app.py
uv run basedpyright --level error src/dbt_osmosis/workbench/app.py tests/core/test_workbench_app.py
PYTHONPYCACHEPREFIX=/tmp/dbt-osmosis-ai-quality/exhaustive-2026-07-05/pycache-feed-tests-2 uv run pytest tests/core/test_workbench_app.py -q -o cache_dir=/tmp/dbt-osmosis-ai-quality/exhaustive-2026-07-05/pytest-cache-feed-tests-2
uv run --no-sync --with semgrep semgrep scan --config p/security-audit --error --json --output /tmp/dbt-osmosis-ai-quality/exhaustive-2026-07-05/semgrep-security-after-feed.json .
uv run --no-sync --with semgrep semgrep scan --config p/default --error --json --output /tmp/dbt-osmosis-ai-quality/exhaustive-2026-07-05/semgrep-default-after-feed.json .
codeql database create /tmp/dbt-osmosis-ai-quality/exhaustive-2026-07-05/feed-fix/codeql-db --language=python --source-root . --overwrite
codeql database analyze /tmp/dbt-osmosis-ai-quality/exhaustive-2026-07-05/feed-fix/codeql-db codeql/python-queries:codeql-suites/python-security-and-quality.qls --format=sarif-latest --output=/tmp/dbt-osmosis-ai-quality/exhaustive-2026-07-05/feed-fix/codeql.sarif
```

## Results

- Ruff formatted one file and then passed.
- basedpyright exited 0 with 0 errors, 0 warnings, 0 notes.
- `tests/core/test_workbench_app.py` passed: 14 passed, 2 warnings.
- Semgrep security-audit exited 0 with 0 findings.
- Semgrep default dropped from 33 findings to 32 findings; the removed finding was the workbench dynamic urllib result.
- CodeQL Python security-and-quality exited 0 with 0 SARIF results.

## What This Supports Or Challenges

This supports closing `.10x/tickets/done/2026-07-05-harden-workbench-feed-fetcher.md`.

The remaining Semgrep default findings are unchanged policy/CI findings:

- 26 mutable GitHub Action references;
- 2 `workflow_run` checkout findings;
- 2 Dependabot cooldown findings;
- 1 uv cooldown finding;
- 1 non-literal import warning in the CLI.

## Limits

The focused tests use fake HTTP(S) connections rather than live network calls. This is intentional; the workbench feed is optional and should fail closed when external access is unavailable.
