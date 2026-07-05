Status: recorded
Created: 2026-07-05
Updated: 2026-07-05
Relates-To: .10x/tickets/done/2026-07-05-normalize-test-module-import-style.md

# Test module import style

## What Was Observed

The CodeQL scan after LLM optional SDK test import cleanup reported 61 total Python security-and-quality results. Five results used rule `py/import-and-import-from`:

- `tests/core/test_demo_fixture_support.py` for `tests.conftest`.
- `tests/core/test_inheritance_behavior.py` for `dbt_osmosis.core.inheritance`.
- `tests/core/test_logger.py` for `dbt_osmosis.core.logger`.
- `tests/core/test_test_suggestions.py` for `dbt_osmosis.core.test_suggestions`.

After normalizing those tests to one import style per target module, the full CodeQL security-and-quality scan reported 56 total results and zero `py/import-and-import-from` results.

## Procedure

- Replaced the direct `_run_dbt_commands` import with `test_conftest._run_dbt_commands`.
- Replaced inline inheritance module imports used only for monkeypatching with string monkeypatch targets.
- Kept the logger module import and created local aliases from it instead of mixing module and from-import styles.
- Kept the test-suggestions module import and created local aliases from it instead of mixing module and from-import styles.
- Ran focused affected tests:

```bash
PYTHONPYCACHEPREFIX=/tmp/dbt-osmosis-ai-quality/continuation/pycache-import-style-tests \
  uv run pytest \
  tests/core/test_demo_fixture_support.py \
  tests/core/test_inheritance_behavior.py \
  tests/core/test_logger.py \
  tests/core/test_test_suggestions.py \
  -o cache_dir=/tmp/dbt-osmosis-ai-quality/continuation/pytest-cache-import-style
```

Result: 112 passed, 2 warnings.

- Ran Ruff on touched files:

```bash
uvx ruff==0.15.17 check \
  tests/core/test_demo_fixture_support.py \
  tests/core/test_inheritance_behavior.py \
  tests/core/test_logger.py \
  tests/core/test_test_suggestions.py
```

Result: all checks passed.

- Rebuilt and analyzed a CodeQL database:

```bash
rm -rf /tmp/dbt-osmosis-ai-quality/continuation/codeql-db-import-style-after \
  /tmp/dbt-osmosis-ai-quality/continuation/codeql-results-import-style-after.sarif

codeql database create /tmp/dbt-osmosis-ai-quality/continuation/codeql-db-import-style-after \
  --language=python \
  --source-root . \
  --overwrite

codeql database analyze \
  /tmp/dbt-osmosis-ai-quality/continuation/codeql-db-import-style-after \
  codeql/python-queries:codeql-suites/python-security-and-quality.qls \
  --format=sarif-latest \
  --output=/tmp/dbt-osmosis-ai-quality/continuation/codeql-results-import-style-after.sarif
```

Parsed result summary:

```text
total: 56
py/import-and-import-from: 0
```

## What This Supports Or Challenges

This supports closing `.10x/tickets/done/2026-07-05-normalize-test-module-import-style.md`:

- `ACC-001`: The targeted CodeQL rule is gone.
- `ACC-002`: Focused affected tests passed.
- `ACC-003`: Ruff passed.
- `ACC-004`: The implementation diff only normalizes imports and the direct references required by those import changes.

## Limits

The CodeQL after-scan still reports a pre-existing `py/mixed-returns` finding in `tests/core/test_test_suggestions.py`, which is outside this ticket.
