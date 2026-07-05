Status: recorded
Created: 2026-07-05
Updated: 2026-07-05
Relates-To: .10x/tickets/done/2026-07-05-collapse-unnecessary-callable-wrappers.md

# Callable wrapper collapse

## What Was Observed

The CodeQL scan after LLM cleanup reported 71 total Python security-and-quality results. Six results used rule `py/unnecessary-lambda`:

- `src/dbt_osmosis/core/settings.py`: three `lambda v: bool(v)` predicates.
- `src/dbt_osmosis/workbench/app.py`: two `lambda: run_query()` hotkey callbacks.
- `tests/core/test_inheritance_behavior.py`: one `lambda: object()` monkeypatch helper.

After replacing those wrappers with equivalent callables, the full CodeQL security-and-quality scan reported 65 total results and zero `py/unnecessary-lambda` results.

## Procedure

- Replaced `lambda v: bool(v)` with `bool` in `YamlRefactorContext` config lookups.
- Replaced `lambda: run_query()` with `run_query` for the workbench run-query hotkeys.
- Replaced `lambda: object()` with `object` in the focused inheritance behavior test.
- Ran focused tests:

```bash
PYTHONPYCACHEPREFIX=/tmp/dbt-osmosis-ai-quality/continuation/pycache-lambda-tests \
  uv run pytest \
  tests/core/test_settings.py \
  tests/core/test_workbench_app.py \
  tests/core/test_inheritance_behavior.py::test_semantic_analysis_tag_merge_preserves_existing_then_suggested_order \
  -o cache_dir=/tmp/dbt-osmosis-ai-quality/continuation/pytest-cache-lambda
```

Result: 66 passed, 2 warnings.

- Ran Ruff on the touched files:

```bash
uvx ruff==0.15.17 check \
  src/dbt_osmosis/core/settings.py \
  src/dbt_osmosis/workbench/app.py \
  tests/core/test_inheritance_behavior.py
```

Result: all checks passed.

- Rebuilt and analyzed a CodeQL database:

```bash
rm -rf /tmp/dbt-osmosis-ai-quality/continuation/codeql-db-lambda-after \
  /tmp/dbt-osmosis-ai-quality/continuation/codeql-results-lambda-after.sarif

codeql database create /tmp/dbt-osmosis-ai-quality/continuation/codeql-db-lambda-after \
  --language=python \
  --source-root . \
  --overwrite

codeql database analyze \
  /tmp/dbt-osmosis-ai-quality/continuation/codeql-db-lambda-after \
  codeql/python-queries:codeql-suites/python-security-and-quality.qls \
  --format=sarif-latest \
  --output=/tmp/dbt-osmosis-ai-quality/continuation/codeql-results-lambda-after.sarif
```

Parsed result summary:

```text
total: 65
py/unnecessary-lambda: 0
```

## What This Supports Or Challenges

This supports closing `.10x/tickets/done/2026-07-05-collapse-unnecessary-callable-wrappers.md`:

- `ACC-001`: The targeted CodeQL rule is gone.
- `ACC-002`: Settings tests passed.
- `ACC-003`: Workbench and focused inheritance behavior tests passed.
- `ACC-004`: Ruff passed.
- `ACC-005`: The implementation diff is limited to replacing wrapper lambdas with equivalent callables.

## Limits

The CodeQL after-scan still reports older findings in touched files: one core import-cycle finding in `settings.py` and two import-style findings in `tests/core/test_inheritance_behavior.py`. Those rules are outside this ticket.
