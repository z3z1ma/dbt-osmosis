Status: recorded
Created: 2026-07-05
Updated: 2026-07-05
Relates-To: .10x/tickets/done/2026-07-05-stabilize-llm-optional-sdk-test-imports.md

# LLM optional SDK test imports

## What Was Observed

The CodeQL scan after callable wrapper cleanup reported 65 total Python security-and-quality results. Four results used rule `py/uninitialized-local-variable` in `tests/core/test_llm.py`, where optional OpenAI SDK retry tests imported `openai` in a `try` block and skipped in `except ImportError`.

After replacing those conditional imports with `pytest.importorskip("openai", reason="openai not installed")`, the full CodeQL security-and-quality scan reported 61 total results and zero findings in `tests/core/test_llm.py`.

## Procedure

- Replaced the four optional OpenAI SDK retry-test import blocks with initialized module bindings from `pytest.importorskip`.
- Ran focused LLM tests:

```bash
PYTHONPYCACHEPREFIX=/tmp/dbt-osmosis-ai-quality/continuation/pycache-llm-import-tests \
  uv run pytest tests/core/test_llm.py \
  -o cache_dir=/tmp/dbt-osmosis-ai-quality/continuation/pytest-cache-llm-import
```

Result: 30 passed, 9 skipped, 2 warnings. The four OpenAI SDK retry tests continued to skip because the optional `openai` package is not installed locally.

- Ran Ruff on the touched file:

```bash
uvx ruff==0.15.17 check tests/core/test_llm.py
```

Result: all checks passed.

- Rebuilt and analyzed a CodeQL database:

```bash
rm -rf /tmp/dbt-osmosis-ai-quality/continuation/codeql-db-llm-test-import-after \
  /tmp/dbt-osmosis-ai-quality/continuation/codeql-results-llm-test-import-after.sarif

codeql database create /tmp/dbt-osmosis-ai-quality/continuation/codeql-db-llm-test-import-after \
  --language=python \
  --source-root . \
  --overwrite

codeql database analyze \
  /tmp/dbt-osmosis-ai-quality/continuation/codeql-db-llm-test-import-after \
  codeql/python-queries:codeql-suites/python-security-and-quality.qls \
  --format=sarif-latest \
  --output=/tmp/dbt-osmosis-ai-quality/continuation/codeql-results-llm-test-import-after.sarif
```

Parsed result summary:

```text
total: 61
tests/core/test_llm.py results: 0
py/uninitialized-local-variable: 0
```

## What This Supports Or Challenges

This supports closing `.10x/tickets/done/2026-07-05-stabilize-llm-optional-sdk-test-imports.md`:

- `ACC-001`: Optional retry tests now bind `openai` through `pytest.importorskip`.
- `ACC-002`: Focused LLM tests passed with equivalent optional dependency skips.
- `ACC-003`: Ruff passed.
- `ACC-004`: Targeted CodeQL findings cleared.

## Limits

The optional OpenAI SDK is not installed in the local environment, so the affected retry tests still skip locally. This ticket only improves test import structure and static-analysis clarity.
