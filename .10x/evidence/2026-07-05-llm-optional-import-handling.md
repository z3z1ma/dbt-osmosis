Status: recorded
Created: 2026-07-05
Updated: 2026-07-05
Relates-To: .10x/tickets/done/2026-07-05-tighten-llm-optional-import-handling.md

# LLM optional import handling

## What Was Observed

The CodeQL scan after schema export cleanup reported 73 total Python security-and-quality results. Two results pointed at `src/dbt_osmosis/core/llm.py`:

- `py/unused-global-variable` for package-level `openai`.
- `py/empty-except` for the malformed `Retry-After` fallback in `_call_with_retry`.

After the LLM cleanup, the full CodeQL security-and-quality scan reported 71 total results and zero results for `src/dbt_osmosis/core/llm.py`.

## Procedure

- Inspected source references for package-level `dbt_osmosis.core.llm.openai`; the only source/test binding was the fallback assignment inside `src/dbt_osmosis/core/llm.py`.
- Changed the optional OpenAI import to bind `RateLimitError` directly:

```python
from openai import RateLimitError as _OpenAIRateLimitError
```

- Removed the package-level fallback `openai = None` binding.
- Added an explicit comment documenting that malformed `Retry-After` headers keep the initialized exponential backoff delay.
- Ran focused LLM tests:

```bash
PYTHONPYCACHEPREFIX=/tmp/dbt-osmosis-ai-quality/continuation/pycache-llm-tests \
  uv run pytest tests/core/test_llm.py \
  -o cache_dir=/tmp/dbt-osmosis-ai-quality/continuation/pytest-cache-llm
```

Result: 30 passed, 9 skipped, 2 warnings. Skips were optional `openai` and `azure-identity` dependency paths in the local environment.

- Ran Ruff on the touched Python file:

```bash
uvx ruff==0.15.17 check src/dbt_osmosis/core/llm.py
```

Result: all checks passed.

- Rebuilt and analyzed a CodeQL database:

```bash
rm -rf /tmp/dbt-osmosis-ai-quality/continuation/codeql-db-llm-after \
  /tmp/dbt-osmosis-ai-quality/continuation/codeql-results-llm-after.sarif

codeql database create /tmp/dbt-osmosis-ai-quality/continuation/codeql-db-llm-after \
  --language=python \
  --source-root . \
  --overwrite

codeql database analyze \
  /tmp/dbt-osmosis-ai-quality/continuation/codeql-db-llm-after \
  codeql/python-queries:codeql-suites/python-security-and-quality.qls \
  --format=sarif-latest \
  --output=/tmp/dbt-osmosis-ai-quality/continuation/codeql-results-llm-after.sarif
```

Parsed result summary:

```text
total: 71
src/dbt_osmosis/core/llm.py results: 0
py/empty-except: 0
py/unused-global-variable: 0
```

## What This Supports Or Challenges

This supports closing `.10x/tickets/done/2026-07-05-tighten-llm-optional-import-handling.md`:

- `ACC-001`: The package-level `openai` binding is removed.
- `ACC-002`: The `wait_time` fallback remains the already initialized exponential delay when `Retry-After` cannot be parsed.
- `ACC-003`: Focused LLM tests passed.
- `ACC-004`: Ruff passed.
- `ACC-005`: The targeted CodeQL findings cleared with no remaining `core/llm.py` findings.

## Limits

The local environment does not install optional `openai` or `azure-identity`, so optional dependency tests that require those packages were skipped. The ticket did not address test-only CodeQL findings in `tests/core/test_llm.py`.
