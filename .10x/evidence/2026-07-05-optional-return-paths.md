Status: recorded
Created: 2026-07-05
Updated: 2026-07-05
Relates-To: .10x/tickets/done/2026-07-05-make-optional-return-paths-explicit.md

# Optional return paths

## What Was Observed

The CodeQL scan after test import-style cleanup reported 56 total Python security-and-quality results. Two results used rule `py/mixed-returns`:

- `src/dbt_osmosis/sql/proxy.py` for `_regex_parse_to_complete_dict`.
- `tests/core/test_test_suggestions.py` for the `sample_node` fixture.

After making both paths explicit, the full CodeQL security-and-quality scan reported 54 total results and zero `py/mixed-returns` results. No results remained for either touched file.

## Procedure

- Added `return None` to `_regex_parse_to_complete_dict` when the SQL regex does not match completely.
- Added a defensive `raise AssertionError("pytest.skip did not raise")` after `pytest.skip` in the `sample_node` fixture.
- Ran focused affected tests:

```bash
PYTHONPYCACHEPREFIX=/tmp/dbt-osmosis-ai-quality/continuation/pycache-mixed-return-tests \
  uv run pytest tests/core/test_sql_proxy.py tests/core/test_test_suggestions.py \
  -o cache_dir=/tmp/dbt-osmosis-ai-quality/continuation/pytest-cache-mixed-return
```

Result: 44 passed, 2 warnings.

- Ran Ruff on touched files:

```bash
uvx ruff==0.15.17 check \
  src/dbt_osmosis/sql/proxy.py \
  tests/core/test_test_suggestions.py
```

Result: all checks passed.

- Rebuilt and analyzed a CodeQL database:

```bash
rm -rf /tmp/dbt-osmosis-ai-quality/continuation/codeql-db-mixed-return-after \
  /tmp/dbt-osmosis-ai-quality/continuation/codeql-results-mixed-return-after.sarif

codeql database create /tmp/dbt-osmosis-ai-quality/continuation/codeql-db-mixed-return-after \
  --language=python \
  --source-root . \
  --overwrite

codeql database analyze \
  /tmp/dbt-osmosis-ai-quality/continuation/codeql-db-mixed-return-after \
  codeql/python-queries:codeql-suites/python-security-and-quality.qls \
  --format=sarif-latest \
  --output=/tmp/dbt-osmosis-ai-quality/continuation/codeql-results-mixed-return-after.sarif
```

Parsed result summary:

```text
total: 54
py/mixed-returns: 0
touched-file findings: 0
```

## What This Supports Or Challenges

This supports closing `.10x/tickets/done/2026-07-05-make-optional-return-paths-explicit.md`:

- `ACC-001`: SQL regex parsing has an explicit no-match `None`.
- `ACC-002`: The fixture no longer has an implicit fallthrough after `pytest.skip`.
- `ACC-003`: Focused tests passed.
- `ACC-004`: Ruff passed.
- `ACC-005`: Targeted CodeQL findings cleared.

## Limits

This ticket did not address remaining CodeQL cyclic import, ineffectual statement, or special-method findings.
