Status: recorded
Created: 2026-07-05
Updated: 2026-07-05
Relates-To: .10x/tickets/done/2026-07-05-clear-pydoclint-doc301.md

# pydoclint DOC301 cleanup evidence

## What Was Observed

The exhaustive pydoclint pass reported 19 DOC301 findings. All findings were redundant `__init__` docstrings under classes, which pydoclint expects to be documented on the class instead.

## Procedure

Removed the redundant constructor docstrings from:

- `src/dbt_osmosis/core/diff.py`
- `src/dbt_osmosis/core/introspection.py`
- `src/dbt_osmosis/core/migration.py`
- `src/dbt_osmosis/core/schema/validation.py`
- `src/dbt_osmosis/core/sql_lint.py`
- `src/dbt_osmosis/core/test_suggestions.py`
- `tests/core/test_config_resolution.py`
- `tests/core/test_property_accessor.py`

No constructor signatures, defaults, or runtime logic changed.

Commands run:

```text
uv run --no-sync --with pydoclint pydoclint src tests
uvx ruff==0.15.17 check <touched files>
uvx ruff==0.15.17 format --check <touched files>
uv run basedpyright --level error <touched source files>
PYTHONPYCACHEPREFIX=/tmp/dbt-osmosis-ai-quality/exhaustive-2026-07-05/pycache-doc301-tests uv run pytest tests/core/test_config_resolution.py tests/core/test_property_accessor.py -q -o cache_dir=/tmp/dbt-osmosis-ai-quality/exhaustive-2026-07-05/pytest-cache-doc301-tests
```

## Results

- pydoclint exited 0 and reported `No violations`.
- Ruff exited 0 and reported all touched files passed and were already formatted.
- basedpyright exited 0 with 0 errors, 0 warnings, 0 notes.
- Focused pytest exited 0: 79 passed, 3 skipped, 2 warnings.

## What This Supports Or Challenges

This supports closing `.10x/tickets/done/2026-07-05-clear-pydoclint-doc301.md` and updates the metric vector from pydoclint failing on 19 DOC301 findings to pydoclint clean.

## Limits

This did not add new docstrings. It only removed pydoclint-rejected duplicate constructor docstrings.
