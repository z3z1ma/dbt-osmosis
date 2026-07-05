Status: recorded
Created: 2026-07-05
Updated: 2026-07-05
Relates-To: .10x/tickets/done/2026-07-05-ratify-type-checker-adoption-scope.md

# Type tools zero evidence

## What was observed

The exact default type-tool commands now exit zero:

- `uv run --no-sync --with ty ty check`
- `uv run --no-sync --with mypy mypy .`

The configured production type surface is `src/dbt_osmosis/core` plus `src/dbt_osmosis/cli`. Tests, the optional Streamlit workbench, and optional SQL proxy are excluded from ty/mypy default scope and remain covered by pytest and existing smoke tests.

## Procedure

- `uv run --no-sync --with ty ty check` → exit 0; all checks passed.
- `uv run --no-sync --with mypy mypy .` → exit 0; no issues found in 41 source files.
- `uv run basedpyright --level error` → exit 0; 0 errors, 0 warnings, 0 notes.
- `uvx ruff==0.15.17 check ...` for changed source and `pyproject.toml` → exit 0.
- `uvx ruff==0.15.17 format --check ...` for changed source and `pyproject.toml` → exit 0.
- `uv run pytest tests/core/test_cli.py tests/core/test_diff.py tests/core/test_migration.py tests/core/test_path_management.py tests/core/test_inheritance_behavior.py tests/core/test_transforms.py tests/core/test_schema.py tests/core/test_validation.py tests/core/test_sync_operations.py tests/core/test_llm.py` → exit 0; 275 passed, 10 skipped, 2 warnings.

## What this supports or challenges

This supports closing `.10x/tickets/done/2026-07-05-ratify-type-checker-adoption-scope.md`. The project now has zero-exit `ty`, mypy, and basedpyright checks over the production core/CLI surface.

## Limits

The mypy and ty default scopes intentionally exclude tests and optional-extra surfaces that require different dependency environments. This is encoded in repository configuration, not command-line `exit-zero` flags or baselines.
