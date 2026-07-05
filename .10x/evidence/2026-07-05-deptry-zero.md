Status: recorded
Created: 2026-07-05
Updated: 2026-07-05
Relates-To: .10x/tickets/done/2026-07-05-make-deptry-zero.md

# Deptry zero evidence

## What was observed

Deptry now exits zero using repository configuration.

## Procedure

- `uv run --no-sync --with deptry deptry .` → exit 0; `Success! No dependency issues found.`
- `uv run pytest tests/test_package_metadata.py` → exit 0; 9 passed, 2 warnings.
- `uvx ruff==0.15.17 check pyproject.toml tests/test_package_metadata.py` → exit 0.
- `uvx ruff==0.15.17 format --check pyproject.toml tests/test_package_metadata.py` → exit 0.
- `uv lock` → exit 0; resolved 178 packages.

## What this supports or challenges

This supports `.10x/tickets/done/2026-07-05-make-deptry-zero.md` acceptance. The project metadata now distinguishes direct imports, optional extras, first-party imports, and dev tooling more accurately.

## Limits

Deptry still prints informational assumptions for some package-to-module mappings, but it reports no dependency issues and exits 0.
