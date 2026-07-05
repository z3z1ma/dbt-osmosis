Status: done
Created: 2026-07-05
Updated: 2026-07-05
Parent: None
Depends-On: None

# Clarify schema package exports

## Scope

Replace wildcard imports in `src/dbt_osmosis/core/schema/__init__.py` with explicit imports so `__all__` names are defined directly in the package module.

In scope:

- Preserve the existing exported names.
- Avoid changing reader, writer, parser, or validation behavior.
- Verify runtime imports still expose the same names.
- Verify CodeQL no longer reports `py/undefined-export` for schema package exports, or record any residual reason.

Out of scope:

- Redesigning the schema package public API.
- Moving private cache/read/write helpers.
- Changing call sites that already import concrete schema modules.

## Acceptance Criteria

- ACC-001: `from dbt_osmosis.core.schema import _YAML_BUFFER_CACHE, _read_yaml, _write_yaml` succeeds.
- ACC-002: Focused schema/package metadata tests pass.
- ACC-003: Ruff passes on the touched Python file.
- ACC-004: CodeQL `py/undefined-export` no longer reports `src/dbt_osmosis/core/schema/__init__.py`.
- ACC-005: Diff is limited to the package export file and 10x records unless tests prove otherwise.

## Closure Evidence

- ACC-001: Recorded in `.10x/evidence/2026-07-05-schema-package-export-clarity.md`; runtime import of `_YAML_BUFFER_CACHE`, `_read_yaml`, and `_write_yaml` succeeded.
- ACC-002: Recorded in `.10x/evidence/2026-07-05-schema-package-export-clarity.md`; `tests/core/test_validation.py` and `tests/core/test_schema.py` passed with 80 tests.
- ACC-003: Recorded in `.10x/evidence/2026-07-05-schema-package-export-clarity.md`; Ruff passed for `src/dbt_osmosis/core/schema/__init__.py`.
- ACC-004: Recorded in `.10x/evidence/2026-07-05-schema-package-export-clarity.md`; full CodeQL security-and-quality scan dropped from 76 to 73 results and has zero `py/undefined-export` results.
- ACC-005: Confirmed by review in `.10x/reviews/2026-07-05-schema-package-export-clarity-review.md`; implementation diff is limited to `src/dbt_osmosis/core/schema/__init__.py` plus 10x records.

## Evidence Expectations

- Capture before CodeQL finding from the continuation scan.
- Capture runtime import check and focused tests.
- Capture focused CodeQL or SARIF after-check for `py/undefined-export`.

## Progress and Notes

- 2026-07-05: CodeQL security-and-quality scan reported `py/undefined-export` for `_YAML_BUFFER_CACHE`, `_read_yaml`, and `_write_yaml` in `src/dbt_osmosis/core/schema/__init__.py`.
- 2026-07-05: Runtime inspection confirmed these names are present today via wildcard imports, so the fix is static-boundary clarity rather than behavior repair.
- 2026-07-05: Started execution on branch `codex/quality-optimizer-hill-climb`.
- 2026-07-05: Replaced wildcard imports with explicit imports while preserving the existing `__all__` export list.
- 2026-07-05: Verified runtime imports, focused schema tests, Ruff, and full CodeQL after-scan.
- 2026-07-05: Closure review passed with no blocking findings.

## Blockers

None known.

## References

- `/tmp/dbt-osmosis-ai-quality/continuation/codeql-results.sarif`
- `/tmp/dbt-osmosis-ai-quality/continuation/codeql-results-after.sarif`
- `.10x/evidence/2026-07-05-schema-package-export-clarity.md`
- `.10x/reviews/2026-07-05-schema-package-export-clarity-review.md`
- `src/dbt_osmosis/core/schema/__init__.py`
