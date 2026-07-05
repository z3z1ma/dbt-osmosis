Status: done
Created: 2026-07-05
Updated: 2026-07-05
Parent: None
Depends-On: None

# Normalize test module import style

## Scope

Address CodeQL `py/import-and-import-from` findings in tests by using one import style per target module.

In scope:

- Normalize `tests.conftest` imports in `tests/core/test_demo_fixture_support.py`.
- Normalize `dbt_osmosis.core.inheritance` imports in `tests/core/test_inheritance_behavior.py`.
- Normalize `dbt_osmosis.core.logger` imports in `tests/core/test_logger.py`.
- Normalize `dbt_osmosis.core.test_suggestions` imports in `tests/core/test_test_suggestions.py`.
- Verify focused affected tests, Ruff, and CodeQL rule delta.

Out of scope:

- Changing production imports.
- Refactoring test assertions beyond import style.
- Addressing CodeQL cyclic import, mixed return, unexpected raise, or ineffectual statement findings.

## Acceptance Criteria

- ACC-001: CodeQL no longer reports `py/import-and-import-from`.
- ACC-002: Focused affected tests pass.
- ACC-003: Ruff passes on touched test files.
- ACC-004: Test behavior remains equivalent; changes are limited to import normalization and direct references needed by that normalization.

## Closure Evidence

- ACC-001: Recorded in `.10x/evidence/2026-07-05-test-module-import-style.md`; full CodeQL security-and-quality scan dropped from 61 to 56 results and has zero `py/import-and-import-from` results.
- ACC-002: Recorded in `.10x/evidence/2026-07-05-test-module-import-style.md`; focused affected tests passed with 112 tests.
- ACC-003: Recorded in `.10x/evidence/2026-07-05-test-module-import-style.md`; Ruff passed for touched test files.
- ACC-004: Confirmed by `.10x/reviews/2026-07-05-test-module-import-style-review.md`; changes are limited to import normalization and equivalent references.

## Evidence Expectations

- Capture before CodeQL findings from `/tmp/dbt-osmosis-ai-quality/continuation/codeql-results-llm-test-import-after.sarif`.
- Capture focused pytest output.
- Capture Ruff output.
- Capture CodeQL after-check or SARIF parse proving the targeted findings cleared.

## Progress and Notes

- 2026-07-05: CodeQL after LLM optional SDK test import cleanup reported five `py/import-and-import-from` findings across four test files.
- 2026-07-05: Source inspection found each finding comes from tests mixing module imports for monkeypatch/facade checks with direct imports for convenience.
- 2026-07-05: Normalized affected tests to a single import style per target module while preserving monkeypatch/facade behavior.
- 2026-07-05: Verified focused affected tests, Ruff, and full CodeQL after-scan.
- 2026-07-05: Closure review passed with no blocking findings.

## Blockers

None known.

## References

- `/tmp/dbt-osmosis-ai-quality/continuation/codeql-results-llm-test-import-after.sarif`
- `/tmp/dbt-osmosis-ai-quality/continuation/codeql-results-import-style-after.sarif`
- `.10x/evidence/2026-07-05-test-module-import-style.md`
- `.10x/reviews/2026-07-05-test-module-import-style-review.md`
- `tests/core/test_demo_fixture_support.py`
- `tests/core/test_inheritance_behavior.py`
- `tests/core/test_logger.py`
- `tests/core/test_test_suggestions.py`
