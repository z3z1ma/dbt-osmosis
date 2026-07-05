Status: recorded
Created: 2026-07-05
Updated: 2026-07-05
Target: .10x/tickets/done/2026-07-05-normalize-test-module-import-style.md
Verdict: pass

# Test module import style review

## Target

Review of import normalization in `tests/core/test_demo_fixture_support.py`, `tests/core/test_inheritance_behavior.py`, `tests/core/test_logger.py`, and `tests/core/test_test_suggestions.py`.

## Assumptions Tested

- Replacing direct `_run_dbt_commands` usage with `test_conftest._run_dbt_commands` preserves the same function call.
- String monkeypatch targets in inheritance tests patch the same module attributes as the prior module object.
- Logger and test-suggestion local aliases reference the same module attributes as the prior direct imports.
- The affected test suites still pass after import normalization.

## Findings

No blocking findings.

## Verdict

Pass. The patch removes mixed import styles without changing assertions, fixtures, or production code, and CodeQL no longer reports `py/import-and-import-from`.

## Residual Risk

The affected test file `tests/core/test_test_suggestions.py` still has a separate `py/mixed-returns` CodeQL finding that predates this ticket and needs separate scope.
