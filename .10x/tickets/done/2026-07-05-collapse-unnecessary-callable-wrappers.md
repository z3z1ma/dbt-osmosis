Status: done
Created: 2026-07-05
Updated: 2026-07-05
Parent: None
Depends-On: None

# Collapse unnecessary callable wrappers

## Scope

Address CodeQL `py/unnecessary-lambda` findings where lambdas are direct wrappers around existing callables.

In scope:

- Replace `lambda v: bool(v)` predicates in `YamlRefactorContext` config lookups with `bool`.
- Replace workbench hotkey wrappers `lambda: run_query()` with `run_query`.
- Replace the test-only `lambda: object()` helper with the callable `object`.
- Verify focused settings, workbench, and inheritance behavior tests.
- Verify CodeQL no longer reports `py/unnecessary-lambda`.

Out of scope:

- Changing config precedence, workbench hotkey semantics, or semantic analysis behavior.
- Refactoring surrounding settings or workbench app structure.
- Addressing other CodeQL rules.

## Acceptance Criteria

- ACC-001: All six CodeQL `py/unnecessary-lambda` findings are removed.
- ACC-002: Settings tests covering source definitions, ignore patterns, and YAML settings pass.
- ACC-003: Focused workbench and inheritance behavior tests pass.
- ACC-004: Ruff passes on touched Python files.
- ACC-005: Diff is limited to replacing wrappers with equivalent callables plus 10x records.

## Closure Evidence

- ACC-001: Recorded in `.10x/evidence/2026-07-05-callable-wrapper-collapse.md`; full CodeQL security-and-quality scan dropped from 71 to 65 results and has zero `py/unnecessary-lambda` results.
- ACC-002: Recorded in `.10x/evidence/2026-07-05-callable-wrapper-collapse.md`; `tests/core/test_settings.py` passed.
- ACC-003: Recorded in `.10x/evidence/2026-07-05-callable-wrapper-collapse.md`; `tests/core/test_workbench_app.py` and the focused inheritance behavior test passed.
- ACC-004: Recorded in `.10x/evidence/2026-07-05-callable-wrapper-collapse.md`; Ruff passed for all touched Python files.
- ACC-005: Confirmed by `.10x/reviews/2026-07-05-callable-wrapper-collapse-review.md`; implementation changes only replace direct callable wrappers.

## Evidence Expectations

- Capture before CodeQL findings from `/tmp/dbt-osmosis-ai-quality/continuation/codeql-results-llm-after.sarif`.
- Capture focused pytest output.
- Capture Ruff output.
- Capture CodeQL after-check or SARIF parse proving `py/unnecessary-lambda` is gone.

## Progress and Notes

- 2026-07-05: CodeQL after the LLM cleanup reported six `py/unnecessary-lambda` findings: three in `src/dbt_osmosis/core/settings.py`, two in `src/dbt_osmosis/workbench/app.py`, and one in `tests/core/test_inheritance_behavior.py`.
- 2026-07-05: Source inspection confirmed the production callables have compatible signatures: `bool` accepts one value and `run_query` accepts no arguments.
- 2026-07-05: Replaced direct wrapper lambdas with `bool`, `run_query`, and `object`.
- 2026-07-05: Verified focused tests, Ruff, and full CodeQL after-scan.
- 2026-07-05: Closure review passed with no blocking findings.

## Blockers

None known.

## References

- `/tmp/dbt-osmosis-ai-quality/continuation/codeql-results-llm-after.sarif`
- `/tmp/dbt-osmosis-ai-quality/continuation/codeql-results-lambda-after.sarif`
- `.10x/evidence/2026-07-05-callable-wrapper-collapse.md`
- `.10x/reviews/2026-07-05-callable-wrapper-collapse-review.md`
- `src/dbt_osmosis/core/settings.py`
- `src/dbt_osmosis/workbench/app.py`
- `tests/core/test_inheritance_behavior.py`
