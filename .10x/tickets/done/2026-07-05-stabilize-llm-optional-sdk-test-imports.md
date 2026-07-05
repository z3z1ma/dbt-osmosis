Status: done
Created: 2026-07-05
Updated: 2026-07-05
Parent: None
Depends-On: None

# Stabilize LLM optional SDK test imports

## Scope

Replace conditional `openai` test imports in `tests/core/test_llm.py` with pytest's native `importorskip` so CodeQL no longer sees possibly uninitialized local variables.

In scope:

- Update the four retry tests that conditionally import `openai`.
- Preserve skip behavior when the optional OpenAI SDK is not installed.
- Verify focused LLM tests and Ruff.
- Verify CodeQL no longer reports `py/uninitialized-local-variable` for those test imports.

Out of scope:

- Installing optional OpenAI or Azure Identity dependencies.
- Changing retry behavior in `src/dbt_osmosis/core/llm.py`.
- Reworking other optional dependency test patterns.

## Acceptance Criteria

- ACC-001: `tests/core/test_llm.py` uses initialized module bindings for optional OpenAI retry tests.
- ACC-002: Focused LLM tests pass or skip the same optional dependency cases.
- ACC-003: Ruff passes on `tests/core/test_llm.py`.
- ACC-004: CodeQL no longer reports `py/uninitialized-local-variable` in `tests/core/test_llm.py`.

## Closure Evidence

- ACC-001: Recorded in `.10x/evidence/2026-07-05-llm-optional-sdk-test-imports.md`; retry tests now bind `openai` with `pytest.importorskip`.
- ACC-002: Recorded in `.10x/evidence/2026-07-05-llm-optional-sdk-test-imports.md`; focused LLM tests passed with the same 30 passed and 9 skipped pattern.
- ACC-003: Recorded in `.10x/evidence/2026-07-05-llm-optional-sdk-test-imports.md`; Ruff passed for `tests/core/test_llm.py`.
- ACC-004: Recorded in `.10x/evidence/2026-07-05-llm-optional-sdk-test-imports.md`; full CodeQL security-and-quality scan dropped from 65 to 61 results and has zero `tests/core/test_llm.py` results.

## Evidence Expectations

- Capture before CodeQL findings from `/tmp/dbt-osmosis-ai-quality/continuation/codeql-results-lambda-after.sarif`.
- Capture focused pytest output.
- Capture Ruff output.
- Capture CodeQL after-check or SARIF parse proving the targeted findings cleared.

## Progress and Notes

- 2026-07-05: CodeQL after callable wrapper cleanup reported four `py/uninitialized-local-variable` findings in `tests/core/test_llm.py` for `openai` inside optional SDK retry tests.
- 2026-07-05: Source inspection showed each test currently uses `try: import openai` followed by `pytest.skip`, which is behaviorally equivalent to `pytest.importorskip("openai")` for the local environment.
- 2026-07-05: Replaced the four conditional OpenAI imports with `pytest.importorskip("openai", reason="openai not installed")`.
- 2026-07-05: Verified focused LLM tests, Ruff, and full CodeQL after-scan.
- 2026-07-05: Closure review passed with no blocking findings.

## Blockers

None known.

## References

- `/tmp/dbt-osmosis-ai-quality/continuation/codeql-results-lambda-after.sarif`
- `/tmp/dbt-osmosis-ai-quality/continuation/codeql-results-llm-test-import-after.sarif`
- `.10x/evidence/2026-07-05-llm-optional-sdk-test-imports.md`
- `.10x/reviews/2026-07-05-llm-optional-sdk-test-imports-review.md`
- `tests/core/test_llm.py`
