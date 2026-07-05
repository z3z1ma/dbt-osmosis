Status: done
Created: 2026-07-05
Updated: 2026-07-05
Parent: None
Depends-On: None

# Tighten LLM optional import handling

## Scope

Address the small production CodeQL findings in `src/dbt_osmosis/core/llm.py` without changing LLM provider semantics.

In scope:

- Remove the unused package-level `openai` binding while preserving optional OpenAI SDK behavior.
- Keep `_OpenAIRateLimitError`, `OpenAI`, and `AzureOpenAI` behavior equivalent for retry and client construction.
- Add explicit handling documentation for malformed `Retry-After` headers or otherwise clear the `py/empty-except` finding.
- Verify focused LLM tests and CodeQL rule deltas.

Out of scope:

- Changing LLM provider selection, required environment variables, retry counts, or retry timing semantics.
- Reworking Azure Identity behavior.
- Fixing test-only CodeQL findings in `tests/core/test_llm.py`.
- Redesigning optional dependency loading across the public facade.

## Acceptance Criteria

- ACC-001: `src/dbt_osmosis/core/llm.py` no longer exposes an unused package-level `openai` binding.
- ACC-002: Malformed `Retry-After` values still fall back to exponential backoff.
- ACC-003: Focused LLM tests pass.
- ACC-004: Ruff passes on touched Python files.
- ACC-005: CodeQL no longer reports `py/empty-except` or `py/unused-global-variable` for `src/dbt_osmosis/core/llm.py`.

## Closure Evidence

- ACC-001: Recorded in `.10x/evidence/2026-07-05-llm-optional-import-handling.md`; package-level `openai` binding was removed and no source/test references rely on it.
- ACC-002: Recorded in `.10x/evidence/2026-07-05-llm-optional-import-handling.md`; retry logic still leaves the initialized exponential delay in place when `Retry-After` is malformed.
- ACC-003: Recorded in `.10x/evidence/2026-07-05-llm-optional-import-handling.md`; focused LLM tests passed with 30 passed and 9 skipped optional-dependency cases.
- ACC-004: Recorded in `.10x/evidence/2026-07-05-llm-optional-import-handling.md`; Ruff passed for `src/dbt_osmosis/core/llm.py`.
- ACC-005: Recorded in `.10x/evidence/2026-07-05-llm-optional-import-handling.md`; full CodeQL security-and-quality scan dropped from 73 to 71 results and has zero results for `src/dbt_osmosis/core/llm.py`.

## Evidence Expectations

- Capture the before CodeQL findings from `/tmp/dbt-osmosis-ai-quality/continuation/codeql-results-after.sarif`.
- Capture focused LLM pytest output.
- Capture Ruff output.
- Capture a CodeQL after-check or SARIF parse proving the targeted rules cleared for `src/dbt_osmosis/core/llm.py`.

## Progress and Notes

- 2026-07-05: CodeQL security-and-quality scan after schema cleanup reported `py/empty-except` at `src/dbt_osmosis/core/llm.py:100` and `py/unused-global-variable` at `src/dbt_osmosis/core/llm.py:26`.
- 2026-07-05: Source inspection found the `openai` module binding is only used to derive `RateLimitError`; callers and tests patch/use `OpenAI`, `AzureOpenAI`, `_OPENAI_AVAILABLE`, and `_OpenAIRateLimitError` instead.
- 2026-07-05: Bound `RateLimitError` directly from the OpenAI SDK and removed the package-level `openai` fallback binding.
- 2026-07-05: Added an explanatory comment for malformed `Retry-After` fallback behavior.
- 2026-07-05: Verified focused tests, Ruff, and full CodeQL after-scan.
- 2026-07-05: Closure review passed with no blocking findings.

## Blockers

None known.

## References

- `/tmp/dbt-osmosis-ai-quality/continuation/codeql-results-after.sarif`
- `/tmp/dbt-osmosis-ai-quality/continuation/codeql-results-llm-after.sarif`
- `.10x/evidence/2026-07-05-llm-optional-import-handling.md`
- `.10x/reviews/2026-07-05-llm-optional-import-handling-review.md`
- `src/dbt_osmosis/core/llm.py`
- `tests/core/test_llm.py`
