Status: done
Created: 2026-07-05
Updated: 2026-07-05
Parent: None
Depends-On: None

# Make optional return paths explicit

## Scope

Address CodeQL `py/mixed-returns` findings by making optional return paths explicit.

In scope:

- Add an explicit `None` return when SQL regex parsing does not produce a complete match.
- Make the test `sample_node` fixture's post-skip path unreachable.
- Verify focused SQL proxy and test suggestion tests.
- Verify Ruff and CodeQL rule delta.

Out of scope:

- Changing SQL proxy parsing semantics.
- Changing test suggestion fixture selection semantics.
- Addressing cyclic import, ineffectual statement, or special-method findings.

## Acceptance Criteria

- ACC-001: `_regex_parse_to_complete_dict` returns a dict for complete matches and explicitly returns `None` otherwise.
- ACC-002: `sample_node` does not implicitly return `None` after `pytest.skip`.
- ACC-003: Focused affected tests pass.
- ACC-004: Ruff passes on touched files.
- ACC-005: CodeQL no longer reports `py/mixed-returns`.

## Closure Evidence

- ACC-001: Recorded in `.10x/evidence/2026-07-05-optional-return-paths.md`; `_regex_parse_to_complete_dict` now explicitly returns `None`.
- ACC-002: Recorded in `.10x/evidence/2026-07-05-optional-return-paths.md`; `sample_node` raises if `pytest.skip` ever returns.
- ACC-003: Recorded in `.10x/evidence/2026-07-05-optional-return-paths.md`; focused SQL proxy and test suggestion tests passed with 44 tests.
- ACC-004: Recorded in `.10x/evidence/2026-07-05-optional-return-paths.md`; Ruff passed for touched files.
- ACC-005: Recorded in `.10x/evidence/2026-07-05-optional-return-paths.md`; full CodeQL security-and-quality scan dropped from 56 to 54 results and has zero `py/mixed-returns` results.

## Evidence Expectations

- Capture before CodeQL findings from `/tmp/dbt-osmosis-ai-quality/continuation/codeql-results-import-style-after.sarif`.
- Capture focused pytest output.
- Capture Ruff output.
- Capture CodeQL after-check or SARIF parse proving the targeted findings cleared.

## Progress and Notes

- 2026-07-05: CodeQL after test import-style cleanup reported two `py/mixed-returns` findings: `src/dbt_osmosis/sql/proxy.py:46` and `tests/core/test_test_suggestions.py:37`.
- 2026-07-05: Source inspection found both are explicit/implicit `None` shape issues, not intended behavior changes.
- 2026-07-05: Added explicit no-match `None` return to SQL regex parsing helper.
- 2026-07-05: Added a defensive assertion after `pytest.skip` in `sample_node`.
- 2026-07-05: Verified focused tests, Ruff, and full CodeQL after-scan.
- 2026-07-05: Closure review passed with no blocking findings.

## Blockers

None known.

## References

- `/tmp/dbt-osmosis-ai-quality/continuation/codeql-results-import-style-after.sarif`
- `/tmp/dbt-osmosis-ai-quality/continuation/codeql-results-mixed-return-after.sarif`
- `.10x/evidence/2026-07-05-optional-return-paths.md`
- `.10x/reviews/2026-07-05-optional-return-paths-review.md`
- `src/dbt_osmosis/sql/proxy.py`
- `tests/core/test_test_suggestions.py`
