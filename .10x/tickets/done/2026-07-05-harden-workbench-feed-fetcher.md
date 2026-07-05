Status: done
Created: 2026-07-05
Updated: 2026-07-05
Parent: None
Depends-On: None

# Harden workbench feed fetcher

## Scope

Replace the opt-in workbench RSS fetcher's `urllib.request.urlopen` path with a narrower HTTP(S)-only implementation so Semgrep security-audit no longer reports dynamic urllib usage.

In scope:

- Keep the external feed opt-in behavior unchanged.
- Keep the existing timeout and response-size cap.
- Reject non-http(s) schemes before any network call.
- Avoid adding a new HTTP client dependency.
- Update focused tests for oversized responses and non-http(s) rejection.

Out of scope:

- Redesigning the workbench feed UI.
- Adding redirects, retry policy, or new feed sources.
- Changing Streamlit state or dashboard layout.

## Acceptance Criteria

- ACC-001: `semgrep scan --config p/security-audit` no longer reports `dynamic-urllib-use-detected` for the workbench feed fetcher.
- ACC-002: Existing feed behavior still returns the disabled/unavailable safe fallback on failures.
- ACC-003: Focused workbench tests pass.
- ACC-004: Ruff and basedpyright error-level remain clean for the touched surface.

## Evidence Expectations

- Focused pytest for `tests/core/test_workbench_app.py`.
- Ruff check for touched files.
- basedpyright error-level for `src/dbt_osmosis/workbench/app.py`.
- Semgrep security-audit rerun or focused confirmation.

## Progress and Notes

- 2026-07-05: Exhaustive optimizer pass found one Semgrep security-audit result: dynamic `urllib.request.urlopen` at `src/dbt_osmosis/workbench/app.py:156`.
- 2026-07-05: Replaced `urllib.request.urlopen` with explicit `http.client.HTTPConnection`/`HTTPSConnection` after parsing and rejecting non-http(s) feed URLs.
- 2026-07-05: Added focused tests for non-http scheme rejection, oversized response handling, HTTP error connection closure, query paths, and root-path fetches.
- 2026-07-05: Verification recorded in `.10x/evidence/2026-07-05-workbench-feed-fetcher-hardening.md`: Ruff passed, basedpyright passed, focused workbench tests passed, Semgrep security-audit dropped to 0 findings, Semgrep default dropped from 33 to 32 findings, and CodeQL remained 0 findings.
- 2026-07-05: Closure review recorded in `.10x/reviews/2026-07-05-workbench-feed-fetcher-hardening-review.md` with verdict pass.

## Blockers

- None.

## References

- `/tmp/dbt-osmosis-ai-quality/exhaustive-2026-07-05/semgrep-security.json`
- `.10x/evidence/2026-07-05-workbench-feed-fetcher-hardening.md`
- `.10x/reviews/2026-07-05-workbench-feed-fetcher-hardening-review.md`
- `src/dbt_osmosis/workbench/app.py`
- `tests/core/test_workbench_app.py`
