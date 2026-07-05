Status: done
Created: 2026-07-05
Updated: 2026-07-05
Parent: None
Depends-On: None

# Clarify protocol contract stubs

## Scope

Address remaining CodeQL `py/ineffectual-statement` findings caused by ellipsis-only bodies in Protocol and contract stubs.

In scope:

- Replace Protocol or overload stub bodies in `src/dbt_osmosis/core/dbt_protocols.py`, `src/dbt_osmosis/core/introspection.py`, `src/dbt_osmosis/core/logger.py`, and `src/dbt_osmosis/workbench/components/dashboard.py` with explicit non-implementation bodies that type check.
- Replace equivalent contract stub bodies in `specs/001-unified-config-resolution/contracts/config-resolver.py`.
- Verify Ruff, basedpyright error-level, focused tests, and CodeQL rule delta.

Out of scope:

- Changing protocol member signatures.
- Changing concrete runtime behavior outside fallback bodies that should not be called directly.
- Addressing CodeQL `py/cyclic-import` findings.
- Refactoring the unified configuration contract.

## Acceptance Criteria

- ACC-001: CodeQL no longer reports `py/ineffectual-statement` for the scoped stub-body locations.
- ACC-002: Protocol and contract signatures remain unchanged.
- ACC-003: Ruff passes for touched source and contract files.
- ACC-004: basedpyright reports no errors for touched source files.
- ACC-005: Focused logger, introspection/config, and workbench tests pass.

## Evidence Expectations

- Capture before CodeQL findings from `/tmp/dbt-osmosis-ai-quality/continuation/codeql-results-map-drain-after.sarif`.
- Capture focused pytest output.
- Capture Ruff and basedpyright output.
- Capture CodeQL after-check or SARIF parse proving the targeted findings cleared.

## Progress and Notes

- 2026-07-05: CodeQL after map-drain cleanup reported 35 `py/ineffectual-statement` findings, all in Protocol, overload, or specification contract stub bodies.
- 2026-07-05: A temporary basedpyright check showed plain `pass` in non-`None` Protocol methods causes `reportReturnType` errors; explicit non-implementation bodies are required.
- 2026-07-05: Replaced ordinary stub bodies with `raise NotImplementedError`; used abstract `pass` bodies for special Protocol methods `__call__` and `__bool__`.
- 2026-07-05: Ruff check and Ruff format check passed for all touched files.
- 2026-07-05: basedpyright error-level passed for all touched source files.
- 2026-07-05: Focused logger, introspection/config, property accessor, and workbench tests passed: 154 passed, 3 skipped, 2 warnings.
- 2026-07-05: CodeQL after-check reported 7 total findings, all `py/cyclic-import`; `py/ineffectual-statement` is now zero.

## Blockers

None known.

## References

- `/tmp/dbt-osmosis-ai-quality/continuation/codeql-results-map-drain-after.sarif`
- `/tmp/dbt-osmosis-ai-quality/continuation/codeql-results-stub-bodies-after.sarif`
- `.10x/evidence/2026-07-05-protocol-contract-stub-clarity.md`
- `.10x/reviews/2026-07-05-protocol-contract-stub-clarity-review.md`
- `src/dbt_osmosis/core/dbt_protocols.py`
- `src/dbt_osmosis/core/introspection.py`
- `src/dbt_osmosis/core/logger.py`
- `src/dbt_osmosis/workbench/components/dashboard.py`
- `specs/001-unified-config-resolution/contracts/config-resolver.py`
