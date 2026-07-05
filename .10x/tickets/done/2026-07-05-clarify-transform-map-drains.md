Status: done
Created: 2026-07-05
Updated: 2026-07-05
Parent: None
Depends-On: None

# Clarify transform map drains

## Scope

Address production CodeQL `py/ineffectual-statement` findings caused by ellipsis loop bodies that drain `ThreadPoolExecutor.map` results.

In scope:

- Replace ellipsis-only loop bodies in `src/dbt_osmosis/core/transforms.py` with `pass`.
- Replace the equivalent ellipsis-only loop body in `src/dbt_osmosis/core/sync_operations.py` with `pass`.
- Verify focused transform pipeline and sync operation tests.
- Verify Ruff, basedpyright error-level for touched core files, and CodeQL rule delta.

Out of scope:

- Changing map execution, concurrency behavior, or candidate node iteration.
- Refactoring transform all-node execution.
- Addressing protocol/spec ellipsis bodies or cyclic imports.

## Acceptance Criteria

- ACC-001: All included map-drain loops still consume the mapped iterator.
- ACC-002: CodeQL no longer reports `py/ineffectual-statement` for the included transform/sync map-drain bodies.
- ACC-003: Focused transform/sync tests pass.
- ACC-004: Ruff passes and basedpyright reports no errors for touched core files.

## Evidence Expectations

- Capture before CodeQL findings from `/tmp/dbt-osmosis-ai-quality/continuation/codeql-results-rshift-after.sarif`.
- Capture focused pytest output.
- Capture Ruff and basedpyright output.
- Capture CodeQL after-check or SARIF parse proving the targeted findings cleared.

## Progress and Notes

- 2026-07-05: CodeQL after rshift protocol cleanup reported 46 `py/ineffectual-statement` findings, including ten transform map-drain ellipses and one sync operation map-drain ellipsis.
- 2026-07-05: Source inspection confirmed these loops intentionally consume `context.pool.map(...)`; replacing `...` with `pass` is behavior-preserving and clearer.
- 2026-07-05: Replaced the eleven scoped ellipsis loop bodies with `pass`.
- 2026-07-05: Focused transform/sync tests passed: 76 passed, 2 warnings.
- 2026-07-05: Ruff passed for the two touched source files.
- 2026-07-05: basedpyright error-level passed for the two touched source files.
- 2026-07-05: CodeQL after-check reported 42 total findings: 7 `py/cyclic-import`, 35 `py/ineffectual-statement`, and zero findings in the touched map-drain locations.

## Blockers

None known.

## References

- `/tmp/dbt-osmosis-ai-quality/continuation/codeql-results-rshift-after.sarif`
- `/tmp/dbt-osmosis-ai-quality/continuation/codeql-results-map-drain-after.sarif`
- `.10x/evidence/2026-07-05-transform-map-drains.md`
- `.10x/reviews/2026-07-05-transform-map-drains-review.md`
- `src/dbt_osmosis/core/transforms.py`
- `src/dbt_osmosis/core/sync_operations.py`
