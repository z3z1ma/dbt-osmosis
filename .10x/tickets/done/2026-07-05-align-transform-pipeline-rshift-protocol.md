Status: done
Created: 2026-07-05
Updated: 2026-07-05
Parent: None
Depends-On: None

# Align transform pipeline rshift protocol

## Scope

Address CodeQL `py/unexpected-raise-in-special-method` in `TransformPipeline.__rshift__` by following Python operator protocol for unsupported operands.

In scope:

- Return `NotImplemented` for unsupported `>>` operands instead of raising `ValueError` inside `__rshift__`.
- Type the method return accordingly.
- Add focused coverage for unsupported pipeline operands.
- Verify focused pipeline/transform tests, Ruff, pyright for core, and CodeQL rule delta.

Out of scope:

- Changing valid transform chaining behavior.
- Redesigning `TransformOperation` or `TransformPipeline`.
- Addressing cyclic import or ineffectual statement findings.

## Acceptance Criteria

- ACC-001: Valid transform chaining with `>>` remains unchanged.
- ACC-002: Unsupported right-hand operands follow Python operator protocol and raise the standard unsupported-operand `TypeError`.
- ACC-003: Focused pipeline/transform tests pass.
- ACC-004: Ruff passes for touched code and basedpyright reports no errors for `src/dbt_osmosis/core/transforms.py`.
- ACC-005: CodeQL no longer reports `py/unexpected-raise-in-special-method`.

## Closure Evidence

- ACC-001: Recorded in `.10x/evidence/2026-07-05-transform-pipeline-rshift-protocol.md`; focused pipeline tests passed.
- ACC-002: Recorded in `.10x/evidence/2026-07-05-transform-pipeline-rshift-protocol.md`; a focused test now asserts unsupported operands raise Python's standard operator `TypeError`.
- ACC-003: Recorded in `.10x/evidence/2026-07-05-transform-pipeline-rshift-protocol.md`; `tests/core/test_pipeline_integration.py` passed with 18 tests.
- ACC-004: Recorded in `.10x/evidence/2026-07-05-transform-pipeline-rshift-protocol.md`; Ruff passed and basedpyright `--level error` reported zero errors.
- ACC-005: Recorded in `.10x/evidence/2026-07-05-transform-pipeline-rshift-protocol.md`; full CodeQL security-and-quality scan dropped from 54 to 53 results and has zero `py/unexpected-raise-in-special-method` results.

## Evidence Expectations

- Capture before CodeQL finding from `/tmp/dbt-osmosis-ai-quality/continuation/codeql-results-mixed-return-after.sarif`.
- Capture focused pytest output.
- Capture Ruff and pyright output.
- Capture CodeQL after-check or SARIF parse proving the targeted finding cleared.

## Progress and Notes

- 2026-07-05: CodeQL after explicit return cleanup reported one `py/unexpected-raise-in-special-method` finding in `src/dbt_osmosis/core/transforms.py`.
- 2026-07-05: Source inspection found `TransformPipeline.__rshift__` raises `ValueError` for unsupported operands. Returning `NotImplemented` lets Python raise the standard operator `TypeError`.
- 2026-07-05: Updated `TransformPipeline.__rshift__` to return `NotImplemented` for unsupported operands and added a focused test for the resulting operator `TypeError`.
- 2026-07-05: Verified focused pipeline tests, Ruff, basedpyright error-level, and full CodeQL after-scan.
- 2026-07-05: basedpyright warning-level still exits non-zero for existing `transforms.py` warnings; no errors were reported.
- 2026-07-05: Closure review passed with no blocking findings.

## Blockers

None known.

## References

- `/tmp/dbt-osmosis-ai-quality/continuation/codeql-results-mixed-return-after.sarif`
- `/tmp/dbt-osmosis-ai-quality/continuation/codeql-results-rshift-after.sarif`
- `.10x/evidence/2026-07-05-transform-pipeline-rshift-protocol.md`
- `.10x/reviews/2026-07-05-transform-pipeline-rshift-protocol-review.md`
- `src/dbt_osmosis/core/transforms.py`
- `tests/core/test_pipeline_integration.py`
