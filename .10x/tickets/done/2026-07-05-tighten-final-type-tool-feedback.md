Status: done
Created: 2026-07-05
Updated: 2026-07-05
Parent: None
Depends-On: None

# Tighten final type tool feedback

## Scope

Address directly actionable `ty` feedback introduced or exposed by the quality optimizer cleanup without attempting broad project-wide `ty` or mypy adoption.

In scope:

- Use `pass` bodies for `_find_first()` overload stubs so overload bodies satisfy `ty` without reintroducing CodeQL findings.
- Add overloads for `TransformPipeline.__rshift__()` so valid transform chains type as `TransformPipeline`, while unsupported operands still return `NotImplemented` for Python operator protocol.
- Keep runtime transform chaining behavior covered by existing pipeline tests.
- Verify Ruff, basedpyright error-level, targeted `ty` checks for the touched source files, focused pipeline/introspection tests, and CodeQL remains clean.

Out of scope:

- Fixing broad project-wide `ty` diagnostics.
- Fixing broad project-wide mypy diagnostics.
- Changing transform execution semantics.
- Changing optional extra dependency policy.

## Acceptance Criteria

- ACC-001: `ty` no longer reports the scoped overload-body or valid `TransformPipeline >> operation` diagnostics in touched files.
- ACC-002: Unsupported pipeline `>> object()` still raises Python's standard unsupported-operand `TypeError`.
- ACC-003: Ruff, basedpyright error-level, focused tests, and CodeQL pass for the touched surface.

## Evidence Expectations

- Capture current project-wide `ty`/mypy limitation as context.
- Capture targeted `ty`, Ruff, basedpyright, focused pytest, and CodeQL output.

## Progress and Notes

- 2026-07-05: Whole-repo `ty check` reports broad pre-existing diagnostics, including optional extras and tests. Two actionable diagnostics map to recent quality cleanup: overload bodies in `_find_first()` and a broad `TransformPipeline.__rshift__()` return type.
- 2026-07-05: Whole-repo mypy reports broad pre-existing diagnostics around optional extras, tests, and current protocol strictness; this ticket does not adopt mypy project-wide.
- 2026-07-05: Replaced `_find_first()` overload bodies with `pass`, which basedpyright and `ty` accept without reintroducing CodeQL findings.
- 2026-07-05: Added `TransformPipeline.__rshift__()` overloads so valid transform chains type as `TransformPipeline` while unsupported operands still return `NotImplemented`.
- 2026-07-05: Fixed the all-node `suggest_improved_documentation()` path to map the wrapped function when forwarding `threshold` and `learning_mode` kwargs.
- 2026-07-05: Ruff, basedpyright error-level, focused pipeline/introspection tests, and CodeQL passed.
- 2026-07-05: Targeted `ty` check on `introspection.py`, `transforms.py`, and `cli/main.py` dropped from 7 diagnostics to 4, leaving only pre-existing CLI diagnostics outside this ticket.

## Blockers

None known.

## References

- `/tmp/dbt-osmosis-ai-quality/continuation/codeql-results-final-type-after.sarif`
- `.10x/evidence/2026-07-05-final-type-tool-feedback.md`
- `.10x/reviews/2026-07-05-final-type-tool-feedback-review.md`
- `src/dbt_osmosis/core/introspection.py`
- `src/dbt_osmosis/core/transforms.py`
- `tests/core/test_pipeline_integration.py`
