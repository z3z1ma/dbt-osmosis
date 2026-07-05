Status: recorded
Created: 2026-07-05
Updated: 2026-07-05
Target: .10x/tickets/done/2026-07-05-collapse-unnecessary-callable-wrappers.md
Verdict: pass

# Callable wrapper collapse review

## Target

Review of the direct callable wrapper removals in `src/dbt_osmosis/core/settings.py`, `src/dbt_osmosis/workbench/app.py`, and `tests/core/test_inheritance_behavior.py`.

## Assumptions Tested

- `bool` is a signature-compatible replacement for `lambda v: bool(v)`.
- `run_query` is a signature-compatible replacement for `lambda: run_query()`.
- `object` is a signature-compatible replacement for `lambda: object()` in the test monkeypatch.
- The change should not alter config precedence, hotkey registration, or semantic tag merge behavior.

## Findings

No blocking findings.

## Verdict

Pass. The patch removes the six unnecessary wrapper lambdas, preserves the same call signatures at each call site, and clears the targeted CodeQL rule with focused tests and Ruff passing.

## Residual Risk

Residual CodeQL import-cycle and import-style findings remain, including some in touched files, but they pre-existed this ticket and require separate scope.
