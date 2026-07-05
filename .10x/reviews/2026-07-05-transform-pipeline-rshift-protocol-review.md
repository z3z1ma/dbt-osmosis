Status: recorded
Created: 2026-07-05
Updated: 2026-07-05
Target: .10x/tickets/done/2026-07-05-align-transform-pipeline-rshift-protocol.md
Verdict: pass

# Transform pipeline rshift protocol review

## Target

Review of `TransformPipeline.__rshift__` operator-protocol cleanup in `src/dbt_osmosis/core/transforms.py`.

## Assumptions Tested

- Valid `>>` chaining behavior remains covered by the existing pipeline integration tests.
- Returning `NotImplemented` for unsupported right-hand operands is the correct Python special-method protocol.
- The invalid-use error type changes from custom `ValueError` to standard unsupported-operand `TypeError`; no tests or inspected references depended on the old `ValueError`.
- Type checking should not gain errors from the widened return type.

## Findings

No blocking findings.

## Verdict

Pass. The patch preserves valid transform chaining, makes unsupported operands follow Python operator protocol, adds focused coverage for invalid operands, and clears the targeted CodeQL finding.

## Residual Risk

External callers that intentionally caught the old `ValueError` for invalid `pipeline >> object()` usage would now see Python's standard operator `TypeError`. That risk is limited to invalid operator use and is consistent with the protocol.
