Status: recorded
Created: 2026-07-05
Updated: 2026-07-05
Target: .10x/tickets/done/2026-07-05-clarify-protocol-contract-stubs.md
Verdict: pass

# Protocol contract stub clarity review

## Target

Review of Protocol, overload, and configuration contract stub-body cleanup across core, workbench, and spec contract files.

## Assumptions Tested

- The scoped bodies are stubs and not concrete execution paths used by normal runtime code.
- Function/property signatures are the compatibility contract; this patch must not change them.
- Plain `pass` is not safe for non-`None` Protocol methods because basedpyright reports missing returns.
- Raising `NotImplementedError` is acceptable for ordinary stubs, while special methods should avoid explicit raise bodies.
- CodeQL should clear `py/ineffectual-statement` without introducing a new `py/unexpected-raise-in-special-method` result.

## Findings

No blocking findings.

## Verdict

Pass. The patch preserves signatures, keeps type checking clean, clears all remaining CodeQL `py/ineffectual-statement` findings, and leaves only pre-existing cyclic-import findings.

## Residual Risk

Explicit subclasses of these Protocol classes that relied on inherited ellipsis-returning fallback methods would now see `NotImplementedError` or abstract special-method requirements. The inspected usage treats these classes as structural type contracts, so that risk is low and better aligned with the intended non-implementation semantics.
