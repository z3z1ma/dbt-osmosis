Status: recorded
Created: 2026-07-05
Updated: 2026-07-05
Target: .10x/tickets/done/2026-07-05-ratify-type-checker-adoption-scope.md
Verdict: pass

# Type tools zero review

## Target

Type-tool adoption changes for ty and mypy, plus source fixes needed to make the configured production surface clean.

## Findings

None.

## Assumptions tested

- The configured ty/mypy scope matches the existing basedpyright production type surface: core and CLI.
- Optional workbench and SQL proxy imports remain runtime-tested elsewhere rather than forced into the default no-extra type environment.
- Source fixes are behavior-preserving: casts at third-party boundaries, timezone-aware timestamp defaults, and explicit type narrowing.

## Verdict

Pass. The exact default ty and mypy commands now exit zero without using `--exit-zero`, broad code suppressions, or generated baselines.

## Residual risk

Optional-extra type checking could be added later with separate dependency-environment commands, but it is not part of the current zero-exit default gate.
