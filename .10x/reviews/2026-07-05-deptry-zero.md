Status: recorded
Created: 2026-07-05
Updated: 2026-07-05
Target: .10x/tickets/done/2026-07-05-make-deptry-zero.md
Verdict: pass

# Deptry zero review

## Target

Dependency metadata and Deptry configuration changes made to clear `deptry .`.

## Findings

None.

## Assumptions tested

- Direct dependencies were added for packages actually imported by source: `agate`, `dbt-common`, `packaging`, and `typing-extensions`.
- Workbench imports `pandas`, so `pandas` belongs in the workbench optional extra.
- `streamlit-ace`, `ipython`, `setuptools`, and dev tools are retained because smoke tests, workbench requirements, or developer workflows still rely on those surfaces even when source code does not import them directly.

## Verdict

Pass. The change clears Deptry without removing intentional extras or adding a broad global rule ignore.

## Residual risk

Future dependency additions should update `[tool.deptry]` only when the dependency is intentionally non-imported tooling or optional-extra surface area.
