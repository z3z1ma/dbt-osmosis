Status: done
Created: 2026-07-05
Updated: 2026-07-05
Parent: None
Depends-On: None

# Make Deptry zero

## Scope

Make `uv run --no-sync --with deptry deptry .` exit zero by correcting dependency metadata and encoding repository intent in Deptry configuration.

In scope:

- Mark `dbt_osmosis` as first-party for Deptry.
- Add direct runtime dependencies for packages imported by source code.
- Add direct optional workbench dependency metadata for `pandas`.
- Treat the `dev` optional dependency group as development dependencies.
- Keep intentionally retained workbench/dev tooling dependencies with narrow Deptry `DEP002` ignores.

Out of scope:

- Removing public optional extras that are covered by package smoke tests.
- Removing dev tooling dependencies used by workflows, pre-commit, tests, or local developer tasks.
- Changing runtime code to satisfy dependency metadata diagnostics.

## Acceptance Criteria

- ACC-001: `uv run --no-sync --with deptry deptry .` exits 0.
- ACC-002: Package metadata tests pass after direct dependency changes.
- ACC-003: Ruff remains clean for changed files.

## Evidence Expectations

- Record Deptry, package metadata test, and Ruff results.

## Progress and Notes

- 2026-07-05: Initial Deptry run reported 223 issues, dominated by first-party misclassification, dev-tool unused dependency findings, and transitive dependency imports.
- 2026-07-05: Added direct metadata for `agate`, `dbt-common`, `packaging`, `typing-extensions`, and workbench `pandas`.
- 2026-07-05: Added `[tool.deptry]` config for first-party modules, module-name mappings, dev optional dependency group, and narrow `DEP002` ignores for retained dev/workbench tooling dependencies.
- 2026-07-05: Refreshed `uv.lock` after metadata changes.

## Blockers

None.

## References

- `.10x/evidence/2026-07-05-deptry-zero.md`
- `.10x/reviews/2026-07-05-deptry-zero.md`
