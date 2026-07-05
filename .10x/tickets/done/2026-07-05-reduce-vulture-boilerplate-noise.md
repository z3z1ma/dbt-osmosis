Status: done
Created: 2026-07-05
Updated: 2026-07-05
Parent: None
Depends-On: None

# Reduce Vulture boilerplate noise

## Scope

Reduce source-only Vulture findings that are caused by local protocol stubs or context-manager boilerplate, without changing runtime behavior or public compatibility surfaces.

In scope:

- Replace protocol abstract bodies with stub ellipses where the method exists only for static shape.
- Rename unused context-manager exception parameters to underscore-prefixed names where they are intentionally ignored.
- Mark unused external callback metadata as intentional without changing callback signatures.
- Re-run Vulture and focused quality checks that cover the touched source.

Out of scope:

- Removing public compatibility re-exports from `src/dbt_osmosis/core/osmosis.py`.
- Changing optional SQL proxy callback signatures that may be owned by `mysql_mimic`.
- Adding Vulture suppressions, allowlists, or baselines.

## Acceptance Criteria

- ACC-001: Source-only Vulture findings decrease for safe boilerplate cases.
- ACC-002: Ruff, pydoclint, basedpyright, and focused tests for the touched surfaces pass.
- ACC-003: Any remaining Vulture findings are classified as compatibility/API-owned rather than silently changed.

## Evidence Expectations

- Record the before/after Vulture count or finding classes.
- Record exact verification commands and exit codes.

## Progress and Notes

- 2026-07-05: Initial source-only Vulture pass reported 13 findings, mostly protocol/context-manager parameters plus compatibility/API-owned symbols.
- 2026-07-05: High-confidence source-only Vulture pass before cleanup reported six findings: protocol keyword parameters, logger protocol parameters, and SQL proxy callback metadata.
- 2026-07-05: Preserved protocol/callback signatures while marking unused parameters as intentional and renaming ignored context-manager exception arguments.
- 2026-07-05: High-confidence source-only Vulture pass now exits 0 with no output.

## Blockers

None.

## References

- `.10x/evidence/2026-07-05-quality-optimizer-final-vector.md`
- `.10x/evidence/2026-07-05-vulture-boilerplate-cleanup.md`
- `.10x/reviews/2026-07-05-vulture-boilerplate-cleanup.md`
