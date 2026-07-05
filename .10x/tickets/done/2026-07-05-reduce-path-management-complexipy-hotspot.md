Status: done
Created: 2026-07-05
Updated: 2026-07-05
Parent: None
Depends-On: None

# Reduce path management Complexipy hotspot

## Scope

Reduce `complexipy` cognitive complexity in `src/dbt_osmosis/core/path_management.py`, starting with `create_missing_source_yamls`, without changing YAML routing behavior, project-root safety checks, or schema read/write helpers.

In scope:

- Extract cohesive helper functions from `create_missing_source_yamls`.
- Preserve existing source YAML creation and existing-source table update behavior.
- Verify with Complexipy, Ruff, ty, mypy, and focused tests.

Out of scope:

- Changing dbt source discovery semantics.
- Bypassing schema reader/writer helpers.
- Adding Complexipy baselines, snapshots, ratchets, or higher thresholds.

## Acceptance Criteria

- ACC-001: `create_missing_source_yamls` complexity decreases materially from the observed score of 84.
- ACC-002: Focused `path_management` tests pass.
- ACC-003: Ruff, ty, and mypy remain clean for the configured production surface.

## Evidence Expectations

- Record before/after Complexipy scores for `src/dbt_osmosis/core/path_management.py`.
- Record exact verification commands and exit codes.

## Progress and Notes

- 2026-07-05: Complexipy reported `create_missing_source_yamls` at 84, the worst current hotspot.
- 2026-07-05: User explicitly rejected ratchet/baseline/threshold shortcuts; this ticket must reduce code complexity directly.
- 2026-07-05: Split vars routing, YAML-template fallback resolution, source specification parsing, source YAML path validation, source relation description, existing-source update, missing-source write, and source-definition processing into focused helpers.
- 2026-07-05: Direct file-level Complexipy check now exits zero for `src/dbt_osmosis/core/path_management.py`.
- 2026-07-05: Focused routing/security/path-management tests pass.

## Blockers

None.

## References

- `/tmp/dbt-osmosis-ai-quality/exhaustive-2026-07-05/final-post-vulture/complexipy-source-failed-plain.txt`
- `.10x/evidence/2026-07-05-path-management-complexipy.md`
- `.10x/reviews/2026-07-05-path-management-complexipy.md`
