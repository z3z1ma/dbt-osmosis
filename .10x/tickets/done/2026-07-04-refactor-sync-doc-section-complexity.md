Status: done
Created: 2026-07-04
Updated: 2026-07-04
Parent: .10x/tickets/done/2026-07-04-quality-optimizer-hill-climb.md
Depends-On: .10x/tickets/done/2026-07-04-remediate-uv-audit-vulnerabilities.md

# Refactor sync doc section complexity

## Scope

Refactor `src/dbt_osmosis/core/sync_operations.py:_sync_doc_section` to reduce complexity while preserving YAML sync behavior.

In scope:

- Extract cohesive helpers inside `sync_operations.py` only when they remove real branching complexity.
- Preserve existing public behavior for:
  - model/source description syncing,
  - catalog data type precedence,
  - scaffold empty configs,
  - skip add data types,
  - skip merge meta,
  - unrendered description preservation,
  - prefer YAML values,
  - fusion compatibility config/meta/tag handling,
  - output-to-upper/lower casing,
  - removal of empty column sections.
- Remove the two redundant casts in `sync_operations.py` only if doing so is included in the same focused edit and stays behavior-neutral.

Out of scope:

- Changing YAML writer/reader behavior.
- Changing configuration precedence.
- Changing schema formatting policy.
- Dependency updates.
- Broad sync pipeline redesign.

## Acceptance Criteria

- ACC-001: `_sync_doc_section` Radon CC drops materially from baseline 78 without hiding behavior in opaque helpers.
- ACC-002: Complexipy no longer flags `_sync_doc_section`, or any residual complexity is justified by preserved behavior.
- ACC-003: `tests/core/test_sync_operations.py` passes.
- ACC-004: Relevant integration tests that exercise YAML inheritance/sync behavior pass.
- ACC-005: Ruff and basedpyright remain clean.
- ACC-006: No user-visible YAML sync behavior changes except behavior-neutral internal structure.

## Evidence Expectations

- Before/after Radon result for `_sync_doc_section`.
- Before/after Complexipy result for `sync_operations.py`.
- Targeted pytest output for sync/inheritance tests.
- Ruff and basedpyright output.

## Progress and Notes

- 2026-07-04: Baseline Radon CC for `_sync_doc_section` is 78, rank F, lines 21-313. Complexipy also flags it as failed.
- 2026-07-04: Started execution after dependency audit remediation was closed and Semgrep findings were triaged.
- 2026-07-04: Worker refactored `_sync_doc_section` into focused helpers in `src/dbt_osmosis/core/sync_operations.py`.
- 2026-07-04: Parent verification recorded in `.10x/evidence/2026-07-04-sync-doc-section-refactor.md`; review recorded in `.10x/reviews/2026-07-04-sync-doc-section-refactor-review.md`.

## Blockers

Execution should wait until the higher-priority dependency vulnerability ticket is handled or deliberately deferred.

## References

- `.10x/research/2026-07-04-quality-optimizer-baseline.md`
- `.10x/evidence/2026-07-04-quality-optimizer-baseline.md`
- `.10x/evidence/2026-07-04-sync-doc-section-refactor.md`
- `.10x/reviews/2026-07-04-sync-doc-section-refactor-review.md`
