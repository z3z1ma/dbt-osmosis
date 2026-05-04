---
id: ticket:c10dry17
kind: ticket
status: closed
change_class: code-behavior
risk_class: medium
created_at: 2026-05-03T21:10:43Z
updated_at: 2026-05-04T12:24:01Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  initiative:
    - initiative:dbt-110-111-hardening
  evidence:
    - evidence:oracle-backlog-scan
    - evidence:c10dry17-dry-run-cache-isolation-validation
  packets:
    - packet:ralph-ticket-c10dry17-20260504T115035Z
  critique:
    - critique:c10dry17-dry-run-cache-isolation-review
  wiki:
    - wiki:yaml-sync-safety
depends_on: []
---

# Summary

Ensure dry-run/check mode cannot leave dirty YAML buffers in global caches, and make tests reset production cache objects instead of replacing them with plain dicts.

# Context

`src/dbt_osmosis/core/schema/writer.py:125-176` and `232-284` detect dry-run changes but do not clear caches in the dry-run branches. `src/dbt_osmosis/core/sync_operations.py:327-333` marks buffers dirty before commit decisions. `tests/conftest.py:279-289` patches `_COLUMN_LIST_CACHE` and `_YAML_BUFFER_CACHE` to `{}`, bypassing production LRU/dirty behavior, and several tests clear only one YAML cache.

# Why Now

Dry-run and `--check` are safety-critical commands. Stale process-global YAML state can make later reads/writes lie, and test cache replacement can hide production bugs needed for dbt 1.10/1.11 confidence.

# Scope

- Make dry-run operate on isolated copies or explicitly discard dirty cache entries after reporting mutations.
- Ensure `_YAML_BUFFER_CACHE` and `_YAML_ORIGINAL_CACHE` are reset together when appropriate.
- Add central test fixtures that clear real cache instances without replacing their types.
- Add regression tests for dry-run followed by real read/write in the same process.

# Out Of Scope

- Redesigning all YAML cache internals.
- Removing cache behavior if a smaller isolation fix is enough.

# Acceptance Criteria

- ACC-001: A dry-run sync does not affect a later read from disk in the same process.
- ACC-002: `--check` still reports mutations accurately after dry-run cache cleanup.
- ACC-003: Tests reset `_YAML_BUFFER_CACHE`, `_YAML_ORIGINAL_CACHE`, and `_COLUMN_LIST_CACHE` together where needed.
- ACC-004: Tests preserve production cache classes unless explicitly testing fallback behavior.
- ACC-005: Regression tests prove preserved sections cannot leak between tests through original-cache state.

# Coverage

Covers:

- ticket:c10dry17#ACC-001
- ticket:c10dry17#ACC-002
- ticket:c10dry17#ACC-003
- ticket:c10dry17#ACC-004
- ticket:c10dry17#ACC-005
- initiative:dbt-110-111-hardening#OBJ-003
- initiative:dbt-110-111-hardening#OBJ-004

# Claim Matrix

| Claim | Evidence | Critique | Status |
| --- | --- | --- | --- |
| ticket:c10dry17#ACC-001 | evidence:c10dry17-dry-run-cache-isolation-validation | critique:c10dry17-dry-run-cache-isolation-review | accepted |
| ticket:c10dry17#ACC-002 | evidence:c10dry17-dry-run-cache-isolation-validation | critique:c10dry17-dry-run-cache-isolation-review | accepted |
| ticket:c10dry17#ACC-003 | evidence:c10dry17-dry-run-cache-isolation-validation | critique:c10dry17-dry-run-cache-isolation-review | accepted |
| ticket:c10dry17#ACC-004 | evidence:c10dry17-dry-run-cache-isolation-validation | critique:c10dry17-dry-run-cache-isolation-review | accepted |
| ticket:c10dry17#ACC-005 | evidence:c10dry17-dry-run-cache-isolation-validation | critique:c10dry17-dry-run-cache-isolation-review | accepted |

# Execution Notes

Keep the production cache locking model intact. Avoid making tests pass by adding more monkeypatching around cache globals.

# Blockers

None.

# Evidence

Evidence recorded:

- evidence:oracle-backlog-scan
- evidence:c10dry17-dry-run-cache-isolation-validation

# Critique Disposition

Risk class: medium

Critique policy: recommended

Policy rationale: Cache behavior is subtle and can create cross-test/process contamination.

Required critique profiles: code-change, test-coverage

Findings: critique:c10dry17-dry-run-cache-isolation-review#FIND-001 is low severity and accepted as a non-blocking residual risk. No medium/high findings.

Disposition status: completed

Deferral / not-required rationale: N/A - critique completed with no blockers.

# Retrospective / Promotion Disposition

Disposition status: completed

Promoted: dry-run cache isolation and test-fixture rules updated in wiki:yaml-sync-safety.

Deferred / not-required rationale: Final initiative-level CI evidence remains deferred to the initiative validation pass.

# Wiki Disposition

wiki:yaml-sync-safety updated with dry-run cache cleanup and cache-fixture guidance.

# Acceptance Decision

Accepted by: OpenCode parent agent.
Accepted at: 2026-05-04T12:24:01Z.
Basis: implementation commit `d9eee85a212485c5d6f2944e52eafc2fad3c345e`, evidence:c10dry17-dry-run-cache-isolation-validation, critique:c10dry17-dry-run-cache-isolation-review, and wiki:yaml-sync-safety promotion.
Residual risks: Full repository suite and GitHub Actions matrix are deferred to final initiative validation; `--check` coverage is helper-level rather than Click integration-level; dry-run sync can still expose transient in-memory state to concurrent same-process readers before finalization discards it; preview-then-apply helper flows should reread YAML after dry-run cleanup before a real write to preserve unmanaged sections.

# Dependencies

Coordinate with ticket:c10race13 if grouping changes cache mutation flow.

# Journal

- 2026-05-03T21:10:43Z: Created from core architecture and tests/fixtures oracle findings.
- 2026-05-04T11:50:35Z: Activated ticket and compiled Ralph packet `packet:ralph-ticket-c10dry17-20260504T115035Z` for test-first dry-run YAML cache isolation and production-cache fixture reset behavior with local-only validation.
- 2026-05-04T12:24:01Z: Ralph iteration consumed. Implementation commit `d9eee85a212485c5d6f2944e52eafc2fad3c345e` isolated dry-run writer/sync cache state, converted cache fixtures to clear production cache instances, and added regression coverage. Local validation passed with `186 passed, 3 skipped`; final critique found no medium/high blockers. Accepted and closed with final initiative-level CI deferred.
