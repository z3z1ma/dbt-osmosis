---
id: ticket:c10dry17
kind: ticket
status: ready
change_class: code-behavior
risk_class: medium
created_at: 2026-05-03T21:10:43Z
updated_at: 2026-05-03T21:10:43Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  initiative:
    - initiative:dbt-110-111-hardening
  evidence:
    - evidence:oracle-backlog-scan
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
| ticket:c10dry17#ACC-001 | evidence:oracle-backlog-scan | None | open |
| ticket:c10dry17#ACC-004 | evidence:oracle-backlog-scan | None | open |

# Execution Notes

Keep the production cache locking model intact. Avoid making tests pass by adding more monkeypatching around cache globals.

# Blockers

None.

# Evidence

Existing evidence: evidence:oracle-backlog-scan. Missing evidence: dry-run regression test output.

# Critique Disposition

Risk class: medium

Critique policy: recommended

Policy rationale: Cache behavior is subtle and can create cross-test/process contamination.

Required critique profiles: code-change, test-coverage

Findings: None - no critique yet.

Disposition status: pending

Deferral / not-required rationale: None.

# Retrospective / Promotion Disposition

Disposition status: pending

Promoted: None - implementation not complete.

Deferred / not-required rationale: Consider wiki/testing note if cache reset becomes canonical.

# Wiki Disposition

N/A - no wiki promotion selected yet.

# Acceptance Decision

Accepted by: Not accepted yet.
Accepted at: N/A.
Basis: Pending tests.
Residual risks: Long-lived CLI/workbench processes may expose additional cache assumptions.

# Dependencies

Coordinate with ticket:c10race13 if grouping changes cache mutation flow.

# Journal

- 2026-05-03T21:10:43Z: Created from core architecture and tests/fixtures oracle findings.
