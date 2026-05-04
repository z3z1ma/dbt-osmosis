---
id: ticket:c10race13
kind: ticket
status: complete_pending_acceptance
change_class: code-behavior
risk_class: high
created_at: 2026-05-03T21:10:43Z
updated_at: 2026-05-04T07:51:02Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  initiative:
    - initiative:dbt-110-111-hardening
  evidence:
    - evidence:oracle-backlog-scan
    - evidence:c10race13-yaml-sync-serialization-verification
  critique:
    - critique:c10race13-yaml-sync-serialization-review
  packets:
    - packet:ralph-ticket-c10race13-20260504T071043Z
depends_on: []
---

# Summary

Serialize YAML sync work by resolved target YAML path so multiple nodes cannot concurrently mutate/write the same ruamel document or collide on the same temp file.

# Context

`src/dbt_osmosis/core/sync_operations.py:664-688` groups only versioned models, while `sync_node_to_yaml()` maps groups through `context.pool.map()` at `sync_operations.py:728-731`. Non-versioned models or sources sharing one schema file can run concurrently. `schema/reader.py` returns shared mutable cached YAML documents, and `schema/writer.py:131` and `238` use fixed `.tmp` paths per target file.

# Why Now

YAML preservation is core project policy. A race that can lose entries or corrupt writes is a high-risk correctness bug, especially when CI increases parallel compatibility coverage.

# Scope

- Resolve each candidate node's current/target YAML path before grouping.
- Group all sync work by YAML target path, not only by versioned model key.
- Update each YAML document once per work item, with no concurrent writers for the same path.
- Ensure temp file strategy cannot collide if multiple writes to different logical paths share suffixes.
- Add regression tests using `threads >= 2` and multiple nodes routed to the same YAML file.

# Out Of Scope

- Rewriting the whole transform pipeline.
- Removing concurrency for independent YAML files.

# Acceptance Criteria

- ACC-001: No two worker threads can mutate/write the same YAML document concurrently during sync.
- ACC-002: Versioned and unversioned models sharing a YAML file are grouped correctly.
- ACC-003: Sources sharing a YAML file are grouped correctly.
- ACC-004: Temp file writes cannot collide for the same target path.
- ACC-005: Regression tests repeatedly sync two or more nodes to the same YAML file with multiple threads and assert all sections survive.
- ACC-006: Existing versioned model grouping behavior remains correct.

# Coverage

Covers:

- ticket:c10race13#ACC-001
- ticket:c10race13#ACC-002
- ticket:c10race13#ACC-003
- ticket:c10race13#ACC-004
- ticket:c10race13#ACC-005
- ticket:c10race13#ACC-006
- initiative:dbt-110-111-hardening#OBJ-003

# Claim Matrix

| Claim | Evidence | Critique | Status |
| --- | --- | --- | --- |
| ticket:c10race13#ACC-001 | evidence:c10race13-yaml-sync-serialization-verification | critique:c10race13-yaml-sync-serialization-review | locally supported; post-commit CI pending |
| ticket:c10race13#ACC-002 | evidence:c10race13-yaml-sync-serialization-verification | critique:c10race13-yaml-sync-serialization-review | locally supported; post-commit CI pending |
| ticket:c10race13#ACC-003 | evidence:c10race13-yaml-sync-serialization-verification | critique:c10race13-yaml-sync-serialization-review | locally supported; post-commit CI pending |
| ticket:c10race13#ACC-004 | evidence:c10race13-yaml-sync-serialization-verification | critique:c10race13-yaml-sync-serialization-review | locally supported; post-commit CI pending |
| ticket:c10race13#ACC-005 | evidence:c10race13-yaml-sync-serialization-verification | critique:c10race13-yaml-sync-serialization-review | locally supported; post-commit CI pending |
| ticket:c10race13#ACC-006 | evidence:c10race13-yaml-sync-serialization-verification | critique:c10race13-yaml-sync-serialization-review | locally supported; post-commit CI pending |

# Execution Notes

Start with grouping semantics before changing cache internals. If grouping by path makes shared mutable cache safe enough, avoid over-engineering locks around every YAML operation.

# Blockers

None.

# Evidence

Existing evidence:

- evidence:oracle-backlog-scan
- evidence:c10race13-yaml-sync-serialization-verification

Evidence disposition: locally sufficient for implementation commit. The local evidence records test-first red checks, path-grouped sync scheduling, same-target repeated threaded survival, unique temp-file collision safety, existing file-mode preservation, related sync/schema/path tests, targeted hooks, and final review.

Missing evidence: post-commit `Tests` and `lint` evidence for the implementation commit before final acceptance/closure.

# Critique Disposition

Risk class: high

Critique policy: mandatory

Policy rationale: Concurrent YAML writes can silently lose user-authored schema content.

Required critique profiles: code-change, test-coverage, data-preservation

Findings: No open medium/high findings in critique:c10race13-yaml-sync-serialization-review. Low residual risks are recorded in the critique and accepted as non-blocking for implementation commit.

Disposition status: completed locally; post-commit CI still pending before closure

Deferral / not-required rationale: None.

# Retrospective / Promotion Disposition

Disposition status: pending

Promoted: None - implementation not complete.

Deferred / not-required rationale: Consider wiki/YAML pipeline note after accepted fix.

# Wiki Disposition

N/A - no wiki promotion selected yet.

# Acceptance Decision

Accepted by: Not accepted yet.
Accepted at: N/A.
Basis: Local implementation evidence and mandatory critique are sufficient for implementation commit. Final acceptance is pending post-commit CI and retrospective / promotion follow-through.
Residual risks: Direct concurrent single-node calls and cross-process final writes can still race; source same-path coverage is grouping-level with source-shaped nodes because the demo fixture has no parsed sources; release/CI evidence is not recorded yet for this source change.

# Dependencies

Coordinate with ticket:c10dry17 if cache behavior changes.

# Journal

- 2026-05-03T21:10:43Z: Created from core architecture oracle finding.
- 2026-05-04T07:10:43Z: Activated ticket and compiled Ralph packet `packet:ralph-ticket-c10race13-20260504T071043Z` for test-first path-grouped YAML sync serialization and temp-file safety.
- 2026-05-04T07:51:02Z: Consumed Ralph output, applied parent review-driven fixes for source/mixed grouping coverage, repeated threaded survival, file mode preservation, deterministic ordering, and umask-free temp creation, recorded local evidence `evidence:c10race13-yaml-sync-serialization-verification`, completed mandatory critique `critique:c10race13-yaml-sync-serialization-review`, and moved ticket to `complete_pending_acceptance` pending implementation commit, post-commit CI, and retrospective / promotion follow-through.
