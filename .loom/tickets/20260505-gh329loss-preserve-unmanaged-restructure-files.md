---
id: ticket:gh329loss
kind: ticket
status: closed
change_class: code-behavior
risk_class: high
created_at: 2026-05-05T06:02:19Z
updated_at: 2026-05-05T07:58:33Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  initiative:
    - initiative:issue-pr-zero
  evidence:
    - evidence:gh329loss-restructure-preservation-validation
  critique:
    - critique:gh329loss-restructure-preservation-review
  packets:
    - packet:ralph-ticket-gh329loss-20260505T062916Z
    - packet:ralph-ticket-gh329loss-20260505T064213Z
external_refs:
  github_issue: https://github.com/z3z1ma/dbt-osmosis/issues/329
  related_github_issue: https://github.com/z3z1ma/dbt-osmosis/issues/306
depends_on: []
---

# Summary

Prevent `yaml refactor` restructure cleanup from deleting schema YAML files that still contain unmanaged top-level content such as `databricks-tags` or `semantic_models` after the last managed model/source/seed is moved out.

# Context

Issue #329 reports loss of Databricks/custom metadata, and issue #306 reports `semantic_model` definitions being removed. Current main preserves unmanaged top-level sections during same-path read/write cycles and has tests for `semantic_models` and arbitrary unknown top-level keys. Oracle triage found a remaining restructure edge: `apply_restructure_plan()` reads superseded files through filtered YAML, removes managed entries, then may unlink the old file based only on filtered managed content. That can lose root-level unmanaged sections when refactor moves the last managed entry out of a file.

# Why Now

Silent YAML data loss is more harmful than visible command failure. The previous hardening initiative improved same-path preservation; this ticket closes the remaining file-deletion path before closing user-reported preservation issues.

# Scope

- Make superseded-file cleanup consider original unmanaged top-level sections before deleting a schema YAML file.
- Preserve root-level unmanaged sections such as `semantic_models`, `macros`, `databricks-tags`, and unknown future dbt sections when managed entries are restructured away.
- Keep safe deletion behavior for truly empty superseded files with no unmanaged content.
- Add regression tests for restructure/refactor movement, not only same-path writer merges.

# Out Of Scope

- Databricks-specific tag generation or validation.
- Configurable model-level preserve lists for keys inside a managed model entry.
- Metadata inheritance filtering, which is owned by ticket:gh333meta.
- Rewriting the parser/writer preservation model beyond the superseded-file cleanup edge.

# Acceptance Criteria

- ACC-001: When refactor moves the last managed model/source/seed out of a schema file that contains root `semantic_models`, the old file is not deleted or the unmanaged section is otherwise preserved without loss.
- ACC-002: The same preservation holds for an arbitrary root key such as `databricks-tags`.
- ACC-003: Superseded schema files with no managed entries and no unmanaged top-level content are still removed when appropriate.
- ACC-004: Existing same-path read/write preservation tests for unmanaged top-level sections still pass.
- ACC-005: Dry-run/check behavior reports truthful mutations without deleting or stale-caching unmanaged sections.

# Coverage

Covers:

- ticket:gh329loss#ACC-001
- ticket:gh329loss#ACC-002
- ticket:gh329loss#ACC-003
- ticket:gh329loss#ACC-004
- ticket:gh329loss#ACC-005
- initiative:issue-pr-zero#OBJ-001
- initiative:issue-pr-zero#OBJ-002
- initiative:issue-pr-zero#OBJ-005

# Claim Matrix

| Claim | Evidence | Critique | Status |
| --- | --- | --- | --- |
| ticket:gh329loss#ACC-001 | evidence:gh329loss-restructure-preservation-validation | critique:gh329loss-restructure-preservation-review | accepted |
| ticket:gh329loss#ACC-002 | evidence:gh329loss-restructure-preservation-validation | critique:gh329loss-restructure-preservation-review | accepted |
| ticket:gh329loss#ACC-003 | evidence:gh329loss-restructure-preservation-validation | critique:gh329loss-restructure-preservation-review | accepted |
| ticket:gh329loss#ACC-004 | evidence:gh329loss-restructure-preservation-validation | critique:gh329loss-restructure-preservation-review | accepted |
| ticket:gh329loss#ACC-005 | evidence:gh329loss-restructure-preservation-validation | critique:gh329loss-restructure-preservation-review | accepted |

# Execution Notes

Ralph iteration 1 fixed the primary superseded-file deletion path and added semantic-model/same-path regressions. Oracle critique found missing-original-cache and dry-run/cache coverage gaps. Ralph iteration 2 added missing-original-cache fallback handling and dry-run/cache regressions. Parent validation and mandatory Oracle critique now support all acceptance criteria locally.

# Blockers

None.

# Evidence

Evidence status: local red/green Ralph evidence, parent focused pytest, Ruff, and whitespace checks support ACC-001 through ACC-005 for the uncommitted implementation diff. Remote CI will be checked at the initiative level after the full batch push per operator direction.

# Critique Disposition

Risk class: high

Critique policy: mandatory

Policy rationale: This ticket addresses a silent data-loss path in schema YAML rewriting.

Required critique profiles:

- yaml-data-loss
- cache-and-dry-run

Findings:

None - critique:gh329loss-restructure-preservation-review returned `pass` with no findings.

Disposition status: completed

Deferral / not-required rationale: N/A.

# Retrospective / Promotion Disposition

Disposition status: completed

Promoted:

None - retrospective found no durable explanation needing wiki/research/spec promotion beyond the ticket, evidence, and critique records.

Deferred / not-required rationale: No broader behavior contract changed; this ticket adds targeted data-loss regression coverage and keeps the existing YAML preservation architecture intact.

# Wiki Disposition

N/A - no wiki promotion selected yet.

# Acceptance Decision

Accepted by: OpenCode parent agent.
Accepted at: 2026-05-05T07:58:33Z.
Basis: Local implementation evidence, focused validation, and mandatory Oracle critique support ACC-001 through ACC-005. The ticket is ready for issue closure with remote CI deferred to the issue-backlog initiative gate per operator direction.
Residual risks: Remote CI not yet checked; empty unmanaged top-level sections may conservatively preserve a superseded file, which is accepted as fail-closed behavior.

# Dependencies

None.

# Journal

- 2026-05-05T06:02:19Z: Created from GitHub issues #329 and #306 after Oracle identified a remaining restructure deletion edge despite current same-path preservation fixes.
- 2026-05-05T06:51:54Z: Ralph iterations 1 and 2 implemented and validated the restructure preservation fix. Mandatory Oracle critique passed with no findings. Retrospective completed with no promotion needed beyond the ticket/evidence/critique records. Moved to complete_pending_acceptance pending final implementation commit packaging.
- 2026-05-05T07:58:33Z: Accepted and closed locally for per-issue packaging. GitHub issues #329 and #306 are ready for commit, push, comment, and closure.
