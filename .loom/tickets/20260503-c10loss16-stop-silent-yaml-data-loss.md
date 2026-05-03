---
id: ticket:c10loss16
kind: ticket
status: ready
change_class: code-behavior
risk_class: high
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

Stop YAML sync from silently deleting duplicate model/version entries or overwriting generated YAML files without an explicit preservation/merge/overwrite policy.

# Context

`src/dbt_osmosis/core/sync_operations.py:470-490` keeps the first duplicate model entry and pops later entries. `src/dbt_osmosis/core/sync_operations.py:505-519` collapses duplicate version entries by assigning to a dict and replacing the versions list. Separate generate/NL paths directly write files, but ticket:c10gen20 owns that broader safe writer migration.

# Why Now

The constitution requires preserving dbt project intent and schema YAML content. Silent deletion of YAML sections is a high-risk data loss bug.

# Scope

- Decide whether duplicate model/version entries should fail validation or merge safely.
- Ensure distinct tests, meta, comments, descriptions, columns, and version blocks are not silently discarded.
- Make duplicate handling visible to users through clear validation errors or safe merge logs.
- Add tests for duplicate model entries and duplicate versions containing distinct content.
- Coordinate with validation so malformed duplicates are caught before destructive sync when possible.

# Out Of Scope

- Migrating generate/NL writes through schema helpers; ticket:c10gen20 owns that.
- Supporting every possible hand-authored duplicate shape if the chosen policy is to fail closed.

# Acceptance Criteria

- ACC-001: Duplicate model entries are not silently deleted during sync.
- ACC-002: Duplicate version entries are not silently deleted during sync.
- ACC-003: If duplicates are invalid, the command fails with an actionable validation error before writing.
- ACC-004: If duplicates are merged, distinct user-authored content is preserved or conflicts are surfaced.
- ACC-005: Regression tests cover duplicate models and duplicate versions with distinct tests/meta/comments.
- ACC-006: The behavior is documented in validation errors or command output well enough for users to fix YAML.

# Coverage

Covers:

- ticket:c10loss16#ACC-001
- ticket:c10loss16#ACC-002
- ticket:c10loss16#ACC-003
- ticket:c10loss16#ACC-004
- ticket:c10loss16#ACC-005
- ticket:c10loss16#ACC-006
- initiative:dbt-110-111-hardening#OBJ-003

# Claim Matrix

| Claim | Evidence | Critique | Status |
| --- | --- | --- | --- |
| ticket:c10loss16#ACC-001 | evidence:oracle-backlog-scan | None | open |
| ticket:c10loss16#ACC-005 | None - duplicate regression tests not written yet | None | open |

# Execution Notes

Prefer fail-closed validation unless there is a simple, demonstrably safe merge. Silent auto-merge of conflicting YAML can be as dangerous as deletion.

# Blockers

Potential behavior decision: fail versus merge duplicates. If unclear after inspection, ask the user or route to a small spec.

# Evidence

Existing evidence: evidence:oracle-backlog-scan. Missing evidence: regression tests and command output.

# Critique Disposition

Risk class: high

Critique policy: mandatory

Policy rationale: This ticket exists to prevent user-authored YAML data loss.

Required critique profiles: code-change, data-preservation, test-coverage

Findings: None - no critique yet.

Disposition status: pending

Deferral / not-required rationale: None.

# Retrospective / Promotion Disposition

Disposition status: pending

Promoted: None - implementation not complete.

Deferred / not-required rationale: A wiki troubleshooting note may be useful if validation behavior changes.

# Wiki Disposition

N/A - no wiki promotion selected yet.

# Acceptance Decision

Accepted by: Not accepted yet.
Accepted at: N/A.
Basis: Pending tests and critique.
Residual risks: Some duplicate YAML may already exist in user projects and need migration guidance.

# Dependencies

Coordinate with ticket:c10ver15 and ticket:c10gen20.

# Journal

- 2026-05-03T21:10:43Z: Created from core architecture oracle finding.
