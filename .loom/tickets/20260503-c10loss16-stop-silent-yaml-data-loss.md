---
id: ticket:c10loss16
kind: ticket
status: closed
change_class: code-behavior
risk_class: high
created_at: 2026-05-03T21:10:43Z
updated_at: 2026-05-04T11:48:07Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  initiative:
    - initiative:dbt-110-111-hardening
  evidence:
    - evidence:oracle-backlog-scan
    - evidence:c10loss16-duplicate-yaml-fail-closed-validation
  critique:
    - critique:c10loss16-duplicate-yaml-fail-closed-review
  wiki:
    - wiki:yaml-sync-safety
  packets:
    - packet:ralph-ticket-c10loss16-20260504T112322Z
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
| ticket:c10loss16#ACC-001 | evidence:c10loss16-duplicate-yaml-fail-closed-validation | critique:c10loss16-duplicate-yaml-fail-closed-review | accepted |
| ticket:c10loss16#ACC-002 | evidence:c10loss16-duplicate-yaml-fail-closed-validation | critique:c10loss16-duplicate-yaml-fail-closed-review | accepted |
| ticket:c10loss16#ACC-003 | evidence:c10loss16-duplicate-yaml-fail-closed-validation | critique:c10loss16-duplicate-yaml-fail-closed-review | accepted |
| ticket:c10loss16#ACC-004 | evidence:c10loss16-duplicate-yaml-fail-closed-validation | critique:c10loss16-duplicate-yaml-fail-closed-review | accepted |
| ticket:c10loss16#ACC-005 | evidence:c10loss16-duplicate-yaml-fail-closed-validation | critique:c10loss16-duplicate-yaml-fail-closed-review | accepted |
| ticket:c10loss16#ACC-006 | evidence:c10loss16-duplicate-yaml-fail-closed-validation | critique:c10loss16-duplicate-yaml-fail-closed-review | accepted |

# Execution Notes

Ralph iteration `packet:ralph-ticket-c10loss16-20260504T112322Z` implemented the fail-closed policy in one bounded pass. Parent reconciliation added all-node preflight so duplicate YAML aborts before any target document is finalized or written.

# Blockers

None.

# Evidence

Existing evidence:

- evidence:oracle-backlog-scan
- evidence:c10loss16-duplicate-yaml-fail-closed-validation

Evidence disposition: sufficient for scoped local acceptance. Evidence covers the test-first red state, duplicate model/version fail-closed behavior, duplicate model validation, all-node preflight no-write behavior, focused and broader local pytest, Ruff, `git diff --check`, targeted pre-commit, post-commit acceptance validation, and mandatory critique.

Missing evidence: Full repository suite and GitHub Actions matrix are deferred to the initiative-level final validation pass per current operator direction not to wait on per-ticket Actions.

# Critique Disposition

Risk class: high

Critique policy: mandatory

Policy rationale: This ticket exists to prevent user-authored YAML data loss.

Required critique profiles: code-change, data-preservation, test-coverage

Findings: No open medium/high findings in critique:c10loss16-duplicate-yaml-fail-closed-review. Low residual risks are recorded in the critique and accepted as non-blocking for this ticket.

Disposition status: completed

Deferral / not-required rationale: None.

# Retrospective / Promotion Disposition

Disposition status: completed

Promoted:

- `wiki:yaml-sync-safety` now records the fail-closed duplicate YAML policy and all-node preflight rule.

Deferred / not-required rationale: No additional research, spec, plan, initiative, constitution, or memory promotion needed. Generated/NL writer migration remains tracked by `ticket:c10gen20`; final CI lessons, if any, belong to initiative-level final validation.

# Wiki Disposition

Completed. Updated `wiki:yaml-sync-safety` with duplicate YAML fail-closed and preflight rules.

# Acceptance Decision

Accepted by: OpenCode
Accepted at: 2026-05-04T11:48:07Z
Basis: Implementation commit `1bf5d4b7f45b749da36fb098133bbf3086c7d0fc`; local evidence:c10loss16-duplicate-yaml-fail-closed-validation; mandatory critique:c10loss16-duplicate-yaml-fail-closed-review with no medium/high findings; retrospective promotion to `wiki:yaml-sync-safety` completed.
Residual risks: Generated/NL writer migration remains out of scope for `ticket:c10gen20`; single-node seed duplicate errors use shared model-entry wording; duplicate version preflight no-write coverage is indirect through the shared helper; final initiative-level CI remains pending and replaces per-ticket Actions waiting.

# Dependencies

Coordinate with ticket:c10ver15 and ticket:c10gen20.

# Journal

- 2026-05-03T21:10:43Z: Created from core architecture oracle finding.
- 2026-05-04T11:23:22Z: Activated ticket and compiled Ralph packet `packet:ralph-ticket-c10loss16-20260504T112322Z` for test-first fail-closed duplicate model/version YAML sync handling with local-only validation.
- 2026-05-04T11:48:07Z: Consumed Ralph output, added all-node preflight after critique, committed implementation `1bf5d4b7f45b749da36fb098133bbf3086c7d0fc`, recorded local validation evidence `evidence:c10loss16-duplicate-yaml-fail-closed-validation`, completed mandatory critique `critique:c10loss16-duplicate-yaml-fail-closed-review`, promoted duplicate sync safety rules to `wiki:yaml-sync-safety`, accepted all scoped claims, deferred full CI matrix to initiative-level final validation, and closed ticket.
