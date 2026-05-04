---
id: ticket:c10ver15
kind: ticket
status: closed
change_class: code-behavior
risk_class: high
created_at: 2026-05-03T21:10:43Z
updated_at: 2026-05-04T11:20:10Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  initiative:
    - initiative:dbt-110-111-hardening
  evidence:
    - evidence:oracle-backlog-scan
    - evidence:c10ver15-versioned-yaml-access-validation
  critique:
    - critique:c10ver15-versioned-yaml-access-review
  wiki:
    - wiki:versioned-model-yaml
  packets:
    - packet:ralph-ticket-c10ver15-20260504T095044Z
depends_on: []
---

# Summary

Teach YAML property access and validation to handle dbt versioned model blocks under `models[].versions[]`, especially version-level columns.

# Context

`src/dbt_osmosis/core/inheritance.py:211-222` returns only the top-level model YAML entry for models/seeds. `PropertyAccessor._get_from_yaml()` depends on that view. dbt versioned model YAML stores version-specific columns under `models[].versions[].columns`. `src/dbt_osmosis/core/schema/validation.py:669-716` validates only `models[].columns`, not `models[].versions[].columns`.

# Why Now

dbt versioned models are part of modern dbt projects. Ignoring version-level YAML can make inheritance, unrendered descriptions, progenitor overrides, and validation wrong under dbt 1.10/1.11.

# Scope

- Update `_get_node_yaml()` or a replacement helper to return the correct version block for `ModelNode.version`.
- Define how top-level model metadata and version-level metadata combine when both exist.
- Update `PropertyAccessor` and inheritance callers to read versioned columns correctly.
- Extend schema validation to validate `versions`, `v`, version-level `columns`, and nested tests.
- Add tests against a versioned model fixture.

# Out Of Scope

- Redesigning dbt model versioning support beyond YAML property access/sync/validation.
- Changing dbt's own versioned model semantics.

# Acceptance Criteria

- ACC-001: For a versioned `ModelNode`, YAML access finds the matching `versions[].v` block.
- ACC-002: Version-level column descriptions/meta/tags/tests can be read by property access and inheritance paths.
- ACC-003: Top-level model metadata fallback behavior is explicit and tested.
- ACC-004: Schema validation flags invalid version entries and invalid version-level columns/tests.
- ACC-005: Tests cover at least one versioned model with a `doc()` description or progenitor override under a version block.

# Coverage

Covers:

- ticket:c10ver15#ACC-001
- ticket:c10ver15#ACC-002
- ticket:c10ver15#ACC-003
- ticket:c10ver15#ACC-004
- ticket:c10ver15#ACC-005
- initiative:dbt-110-111-hardening#OBJ-003

# Claim Matrix

| Claim | Evidence | Critique | Status |
| --- | --- | --- | --- |
| ticket:c10ver15#ACC-001 | evidence:c10ver15-versioned-yaml-access-validation | critique:c10ver15-versioned-yaml-access-review | accepted |
| ticket:c10ver15#ACC-002 | evidence:c10ver15-versioned-yaml-access-validation | critique:c10ver15-versioned-yaml-access-review | accepted |
| ticket:c10ver15#ACC-003 | evidence:c10ver15-versioned-yaml-access-validation | critique:c10ver15-versioned-yaml-access-review | accepted |
| ticket:c10ver15#ACC-004 | evidence:c10ver15-versioned-yaml-access-validation | critique:c10ver15-versioned-yaml-access-review | accepted |
| ticket:c10ver15#ACC-005 | evidence:c10ver15-versioned-yaml-access-validation | critique:c10ver15-versioned-yaml-access-review | accepted |

# Execution Notes

Ralph iteration `packet:ralph-ticket-c10ver15-20260504T095044Z` completed versioned YAML access and validation in one bounded pass. Parent reconciliation added critique-driven fixes for dbt selector controls, selector-preserving sync, blank version description fallback, duplicate/latest-version validation, int/string sync lookup skew, and conservative version identity matching for distinct string versions.

# Blockers

None.

# Evidence

Existing evidence:

- evidence:oracle-backlog-scan
- evidence:c10ver15-versioned-yaml-access-validation

Evidence disposition: sufficient for scoped local acceptance. Evidence covers the test-first red state, selected version YAML access, version-level column property access, unrendered inheritance, dbt-compatible version and selector validation, selector-preserving sync/refactor, focused and broader local pytest, Ruff, `git diff --check`, targeted pre-commit, post-commit acceptance validation, and mandatory critique.

Missing evidence: Full repository suite and GitHub Actions matrix are deferred to the initiative-level final validation pass per current operator direction not to wait on per-ticket Actions.

# Critique Disposition

Risk class: high

Critique policy: mandatory

Policy rationale: Versioned model YAML access can affect correctness and data preservation for real dbt projects.

Required critique profiles: code-change, test-coverage, dbt-compatibility

Findings: No open medium/high findings in critique:c10ver15-versioned-yaml-access-review. Low residual risks are recorded in the critique and accepted as non-blocking for this ticket.

Disposition status: completed

Deferral / not-required rationale: None.

# Retrospective / Promotion Disposition

Disposition status: completed

Promoted:

- `wiki:versioned-model-yaml` now records accepted versioned model YAML access, version identity, selector validation/preservation, and sync/refactor rules.

Deferred / not-required rationale: No additional research, spec, plan, initiative, constitution, or memory promotion needed. Final CI lessons, if any, belong to initiative-level final validation.

# Wiki Disposition

Completed. Created `wiki:versioned-model-yaml` for accepted versioned model YAML handling and future versioned-sync follow-up context.

# Acceptance Decision

Accepted by: OpenCode
Accepted at: 2026-05-04T11:20:10Z
Basis: Implementation commit `ef1d5409bcceee40c3403c06d75bf5cbe4cc4bb1`; local evidence:c10ver15-versioned-yaml-access-validation; mandatory critique:c10ver15-versioned-yaml-access-review with no medium/high findings; retrospective promotion to `wiki:versioned-model-yaml` completed.
Residual risks: `ModelValidator` remains focused rather than a full dbt schema validator; `_get_node_yaml()` returns a shallow read-only mapping with mutable nested cache objects; non-finite numeric version values are not explicitly modeled; final initiative-level CI remains pending and replaces per-ticket Actions waiting.

# Dependencies

Coordinate with ticket:c10loss16 if versioned dedupe changes.

# Journal

- 2026-05-03T21:10:43Z: Created from core architecture oracle finding.
- 2026-05-04T09:50:44Z: Activated ticket and compiled Ralph packet `packet:ralph-ticket-c10ver15-20260504T095044Z` for test-first versioned model YAML property access and validation support with local-only validation.
- 2026-05-04T11:20:10Z: Consumed Ralph output, applied parent critique-driven fixes, committed implementation `ef1d5409bcceee40c3403c06d75bf5cbe4cc4bb1`, recorded local validation evidence `evidence:c10ver15-versioned-yaml-access-validation`, completed mandatory critique `critique:c10ver15-versioned-yaml-access-review`, promoted accepted versioned YAML handling to `wiki:versioned-model-yaml`, accepted all scoped claims, deferred full CI matrix to initiative-level final validation, and closed ticket.
