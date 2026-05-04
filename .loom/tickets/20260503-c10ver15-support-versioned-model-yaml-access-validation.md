---
id: ticket:c10ver15
kind: ticket
status: active
change_class: code-behavior
risk_class: high
created_at: 2026-05-03T21:10:43Z
updated_at: 2026-05-04T09:50:44Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  initiative:
    - initiative:dbt-110-111-hardening
  evidence:
    - evidence:oracle-backlog-scan
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
| ticket:c10ver15#ACC-001 | evidence:oracle-backlog-scan | None | open |
| ticket:c10ver15#ACC-004 | None - validation tests not written yet | None | open |

# Execution Notes

Avoid returning mutable YAML dicts from helpers if callers only need read access. Keep version matching robust to dbt storing versions as int, float, or string-like values.

# Blockers

None.

# Evidence

Existing evidence: evidence:oracle-backlog-scan. Missing evidence: versioned model test output.

# Critique Disposition

Risk class: high

Critique policy: mandatory

Policy rationale: Versioned model YAML access can affect correctness and data preservation for real dbt projects.

Required critique profiles: code-change, test-coverage, dbt-compatibility

Findings: None - no critique yet.

Disposition status: pending

Deferral / not-required rationale: None.

# Retrospective / Promotion Disposition

Disposition status: pending

Promoted: None - implementation not complete.

Deferred / not-required rationale: Consider wiki note if versioned model handling remains subtle.

# Wiki Disposition

N/A - no wiki promotion selected yet.

# Acceptance Decision

Accepted by: Not accepted yet.
Accepted at: N/A.
Basis: Pending implementation evidence and critique.
Residual risks: Type coercion for `v` values may be adapter/dbt-version sensitive.

# Dependencies

Coordinate with ticket:c10loss16 if versioned dedupe changes.

# Journal

- 2026-05-03T21:10:43Z: Created from core architecture oracle finding.
- 2026-05-04T09:50:44Z: Activated ticket and compiled Ralph packet `packet:ralph-ticket-c10ver15-20260504T095044Z` for test-first versioned model YAML property access and validation support with local-only validation.
