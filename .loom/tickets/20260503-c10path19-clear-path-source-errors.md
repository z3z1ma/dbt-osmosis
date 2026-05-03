---
id: ticket:c10path19
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

Replace raw YAML path-template/source-table crashes with clear user-facing errors and safe defaults.

# Context

`src/dbt_osmosis/core/path_management.py:138-166` directly formats user-configured YAML path templates. Invalid placeholders can surface raw `AttributeError` or `KeyError`, and tests currently expect raw exceptions. `src/dbt_osmosis/core/sync_operations.py:437-445` assumes `doc_source["tables"]` exists after `_get_or_create_source()` can return an existing source that lacks `tables`.

# Why Now

Compatibility work should make command failures clearer, not more cryptic. Bad path templates or incomplete source YAML are user-input problems and should fail closed with actionable messages or be repaired safely when unambiguous.

# Scope

- Catch path template formatting errors and raise `PathResolutionError` with node ID, template, and missing attribute/key.
- Keep existing project-root path traversal validation.
- Make source table creation use `setdefault("tables", [])` with type validation.
- Add tests for invalid path template fields and source YAML with missing `tables`.

# Out Of Scope

- Designing a new path template language.
- Auto-repairing malformed `tables` values that are not lists without surfacing validation.

# Acceptance Criteria

- ACC-001: Invalid YAML path template placeholders raise `PathResolutionError`, not raw `AttributeError`/`KeyError`.
- ACC-002: Error messages include the node unique ID and configured template.
- ACC-003: Existing project-root safety checks still reject traversal outside the dbt project.
- ACC-004: Existing source YAML with `sources: [{name: raw}]` can sync a source table by creating `tables: []` when safe.
- ACC-005: Existing source YAML with non-list `tables` fails with a clear validation/error message.

# Coverage

Covers:

- ticket:c10path19#ACC-001
- ticket:c10path19#ACC-002
- ticket:c10path19#ACC-003
- ticket:c10path19#ACC-004
- ticket:c10path19#ACC-005
- initiative:dbt-110-111-hardening#OBJ-003
- initiative:dbt-110-111-hardening#OBJ-006

# Claim Matrix

| Claim | Evidence | Critique | Status |
| --- | --- | --- | --- |
| ticket:c10path19#ACC-001 | evidence:oracle-backlog-scan | None | open |
| ticket:c10path19#ACC-004 | evidence:oracle-backlog-scan | None | open |

# Execution Notes

Keep path error wrapping narrow so real programmer errors are not swallowed. Tests should assert exception type and useful message, not exact full wording.

# Blockers

None.

# Evidence

Existing evidence: evidence:oracle-backlog-scan. Missing evidence: focused error-handling tests.

# Critique Disposition

Risk class: medium

Critique policy: recommended

Policy rationale: User-facing error handling and source YAML mutation should be reviewed for clarity and data preservation.

Required critique profiles: code-change, operator-clarity, test-coverage

Findings: None - no critique yet.

Disposition status: pending

Deferral / not-required rationale: None.

# Retrospective / Promotion Disposition

Disposition status: pending

Promoted: None - implementation not complete.

Deferred / not-required rationale: Not decided.

# Wiki Disposition

N/A - no wiki promotion selected yet.

# Acceptance Decision

Accepted by: Not accepted yet.
Accepted at: N/A.
Basis: Pending tests.
Residual risks: Some path templates may depend on undocumented object attributes.

# Dependencies

None.

# Journal

- 2026-05-03T21:10:43Z: Created from core architecture oracle finding.
