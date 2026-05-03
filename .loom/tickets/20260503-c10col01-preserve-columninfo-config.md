---
id: ticket:c10col01
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
  research:
    - research:dbt-110-111-api-surfaces
  evidence:
    - evidence:oracle-backlog-scan
external_refs:
  dbt_core_110_components: https://raw.githubusercontent.com/dbt-labs/dbt-core/v1.10.0/core/dbt/artifacts/resources/v1/components.py
  dbt_core_111_components: https://raw.githubusercontent.com/dbt-labs/dbt-core/v1.11.0/core/dbt/artifacts/resources/v1/components.py
depends_on: []
---

# Summary

Stop `inject_missing_columns()` from deleting `ColumnInfo.config`, which creates invalid dbt column objects under dbt 1.10.x and 1.11.x and can break later serialization to YAML.

# Context

`src/dbt_osmosis/core/transforms.py:356-370` creates `ColumnInfo.from_dict(gen_col)` and then deletes `node.columns[final_name].config` when the attribute exists. dbt-core v1.10.0 and v1.11.0 define `ColumnInfo.config: ColumnConfig = field(default_factory=ColumnConfig)`. Sync paths such as `src/dbt_osmosis/core/sync_operations.py:72-88` serialize column metadata with `meta.to_dict(omit_none=True)`, which expects the dataclass field to exist.

# Why Now

dbt 1.10/1.11 compatibility is a stated objective of initiative:dbt-110-111-hardening. This is a direct compatibility bug rather than a cleanup preference because it mutates dbt-owned objects into a shape that does not match the versioned source definitions.

# Scope

- Remove the `delattr(..., "config")` behavior from `inject_missing_columns()`.
- Ensure empty dbt-generated `config` output is filtered at serialization/YAML write boundaries rather than by corrupting dbt manifest objects.
- Reuse or centralize safe column serialization if needed, especially if `src/dbt_osmosis/core/inheritance.py:32-55` already contains a safer helper.
- Add focused tests that exercise missing-column injection and sync serialization.

# Out Of Scope

- Broadly redesigning all column serialization.
- Changing where dbt 1.10+ `config.meta` is read/written beyond what is required to keep injected columns valid; ticket:c10meta02 owns the wider `config.meta`/`config.tags` support.
- Changing column naming or data type normalization behavior.

# Acceptance Criteria

- ACC-001: `inject_missing_columns()` no longer deletes `ColumnInfo.config` or any other dbt dataclass field required by dbt 1.10.x/1.11.x.
- ACC-002: Injected missing columns can be serialized through the normal sync path without `AttributeError` or missing-field errors.
- ACC-003: Empty `config: {meta: {}, tags: []}` does not leak into YAML output unless intentionally configured; filtering happens in a serialization/YAML boundary, not by mutating dbt objects invalidly.
- ACC-004: A regression test constructs or obtains a dbt `ColumnInfo` with the dbt 1.10/1.11 shape, injects a missing column, and proves sync serialization succeeds.
- ACC-005: At least one adapter-backed fixture command or focused integration test under dbt 1.10.x and dbt 1.11.x covers a model with a database column missing from YAML.

# Coverage

Covers:

- ticket:c10col01#ACC-001
- ticket:c10col01#ACC-002
- ticket:c10col01#ACC-003
- ticket:c10col01#ACC-004
- ticket:c10col01#ACC-005
- initiative:dbt-110-111-hardening#OBJ-002
- initiative:dbt-110-111-hardening#OBJ-003

# Claim Matrix

| Claim | Evidence | Critique | Status |
| --- | --- | --- | --- |
| ticket:c10col01#ACC-001 | evidence:oracle-backlog-scan, research:dbt-110-111-api-surfaces | None | open |
| ticket:c10col01#ACC-004 | None - implementation test not written yet | None | open |
| ticket:c10col01#ACC-005 | None - matrix/integration evidence not gathered yet | None | open |

# Execution Notes

Start with the smallest fix: remove the deletion and inspect YAML output. If empty config appears, add filtering in the path that converts `ColumnInfo` to a plain dict before ruamel writes YAML. Avoid adding backwards-compatibility shims for invalid `ColumnInfo` objects unless tests show persisted state can contain them.

# Blockers

None.

# Evidence

Existing evidence: evidence:oracle-backlog-scan and research:dbt-110-111-api-surfaces support that dbt 1.10/1.11 define `ColumnInfo.config` and that local code deletes it. Missing evidence: red/green regression output and dbt matrix run after the fix.

# Critique Disposition

Risk class: high

Critique policy: mandatory

Policy rationale: This touches core YAML sync behavior and dbt compatibility, with user-data/output impact.

Required critique profiles: code-change, test-coverage, dbt-compatibility

Findings: None - no critique yet.

Disposition status: pending

Deferral / not-required rationale: None.

# Retrospective / Promotion Disposition

Disposition status: pending

Promoted: None - implementation not complete.

Deferred / not-required rationale: Decide after fix and critique whether a wiki note about safe dbt object serialization is needed.

# Wiki Disposition

N/A - no wiki promotion selected yet.

# Acceptance Decision

Accepted by: Not accepted yet.
Accepted at: N/A.
Basis: Pending implementation evidence and critique.
Residual risks: Unknown until dbt 1.10/1.11 tests run.

# Dependencies

Coordinate with ticket:c10meta02 if the fix changes shared column serialization helpers.

# Journal

- 2026-05-03T21:10:43Z: Created from dbt compatibility and core architecture oracle findings.
