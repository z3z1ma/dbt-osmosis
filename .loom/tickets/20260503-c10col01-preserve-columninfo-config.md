---
id: ticket:c10col01
kind: ticket
status: complete_pending_acceptance
change_class: code-behavior
risk_class: high
created_at: 2026-05-03T21:10:43Z
updated_at: 2026-05-03T22:08:00Z
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
  critique:
    - critique:c10col01-columninfo-config
  packet:
    - packet:ralph-ticket-c10col01-20260503T214308Z
    - packet:ralph-ticket-c10col01-20260503T215123Z
external_refs:
  github_issue: https://github.com/z3z1ma/dbt-osmosis/issues/352
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
| ticket:c10col01#ACC-001 | packet:ralph-ticket-c10col01-20260503T214308Z, `uv run pytest tests/core/test_transforms.py` | critique:c10col01-columninfo-config | supported |
| ticket:c10col01#ACC-002 | packet:ralph-ticket-c10col01-20260503T214308Z, `uv run pytest tests/core/test_transforms.py` | critique:c10col01-columninfo-config | supported |
| ticket:c10col01#ACC-003 | packet:ralph-ticket-c10col01-20260503T215123Z, `uv run pytest tests/core/test_transforms.py` | critique:c10col01-columninfo-config#FIND-002 withdrawn after coverage refinement | supported |
| ticket:c10col01#ACC-004 | packet:ralph-ticket-c10col01-20260503T214308Z, packet:ralph-ticket-c10col01-20260503T215123Z | critique:c10col01-columninfo-config | supported |
| ticket:c10col01#ACC-005 | None - dbt 1.11 adapter-backed matrix/integration evidence not gathered yet | critique:c10col01-columninfo-config#FIND-001 | converted_to_follow_up: ticket:c10ci06 and ticket:c10cfg12 |

# Execution Notes

Implemented in two Ralph iterations. Iteration 1 removed the invalid deletion and added a regression test for sync serialization. Iteration 2 parameterized the regression coverage across classic and fusion-compatible output paths so empty `config` does not leak into YAML-facing output.

# Blockers

None.

# Evidence

Existing evidence: evidence:oracle-backlog-scan and research:dbt-110-111-api-surfaces support that dbt 1.10/1.11 define `ColumnInfo.config` and that local code deleted it.

Implementation evidence:

- packet:ralph-ticket-c10col01-20260503T214308Z recorded red/green evidence: focused regression failed before the fix with `AttributeError: 'ColumnInfo' object has no attribute 'config'`, then passed after removing the deletion.
- packet:ralph-ticket-c10col01-20260503T215123Z recorded coverage refinement for `fusion_compat=False` and `fusion_compat=True`.
- Parent verification: `uv run pytest tests/core/test_transforms.py` passed with `18 passed in 9.76s` on Python 3.13.9.

Missing evidence: dbt 1.11 adapter-backed matrix/integration evidence for ACC-005. That evidence is converted to follow-up under ticket:c10ci06 and ticket:c10cfg12 because those tickets own matrix/config-shape coverage across dbt versions.

# Critique Disposition

Risk class: high

Critique policy: mandatory

Policy rationale: This touches core YAML sync behavior and dbt compatibility, with user-data/output impact.

Required critique profiles: code-change, test-coverage, dbt-compatibility

Findings:

- critique:c10col01-columninfo-config#FIND-001 remains open for missing dbt 1.11 adapter-backed evidence; ticket disposition: converted_to_follow_up to ticket:c10ci06 and ticket:c10cfg12.
- critique:c10col01-columninfo-config#FIND-002 was withdrawn after packet:ralph-ticket-c10col01-20260503T215123Z added fusion-compatible output coverage.

Disposition status: completed

Deferral / not-required rationale: Mandatory critique completed. The only open high finding is not ignored; it is converted to follow-up because the dbt 1.11 adapter-backed evidence belongs with the broader CI/config-shape matrix work.

# Retrospective / Promotion Disposition

Disposition status: pending

Promoted: None yet - run retrospective after the implementation commit per operator request.

Deferred / not-required rationale: Retrospective pending. Likely no wiki promotion is needed for this narrow fix unless later tickets expose a repeated dbt object serialization pattern.

# Wiki Disposition

N/A - no wiki promotion selected yet.

# Acceptance Decision

Accepted by: Not accepted yet.
Accepted at: N/A.
Basis: Implementation evidence and critique exist for ACC-001 through ACC-004. Final acceptance remains pending because ACC-005 was converted to follow-up coverage under ticket:c10ci06 and ticket:c10cfg12.
Residual risks: dbt 1.11 adapter-backed runtime evidence is not yet available for this exact missing-column scenario.

# Dependencies

Coordinate with ticket:c10meta02 if the fix changes shared column serialization helpers.

# Journal

- 2026-05-03T21:10:43Z: Created from dbt compatibility and core architecture oracle findings.
- 2026-05-03T22:08:00Z: Ralph iterations removed the `ColumnInfo.config` deletion and added classic/fusion-compatible sync serialization regression coverage. Mandatory critique completed with dbt 1.11 runtime evidence converted to follow-up under ticket:c10ci06 and ticket:c10cfg12.
