---
id: ticket:c10meta02
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
  dbt_meta_docs: https://docs.getdbt.com/reference/resource-configs/meta
  dbt_columns_docs: https://docs.getdbt.com/reference/resource-properties/columns
depends_on: []
---

# Summary

Make column-level dbt-osmosis settings, property access, and inheritance understand dbt 1.10+ `columns[].config.meta` and `columns[].config.tags` in addition to legacy top-level `meta` and `tags`.

# Context

dbt docs show column `meta` under `config.meta` for v1.10 and backported v1.9 behavior, and the column properties reference shows `config.tags` and `config.meta`. Local code still reads many column-level values from `column.meta` and `column.tags` only. Important paths include `src/dbt_osmosis/core/introspection.py:720-745`, `src/dbt_osmosis/core/introspection.py:1380-1396`, `src/dbt_osmosis/core/transforms.py:230-291`, and `src/dbt_osmosis/core/inheritance.py:276-324`, `385-433`, `439-475`.

# Why Now

dbt 1.10/1.11 users are encouraged away from top-level column `meta`. If `dbt-osmosis` ignores the supported config location, documented settings and inheritance behavior silently fail under the exact versions this initiative must support.

# Scope

- Treat top-level `column.meta` and `column.config.meta` as equivalent read sources with documented precedence.
- Treat top-level `column.tags` and `column.config.tags` as equivalent read sources where inheritance/metadata propagation expects tags.
- Ensure `SettingsResolver.resolve(..., column_name=...)` finds dbt-osmosis options under `columns[].config.meta`.
- Ensure `PropertyAccessor.get_meta(..., column_name=...)` and inheritance graph collection preserve metadata regardless of legacy or dbt 1.10+ placement.
- Preserve output behavior: when fusion/dbt 1.10-compatible YAML output is selected, write metadata under `config`; otherwise avoid surprising legacy output changes unless the behavior is explicitly configured.

# Out Of Scope

- Rebuilding the entire settings resolver precedence stack; ticket:c10res14 owns project vars/supplementary resolver integration.
- Dropping support for legacy top-level `meta`/`tags`.
- Supporting unrelated dbt resource property moves such as `freshness`, `docs`, `group`, or `access` unless they block this ticket's tests.

# Acceptance Criteria

- ACC-001: Column-level dbt-osmosis options under `columns[].config.meta` override node-level settings under dbt 1.10.x and 1.11.x.
- ACC-002: Column `config.meta` is visible through `PropertyAccessor` for manifest-backed column meta reads.
- ACC-003: Column `config.tags` participates in inheritance and tag merging wherever legacy `column.tags` currently participates.
- ACC-004: Legacy top-level column `meta` and `tags` remain supported and have an intentional precedence relationship with `config.meta`/`config.tags`.
- ACC-005: Tests use real dbt-parsed fixture nodes under dbt 1.10.x and 1.11.x, not only mocks.
- ACC-006: Documentation or inline comments explain the compatibility rule at the implementation boundary.

# Coverage

Covers:

- ticket:c10meta02#ACC-001
- ticket:c10meta02#ACC-002
- ticket:c10meta02#ACC-003
- ticket:c10meta02#ACC-004
- ticket:c10meta02#ACC-005
- ticket:c10meta02#ACC-006
- initiative:dbt-110-111-hardening#OBJ-002

# Claim Matrix

| Claim | Evidence | Critique | Status |
| --- | --- | --- | --- |
| ticket:c10meta02#ACC-001 | research:dbt-110-111-api-surfaces | None | open |
| ticket:c10meta02#ACC-005 | None - fixture tests not written yet | None | open |

# Execution Notes

Prefer a small helper that returns merged/effective column meta and tags from a dbt `ColumnInfo` object, then reuse it from resolver, property accessor, and inheritance paths. Avoid sprinkling ad hoc `hasattr(column, "config")` blocks across every caller.

# Blockers

None.

# Evidence

Existing evidence: research:dbt-110-111-api-surfaces and evidence:oracle-backlog-scan. Missing evidence: focused dbt 1.10/1.11 fixture output and regression tests.

# Critique Disposition

Risk class: high

Critique policy: mandatory

Policy rationale: This changes config precedence and inheritance behavior across core YAML workflows.

Required critique profiles: code-change, test-coverage, dbt-compatibility, regression-risk

Findings: None - no critique yet.

Disposition status: pending

Deferral / not-required rationale: None.

# Retrospective / Promotion Disposition

Disposition status: pending

Promoted: None - implementation not complete.

Deferred / not-required rationale: Consider a wiki/config-resolution note after acceptance.

# Wiki Disposition

N/A - no wiki promotion selected yet.

# Acceptance Decision

Accepted by: Not accepted yet.
Accepted at: N/A.
Basis: Pending implementation evidence and critique.
Residual risks: Config precedence regressions until covered by real parsed fixtures.

# Dependencies

Coordinate with ticket:c10cfg12 for fixture coverage and ticket:c10res14 for broader resolver integration.

# Journal

- 2026-05-03T21:10:43Z: Created from dbt compatibility oracle and dbt docs/source research.
