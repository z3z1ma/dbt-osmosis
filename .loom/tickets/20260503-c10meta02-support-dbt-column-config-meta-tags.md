---
id: ticket:c10meta02
kind: ticket
status: closed
change_class: code-behavior
risk_class: high
created_at: 2026-05-03T21:10:43Z
updated_at: 2026-05-04T03:29:15Z
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
    - evidence:c10meta02-column-config-meta-tags
    - evidence:c10col01-c10meta02-main-ci-success
  critique:
    - critique:c10meta02-column-config-meta-tags
  packet:
    - packet:ralph-ticket-c10meta02-20260503T215906Z
    - packet:ralph-ticket-c10meta02-20260503T220442Z
    - packet:ralph-ticket-c10meta02-20260503T221219Z
external_refs:
  github_issue: https://github.com/z3z1ma/dbt-osmosis/issues/353
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
| ticket:c10meta02#ACC-001 | evidence:c10meta02-column-config-meta-tags; evidence:c10col01-c10meta02-main-ci-success | critique:c10meta02-column-config-meta-tags | supported |
| ticket:c10meta02#ACC-002 | evidence:c10meta02-column-config-meta-tags; evidence:c10col01-c10meta02-main-ci-success | critique:c10meta02-column-config-meta-tags#FIND-002 withdrawn after fix | supported |
| ticket:c10meta02#ACC-003 | evidence:c10meta02-column-config-meta-tags; evidence:c10col01-c10meta02-main-ci-success | critique:c10meta02-column-config-meta-tags#FIND-003 withdrawn after fix | supported |
| ticket:c10meta02#ACC-004 | evidence:c10meta02-column-config-meta-tags; evidence:c10col01-c10meta02-main-ci-success | critique:c10meta02-column-config-meta-tags | supported |
| ticket:c10meta02#ACC-005 | evidence:c10ci06-main-ci-success covers broad dbt 1.10/1.11 matrix execution; exact parsed-fixture evidence remains with ticket:c10cfg12 | critique:c10meta02-column-config-meta-tags#FIND-001 | converted_to_follow_up: ticket:c10ci06 closed; ticket:c10cfg12 ready follow-up |
| ticket:c10meta02#ACC-006 | evidence:c10meta02-column-config-meta-tags; evidence:c10col01-c10meta02-main-ci-success | critique:c10meta02-column-config-meta-tags | supported |

# Execution Notes

Implemented with helper functions that return effective column meta/tags from legacy top-level fields plus dbt 1.10+ `config.meta` / `config.tags`. The helpers are reused by `SettingsResolver`, `PropertyAccessor`, and inheritance graph normalization. Config metadata wins exact meta-key conflicts; tag merging preserves legacy order and appends new config tags without duplicates.

# Blockers

None.

# Evidence

Existing evidence: research:dbt-110-111-api-surfaces and evidence:oracle-backlog-scan.

Implementation evidence:

- packet:ralph-ticket-c10meta02-20260503T215906Z recorded red/green tests for column `config.meta` settings resolution, manifest property access, and inheritance.
- packet:ralph-ticket-c10meta02-20260503T220442Z recorded YAML-source property access coverage; strict red state was not observed because the prior uncommitted implementation already satisfied the new test.
- packet:ralph-ticket-c10meta02-20260503T221219Z recorded red/green fallback regression coverage after critique found the YAML fallback issue.
- evidence:c10meta02-column-config-meta-tags preserves the observed red/green and post-commit validation output.
- evidence:c10col01-c10meta02-main-ci-success records final `main` `Tests` workflow success after the implementation commit landed.
- Parent verification: `uv run pytest tests/core/test_settings_resolver.py tests/core/test_property_accessor.py tests/core/test_inheritance_behavior.py` passed with `50 passed, 3 skipped in 13.05s`.
- Parent verification: `uv run ruff check ...` passed.
- Parent verification: `uv run pyright src/dbt_osmosis/core/introspection.py src/dbt_osmosis/core/inheritance.py` reported `0 errors`.

Missing evidence: exact dbt 1.11 adapter-backed parsed-fixture evidence for ACC-005. That evidence is converted to follow-up under ticket:c10cfg12; broad dbt 1.10/1.11 matrix execution is covered by closed ticket:c10ci06.

# Critique Disposition

Risk class: high

Critique policy: mandatory

Policy rationale: This changes config precedence and inheritance behavior across core YAML workflows.

Required critique profiles: code-change, test-coverage, dbt-compatibility, regression-risk

Findings:

- critique:c10meta02-column-config-meta-tags#FIND-001 remains open for missing exact dbt 1.11 parsed-fixture evidence; ticket disposition: converted_to_follow_up to closed ticket:c10ci06 for broad matrix execution and ready ticket:c10cfg12 for exact parsed-fixture follow-up.
- critique:c10meta02-column-config-meta-tags#FIND-002 was withdrawn after packet:ralph-ticket-c10meta02-20260503T221219Z fixed YAML-source fallback behavior.
- critique:c10meta02-column-config-meta-tags#FIND-003 was withdrawn after packet:ralph-ticket-c10meta02-20260503T221219Z added older-dbt-safe test config setup.

Disposition status: completed

Deferral / not-required rationale: Mandatory critique completed. The open evidence gap is not ignored; it is converted to follow-up because broad dbt 1.11 matrix evidence belongs with ticket:c10ci06 and exact parsed-fixture coverage belongs with ticket:c10cfg12.

# Retrospective / Promotion Disposition

Disposition status: completed

Promoted: evidence:c10meta02-column-config-meta-tags preserves red/green and post-commit validation output.

Deferred / not-required rationale: Wiki promotion deferred until the broader resolver/config-shape tickets finish, so the accepted explanation can cover the full precedence model rather than this partial compatibility slice.

# Wiki Disposition

N/A - no wiki promotion selected yet.

# Acceptance Decision

Accepted by: OpenCode.
Accepted at: 2026-05-04T03:29:15Z.
Basis: Implementation evidence and mandatory critique support ACC-001 through ACC-004 and ACC-006; final `main` CI passed after the implementation commit; ACC-005's exact fixture gap is explicitly converted to ticket:c10cfg12 with broad dbt 1.10/1.11 matrix coverage already accepted under ticket:c10ci06.
Residual risks: Exact dbt 1.11 parsed-fixture behavior is not yet verified in this ticket's evidence and remains follow-up work under ticket:c10cfg12.

# Dependencies

Coordinate with ticket:c10cfg12 for fixture coverage and ticket:c10res14 for broader resolver integration.

# Journal

- 2026-05-03T21:10:43Z: Created from dbt compatibility oracle and dbt docs/source research.
- 2026-05-03T22:36:00Z: Ralph iterations added effective column config metadata/tag helpers, settings/property/inheritance coverage, YAML-source fallback protection, and mandatory critique. dbt 1.11 parsed-fixture evidence converted to follow-up under ticket:c10cfg12 and ticket:c10ci06.
- 2026-05-03T22:45:00Z: Retrospective completed. Promoted validation output to evidence:c10meta02-column-config-meta-tags; wiki explanation deferred until broader config-resolution tickets settle the full precedence model.
- 2026-05-04T03:29:15Z: Accepted and closed after final `main` `Tests` workflow success was recorded in `evidence:c10col01-c10meta02-main-ci-success`; exact dbt 1.11 parsed-fixture evidence remains converted to ticket:c10cfg12.
