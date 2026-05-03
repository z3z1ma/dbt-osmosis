---
id: ticket:c10res14
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

Finish the context-aware `SettingsResolver` so documented precedence actually includes supplementary `dbt-osmosis.yml` and project vars, then retire legacy `_get_setting_for_node()` call sites.

# Context

`src/dbt_osmosis/core/introspection.py:690-802` defines `SettingsResolver.resolve()` but it is node-only and does not call `ProjectVarsSource` or `SupplementaryFileSource` from `introspection.py:430-650`. Legacy `_get_setting_for_node()` still drives behavior in `transforms.py`, `inheritance.py`, `sync_operations.py`, and `plugins.py`. `src/dbt_osmosis/core/settings.py:234-257` and `338-346` duplicate partial config loading.

# Why Now

The constitution and repo guidance say configuration resolution should be centralized. Compatibility work around dbt 1.10+ config locations will be fragile if the resolver remains partially dead and call sites keep using older ad hoc logic.

# Scope

- Give the resolver enough context to include column meta, node meta/config, `config.extra`, supplementary `dbt-osmosis.yml`, vars, and fallback defaults in the documented precedence order.
- Preserve explicit falsey values such as `False`, `0`, and empty strings where they are valid settings.
- Migrate core call sites away from `_get_setting_for_node()`.
- Remove or clearly mark dead source classes if they remain intentionally unused.
- Add integration tests that prove real transform behavior changes from supplementary file and vars settings.

# Out Of Scope

- Inventing new config precedence not already documented or demonstrated.
- Changing behavior for unrelated CLI options unless they currently violate documented precedence.

# Acceptance Criteria

- ACC-001: A single resolver path handles documented precedence from column-level settings through fallback defaults.
- ACC-002: Supplementary `dbt-osmosis.yml` settings influence real transform/sync behavior, not only isolated source-class tests.
- ACC-003: Project vars settings influence real transform/sync behavior according to documented precedence.
- ACC-004: Core production call sites no longer call deprecated `_get_setting_for_node()`.
- ACC-005: Tests cover falsey explicit settings so they are not treated as missing.
- ACC-006: Deprecated/dead resolver pieces are removed, documented, or isolated.

# Coverage

Covers:

- ticket:c10res14#ACC-001
- ticket:c10res14#ACC-002
- ticket:c10res14#ACC-003
- ticket:c10res14#ACC-004
- ticket:c10res14#ACC-005
- ticket:c10res14#ACC-006
- initiative:dbt-110-111-hardening#OBJ-002
- initiative:dbt-110-111-hardening#OBJ-008

# Claim Matrix

| Claim | Evidence | Critique | Status |
| --- | --- | --- | --- |
| ticket:c10res14#ACC-002 | evidence:oracle-backlog-scan | None | open |
| ticket:c10res14#ACC-004 | None - migration not done yet | None | open |

# Execution Notes

This ticket is larger than the narrow config.meta fixes. Consider breaking implementation into one Ralph packet for resolver context and another for call-site migration if scope grows.

# Blockers

Potential blocker: if documented precedence is ambiguous, route to a spec or update this ticket before changing behavior.

# Evidence

Existing evidence: evidence:oracle-backlog-scan. Missing evidence: integration tests against real `YamlRefactorContext`.

# Critique Disposition

Risk class: high

Critique policy: mandatory

Policy rationale: Config precedence affects many commands and can silently change output.

Required critique profiles: code-change, test-coverage, regression-risk

Findings: None - no critique yet.

Disposition status: pending

Deferral / not-required rationale: None.

# Retrospective / Promotion Disposition

Disposition status: pending

Promoted: None - implementation not complete.

Deferred / not-required rationale: Wiki promotion likely if resolver behavior becomes the canonical implementation guide.

# Wiki Disposition

N/A - no wiki promotion selected yet.

# Acceptance Decision

Accepted by: Not accepted yet.
Accepted at: N/A.
Basis: Pending implementation evidence and critique.
Residual risks: Precedence changes may reveal undocumented user dependencies.

# Dependencies

Coordinate with ticket:c10meta02, ticket:c10opt03, and ticket:c10cfg12.

# Journal

- 2026-05-03T21:10:43Z: Created from core architecture and dbt compatibility oracle findings.
