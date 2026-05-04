---
id: ticket:c10res14
kind: ticket
status: closed
change_class: code-behavior
risk_class: high
created_at: 2026-05-03T21:10:43Z
updated_at: 2026-05-04T09:45:38Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  initiative:
    - initiative:dbt-110-111-hardening
  evidence:
    - evidence:oracle-backlog-scan
    - evidence:c10res14-context-aware-resolver-validation
  critique:
    - critique:c10res14-context-aware-settings-resolver-review
  wiki:
    - wiki:config-resolution
  packets:
    - packet:ralph-ticket-c10res14-20260504T082226Z
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
| ticket:c10res14#ACC-001 | evidence:c10res14-context-aware-resolver-validation | critique:c10res14-context-aware-settings-resolver-review | accepted |
| ticket:c10res14#ACC-002 | evidence:c10res14-context-aware-resolver-validation | critique:c10res14-context-aware-settings-resolver-review | accepted |
| ticket:c10res14#ACC-003 | evidence:c10res14-context-aware-resolver-validation | critique:c10res14-context-aware-settings-resolver-review | accepted |
| ticket:c10res14#ACC-004 | evidence:c10res14-context-aware-resolver-validation | critique:c10res14-context-aware-settings-resolver-review | accepted |
| ticket:c10res14#ACC-005 | evidence:c10res14-context-aware-resolver-validation | critique:c10res14-context-aware-settings-resolver-review | accepted |
| ticket:c10res14#ACC-006 | evidence:c10res14-context-aware-resolver-validation | critique:c10res14-context-aware-settings-resolver-review | accepted |

# Execution Notes

Ralph iteration `packet:ralph-ticket-c10res14-20260504T082226Z` completed the resolver context and production call-site migration in one bounded pass. Parent reconciliation added critique-driven fixes for precise dtype context, project-vars shape coverage, context debug APIs, supplementary parse caching, all-node fan-out precedence, and inheritance plugin context transport.

# Blockers

None.

# Evidence

Existing evidence:

- evidence:oracle-backlog-scan
- evidence:c10res14-context-aware-resolver-validation

Evidence disposition: sufficient for scoped local acceptance. Evidence covers the test-first red state, context-backed resolver behavior, supplementary-file transform behavior, project-vars sync and introspection behavior, inherited prefix behavior, falsey preservation, migrated production call sites, focused and broader local pytest, Ruff, `git diff --check`, targeted pre-commit, and mandatory critique.

Missing evidence: Full repository suite and GitHub Actions matrix are deferred to the initiative-level final validation pass per current operator direction not to wait on per-ticket Actions.

# Critique Disposition

Risk class: high

Critique policy: mandatory

Policy rationale: Config precedence affects many commands and can silently change output.

Required critique profiles: code-change, test-coverage, regression-risk

Findings: No open medium/high findings in critique:c10res14-context-aware-settings-resolver-review. Low residual risks are recorded in the critique and accepted as non-blocking for this ticket.

Disposition status: completed

Deferral / not-required rationale: None.

# Retrospective / Promotion Disposition

Disposition status: completed

Promoted:

- `wiki:config-resolution` now records accepted resolver precedence, context-source rules, falsey handling, plugin context boundary, and debug helper usage.

Deferred / not-required rationale: No additional research, spec, plan, initiative, constitution, or memory promotion needed. Final CI lessons, if any, belong to initiative-level final validation.

# Wiki Disposition

Completed. Created `wiki:config-resolution` for the accepted settings resolver behavior and production call-site guidance.

# Acceptance Decision

Accepted by: OpenCode
Accepted at: 2026-05-04T09:45:38Z
Basis: Implementation commit `e4047ad46529dcecc40a6a68b27e8fcd5716b314`; local evidence:c10res14-context-aware-resolver-validation; mandatory critique:c10res14-context-aware-settings-resolver-review with no medium/high findings; retrospective promotion to `wiki:config-resolution` completed.
Residual risks: `CONTEXT_SETTINGS` is an API footgun if future direct resolver callers omit matching fallback values; explicit `None` remains treated as missing in class-backed sources; third-party plugins may need to tolerate full `YamlRefactorContext`; final initiative-level CI remains pending and replaces per-ticket Actions waiting.

# Dependencies

Coordinate with ticket:c10meta02, ticket:c10opt03, and ticket:c10cfg12.

# Journal

- 2026-05-03T21:10:43Z: Created from core architecture and dbt compatibility oracle findings.
- 2026-05-04T08:22:26Z: Activated ticket and compiled Ralph packet `packet:ralph-ticket-c10res14-20260504T082226Z` for test-first context-aware resolver completion and core call-site migration with local-only validation.
- 2026-05-04T09:45:38Z: Consumed Ralph output, applied parent critique-driven fixes, committed implementation `e4047ad46529dcecc40a6a68b27e8fcd5716b314`, recorded local validation evidence `evidence:c10res14-context-aware-resolver-validation`, completed mandatory critique `critique:c10res14-context-aware-settings-resolver-review`, promoted accepted resolver behavior to `wiki:config-resolution`, accepted all scoped claims, deferred full CI matrix to initiative-level final validation, and closed ticket.
