---
id: ticket:gh328scaf
kind: ticket
status: ready
change_class: code-behavior
risk_class: medium
created_at: 2026-05-05T06:02:19Z
updated_at: 2026-05-05T06:02:19Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  initiative:
    - initiative:issue-pr-zero
external_refs:
  github_issue: https://github.com/z3z1ma/dbt-osmosis/issues/328
depends_on: []
---

# Summary

Honor `scaffold-empty-configs` when configured in dbt project `+dbt-osmosis-options`, not only when passed as the CLI `--scaffold-empty-configs` flag.

# Context

Issue #328 reports that `dbt-osmosis yaml refactor --scaffold-empty-configs` scaffolds empty config fields, while `+dbt-osmosis-options: {scaffold-empty-configs: true}` in `dbt_project.yml` does not. Oracle triage found that `_sync_doc_section()` directly checks `context.settings.scaffold_empty_configs` instead of resolving the per-node setting through `SettingsResolver` in the paths that decide whether to keep empty placeholders.

# Why Now

The previous settings-resolution hardening made project/node/column option precedence a core contract. A setting that only works as a CLI flag violates that contract and creates confusing behavior for pre-commit and team-wide YAML workflows.

# Scope

- Use context-aware setting resolution where YAML sync decides whether to keep or remove empty placeholder fields.
- Preserve the CLI flag behavior.
- Preserve default false behavior when neither CLI nor config enables scaffolding.
- Add focused tests for dbt project config and CLI/global behavior.

# Out Of Scope

- Redesigning configuration precedence.
- Adding new scaffold-related settings.
- Broad YAML formatting changes unrelated to empty field retention.

# Acceptance Criteria

- ACC-001: `+dbt-osmosis-options: {scaffold-empty-configs: true}` causes `yaml refactor` to keep/scaffold empty fields without the CLI flag.
- ACC-002: CLI `--scaffold-empty-configs` remains functional.
- ACC-003: Default behavior remains unchanged when the setting is absent or false.
- ACC-004: Tests cover at least one node/project config path and one column cleanup path where empty fields would otherwise be removed.
- ACC-005: The implementation uses `SettingsResolver` or existing `resolve_setting()` paths rather than a new ad hoc config lookup.

# Coverage

Covers:

- ticket:gh328scaf#ACC-001
- ticket:gh328scaf#ACC-002
- ticket:gh328scaf#ACC-003
- ticket:gh328scaf#ACC-004
- ticket:gh328scaf#ACC-005
- initiative:issue-pr-zero#OBJ-001
- initiative:issue-pr-zero#OBJ-002
- initiative:issue-pr-zero#OBJ-005

# Claim Matrix

| Claim | Evidence | Critique | Status |
| --- | --- | --- | --- |
| ticket:gh328scaf#ACC-001 | None yet | None yet | open |
| ticket:gh328scaf#ACC-002 | None yet | None yet | open |
| ticket:gh328scaf#ACC-003 | None yet | None yet | open |
| ticket:gh328scaf#ACC-004 | None yet | None yet | open |
| ticket:gh328scaf#ACC-005 | None yet | None yet | open |

# Execution Notes

Relevant code paths include `src/dbt_osmosis/core/sync_operations.py` checks for `context.settings.scaffold_empty_configs`, `src/dbt_osmosis/core/settings.py` setting declaration, and `src/dbt_osmosis/core/introspection.py` setting resolution.

# Blockers

None.

# Evidence

Expected evidence: red/green focused tests for config-driven scaffolding and a targeted settings/sync pytest run. Remote CI evidence is needed before closure.

# Critique Disposition

Risk class: medium

Critique policy: recommended

Policy rationale: The change affects config precedence for YAML output, but it should be narrow and covered by tests.

Required critique profiles:

- config-resolution
- yaml-output

Findings:

None - no critique yet.

Disposition status: pending

Deferral / not-required rationale: N/A.

# Retrospective / Promotion Disposition

Disposition status: pending

Promoted:

None yet.

Deferred / not-required rationale: N/A.

# Wiki Disposition

N/A - no wiki promotion selected yet.

# Acceptance Decision

Accepted by: Pending implementation and evidence.
Accepted at: N/A.
Basis: N/A.
Residual risks: N/A.

# Dependencies

None.

# Journal

- 2026-05-05T06:02:19Z: Created from GitHub issue #328 and Oracle triage as a validated settings-resolution bug.
