---
id: ticket:gh328scaf
kind: ticket
status: closed
change_class: code-behavior
risk_class: medium
created_at: 2026-05-05T06:02:19Z
updated_at: 2026-05-05T08:02:39Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  initiative:
    - initiative:issue-pr-zero
  evidence:
    - evidence:gh328scaf-config-resolution-validation
  critique:
    - critique:gh328scaf-config-resolution-review
  packets:
    - packet:ralph-ticket-gh328scaf-20260505T065313Z
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
| ticket:gh328scaf#ACC-001 | evidence:gh328scaf-config-resolution-validation | critique:gh328scaf-config-resolution-review | accepted |
| ticket:gh328scaf#ACC-002 | evidence:gh328scaf-config-resolution-validation | critique:gh328scaf-config-resolution-review | accepted |
| ticket:gh328scaf#ACC-003 | evidence:gh328scaf-config-resolution-validation | critique:gh328scaf-config-resolution-review | accepted |
| ticket:gh328scaf#ACC-004 | evidence:gh328scaf-config-resolution-validation | critique:gh328scaf-config-resolution-review | accepted |
| ticket:gh328scaf#ACC-005 | evidence:gh328scaf-config-resolution-validation | critique:gh328scaf-config-resolution-review | accepted |

# Execution Notes

Ralph implemented config-aware scaffold resolution in `_sync_doc_section()` using existing `resolve_setting()` paths and preserved CLI/default fallback behavior.

# Blockers

None.

# Evidence

Evidence status: local red/green Ralph evidence, parent focused sync pytest, full sync-operation pytest, Ruff, and whitespace checks support ACC-001 through ACC-005 for the uncommitted implementation diff. Remote CI will be checked at the initiative level after the full batch push per operator direction.

# Critique Disposition

Risk class: medium

Critique policy: recommended

Policy rationale: The change affects config precedence for YAML output, but it should be narrow and covered by tests.

Required critique profiles:

- config-resolution
- yaml-output

Findings:

None - critique:gh328scaf-config-resolution-review returned `pass` with no findings.

Disposition status: completed

Deferral / not-required rationale: N/A.

# Retrospective / Promotion Disposition

Disposition status: completed

Promoted:

None - retrospective found no durable explanation needing wiki/research/spec promotion beyond this ticket, evidence, and critique.

Deferred / not-required rationale: Behavior is a narrow bug fix to existing config-resolution expectations.

# Wiki Disposition

N/A - no wiki promotion selected yet.

# Acceptance Decision

Accepted by: OpenCode parent agent.
Accepted at: 2026-05-05T08:02:39Z.
Basis: Local implementation evidence, focused validation, and Oracle critique support ACC-001 through ACC-005. The ticket is ready for issue closure with remote CI deferred to the issue-backlog initiative gate per operator direction.
Residual risks: Remote CI not yet checked; coverage is unit-level rather than a full CLI/dbt-project integration run.

# Dependencies

None.

# Journal

- 2026-05-05T06:02:19Z: Created from GitHub issue #328 and Oracle triage as a validated settings-resolution bug.
- 2026-05-05T06:59:39Z: Ralph implemented config-aware scaffold resolution and parent validation passed. Oracle critique accepted with no findings. Retrospective completed with no promotion needed beyond ticket/evidence/critique records. Moved to complete_pending_acceptance pending final implementation commit packaging.
- 2026-05-05T08:02:39Z: Accepted and closed locally for per-issue packaging. GitHub issue #328 is ready for commit, push, comment, and closure.
