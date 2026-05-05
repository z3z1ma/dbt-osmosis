---
id: ticket:gh311wrap
kind: ticket
status: closed
change_class: code-behavior
risk_class: medium
created_at: 2026-05-05T06:02:19Z
updated_at: 2026-05-05T08:14:39Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  initiative:
    - initiative:issue-pr-zero
  evidence:
    - evidence:gh311wrap-nested-description-wrap-validation
  critique:
    - critique:gh311wrap-nested-description-wrap-review
  packets:
    - packet:ralph-ticket-gh311wrap-20260505T074418Z
external_refs:
  github_issue: https://github.com/z3z1ma/dbt-osmosis/issues/311
  related_github_pr: https://github.com/z3z1ma/dbt-osmosis/pull/346
depends_on: []
---

# Summary

Prevent trailing whitespace when ruamel wraps nested column descriptions near the configured YAML width.

# Context

Issue #311 reports that descriptions around 80-83 characters are emitted as plain scalars with a line break and trailing whitespace instead of folded `>-` style. Current main reproduces the behavior for nested column descriptions: lengths 80-87 dump as plain scalars with a trailing-space line when `width=100`, while length 88+ uses folded style. PR #346 is adjacent but open, failing, and broader than this specific nested-threshold bug.

# Why Now

Formatter oscillation and trailing whitespace create noisy diffs and can break pre-commit workflows. The bug is small, reproducible, and user-visible.

# Scope

- Adjust the YAML string style threshold or representation logic for nested column descriptions so ruamel does not emit trailing whitespace.
- Add regression tests for nested column description lengths around the failing boundary.
- Preserve existing boolean-like string quoting and multiline literal behavior.
- Decide separately whether any narrow part of PR #346 is useful; do not inherit its broad unrelated scope by default.

# Out Of Scope

- A broad YAML formatter rewrite.
- Rename-description transforms or whitespace-only inheritance behavior from PR #346.
- Changing all scalar style choices outside the minimal no-trailing-whitespace fix.

# Acceptance Criteria

- ACC-001: Nested column descriptions with lengths 80-87 at default width emit no trailing whitespace.
- ACC-002: The reported 83-character example emits folded style or another representation with no trailing whitespace.
- ACC-003: Manually folded descriptions do not degrade into trailing-space plain scalars on rewrite.
- ACC-004: Existing YAML scalar tests for boolean-like strings, multiline literals, and long folded descriptions still pass.
- ACC-005: `git diff --check` passes after the new regression fixture is written.

# Coverage

Covers:

- ticket:gh311wrap#ACC-001
- ticket:gh311wrap#ACC-002
- ticket:gh311wrap#ACC-003
- ticket:gh311wrap#ACC-004
- ticket:gh311wrap#ACC-005
- initiative:issue-pr-zero#OBJ-001
- initiative:issue-pr-zero#OBJ-002
- initiative:issue-pr-zero#OBJ-005

# Claim Matrix

| Claim | Evidence | Critique | Status |
| --- | --- | --- | --- |
| ticket:gh311wrap#ACC-001 | evidence:gh311wrap-nested-description-wrap-validation | critique:gh311wrap-nested-description-wrap-review | accepted |
| ticket:gh311wrap#ACC-002 | evidence:gh311wrap-nested-description-wrap-validation | critique:gh311wrap-nested-description-wrap-review | accepted |
| ticket:gh311wrap#ACC-003 | evidence:gh311wrap-nested-description-wrap-validation | critique:gh311wrap-nested-description-wrap-review | accepted |
| ticket:gh311wrap#ACC-004 | evidence:gh311wrap-nested-description-wrap-validation | critique:gh311wrap-nested-description-wrap-review | accepted |
| ticket:gh311wrap#ACC-005 | evidence:gh311wrap-nested-description-wrap-validation | critique:gh311wrap-nested-description-wrap-review | accepted |

# Execution Notes

Ralph added a nested column description regression for lengths 80-87 and adjusted the YAML scalar threshold to account for nested indentation before plain scalar wrapping can create trailing whitespace. PR #346 remains related context, not a dependency.

# Blockers

None.

# Evidence

Evidence status: local red/green Ralph evidence, parent focused schema pytest, Ruff, format check, and whitespace check support ACC-001 through ACC-005. Remote CI will be checked at the initiative level after the issue-backlog batch per operator direction.

# Critique Disposition

Risk class: medium

Critique policy: recommended

Policy rationale: YAML formatting behavior is user-visible and can create diff churn, but the target fix is narrow.

Required critique profiles:

- yaml-formatting

Findings:

None - critique:gh311wrap-nested-description-wrap-review returned `pass` with no findings.

Disposition status: completed

Deferral / not-required rationale: N/A.

# Retrospective / Promotion Disposition

Disposition status: completed

Promoted:

None - retrospective found no durable explanation needing wiki/research/spec promotion beyond this ticket, evidence, and critique.

Deferred / not-required rationale: Behavior is a narrow YAML formatting bug fix with focused regression coverage.

# Wiki Disposition

N/A - no wiki promotion selected yet.

# Acceptance Decision

Accepted by: OpenCode parent agent.
Accepted at: 2026-05-05T08:14:39Z.
Basis: Local red/green implementation evidence, focused validation, and Oracle critique support ACC-001 through ACC-005. The ticket is ready for issue closure with remote CI deferred to the issue-backlog initiative gate per operator direction.
Residual risks: Remote CI not yet checked; parser logic relies on ruamel's current `object_keeper` traversal state; source-table column descriptions are expected to follow the same path but are not explicitly tested.

# Dependencies

Related but not dependent: PR #346.

# Journal

- 2026-05-05T06:02:19Z: Created from GitHub issue #311, local reproduction, and Oracle triage as a validated YAML formatting bug.
- 2026-05-05T07:44:19Z: Compiled Ralph packet packet:ralph-ticket-gh311wrap-20260505T074418Z and moved ticket to active for the nested description trailing-whitespace implementation iteration.
- 2026-05-05T08:14:39Z: Ralph implemented the nested description wrapping fix. Parent validation passed and Oracle critique accepted with no findings. Retrospective completed with no promotion needed beyond ticket/evidence/critique records. Accepted and closed locally for issue packaging.
