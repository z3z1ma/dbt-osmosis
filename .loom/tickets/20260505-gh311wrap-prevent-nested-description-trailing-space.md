---
id: ticket:gh311wrap
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
| ticket:gh311wrap#ACC-001 | None yet | None yet | open |
| ticket:gh311wrap#ACC-002 | None yet | None yet | open |
| ticket:gh311wrap#ACC-003 | None yet | None yet | open |
| ticket:gh311wrap#ACC-004 | None yet | None yet | open |
| ticket:gh311wrap#ACC-005 | None yet | None yet | open |

# Execution Notes

Parent triage reproduced the bug with `create_yaml_instance(width=100)`: lengths 80-87 were plain scalars with trailing whitespace, and length 88+ used folded style. Oracle identified `src/dbt_osmosis/core/schema/parser.py` threshold logic as ignoring nested indentation. PR #346 should be treated as related context, not as the owner of this ticket.

# Blockers

None.

# Evidence

Expected evidence: red/green formatter regression around the nested description threshold, focused `tests/core/test_schema.py` run, `git diff --check`, and remote CI before closure.

# Critique Disposition

Risk class: medium

Critique policy: recommended

Policy rationale: YAML formatting behavior is user-visible and can create diff churn, but the target fix is narrow.

Required critique profiles:

- yaml-formatting

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

Related but not dependent: PR #346.

# Journal

- 2026-05-05T06:02:19Z: Created from GitHub issue #311, local reproduction, and Oracle triage as a validated YAML formatting bug.
