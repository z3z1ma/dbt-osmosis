---
id: ticket:gh326skip
kind: ticket
status: closed
change_class: code-behavior
risk_class: medium
created_at: 2026-05-05T06:02:19Z
updated_at: 2026-05-05T08:07:56Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  initiative:
    - initiative:issue-pr-zero
  evidence:
    - evidence:gh326skip-description-inheritance-validation
  critique:
    - critique:gh326skip-description-inheritance-review
  packets:
    - packet:ralph-ticket-gh326skip-20260505T072053Z
external_refs:
  github_issue: https://github.com/z3z1ma/dbt-osmosis/issues/326
depends_on: []
---

# Summary

Add an option to skip column description inheritance so empty child descriptions can remain empty instead of receiving upstream documentation.

# Context

Issue #326 reports that inherited upstream descriptions can become inaccurate when a column changes meaning, grain, or filtering in downstream models. Current behavior always considers `description` inheritable, unless the child already has a description and `force-inherit-descriptions` is false. There is no inverse option to prevent inheritance into empty child descriptions.

# Why Now

Description inheritance is a core dbt-osmosis feature, but teams need a safe opt-out for projects where inaccurate inherited text is worse than missing documentation.

# Scope

- Add a CLI/config setting such as `skip-inherit-descriptions`.
- Prevent description inheritance when the setting is enabled.
- Preserve tag and meta inheritance behavior when only description inheritance is skipped.
- Define deterministic behavior when skip and force options both appear.
- Add tests for CLI/global setting and config-resolution paths.

# Out Of Scope

- Changing default description inheritance behavior.
- Validating semantic quality of descriptions.
- Rename-aware documentation synthesis.
- Skipping tags, meta, or arbitrary additional inherited keys.

# Acceptance Criteria

- ACC-001: Empty child column descriptions remain empty when skip-description inheritance is enabled.
- ACC-002: Existing child descriptions are preserved when skip-description inheritance is enabled.
- ACC-003: Tags and meta still inherit normally when only description inheritance is skipped.
- ACC-004: Enabling both force and skip description inheritance has clear deterministic behavior, preferably a user-facing error or documented skip precedence.
- ACC-005: The option works through CLI and dbt-osmosis config resolution.

# Coverage

Covers:

- ticket:gh326skip#ACC-001
- ticket:gh326skip#ACC-002
- ticket:gh326skip#ACC-003
- ticket:gh326skip#ACC-004
- ticket:gh326skip#ACC-005
- initiative:issue-pr-zero#OBJ-001
- initiative:issue-pr-zero#OBJ-002
- initiative:issue-pr-zero#OBJ-005

# Claim Matrix

| Claim | Evidence | Critique | Status |
| --- | --- | --- | --- |
| ticket:gh326skip#ACC-001 | evidence:gh326skip-description-inheritance-validation | critique:gh326skip-description-inheritance-review | accepted |
| ticket:gh326skip#ACC-002 | evidence:gh326skip-description-inheritance-validation | critique:gh326skip-description-inheritance-review | accepted |
| ticket:gh326skip#ACC-003 | evidence:gh326skip-description-inheritance-validation | critique:gh326skip-description-inheritance-review | accepted |
| ticket:gh326skip#ACC-004 | evidence:gh326skip-description-inheritance-validation | critique:gh326skip-description-inheritance-review | accepted |
| ticket:gh326skip#ACC-005 | evidence:gh326skip-description-inheritance-validation | critique:gh326skip-description-inheritance-review | accepted |

# Execution Notes

Ralph added `skip-inherit-descriptions`, wired it into CLI/settings, and removes `description` from the inheritable set before force inheritance can overwrite or fill it. Skip wins deterministically over force while tag/meta inheritance remains unchanged.

# Blockers

None.

# Evidence

Evidence status: local red/green Ralph evidence, parent inheritance/settings pytest, Ruff, format check, CLI help observations, and whitespace check support ACC-001 through ACC-005 for the uncommitted implementation diff. Remote CI will be checked at the initiative level after the full batch push per operator direction.

# Critique Disposition

Risk class: medium

Critique policy: recommended

Policy rationale: This changes a central inheritance behavior, although only behind an opt-in setting.

Required critique profiles:

- inheritance-behavior
- cli-config

Findings:

None - critique:gh326skip-description-inheritance-review returned `pass` with no findings.

Disposition status: completed

Deferral / not-required rationale: N/A.

# Retrospective / Promotion Disposition

Disposition status: completed

Promoted:

None - retrospective found no durable explanation needing wiki/research/spec promotion beyond this ticket, evidence, and critique.

Deferred / not-required rationale: New option behavior is captured in the ticket and tests; broader docs can be handled during release documentation if desired.

# Wiki Disposition

N/A - no wiki promotion selected yet.

# Acceptance Decision

Accepted by: OpenCode parent agent.
Accepted at: 2026-05-05T08:07:56Z.
Basis: Local implementation evidence, focused validation, and Oracle critique support ACC-001 through ACC-005. The ticket is ready for issue closure with remote CI deferred to the issue-backlog initiative gate per operator direction.
Residual risks: Remote CI not yet checked; no dedicated end-to-end CLI invocation test was added beyond parent help observations; user-facing docs beyond CLI help do not yet explain skip-vs-force precedence.

# Dependencies

None.

# Journal

- 2026-05-05T06:02:19Z: Created from GitHub issue #326 and Oracle triage as a validated inheritance opt-out feature ticket.
- 2026-05-05T07:41:32Z: Ralph implemented `skip-inherit-descriptions`. Parent validation passed and Oracle critique accepted with no findings. Retrospective completed with no promotion needed beyond ticket/evidence/critique records. Moved to complete_pending_acceptance pending final implementation commit packaging.
- 2026-05-05T08:07:56Z: Accepted and closed locally for combined inheritance-options packaging with ticket:gh333meta. GitHub issue #326 is ready for commit, push, comment, and closure.
