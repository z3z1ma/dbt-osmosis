---
id: ticket:gh326skip
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
| ticket:gh326skip#ACC-001 | None yet | None yet | open |
| ticket:gh326skip#ACC-002 | None yet | None yet | open |
| ticket:gh326skip#ACC-003 | None yet | None yet | open |
| ticket:gh326skip#ACC-004 | None yet | None yet | open |
| ticket:gh326skip#ACC-005 | None yet | None yet | open |

# Execution Notes

Relevant code path: `src/dbt_osmosis/core/transforms.py` starts `inheritable = ["description"]` and only removes it when a local description exists and force inheritance is disabled. Implementation should avoid changing tag/meta inheritance.

# Blockers

None.

# Evidence

Expected evidence: red/green inheritance tests covering empty child descriptions, existing child descriptions, tag/meta inheritance, and option conflict behavior. Remote CI evidence is needed before closure.

# Critique Disposition

Risk class: medium

Critique policy: recommended

Policy rationale: This changes a central inheritance behavior, although only behind an opt-in setting.

Required critique profiles:

- inheritance-behavior
- cli-config

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

- 2026-05-05T06:02:19Z: Created from GitHub issue #326 and Oracle triage as a validated inheritance opt-out feature ticket.
