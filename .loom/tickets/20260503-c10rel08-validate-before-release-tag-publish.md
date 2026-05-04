---
id: ticket:c10rel08
kind: ticket
status: closed
change_class: release-packaging
risk_class: high
created_at: 2026-05-03T21:10:43Z
updated_at: 2026-05-04T02:40:27Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  initiative:
    - initiative:dbt-110-111-hardening
  evidence:
    - evidence:oracle-backlog-scan
    - evidence:c10rel08-local-release-workflow-validation
    - evidence:c10rel08-main-release-detached-head-failure
    - evidence:c10rel08-main-release-success
  critique:
    - critique:c10rel08-release-workflow-hardening
  packets:
    - packet:ralph-ticket-c10rel08-20260504T012824Z
  wiki:
    - wiki:release-publishing-workflow
depends_on: []
---

# Summary

Rework the release workflow so build, metadata, install smoke, tests, and docs validation happen before any irreversible version tag or PyPI publish.

# Context

`.github/workflows/release.yml:39-58` detects and tags a new version before package build and publish. A bad version bump can leave a tag behind even if `uvx hatchling build` or publishing fails. Release notes are published later by Release Drafter, while changelog fragments exist separately.

# Why Now

The build and release system is part of the compatibility promise. Publishing or tagging without validation makes it possible to ship broken dbt 1.10/1.11 support or broken package metadata.

# Scope

- Move build, metadata check, wheel install smoke, entrypoint smoke, and relevant tests before tag creation or publish.
- Add explicit workflow permissions.
- Prefer trusted publishing/OIDC if feasible.
- Ensure failed validation cannot create a version tag or GitHub release.
- Reconcile release notes/changelog ownership enough that generated release notes match the project process.

# Out Of Scope

- Performing an actual release.
- Large changelog process redesign beyond preventing split-brain release output; ticket:c10pkg10 may own broader packaging/dependency cleanup.

# Acceptance Criteria

- ACC-001: Release workflow builds sdist/wheel and validates metadata before tag creation.
- ACC-002: A clean install from the built wheel verifies `dbt-osmosis --help`, `python -m dbt_osmosis`, and importing `dbt_osmosis.cli.main`.
- ACC-003: Required test/docs/package checks must pass before PyPI publish.
- ACC-004: A failed build or test cannot leave a new version tag or published GitHub release.
- ACC-005: Workflow permissions and PyPI auth are explicit and least-privilege for the chosen publishing method.
- ACC-006: Release notes/changelog source of truth is documented or enforced.

# Coverage

Covers:

- ticket:c10rel08#ACC-001
- ticket:c10rel08#ACC-002
- ticket:c10rel08#ACC-003
- ticket:c10rel08#ACC-004
- ticket:c10rel08#ACC-005
- ticket:c10rel08#ACC-006
- initiative:dbt-110-111-hardening#OBJ-005

# Claim Matrix

| Claim | Evidence | Critique | Status |
| --- | --- | --- | --- |
| ticket:c10rel08#ACC-001 | evidence:c10rel08-local-release-workflow-validation; evidence:c10rel08-main-release-success | critique:c10rel08-release-workflow-hardening | accepted |
| ticket:c10rel08#ACC-002 | evidence:c10rel08-local-release-workflow-validation; evidence:c10rel08-main-release-success | critique:c10rel08-release-workflow-hardening | accepted |
| ticket:c10rel08#ACC-003 | evidence:c10rel08-local-release-workflow-validation; evidence:c10rel08-main-release-detached-head-failure; evidence:c10rel08-main-release-success | critique:c10rel08-release-workflow-hardening | accepted |
| ticket:c10rel08#ACC-004 | evidence:oracle-backlog-scan; evidence:c10rel08-local-release-workflow-validation; evidence:c10rel08-main-release-detached-head-failure; evidence:c10rel08-main-release-success | critique:c10rel08-release-workflow-hardening | accepted |
| ticket:c10rel08#ACC-005 | evidence:c10rel08-local-release-workflow-validation; evidence:c10rel08-main-release-success | critique:c10rel08-release-workflow-hardening | accepted with external PyPI-token residual risk |
| ticket:c10rel08#ACC-006 | evidence:c10rel08-local-release-workflow-validation; evidence:c10rel08-main-release-success; wiki:release-publishing-workflow | critique:c10rel08-release-workflow-hardening | accepted |

# Execution Notes

Use non-destructive dry-run or workflow reasoning for validation. Do not push tags or publish artifacts while implementing this ticket unless explicitly authorized by the user.

# Blockers

None.

# Evidence

Existing evidence: evidence:oracle-backlog-scan; evidence:c10rel08-local-release-workflow-validation; evidence:c10rel08-main-release-detached-head-failure; evidence:c10rel08-main-release-success. Missing evidence: none for ticket acceptance.

# Critique Disposition

Risk class: high

Critique policy: mandatory

Policy rationale: Release workflow changes can affect irreversible tags and published packages.

Required critique profiles: release-packaging, operator-clarity, security

Findings: None open. Pre-final findings about validation credentials, missing full Tests workflow gate, broad `workflow_run` triggers, and detached-SHA tag detection were resolved before final verdict.

Disposition status: completed

Deferral / not-required rationale: None. Mandatory critique passed in `critique:c10rel08-release-workflow-hardening`, including the branch-attach follow-up after the first live Release run failed in post-validation tag detection.

# Retrospective / Promotion Disposition

Disposition status: completed

Promoted: `wiki:release-publishing-workflow` explains the accepted post-Tests release publishing gate, least-privilege validation boundary, same-repository push guards, local branch attach requirement, no-tag behavior, and residual release risks.

Deferred / not-required rationale: no research/spec/plan/constitution promotion needed; the ticket-owned acceptance criteria were sufficient and no durable policy changed.

# Wiki Disposition

Completed: `wiki:release-publishing-workflow`.

# Acceptance Decision

Accepted by: OpenCode.
Accepted at: 2026-05-04T02:40:27Z.
Basis: `evidence:c10rel08-main-release-success` shows commit `8058f91d3faf90300ad571db11522bee7e02c45d` passed `lint`, `Tests`, and Release. Mandatory critique passed in `critique:c10rel08-release-workflow-hardening`, including the branch-attach follow-up.
Residual risks: PyPI token scope/project restriction is external to repository evidence. The observed Release run covered no-tag behavior, not a real PyPI/GitHub release publish. Tag creation, PyPI publishing, and GitHub release publishing are not atomic if a future tagged release fails after tag creation.

# Dependencies

Coordinate with ticket:c10docs09 and ticket:c10pkg10 if package/docs checks become release gates.

# Journal

- 2026-05-03T21:10:43Z: Created from CI/build oracle finding.
- 2026-05-04T01:28:24Z: Activated ticket and compiled Ralph packet `packet:ralph-ticket-c10rel08-20260504T012824Z` for release workflow validation-before-tag/publish implementation.
- 2026-05-04T01:35:08Z: Consumed Ralph packet after `.github/workflows/release.yml` was updated to gate version tagging, PyPI publishing, and GitHub release-note publishing behind lock/package/test/docs/build/metadata/wheel-smoke checks; recorded local evidence and moved to mandatory critique.
- 2026-05-04T01:50:21Z: Mandatory critique passed after parent follow-up split release validation into a read-only `validate` job, restricted write credentials to the post-validation `release` job, and narrowed `workflow_run` to successful same-repository push runs of `Tests` on `main`. Final `main` GitHub Actions evidence remains pending before acceptance.
- 2026-05-04T02:15:26Z: First pushed Release run validated successfully but failed in post-validation tag detection because `action-detect-and-tag-new-version` expected local `refs/heads/main` and the workflow checked out the tested SHA detached. Recorded red evidence and added a local branch attach step before tag detection.
- 2026-05-04T02:18:25Z: Mandatory follow-up critique passed the branch-attach fix with no open findings. Final green Release evidence remains pending after the next push.
- 2026-05-04T02:40:27Z: Follow-up commit `8058f91d3faf90300ad571db11522bee7e02c45d` passed `lint`, `Tests`, and Release on `main`; Release validated the tested commit, completed tag detection and no-tag developmental build, and skipped PyPI/Release Drafter publish steps. Promoted accepted workflow explanation to `wiki:release-publishing-workflow` and closed the ticket.
