---
id: ticket:c10rel08
kind: ticket
status: active
change_class: release-packaging
risk_class: high
created_at: 2026-05-03T21:10:43Z
updated_at: 2026-05-04T01:50:21Z
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
  critique:
    - critique:c10rel08-release-workflow-hardening
  packets:
    - packet:ralph-ticket-c10rel08-20260504T012824Z
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
| ticket:c10rel08#ACC-001 | evidence:c10rel08-local-release-workflow-validation | critique:c10rel08-release-workflow-hardening | locally supported; final main CI pending |
| ticket:c10rel08#ACC-002 | evidence:c10rel08-local-release-workflow-validation | critique:c10rel08-release-workflow-hardening | locally supported; final main CI pending |
| ticket:c10rel08#ACC-003 | evidence:c10rel08-local-release-workflow-validation | critique:c10rel08-release-workflow-hardening | partially supported; full workflow CI pending |
| ticket:c10rel08#ACC-004 | evidence:oracle-backlog-scan; evidence:c10rel08-local-release-workflow-validation | critique:c10rel08-release-workflow-hardening | locally supported; final main CI pending |
| ticket:c10rel08#ACC-005 | evidence:c10rel08-local-release-workflow-validation | critique:c10rel08-release-workflow-hardening | supported by workflow review; final main CI pending |
| ticket:c10rel08#ACC-006 | evidence:c10rel08-local-release-workflow-validation | critique:c10rel08-release-workflow-hardening | supported by workflow review; final main CI pending |

# Execution Notes

Use non-destructive dry-run or workflow reasoning for validation. Do not push tags or publish artifacts while implementing this ticket unless explicitly authorized by the user.

# Blockers

Final GitHub Actions evidence is pending before acceptance. Trusted publishing was not adopted because it may require external PyPI/repository setup; token-based PyPI publishing remains explicit.

# Evidence

Existing evidence: evidence:oracle-backlog-scan; evidence:c10rel08-local-release-workflow-validation. Missing evidence: final GitHub Actions evidence from `main`.

# Critique Disposition

Risk class: high

Critique policy: mandatory

Policy rationale: Release workflow changes can affect irreversible tags and published packages.

Required critique profiles: release-packaging, operator-clarity, security

Findings: None open. Pre-final findings about validation credentials, missing full Tests workflow gate, and broad `workflow_run` triggers were resolved before final verdict.

Disposition status: completed

Deferral / not-required rationale: None. Mandatory critique passed in `critique:c10rel08-release-workflow-hardening`.

# Retrospective / Promotion Disposition

Disposition status: pending

Promoted: None - implementation not complete.

Deferred / not-required rationale: Consider wiki/release process update after acceptance.

# Wiki Disposition

N/A - no wiki promotion selected yet.

# Acceptance Decision

Accepted by: Not accepted yet.
Accepted at: N/A.
Basis: Pending workflow evidence and review.
Residual risks: GitHub/PyPI environment behavior may require live verification.

# Dependencies

Coordinate with ticket:c10docs09 and ticket:c10pkg10 if package/docs checks become release gates.

# Journal

- 2026-05-03T21:10:43Z: Created from CI/build oracle finding.
- 2026-05-04T01:28:24Z: Activated ticket and compiled Ralph packet `packet:ralph-ticket-c10rel08-20260504T012824Z` for release workflow validation-before-tag/publish implementation.
- 2026-05-04T01:35:08Z: Consumed Ralph packet after `.github/workflows/release.yml` was updated to gate version tagging, PyPI publishing, and GitHub release-note publishing behind lock/package/test/docs/build/metadata/wheel-smoke checks; recorded local evidence and moved to mandatory critique.
- 2026-05-04T01:50:21Z: Mandatory critique passed after parent follow-up split release validation into a read-only `validate` job, restricted write credentials to the post-validation `release` job, and narrowed `workflow_run` to successful same-repository push runs of `Tests` on `main`. Final `main` GitHub Actions evidence remains pending before acceptance.
