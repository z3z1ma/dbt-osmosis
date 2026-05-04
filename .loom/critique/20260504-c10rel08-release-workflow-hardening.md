---
id: critique:c10rel08-release-workflow-hardening
kind: critique
status: final
created_at: 2026-05-04T01:50:21Z
updated_at: 2026-05-04T01:50:21Z
scope:
  kind: repository
  repositories:
    - repo:root
review_target: "ticket:c10rel08 release workflow hardening diff"
links:
  tickets:
    - ticket:c10rel08
  packets:
    - packet:ralph-ticket-c10rel08-20260504T012824Z
  evidence:
    - evidence:c10rel08-local-release-workflow-validation
external_refs:
  github_issue: https://github.com/z3z1ma/dbt-osmosis/issues/359
---

# Summary

Mandatory high-risk release-packaging critique for the `ticket:c10rel08` release workflow hardening.

# Review Target

Reviewed the current uncommitted `.github/workflows/release.yml` diff after the Ralph implementation and parent follow-up fixes. The workflow now triggers from successful same-repository push runs of the `Tests` workflow on `main`, validates in a read-only job, and reserves write credentials for a separate tag/publish job after validation.

# Verdict

`pass`.

The final workflow shape resolves the blockers found during critique: validation no longer runs with persisted write credentials, Release is gated on the accepted `Tests` workflow, and `workflow_run` is narrowed to successful same-repository push runs on `main`.

# Findings

None open.

Pre-final review challenges resolved before this verdict:

- Validation originally had workflow-level `contents: write` and checkout-persisted credentials before validation completed. Parent split validation and release into separate jobs, made validation read-only, and set `persist-credentials: false` in validation checkout.
- Release originally ran directly on push without waiting for the full `Tests` workflow. Parent changed Release to trigger from successful `Tests` workflow runs.
- The first `workflow_run` follow-up could have been triggered by scheduled or pull request `Tests` runs. Parent added explicit guards for `workflow_run.event == 'push'`, `head_branch == 'main'`, and same-repository head source.

# Evidence Reviewed

- Current `.github/workflows/release.yml`.
- Current workflow diff.
- `ticket:c10rel08`.
- `packet:ralph-ticket-c10rel08-20260504T012824Z`.
- `evidence:c10rel08-local-release-workflow-validation`.
- Parent-reported `pre-commit run --files .github/workflows/release.yml` pass after the final guard fix.
- Read-only reviewer checks: `actionlint .github/workflows/release.yml` and `git diff --check -- .github/workflows/release.yml`, both with no errors.

# Residual Risks

- Live GitHub/PyPI behavior still needs post-push workflow evidence.
- PyPI token scope/project restriction is external and not verifiable from repository files.
- If PyPI publish fails after tag creation, a validated tag may still exist; this ticket's gate prevents validation failures from tagging, not atomic publish rollback.
- Release Drafter remains the GitHub release notes source; changelog/release-note process remains process-fragile but documented enough for this ticket.

# Required Follow-up

- Commit and push the release workflow hardening.
- Capture final GitHub Actions evidence from `main` before acceptance or closure.

# Acceptance Recommendation

`ticket-acceptance-review-needed`.

No critique blocker remains, but the ticket still needs final `main` workflow evidence.
