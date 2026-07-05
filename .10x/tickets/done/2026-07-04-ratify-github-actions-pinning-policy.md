Status: done
Created: 2026-07-04
Updated: 2026-07-05
Parent: .10x/tickets/done/2026-07-04-quality-optimizer-hill-climb.md
Depends-On: .10x/tickets/done/2026-07-04-triage-semgrep-default-findings.md

# Ratify GitHub Actions Pinning Policy

## Scope

Decide whether this project should pin third-party and first-party GitHub Actions references to full 40-character commit SHAs.

In scope:

- 26 Semgrep `github-actions-mutable-action-tag` findings across:
  - `.github/workflows/labeler.yml`
  - `.github/workflows/lint.yml`
  - `.github/workflows/release.yml`
  - `.github/workflows/tests.yml`
- Exact pinning policy:
  - all actions vs selected high-risk release actions,
  - full SHA with comments naming source tags vs tag refs,
  - update cadence and who refreshes pins.

Out of scope:

- Changing release authority, publish conditions, PyPI credentials, or workflow permissions.
- Dependency cooldown policy.
- Semgrep suppression policy.

## Acceptance Criteria

- ACC-001: User ratifies whether all mutable GitHub Action references should be pinned to full SHAs.
- ACC-002: If pinning is required, the implementation scope names which workflows/actions are included.
- ACC-003: Update/maintenance expectations are clear enough to avoid silently freezing CI dependencies forever.
- ACC-004: Implementation resolves the current tag SHAs from upstream, updates workflow `uses:` references, and records verification.

## Progress and Notes

- 2026-07-04: Semgrep default rules reported 26 mutable action references: 2 in `labeler.yml`, 2 in `lint.yml`, 10 in `release.yml`, and 12 in `tests.yml`.
- 2026-07-04: Pinning is a security hardening improvement but changes maintenance and update behavior. The exact policy is blocked until ratified.
- 2026-07-05: User explicitly ratified fixing any nonzero tool output to zero exit code, including policy/config changes required by the tools. Treat all current Semgrep mutable-action findings as authorized implementation scope.
- 2026-07-05: Resolved each reported action tag to its upstream full commit SHA and pinned every workflow `uses:` reference to that SHA, preserving the source tag in an inline comment.
- 2026-07-05: Removed direct `workflow_run.head_sha` checkout from the release workflow; it now checks out main and asserts the checked-out commit matches the successful Tests workflow SHA.
- 2026-07-05: Semgrep `p/default` and `p/security-audit` both exit 0 with 0 findings.

## Blockers

None.

## References

- `.10x/tickets/done/2026-07-04-triage-semgrep-default-findings.md`
- `.10x/evidence/2026-07-04-semgrep-triage.md`
- `.10x/evidence/2026-07-05-semgrep-policy-zero.md`
- `.10x/reviews/2026-07-05-semgrep-policy-zero.md`
