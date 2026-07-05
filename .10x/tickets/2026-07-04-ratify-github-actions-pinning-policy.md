Status: blocked
Created: 2026-07-04
Updated: 2026-07-04
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

## Blockers

Blocked on user ratification of GitHub Actions pinning policy.

Recommended ratification question:

Should this repository pin every GitHub Action reference reported by Semgrep to the current full commit SHA, keeping a trailing comment with the original tag for maintainability and requiring explicit future pin refreshes?

## References

- `.10x/tickets/done/2026-07-04-triage-semgrep-default-findings.md`
- `.10x/evidence/2026-07-04-semgrep-triage.md`
