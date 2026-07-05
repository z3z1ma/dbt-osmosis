Status: done
Created: 2026-07-04
Updated: 2026-07-05
Parent: .10x/tickets/done/2026-07-04-quality-optimizer-hill-climb.md
Depends-On: .10x/tickets/done/2026-07-04-triage-semgrep-default-findings.md

# Ratify Dependency Cooldown Policy

## Scope

Decide whether this project should delay newly published dependency versions before automated or local resolution.

In scope:

- Dependabot cooldown policy for `.github/dependabot.yml` ecosystems:
  - `pip` at `/`
  - `npm` at `/docs`
- uv resolver cooldown policy for `[tool.uv]` in `pyproject.toml`.
- Exact cooldown duration and override expectations.

Out of scope:

- General dependency update cadence redesign.
- Dependency vulnerability remediation that is already covered by audit-specific tickets.
- GitHub Actions SHA pinning policy.

## Acceptance Criteria

- ACC-001: User ratifies whether dependency cooldowns are required.
- ACC-002: If required, the exact duration is ratified for Dependabot and uv separately or as one shared value.
- ACC-003: Override expectations are clear for urgent security fixes and release-critical dependency updates.
- ACC-004: Implementation ticket updates `.github/dependabot.yml` and/or `pyproject.toml` only after the policy is ratified.

## Progress and Notes

- 2026-07-04: Semgrep default rules reported missing Dependabot cooldowns at `.github/dependabot.yml:8` and `.github/dependabot.yml:12`, and missing uv cooldown at `pyproject.toml:71`.
- 2026-07-04: Semgrep recommends 7 days for both systems, but 10x treats dependency cooldown duration as a policy default requiring explicit ratification before implementation.
- 2026-07-05: User explicitly ratified fixing any nonzero tool output to zero exit code, including policy/config changes required by the tools. Treat Semgrep's 7-day dependency cooldown recommendation as authorized implementation scope.
- 2026-07-05: Added `cooldown.default-days: 7` to both Dependabot ecosystems and `exclude-newer = "7 days"` under `[tool.uv]`. `uv.lock` was refreshed to record the resolver cooldown metadata.
- 2026-07-05: Semgrep `p/default` and `p/security-audit` both exit 0 with 0 findings.

## Blockers

None.

## References

- `.10x/tickets/done/2026-07-04-triage-semgrep-default-findings.md`
- `.10x/evidence/2026-07-04-semgrep-triage.md`
- `.10x/evidence/2026-07-05-semgrep-policy-zero.md`
- `.10x/reviews/2026-07-05-semgrep-policy-zero.md`
