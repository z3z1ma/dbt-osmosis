Status: done
Created: 2026-07-04
Updated: 2026-07-05
Parent: None
Depends-On: None

# Remediate docs npm audit vulnerabilities

## Scope

Update the docs site npm dependency lock with the smallest safe changes needed to clear or materially reduce current `npm --prefix docs audit` vulnerabilities.

In scope:

- Upgrade Docusaurus docs dependencies along the non-major path reported by npm audit, currently expected around `@docusaurus/*@3.10.1`.
- Update `docs/package.json` and `docs/package-lock.json` only as needed by the resolver.
- Reassess existing `overrides` after the update; remove or adjust only if the resolver proves they are obsolete or harmful.
- Verify docs install/build and npm audit outcome.

Out of scope:

- Migrating docs framework, changing docs content, or redesigning the docs site.
- Updating unrelated root Python dependencies.
- GitHub Actions pinning or dependency cooldown policy.

## Acceptance Criteria

- ACC-001: `npm --prefix docs audit --json` exits 0, or any remaining advisories are explicitly classified with a durable owner/no-action rationale.
- ACC-002: `npm --prefix docs ci` exits 0 from the updated lockfile.
- ACC-003: `npm --prefix docs run build` exits 0.
- ACC-004: The diff is limited to docs dependency metadata unless a directly required compatibility fix is proven.
- ACC-005: Any Python/root gates affected by metadata changes are rerun or explicitly ruled unaffected.

## Evidence Expectations

- Capture before/after npm audit summaries.
- Capture npm install/update command used.
- Capture docs build output.
- Capture diff summary for docs dependency files.

## Progress and Notes

- 2026-07-04: GitHub Dependabot alert inspection plus local `npm --prefix docs audit --json` reproduced 26 docs npm vulnerabilities: 2 high, 23 moderate, 1 low.
- 2026-07-04: npm audit reports a non-major remediation path through Docusaurus `3.10.1`.
- 2026-07-04: Started execution on branch `codex/quality-optimizer-hill-climb`.
- 2026-07-05: Checked Docusaurus package metadata and rejected the apparent `3.10.1` path for this ticket because `@docusaurus/core@3.9+` requires Node `>=20.0` while the repo docs contract still includes Node 18.
- 2026-07-05: Updated the existing docs `qs` override from `6.14.1` to `6.15.3` and regenerated `docs/package-lock.json`.
- 2026-07-05: After the safe remediation, `npm --prefix docs audit --json` reports 20 vulnerabilities: 19 moderate and 1 high. Remaining advisories are owned by `.10x/tickets/2026-07-05-ratify-docs-node20-docusaurus-upgrade.md`.
- 2026-07-05: `npm --prefix docs ci` and `npm --prefix docs run build` both exited 0. Evidence recorded in `.10x/evidence/2026-07-05-docs-npm-audit-remediation.md`.
- 2026-07-05: Python/root gates were ruled unaffected because the diff is limited to docs npm dependency metadata and 10x records.
- 2026-07-05: Closure review recorded in `.10x/reviews/2026-07-05-docs-npm-audit-remediation-review.md`.

## Blockers

None for this ticket. Full docs audit clearance is blocked on the docs Node 20 support decision owned by `.10x/tickets/2026-07-05-ratify-docs-node20-docusaurus-upgrade.md`.

## References

- `.10x/evidence/2026-07-04-github-dependabot-alerts.md`
- `.10x/evidence/2026-07-05-docs-npm-audit-remediation.md`
- `.10x/reviews/2026-07-05-docs-npm-audit-remediation-review.md`
- `.10x/tickets/2026-07-05-ratify-docs-node20-docusaurus-upgrade.md`
- `.10x/tickets/done/2026-07-04-remediate-uv-audit-vulnerabilities.md`
