Status: open
Created: 2026-07-04
Updated: 2026-07-04
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

## Blockers

None known. This is executable after a separate Inner Loop entry; it was not implemented inside the quality optimizer closure because it is a newly discovered non-trivial ticket.

## References

- `.10x/evidence/2026-07-04-github-dependabot-alerts.md`
- `.10x/tickets/done/2026-07-04-remediate-uv-audit-vulnerabilities.md`
