Status: blocked
Created: 2026-07-05
Updated: 2026-07-05
Parent: None
Depends-On: None

# Ratify docs Node 20 Docusaurus upgrade

## Scope

Decide whether the docs toolchain may drop Node 18 support and move to a Node 20+ Docusaurus line in order to fully clear the remaining docs npm audit advisories.

In scope:

- Confirm whether `docs/package.json` may change from `engines.node >=18.0` to `>=20.0`.
- Confirm whether `.github/workflows/tests.yml` may remove Node 18 from the docs build matrix.
- Confirm whether AGENTS/project guidance should be updated from docs Node `>=18` to the ratified minimum.
- If ratified, open or execute a follow-up remediation ticket for Docusaurus `3.10.1` or the then-current fixed version.

Out of scope:

- Changing Python runtime support.
- Changing the docs framework.
- Forcing transitive major overrides that do not match the Docusaurus dependency contract.

## Acceptance Criteria

- ACC-001: The user explicitly confirms or rejects dropping docs Node 18 support.
- ACC-002: If confirmed, the ratified Node minimum and Docusaurus target are stated concretely.
- ACC-003: If rejected, the remaining docs npm audit advisories stay owned by this ticket or a superseding no-action record.

## Evidence Expectations

- Cite the docs audit evidence showing the remaining advisories.
- Cite npm metadata showing the Node engine change for Docusaurus `3.9+`.
- Cite the current docs CI matrix that includes Node 18.

## Progress and Notes

- 2026-07-05: Local npm metadata shows `@docusaurus/core@3.7.0` and `3.8.1` support Node `>=18.0`, while `3.9.0`, `3.9.2`, and `3.10.1` require Node `>=20.0`.
- 2026-07-05: `.github/workflows/tests.yml` docs CI matrix currently includes Node 18 and Node 24.
- 2026-07-05: `npm --prefix docs audit fix --package-lock-only` reports the remaining 20 advisories require `@docusaurus/preset-classic@3.10.1`; that path is not executable without the Node support decision.

## Blockers

- User ratification required: changing the docs minimum supported Node version from 18 to 20 is a platform support decision, not a safe dependency default.

## References

- `.10x/evidence/2026-07-05-docs-npm-audit-remediation.md`
- `.10x/tickets/done/2026-07-04-remediate-docs-npm-audit-vulnerabilities.md`
- `docs/package.json`
- `.github/workflows/tests.yml`
