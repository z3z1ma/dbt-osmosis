---
id: ticket:c10docs09
kind: ticket
status: ready
change_class: release-packaging
risk_class: high
created_at: 2026-05-03T21:10:43Z
updated_at: 2026-05-03T21:10:43Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  initiative:
    - initiative:dbt-110-111-hardening
  evidence:
    - evidence:oracle-backlog-scan
depends_on: []
---

# Summary

Fix the Docusaurus docs build surface and add CI coverage so docs do not silently rot while CLI/config behavior changes for dbt 1.10/1.11.

# Context

`docs/docusaurus.config.js:4` uses ESM `import`, while `docs/docusaurus.config.js:34`, `39`, and `125` still use CommonJS `require.resolve` and `module.exports`. `docs/package.json:17-23` pins Docusaurus 3.7.0 with React 19 dependencies, which the oracle audit flagged as peer-dependency risk. No current workflow builds docs.

# Why Now

The dbt 1.10/1.11 backlog will require user-facing docs updates. If the docs site cannot build in CI, compatibility and migration guidance can ship broken or go unvalidated.

# Scope

- Convert Docusaurus config to a consistent module style.
- Align React/React-DOM versions with the selected Docusaurus version, or upgrade Docusaurus to a React-19-compatible line.
- Add a docs CI job that runs `npm ci`, dependency validation, and `npm run build` in `docs/`.
- Ensure Node version matches documented `engines` and CI setup.
- Add Dependabot coverage for docs npm dependencies if not handled by another ticket.

# Out Of Scope

- Rewriting docs content except build-related fixes.
- Redesigning the docs site visual system.

# Acceptance Criteria

- ACC-001: `npm --prefix docs ci` succeeds on Node 18 and current LTS.
- ACC-002: `npm --prefix docs run build` succeeds in CI.
- ACC-003: Docusaurus config no longer mixes incompatible ESM/CJS runtime patterns.
- ACC-004: React and Docusaurus peer dependencies are valid according to `npm ls` or an equivalent check.
- ACC-005: CI includes a docs build job on pull requests.
- ACC-006: Dependabot or another update mechanism covers docs npm dependencies.

# Coverage

Covers:

- ticket:c10docs09#ACC-001
- ticket:c10docs09#ACC-002
- ticket:c10docs09#ACC-003
- ticket:c10docs09#ACC-004
- ticket:c10docs09#ACC-005
- ticket:c10docs09#ACC-006
- initiative:dbt-110-111-hardening#OBJ-005

# Claim Matrix

| Claim | Evidence | Critique | Status |
| --- | --- | --- | --- |
| ticket:c10docs09#ACC-003 | evidence:oracle-backlog-scan | None | open |
| ticket:c10docs09#ACC-002 | None - docs build not run after fix yet | None | open |

# Execution Notes

Prefer the smallest config/dependency change that gets Docusaurus building. Do not fold broad docs content rewrites into this ticket.

# Blockers

None.

# Evidence

Existing evidence: evidence:oracle-backlog-scan. Missing evidence: successful docs install/build logs.

# Critique Disposition

Risk class: high

Critique policy: mandatory

Policy rationale: Docs build failure can block releases and migration guidance; dependency changes affect supply-chain surface.

Required critique profiles: release-packaging, operator-clarity

Findings: None - no critique yet.

Disposition status: pending

Deferral / not-required rationale: None.

# Retrospective / Promotion Disposition

Disposition status: pending

Promoted: None - implementation not complete.

Deferred / not-required rationale: Consider updating wiki/release docs if docs CI becomes a release gate.

# Wiki Disposition

N/A - no wiki promotion selected yet.

# Acceptance Decision

Accepted by: Not accepted yet.
Accepted at: N/A.
Basis: Pending CI/build evidence.
Residual risks: Docusaurus ecosystem peer dependency changes can recur.

# Dependencies

Coordinate with ticket:c10rel08 if docs build becomes a release gate.

# Journal

- 2026-05-03T21:10:43Z: Created from CI/build oracle finding.
