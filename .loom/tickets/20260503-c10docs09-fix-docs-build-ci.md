---
id: ticket:c10docs09
kind: ticket
status: closed
change_class: release-packaging
risk_class: high
created_at: 2026-05-03T21:10:43Z
updated_at: 2026-05-04T03:24:38Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  initiative:
    - initiative:dbt-110-111-hardening
  evidence:
    - evidence:oracle-backlog-scan
    - evidence:c10docs09-local-docs-ci-validation
    - evidence:c10docs09-main-docs-ci-success
  critique:
    - critique:c10docs09-docs-ci-hardening
  packets:
    - packet:ralph-ticket-c10docs09-20260504T025259Z
  wiki:
    - wiki:ci-compatibility-matrix
    - wiki:release-publishing-workflow
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
| ticket:c10docs09#ACC-001 | evidence:c10docs09-local-docs-ci-validation; evidence:c10docs09-main-docs-ci-success | critique:c10docs09-docs-ci-hardening | supported; local install passed and GitHub Actions Node 18/24 `npm ci` passed |
| ticket:c10docs09#ACC-002 | evidence:c10docs09-local-docs-ci-validation; evidence:c10docs09-main-docs-ci-success | critique:c10docs09-docs-ci-hardening | supported; local build passed and GitHub Actions Node 18/24 docs builds passed |
| ticket:c10docs09#ACC-003 | evidence:oracle-backlog-scan; evidence:c10docs09-local-docs-ci-validation | critique:c10docs09-docs-ci-hardening | supported; config is consistently CommonJS |
| ticket:c10docs09#ACC-004 | evidence:c10docs09-local-docs-ci-validation; evidence:c10docs09-main-docs-ci-success | critique:c10docs09-docs-ci-hardening | supported with documented peer-only TypeScript lockfile drift |
| ticket:c10docs09#ACC-005 | evidence:c10docs09-local-docs-ci-validation; evidence:c10docs09-main-docs-ci-success | critique:c10docs09-docs-ci-hardening | supported; docs job is in the pull_request-capable `Tests` workflow and passed on push |
| ticket:c10docs09#ACC-006 | evidence:c10docs09-local-docs-ci-validation; evidence:c10docs09-main-docs-ci-success | critique:c10docs09-docs-ci-hardening | supported; `/docs` npm Dependabot coverage is configured and dynamic run passed |

# Execution Notes

Prefer the smallest config/dependency change that gets Docusaurus building. Do not fold broad docs content rewrites into this ticket.

# Blockers

None.

# Evidence

Sufficient evidence: evidence:oracle-backlog-scan; evidence:c10docs09-local-docs-ci-validation; evidence:c10docs09-main-docs-ci-success. Missing evidence: none for this ticket's acceptance criteria.

# Critique Disposition

Risk class: high

Critique policy: mandatory

Policy rationale: Docs build failure can block releases and migration guidance; dependency changes affect supply-chain surface.

Required critique profiles: release-packaging, operator-clarity

Findings: Low finding about peer-only TypeScript lockfile drift is documented and accepted as a residual for this ticket. The Node 18/24 CI evidence gate was resolved by `evidence:c10docs09-main-docs-ci-success`.

Disposition status: completed

Deferral / not-required rationale: None. Mandatory critique passed for commit/push trial in `critique:c10docs09-docs-ci-hardening`.

# Retrospective / Promotion Disposition

Disposition status: completed

Promoted: `wiki:ci-compatibility-matrix` now records the docs build CI job as part of the `Tests` workflow, and `wiki:release-publishing-workflow` now records that Release is indirectly gated by the docs job before release-local validation runs.

Deferred / not-required rationale: Existing docs npm audit findings and broader dependency hygiene remain outside this ticket's scope.

# Wiki Disposition

Completed: updated `wiki:ci-compatibility-matrix` and `wiki:release-publishing-workflow` with the accepted docs CI/release-gate shape.

# Acceptance Decision

Accepted by: OpenCode.
Accepted at: 2026-05-04T03:24:38Z.
Basis: Local docs install/dependency/build evidence, mandatory critique, final GitHub Actions `Tests` success with Node 18/24 docs jobs, and docs npm Dependabot dynamic success.
Residual risks: Docusaurus ecosystem peer dependency changes can recur; existing docs npm audit findings remain out of scope; peer-only TypeScript lockfile drift is documented as a non-blocking residual.

# Dependencies

Coordinated with ticket:c10rel08 through `wiki:release-publishing-workflow`; docs build now gates Release through the upstream `Tests` workflow and release-local validation.

# Journal

- 2026-05-03T21:10:43Z: Created from CI/build oracle finding.
- 2026-05-04T02:52:59Z: Activated ticket and compiled Ralph packet `packet:ralph-ticket-c10docs09-20260504T025259Z` for docs config/dependency/CI implementation.
- 2026-05-04T03:02:26Z: Consumed Ralph implementation after docs config was made consistently CommonJS, React/React-DOM were aligned to 18.3.1, the `Tests` workflow gained a Node 18/24 docs job, Dependabot gained `/docs` npm coverage, and local docs install/dependency/build plus hooks passed. Moved to mandatory critique.
- 2026-05-04T03:07:28Z: Mandatory critique passed for commit/push trial. Low TypeScript lockfile drift was documented as peer-only and non-blocking; Node 18/24 GitHub Actions docs evidence remains pending before acceptance.
- 2026-05-04T03:24:38Z: Final GitHub Actions evidence passed for commit `12e9dfee122db41ddb8f85072e1904ecd079dd00`: full `Tests` workflow succeeded, docs builds passed on Node 18 and Node 24, and the docs npm Dependabot dynamic run succeeded. Retrospective promoted docs CI and release-gate notes into wiki; accepted and closed the ticket.
