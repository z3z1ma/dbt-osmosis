---
id: evidence:c10docs09-local-docs-ci-validation
kind: evidence
status: recorded
created_at: 2026-05-04T03:02:26Z
updated_at: 2026-05-04T03:24:38Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  tickets:
    - ticket:c10docs09
  packets:
    - packet:ralph-ticket-c10docs09-20260504T025259Z
  critique:
    - critique:c10docs09-docs-ci-hardening
  evidence:
    - evidence:c10docs09-main-docs-ci-success
external_refs:
  github_issue: https://github.com/z3z1ma/dbt-osmosis/issues/360
---

# Summary

Observed local validation for the `ticket:c10docs09` docs build CI implementation. The docs configuration is now consistently CommonJS, React/React-DOM are aligned to `18.3.1` for Docusaurus 3.7.0, the `Tests` workflow includes a Node 18/24 docs build job, and Dependabot covers `/docs` npm dependencies.

# Procedure

Observed at: 2026-05-04T03:02:26Z.

Source state: local worktree at `5512c32faa57f5dd615a5eeece585e0b18b5b967` plus uncommitted `ticket:c10docs09` changes to docs config/package files, `.github/workflows/tests.yml`, `.github/dependabot.yml`, this evidence record, the Ralph packet, and the ticket.

Procedure:

- Ralph child inspected the pre-change Docusaurus config and observed the mixed ESM/CommonJS shape.
- Ralph child ran pre-change docs checks where feasible: `npm --prefix docs ci` passed with React 19 peer override warnings and `npm --prefix docs run build` passed; `npm --prefix docs ls` before install failed because `node_modules` was absent.
- Ralph child changed only files in child write scope.
- Parent inspected the diff, adjusted the docs CI `setup-node` action to `actions/setup-node@v4.0.4`, and added npm cache configuration using `docs/package-lock.json`.
- Parent ran docs install, dependency validation, build, and pre-commit checks locally.

Expected result when applicable: docs install, peer/dependency validation, and docs build should pass locally; CI should include Node 18/current-LTS docs build coverage; Docusaurus config should use one module runtime style; Dependabot should cover docs npm dependencies.

Actual observed result: local docs install, `npm ls`, docs build, and hooks passed. `npm ci` no longer emitted the React 19 peer override warnings after React/React-DOM were changed to `^18.3.1`. `npm audit` still reported existing vulnerabilities. Critique noted that npm also moved the peer-only TypeScript lock entry from `5.7.2` to `6.0.3`; parent inspected `npm --prefix docs explain typescript` and confirmed TypeScript is an optional peer selected through the Docusaurus dependency chain.

Procedure verdict / exit code: pass for local docs install, dependency tree, docs build, and hooks; mandatory critique and GitHub Actions evidence remain pending.

# Artifacts

Ralph child reported:

- Local Node: `v22.22.1`; npm: `10.9.4`.
- Before install, `npm --prefix docs ls` failed because `node_modules` was absent.
- Before change, `npm --prefix docs ci` passed with React 19 peer override warnings.
- Before change, `npm --prefix docs run build` passed.
- After change, `npm --prefix docs ci` passed.
- After change, `npm --prefix docs ls` passed with React and React DOM `18.3.1`.
- After change, `npm --prefix docs run build` passed.
- `pre-commit run --files docs/docusaurus.config.js docs/package.json docs/package-lock.json .github/workflows/tests.yml .github/dependabot.yml` passed.

Parent verification reran:

- `npm --prefix docs ci` - passed; installed 1292 packages and reported 30 audit vulnerabilities.
- `npm --prefix docs ls` - passed; reported Docusaurus packages `3.7.0`, React `18.3.1`, and React DOM `18.3.1`.
- `npm --prefix docs run build` - passed and generated static files in `docs/build`.
- `pre-commit run --files docs/docusaurus.config.js docs/package.json docs/package-lock.json .github/workflows/tests.yml .github/dependabot.yml` - passed, including YAML checks and actionlint.
- `npm --prefix docs explain typescript` - TypeScript is peer-only through Docusaurus' webpack/cosmiconfig dependency chain.

# Supports Claims

- `ticket:c10docs09#ACC-001` - local `npm --prefix docs ci` passed; CI now encodes Node 18 and Node 24 docs install coverage.
- `ticket:c10docs09#ACC-002` - local `npm --prefix docs run build` passed and CI now encodes docs build coverage.
- `ticket:c10docs09#ACC-003` - `docs/docusaurus.config.js` is consistently CommonJS with `require(...)`, `require.resolve(...)`, and `module.exports`.
- `ticket:c10docs09#ACC-004` - local `npm --prefix docs ls` passed with Docusaurus 3.7.0 and React/React-DOM 18.3.1.
- `ticket:c10docs09#ACC-005` - `.github/workflows/tests.yml` includes a docs job under the existing push/pull_request/schedule workflow trigger.
- `ticket:c10docs09#ACC-006` - `.github/dependabot.yml` includes npm updates for `/docs`.

# Challenges Claims

- `ticket:c10docs09#ACC-001` and `ticket:c10docs09#ACC-002` remain partially supported until GitHub Actions runs the Node 18/24 docs job.

# Environment

Commit: `5512c32faa57f5dd615a5eeece585e0b18b5b967` plus uncommitted c10docs09 changes.

Branch: local worktree branch `loom/dbt-110-111-hardening`; delivery target remains `main`.

Runtime: local macOS/darwin, Node `v22.22.1`, npm `10.9.4`.

Relevant config: `docs/docusaurus.config.js`, `docs/package.json`, `docs/package-lock.json`, `.github/workflows/tests.yml`, `.github/dependabot.yml`.

# Validity

Valid for: supporting mandatory critique and commit/push trial for `ticket:c10docs09`.

Fresh enough for: review of the current uncommitted docs config/dependency/CI diff.

Recheck when: docs dependencies, Docusaurus config, docs CI matrix, Node support, or Dependabot configuration changes.

Invalidated by: failing GitHub Actions docs job on Node 18 or Node 24, hidden peer dependency issue found by critique, or a later docs dependency lockfile change.

Supersedes / superseded by: superseded for Node 18/24 GitHub Actions claims by `evidence:c10docs09-main-docs-ci-success`.

# Limitations

- Local verification used Node `v22.22.1`; Node 18 and Node 24 validation is encoded in CI but was not executed locally.
- `npm audit` still reports 30 vulnerabilities; this ticket did not expand into dependency/security remediation beyond the React/Docusaurus peer-alignment fix and existing overrides.
- `docs/package-lock.json` also changed peer-only TypeScript from `5.7.2` to `6.0.3`; mandatory critique accepted this as documented non-blocking lockfile drift.
- No docs content correctness review was performed; this evidence covers build/config/dependency behavior.

# Result

The local docs build CI implementation checks passed.

# Interpretation

The current diff appears to satisfy the docs build CI hardening shape locally, but because the change is high-risk release packaging, mandatory critique and final GitHub Actions evidence are still required before acceptance.

# Related Records

- `ticket:c10docs09`
- `packet:ralph-ticket-c10docs09-20260504T025259Z`
- `ticket:c10rel08`
