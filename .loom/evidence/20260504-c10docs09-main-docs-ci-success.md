---
id: evidence:c10docs09-main-docs-ci-success
kind: evidence
status: recorded
created_at: 2026-05-04T03:24:38Z
updated_at: 2026-05-04T03:24:38Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  tickets:
    - ticket:c10docs09
  evidence:
    - evidence:c10docs09-local-docs-ci-validation
  critique:
    - critique:c10docs09-docs-ci-hardening
  packets:
    - packet:ralph-ticket-c10docs09-20260504T025259Z
external_refs:
  github_tests_run: https://github.com/z3z1ma/dbt-osmosis/actions/runs/25299065036
  github_docs_node18_job: https://github.com/z3z1ma/dbt-osmosis/actions/runs/25299065036/job/74162683941
  github_docs_node24_job: https://github.com/z3z1ma/dbt-osmosis/actions/runs/25299065036/job/74162683937
  github_docs_dependabot_run: https://github.com/z3z1ma/dbt-osmosis/actions/runs/25299066323
  github_issue: https://github.com/z3z1ma/dbt-osmosis/issues/360
---

# Summary

Observed final GitHub Actions validation for the `ticket:c10docs09` docs build CI change after commit `12e9dfee122db41ddb8f85072e1904ecd079dd00` landed on `main`. The full `Tests` workflow completed successfully, including the new docs build jobs on Node 18 and Node 24, and the docs npm Dependabot dynamic run completed successfully.

# Procedure

Observed at: 2026-05-04T03:24:38Z.

Source state: commit `12e9dfee122db41ddb8f85072e1904ecd079dd00` on GitHub `main`.

Procedure:

- Queried `gh run view 25299065036 --json status,conclusion,url,headSha,workflowName,event,jobs`.
- Inspected the `Build docs (Node 18)` and `Build docs (Node 24)` jobs in that run.
- Queried `gh run view 25299066323 --json status,conclusion,url,headSha,workflowName,event,jobs` for the docs npm Dependabot dynamic run.

Expected result when applicable: the pushed commit should pass the full `Tests` workflow; the docs jobs should run `npm --prefix docs ci`, `npm --prefix docs ls`, and `npm --prefix docs run build` successfully on Node 18 and current LTS; the docs npm Dependabot update mechanism should be active.

Actual observed result: the `Tests` workflow completed with `conclusion: success` for commit `12e9dfee122db41ddb8f85072e1904ecd079dd00`. `Build docs (Node 18)` and `Build docs (Node 24)` both completed successfully, including install, dependency check, and build steps. The docs npm `Dependabot Updates` dynamic run completed successfully.

Procedure verdict / exit code: pass for final GitHub Actions docs CI and docs npm Dependabot coverage observation.

# Artifacts

- `Tests` run `25299065036`: `status: completed`, `conclusion: success`, event `push`, commit `12e9dfee122db41ddb8f85072e1904ecd079dd00`.
- `Build docs (Node 18)` job `74162683941`: completed successfully at 2026-05-04T03:10:29Z; `Install docs dependencies`, `Check docs dependencies`, and `Build docs` steps all succeeded.
- `Build docs (Node 24)` job `74162683937`: completed successfully at 2026-05-04T03:10:17Z; `Install docs dependencies`, `Check docs dependencies`, and `Build docs` steps all succeeded.
- Docs npm `Dependabot Updates` run `25299066323`: completed successfully at 2026-05-04T03:11:08Z for commit `12e9dfee122db41ddb8f85072e1904ecd079dd00`.
- A separate root pip `Dependabot Updates` dynamic run was observed failing on a basedpyright declaration lookup; that run is outside the docs npm update mechanism and does not challenge the docs-specific claims here.

# Supports Claims

- `ticket:c10docs09#ACC-001` - GitHub Actions ran `npm --prefix docs ci` successfully on Node 18 and Node 24.
- `ticket:c10docs09#ACC-002` - GitHub Actions ran `npm --prefix docs run build` successfully on Node 18 and Node 24, and the full `Tests` workflow passed.
- `ticket:c10docs09#ACC-004` - GitHub Actions ran `npm --prefix docs ls` successfully on Node 18 and Node 24; local evidence covers the concrete Docusaurus/React versions.
- `ticket:c10docs09#ACC-005` - the push-triggered `Tests` workflow exercised the new docs jobs; local evidence covers that the same workflow includes `pull_request` coverage.
- `ticket:c10docs09#ACC-006` - the docs npm Dependabot dynamic run completed successfully; local evidence covers the `/docs` npm entry in `.github/dependabot.yml`.

# Challenges Claims

None - this observation did not show a docs CI or docs npm update failure.

# Environment

Commit: `12e9dfee122db41ddb8f85072e1904ecd079dd00`.

Branch: GitHub `main`; local delivery branch `loom/dbt-110-111-hardening` pushed to `origin/main`.

Runtime: GitHub Actions hosted runners; docs jobs used Node 18 and Node 24 through `actions/setup-node@v4.0.4`.

Relevant config: `.github/workflows/tests.yml`, `.github/dependabot.yml`, `docs/package.json`, `docs/package-lock.json`, `docs/docusaurus.config.js`.

External service / harness / data source when applicable: GitHub Actions via `gh` CLI.

# Validity

Valid for: final acceptance evidence for `ticket:c10docs09` at commit `12e9dfee122db41ddb8f85072e1904ecd079dd00`.

Fresh enough for: closing the docs build CI hardening ticket after mandatory critique.

Recheck when: docs dependencies, Docusaurus config, docs CI matrix, Node support policy, Dependabot config, or release gating changes.

Invalidated by: a later failing docs CI run after relevant source/config/dependency changes, removal of the docs jobs, or a discovered mismatch between Dependabot configuration and actual docs npm update coverage.

Supersedes / superseded by: supersedes the CI-pending portions of `evidence:c10docs09-local-docs-ci-validation` for Node 18/24 GitHub Actions claims.

# Limitations

- This evidence observes the push-triggered `Tests` workflow, not a pull-request event; `evidence:c10docs09-local-docs-ci-validation` covers that the docs job is defined under the workflow's existing `pull_request` trigger.
- This evidence does not review docs content accuracy or visual output.
- This evidence does not remediate the existing docs npm audit findings.
- This evidence does not prove unrelated root pip Dependabot dynamic checks.

# Result

Final GitHub Actions docs CI and docs npm Dependabot coverage observations passed for `ticket:c10docs09`.

# Interpretation

The previously pending Node 18/24 docs CI acceptance gate is now satisfied for the pushed `main` commit. This evidence supports ticket acceptance when combined with local validation and mandatory critique.

# Related Records

- `ticket:c10docs09`
- `evidence:c10docs09-local-docs-ci-validation`
- `critique:c10docs09-docs-ci-hardening`
- `packet:ralph-ticket-c10docs09-20260504T025259Z`
- `wiki:ci-compatibility-matrix`
- `wiki:release-publishing-workflow`
