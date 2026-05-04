---
id: evidence:c10col01-c10meta02-main-ci-success
kind: evidence
status: recorded
created_at: 2026-05-04T03:29:15Z
updated_at: 2026-05-04T03:29:15Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  tickets:
    - ticket:c10col01
    - ticket:c10meta02
    - ticket:c10ci06
    - ticket:c10cfg12
  evidence:
    - evidence:c10col01-columninfo-config-red-green
    - evidence:c10meta02-column-config-meta-tags
    - evidence:c10ci06-main-ci-success
  critique:
    - critique:c10col01-columninfo-config
    - critique:c10meta02-column-config-meta-tags
external_refs:
  github_tests_run: https://github.com/z3z1ma/dbt-osmosis/actions/runs/25299065036
  github_issue_c10col01: https://github.com/z3z1ma/dbt-osmosis/issues/352
  github_issue_c10meta02: https://github.com/z3z1ma/dbt-osmosis/issues/353
---

# Summary

Observed final `main` GitHub Actions validation after the `ticket:c10col01` and `ticket:c10meta02` implementation commits were included in commit `12e9dfee122db41ddb8f85072e1904ecd079dd00`. The full `Tests` workflow completed successfully across the dbt/Python matrix and compatibility jobs. This evidence refreshes acceptance support for the implemented code paths while preserving the ticket-owned conversion of exact dbt 1.11 parsed-fixture gaps to `ticket:c10cfg12`.

# Procedure

Observed at: 2026-05-04T03:29:15Z.

Source state: commit `12e9dfee122db41ddb8f85072e1904ecd079dd00` on GitHub `main`, which includes implementation commits `ba92d07d263090cd06eaa8939f1f727ed5420512` and `68d72bb96d0abc8268a00936591a2b3862bac71e` plus later CI hardening commits.

Procedure:

- Queried `gh run view 25299065036 --json status,conclusion,url,headSha,workflowName,event,jobs`.
- Confirmed the workflow returned `status: completed` and `conclusion: success` for commit `12e9dfee122db41ddb8f85072e1904ecd079dd00`.
- Inspected the earlier ticket records to verify mandatory critiques and ticket-owned finding dispositions were already recorded.

Expected result when applicable: the full `Tests` workflow should pass on `main` after both implementation commits; exact dbt 1.11 parsed-fixture gaps should remain explicitly converted to follow-up rather than being silently claimed by this evidence.

Actual observed result: the `Tests` workflow completed successfully for commit `12e9dfee122db41ddb8f85072e1904ecd079dd00`. The run included the dbt/Python pytest matrix, latest dbt 1.10/1.11 compatibility jobs, pip install smoke, lock freshness check, and docs jobs.

Procedure verdict / exit code: pass for final `main` CI freshness; partial for exact parsed-fixture claims, which remain converted follow-up work.

# Artifacts

- `Tests` run `25299065036`: `status: completed`, `conclusion: success`, event `push`, commit `12e9dfee122db41ddb8f85072e1904ecd079dd00`.
- `ticket:c10col01` implementation commit: `ba92d07d263090cd06eaa8939f1f727ed5420512`.
- `ticket:c10meta02` implementation commit: `68d72bb96d0abc8268a00936591a2b3862bac71e`.
- `ticket:c10ci06` already accepted `evidence:c10ci06-main-ci-success` for broad dbt 1.10/1.11 matrix coverage.
- `ticket:c10cfg12` remains the follow-up owner for real parsed fixture coverage gaps.

# Supports Claims

- `ticket:c10col01#ACC-001` through `ticket:c10col01#ACC-004` - current `main` CI passed after the local red/green implementation evidence was committed.
- `ticket:c10meta02#ACC-001` through `ticket:c10meta02#ACC-004` and `ticket:c10meta02#ACC-006` - current `main` CI passed after the local red/green implementation evidence was committed.
- `critique:c10col01-columninfo-config#FIND-001` and `critique:c10meta02-column-config-meta-tags#FIND-001` - supports that the broad matrix side of the converted follow-up is covered by `ticket:c10ci06`; exact parsed-fixture evidence remains with `ticket:c10cfg12`.

# Challenges Claims

None - this observation did not show a regression in the current `main` CI state.

# Environment

Commit: `12e9dfee122db41ddb8f85072e1904ecd079dd00`.

Branch: GitHub `main`; local delivery branch `loom/dbt-110-111-hardening` pushed to `origin/main`.

Runtime: GitHub Actions hosted runners; matrix runtimes are defined by `.github/workflows/tests.yml`.

Relevant config: `.github/workflows/tests.yml`, changed core/test files from `ticket:c10col01` and `ticket:c10meta02`.

External service / harness / data source when applicable: GitHub Actions via `gh` CLI.

# Validity

Valid for: acceptance freshness for `ticket:c10col01` and `ticket:c10meta02` implementation commits on `main`.

Fresh enough for: closing `ticket:c10col01` and `ticket:c10meta02` with explicit follow-up disposition for exact parsed-fixture gaps.

Recheck when: column injection, column config metadata/tag handling, dbt matrix workflow, parsed fixture coverage, or follow-up ticket dispositions change.

Invalidated by: a later failing CI run after relevant source changes, reopening of critique findings without disposition, or removal of the follow-up owner for exact parsed-fixture evidence.

Supersedes / superseded by: not superseded.

# Limitations

- This evidence does not itself prove `ticket:c10col01#ACC-005` or `ticket:c10meta02#ACC-005` exact real parsed-fixture scenarios.
- The exact fixture evidence gaps remain explicitly converted to `ticket:c10cfg12` with broad matrix execution already covered by closed `ticket:c10ci06`.
- This evidence does not review user-facing docs, package metadata, or unrelated dependency update jobs.

# Result

Final `main` CI passed after the `ticket:c10col01` and `ticket:c10meta02` implementation commits were included.

# Interpretation

The implementation portions of `ticket:c10col01` and `ticket:c10meta02` are fresh enough for acceptance when combined with their local red/green evidence and mandatory critiques. Exact dbt 1.11 parsed-fixture evidence remains follow-up work, not a claim satisfied by this evidence.

# Related Records

- `ticket:c10col01`
- `ticket:c10meta02`
- `ticket:c10ci06`
- `ticket:c10cfg12`
- `evidence:c10col01-columninfo-config-red-green`
- `evidence:c10meta02-column-config-meta-tags`
- `evidence:c10ci06-main-ci-success`
- `critique:c10col01-columninfo-config`
- `critique:c10meta02-column-config-meta-tags`
