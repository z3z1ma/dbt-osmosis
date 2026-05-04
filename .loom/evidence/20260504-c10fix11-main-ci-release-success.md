---
id: evidence:c10fix11-main-ci-release-success
kind: evidence
status: recorded
created_at: 2026-05-04T06:21:20Z
updated_at: 2026-05-04T06:21:20Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  tickets:
    - ticket:c10fix11
  evidence:
    - evidence:c10fix11-fixture-isolation-verification
  critique:
    - critique:c10fix11-fixture-isolation-review
external_refs:
  github_actions:
    - https://github.com/z3z1ma/dbt-osmosis/actions/runs/25302762583
    - https://github.com/z3z1ma/dbt-osmosis/actions/runs/25302762587
    - https://github.com/z3z1ma/dbt-osmosis/actions/runs/25303037463
---

# Summary

Observed successful post-commit GitHub Actions validation for `ticket:c10fix11` implementation commit `9484b98d62eead35d8ed3dec1eb913fe6fe9de5e`. This evidence supplements local fixture-isolation verification with main-branch CI and Release validation; it does not decide ticket closure by itself.

# Procedure

Observed at: 2026-05-04T06:21:20Z
Source state: `origin/main` at `9484b98d62eead35d8ed3dec1eb913fe6fe9de5e`.
Procedure: Queried GitHub Actions with `gh run list --branch main --limit 8` and `gh run view` for the post-push `Tests`, `lint`, and downstream `Release` runs.
Expected result when applicable: `Tests`, `lint`, and downstream `Release` validation should complete successfully for implementation commit `9484b98d62eead35d8ed3dec1eb913fe6fe9de5e`.
Actual observed result: All three runs completed with conclusion `success`.
Procedure verdict / exit code: pass; `gh run view` reported `status: completed` and `conclusion: success` for each cited run.

# Artifacts

- `Tests` run `25302762583`: event `push`, head SHA `9484b98d62eead35d8ed3dec1eb913fe6fe9de5e`, created `2026-05-04T05:31:58Z`, updated `2026-05-04T05:41:36Z`, conclusion `success`, URL `https://github.com/z3z1ma/dbt-osmosis/actions/runs/25302762583`.
- `lint` run `25302762587`: event `push`, head SHA `9484b98d62eead35d8ed3dec1eb913fe6fe9de5e`, created `2026-05-04T05:31:58Z`, updated `2026-05-04T05:33:02Z`, conclusion `success`, URL `https://github.com/z3z1ma/dbt-osmosis/actions/runs/25302762587`.
- `Release` run `25303037463`: event `workflow_run`, head SHA `9484b98d62eead35d8ed3dec1eb913fe6fe9de5e`, created `2026-05-04T05:41:38Z`, updated `2026-05-04T05:52:41Z`, conclusion `success`, URL `https://github.com/z3z1ma/dbt-osmosis/actions/runs/25303037463`.
- `gh run list --branch main --limit 8` showed all three post-push runs for `test: isolate demo fixture artifacts` completed successfully.

# Supports Claims

- ticket:c10fix11#ACC-001: post-commit `Tests` run passed the matrix containing fixture profile path tests and integration smoke.
- ticket:c10fix11#ACC-002: post-commit `Tests` run passed with source-artifact guard steps and isolated core manifest fixtures.
- ticket:c10fix11#ACC-003: post-commit `Tests` run passed across the dbt compatibility matrix using temp demo parse copies.
- ticket:c10fix11#ACC-004: post-commit `Tests` run passed the safe integration smoke path without destructive cleanup.
- ticket:c10fix11#ACC-005: post-commit `Tests` run passed fixture-copy exclusion regression coverage.
- ticket:c10fix11#ACC-006: post-commit `Tests` run passed source-artifact guard coverage for repo-root `test.db` and source `demo_duckdb/target`.
- initiative:dbt-110-111-hardening#OBJ-004: `Tests`, `lint`, and downstream `Release` validation succeeded after the fixture isolation implementation landed on `main`.

# Challenges Claims

None - the observed CI and Release results matched the expected successful post-commit validation for the cited claims.

# Environment

Commit: `9484b98d62eead35d8ed3dec1eb913fe6fe9de5e`
Branch: `main`
Runtime: GitHub-hosted Actions runners for `Tests`, `lint`, and `Release` workflows.
OS: GitHub Actions Ubuntu runners.
Relevant config: `.github/workflows/tests.yml`, `.github/workflows/release.yml`, `.github/workflows/lint.yml`, `Taskfile.yml`, fixture tests, and integration smoke script at the cited commit.
External service / harness / data source when applicable: GitHub Actions via `gh` CLI.

# Validity

Valid for: post-commit CI and Release validation of `ticket:c10fix11` implementation commit `9484b98d62eead35d8ed3dec1eb913fe6fe9de5e` on `main`.
Fresh enough for: final ticket acceptance review and closure consideration for `ticket:c10fix11`.
Recheck when: source changes after `9484b98d62eead35d8ed3dec1eb913fe6fe9de5e`, workflow configuration changes, dbt compatibility matrix changes, or GitHub reruns replace these observations.
Invalidated by: a later failed required run for the same commit, source changes that alter fixture behavior, or evidence that the cited runs did not execute the relevant matrix/jobs.
Supersedes / superseded by: Supplements `evidence:c10fix11-fixture-isolation-verification` with post-commit CI and Release validation.

# Limitations

- GitHub Actions success does not prove every future dbt adapter/version combination; it covers the repository's configured matrix at the cited commit.
- The `Release` workflow still has its own validation scope and does not publish a package from this ticket by itself.
- This evidence does not update stale wiki or operator guidance; retrospective follow-through owns that reconciliation.

# Result

The observed main-branch `Tests`, `lint`, and downstream `Release` runs all succeeded for implementation commit `9484b98d62eead35d8ed3dec1eb913fe6fe9de5e`.

# Interpretation

The evidence supports moving `ticket:c10fix11` from local verification to final acceptance review. It does not itself close the ticket or replace retrospective / promotion disposition.

# Related Records

- ticket:c10fix11
- evidence:c10fix11-fixture-isolation-verification
- critique:c10fix11-fixture-isolation-review
- commit `9484b98d62eead35d8ed3dec1eb913fe6fe9de5e`
