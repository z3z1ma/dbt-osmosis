---
id: evidence:c10bounds29-main-ci-success
kind: evidence
status: recorded
created_at: 2026-05-05T00:12:42Z
updated_at: 2026-05-05T00:12:42Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  ticket:
    - ticket:c10bounds29
  evidence:
    - evidence:c10bounds29-support-policy-validation
  critique:
    - critique:c10bounds29-support-policy-review
external_refs:
  implementation_commit: https://github.com/z3z1ma/dbt-osmosis/commit/38d95c22db743f8df3f88059b13f359793ceb057
  github_issue: https://github.com/z3z1ma/dbt-osmosis/issues/380
  labeler_run: https://github.com/z3z1ma/dbt-osmosis/actions/runs/25349768244
  lint_run: https://github.com/z3z1ma/dbt-osmosis/actions/runs/25349768242
  tests_run: https://github.com/z3z1ma/dbt-osmosis/actions/runs/25349768249
  release_run: https://github.com/z3z1ma/dbt-osmosis/actions/runs/25350143083
---

# Summary

Observed green GitHub Actions workflows on `origin/main` for c10bounds29 implementation commit `38d95c22db743f8df3f88059b13f359793ceb057`.

# Procedure

Observed at: 2026-05-05T00:12:42Z

Source state: `origin/main` at commit `38d95c22db743f8df3f88059b13f359793ceb057`.

Procedure: pushed commit `38d95c22db743f8df3f88059b13f359793ceb057` to `origin/main`, watched the push-triggered lint and Tests workflows with `gh run watch --exit-status`, queried run summaries with `gh run view --json databaseId,name,status,conclusion,event,headSha,url,createdAt,updatedAt`, then watched the follow-up Release workflow.

Expected result when applicable: Labeler, lint, Tests, and Release workflows complete successfully for the pushed implementation commit. The new `future-dbt-canary` job should be skipped on push because it is scheduled/manual canary coverage.

Actual observed result: Labeler, lint, Tests, and Release completed with `conclusion: success` for head SHA `38d95c22db743f8df3f88059b13f359793ceb057`. The Tests workflow showed `Canary latest dbt Core compatibility` skipped on push as intended.

Procedure verdict / exit code: pass; watched workflows exited successfully.

# Artifacts

GitHub Actions run summaries observed with `gh run view --json databaseId,name,status,conclusion,event,headSha,url,createdAt,updatedAt`:

```json
{"conclusion":"success","createdAt":"2026-05-04T23:45:50Z","databaseId":25349768244,"event":"push","headSha":"38d95c22db743f8df3f88059b13f359793ceb057","name":"Labeler","status":"completed","updatedAt":"2026-05-04T23:45:58Z","url":"https://github.com/z3z1ma/dbt-osmosis/actions/runs/25349768244"}
```

```json
{"conclusion":"success","createdAt":"2026-05-04T23:45:50Z","databaseId":25349768242,"event":"push","headSha":"38d95c22db743f8df3f88059b13f359793ceb057","name":"lint","status":"completed","updatedAt":"2026-05-04T23:46:58Z","url":"https://github.com/z3z1ma/dbt-osmosis/actions/runs/25349768242"}
```

```json
{"conclusion":"success","createdAt":"2026-05-04T23:45:50Z","databaseId":25349768249,"event":"push","headSha":"38d95c22db743f8df3f88059b13f359793ceb057","name":"Tests","status":"completed","updatedAt":"2026-05-04T23:56:57Z","url":"https://github.com/z3z1ma/dbt-osmosis/actions/runs/25349768249"}
```

```json
{"conclusion":"success","createdAt":"2026-05-04T23:56:59Z","databaseId":25350143083,"event":"workflow_run","headSha":"38d95c22db743f8df3f88059b13f359793ceb057","name":"Release","status":"completed","updatedAt":"2026-05-05T00:12:22Z","url":"https://github.com/z3z1ma/dbt-osmosis/actions/runs/25350143083"}
```

The Tests workflow completed lockfile, docs builds, latest dbt 1.10/1.11 compatibility jobs, pip install smoke, and Python/dbt matrix pytest/integration jobs. The scheduled/manual latest-dbt canary was skipped on push. Release validation included Ruff, basedpyright, demo manifest parse, pytest, docs build, package build, package metadata validation, wheel smoke test, and artifact upload. GitHub Actions emitted non-blocking Node.js 20 deprecation and cache warnings; the workflow conclusions remained `success`.

# Supports Claims

- ticket:c10bounds29#ACC-001 through ticket:c10bounds29#ACC-005: the implementation commit passed remote push workflows and Release validation on `origin/main`.
- initiative:dbt-110-111-hardening#OBJ-001: remote CI passed with explicit dbt 1.10/1.11 compatibility validation and the new future-dbt canary definition present.
- initiative:dbt-110-111-hardening#OBJ-005: release and package validation passed with the open dbt support policy and warning-filter changes.

# Challenges Claims

None observed for the implementation commit. The skipped canary limits future-minor runtime evidence on push, as intended by the canary policy.

# Environment

Commit: `38d95c22db743f8df3f88059b13f359793ceb057`

Branch: `main` on `origin`

Runtime: GitHub Actions hosted runners

OS: GitHub Actions matrix and Release runners

Relevant config: repository workflows at commit `38d95c22db743f8df3f88059b13f359793ceb057`

External service / harness / data source when applicable: GitHub Actions via `gh` CLI

# Validity

Valid for: remote CI state for commit `38d95c22db743f8df3f88059b13f359793ceb057` on `origin/main`.

Fresh enough for: ticket:c10bounds29 acceptance and closure.

Recheck when: another commit supersedes this work, workflows are rerun with different results, a scheduled/manual future-dbt canary produces a finding, or support policy changes.

Invalidated by: failed rerun of the cited workflows or a superseding commit that changes dbt support policy, warning filters, CI canary behavior, docs, or package metadata tests.

Supersedes / superseded by: none.

# Limitations

This evidence records remote CI success for the implementation commit. It does not execute the new scheduled/manual future-dbt canary, does not prove unreviewed future dbt minors pass, and does not remove the critique-accepted low risks around resolver backtracking and matrix-combination wording.

# Result

Commit `38d95c22db743f8df3f88059b13f359793ceb057` passed Labeler, lint, Tests, and Release workflows on `origin/main`.

# Interpretation

This evidence supports ticket closure when combined with local test-first evidence and mandatory critique with ticket-owned low-risk dispositions. It does not convert future dbt minors into audited support.

# Related Records

- ticket:c10bounds29
- evidence:c10bounds29-support-policy-validation
- critique:c10bounds29-support-policy-review
- packet:ralph-ticket-c10bounds29-20260504T233210Z
- packet:critique-ticket-c10bounds29-20260504T233947Z
- initiative:dbt-110-111-hardening
