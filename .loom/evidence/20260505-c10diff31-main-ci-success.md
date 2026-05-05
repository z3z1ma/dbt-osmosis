---
id: evidence:c10diff31-main-ci-success
kind: evidence
status: recorded
created_at: 2026-05-05T01:19:29Z
updated_at: 2026-05-05T01:19:29Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  ticket:
    - ticket:c10diff31
  evidence:
    - evidence:c10diff31-schema-diff-validation
  critique:
    - critique:c10diff31-schema-diff-review
external_refs:
  implementation_commit: https://github.com/z3z1ma/dbt-osmosis/commit/d16c266cba288cb6dfb41b45e31be5b3def230f1
  github_issue: https://github.com/z3z1ma/dbt-osmosis/issues/382
  labeler_run: https://github.com/z3z1ma/dbt-osmosis/actions/runs/25351978236
  lint_run: https://github.com/z3z1ma/dbt-osmosis/actions/runs/25351978249
  tests_run: https://github.com/z3z1ma/dbt-osmosis/actions/runs/25351978340
  release_run: https://github.com/z3z1ma/dbt-osmosis/actions/runs/25352329283
---

# Summary

Observed green GitHub Actions workflows on `origin/main` for c10diff31 implementation commit `d16c266cba288cb6dfb41b45e31be5b3def230f1`.

# Procedure

Observed at: 2026-05-05T01:19:29Z

Source state: `origin/main` at commit `d16c266cba288cb6dfb41b45e31be5b3def230f1`.

Procedure: pushed commit `d16c266cba288cb6dfb41b45e31be5b3def230f1` to `origin/main`, watched the push-triggered Labeler, lint, and Tests workflows with `gh run watch --exit-status`, queried run summaries with `gh run view --json databaseId,name,status,conclusion,event,headSha,url,createdAt,updatedAt`, then watched the follow-up Release workflow.

Expected result when applicable: Labeler, lint, Tests, and Release workflows complete successfully for the pushed implementation commit. The scheduled/manual latest-dbt canary should be skipped on push.

Actual observed result: Labeler, lint, Tests, and Release completed with `conclusion: success` for head SHA `d16c266cba288cb6dfb41b45e31be5b3def230f1`. The Tests workflow showed `Canary latest dbt Core compatibility` skipped on push as intended.

Procedure verdict / exit code: pass; watched workflows exited successfully.

# Artifacts

GitHub Actions run summaries observed with `gh run view --json databaseId,name,status,conclusion,event,headSha,url,createdAt,updatedAt`:

```json
{"conclusion":"success","createdAt":"2026-05-05T00:54:13Z","databaseId":25351978236,"event":"push","headSha":"d16c266cba288cb6dfb41b45e31be5b3def230f1","name":"Labeler","status":"completed","updatedAt":"2026-05-05T00:54:23Z","url":"https://github.com/z3z1ma/dbt-osmosis/actions/runs/25351978236"}
```

```json
{"conclusion":"success","createdAt":"2026-05-05T00:54:13Z","databaseId":25351978249,"event":"push","headSha":"d16c266cba288cb6dfb41b45e31be5b3def230f1","name":"lint","status":"completed","updatedAt":"2026-05-05T00:55:27Z","url":"https://github.com/z3z1ma/dbt-osmosis/actions/runs/25351978249"}
```

```json
{"conclusion":"success","createdAt":"2026-05-05T00:54:13Z","databaseId":25351978340,"event":"push","headSha":"d16c266cba288cb6dfb41b45e31be5b3def230f1","name":"Tests","status":"completed","updatedAt":"2026-05-05T01:05:53Z","url":"https://github.com/z3z1ma/dbt-osmosis/actions/runs/25351978340"}
```

```json
{"conclusion":"success","createdAt":"2026-05-05T01:05:55Z","databaseId":25352329283,"event":"workflow_run","headSha":"d16c266cba288cb6dfb41b45e31be5b3def230f1","name":"Release","status":"completed","updatedAt":"2026-05-05T01:19:19Z","url":"https://github.com/z3z1ma/dbt-osmosis/actions/runs/25352329283"}
```

The Tests workflow completed lockfile, docs builds, latest dbt 1.10/1.11 compatibility jobs, pip install smoke, and Python/dbt matrix pytest/integration jobs. The scheduled/manual latest-dbt canary was skipped on push. Release validation included Ruff, basedpyright, demo manifest parse, pytest, docs build, package build, package metadata validation, wheel smoke test, and artifact upload. GitHub Actions emitted non-blocking Node.js 20 deprecation and cache warnings; workflow conclusions remained `success`.

# Supports Claims

- ticket:c10diff31#ACC-001 through ticket:c10diff31#ACC-005: the implementation commit passed remote push workflows and Release validation on `origin/main`.
- initiative:dbt-110-111-hardening#OBJ-006: remote CI and Release validation support the user-facing schema diff behavior fix.

# Challenges Claims

None observed for the implementation commit.

# Environment

Commit: `d16c266cba288cb6dfb41b45e31be5b3def230f1`

Branch: `main` on `origin`

Runtime: GitHub Actions hosted runners

OS: GitHub Actions matrix and Release runners

Relevant config: repository workflows at commit `d16c266cba288cb6dfb41b45e31be5b3def230f1`

External service / harness / data source when applicable: GitHub Actions via `gh` CLI

# Validity

Valid for: remote CI state for commit `d16c266cba288cb6dfb41b45e31be5b3def230f1` on `origin/main`.

Fresh enough for: ticket:c10diff31 acceptance and closure.

Recheck when: another commit supersedes this work, workflows are rerun with different results, schema diff behavior changes, or a critique finding challenges this validation.

Invalidated by: failed rerun of the cited workflows or a superseding commit that changes schema diff type comparison, rename matching, tests, or workflow validation.

Supersedes / superseded by: none.

# Limitations

This evidence records remote CI success for the implementation commit. It does not execute the scheduled/manual latest-dbt canary, prove exotic semantic-whitespace type strings, or prove globally optimal rename assignment.

# Result

Commit `d16c266cba288cb6dfb41b45e31be5b3def230f1` passed Labeler, lint, Tests, and Release workflows on `origin/main`.

# Interpretation

This evidence supports ticket closure when combined with local test-first evidence and recommended critique with ticket-owned low-risk dispositions. It does not broaden the ticket's type alias policy or rename matching strategy beyond the accepted scoped behavior.

# Related Records

- ticket:c10diff31
- evidence:c10diff31-schema-diff-validation
- critique:c10diff31-schema-diff-review
- packet:ralph-ticket-c10diff31-20260505T003927Z
- packet:critique-ticket-c10diff31-20260505T004715Z
- initiative:dbt-110-111-hardening
