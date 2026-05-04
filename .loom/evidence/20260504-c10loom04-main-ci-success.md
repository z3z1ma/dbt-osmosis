---
id: evidence:c10loom04-main-ci-success
kind: evidence
status: recorded
created_at: 2026-05-04T21:47:54Z
updated_at: 2026-05-04T21:47:54Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  ticket:
    - ticket:c10loom04
  evidence:
    - evidence:c10loom04-dbt-loom-parser-validation
  critique:
    - critique:c10loom04-dbt-loom-parser-review
external_refs:
  github_issue: https://github.com/z3z1ma/dbt-osmosis/issues/355
  labeler_run: https://github.com/z3z1ma/dbt-osmosis/actions/runs/25344274925
  lint_run: https://github.com/z3z1ma/dbt-osmosis/actions/runs/25344274907
  tests_run: https://github.com/z3z1ma/dbt-osmosis/actions/runs/25344274906
  release_run: https://github.com/z3z1ma/dbt-osmosis/actions/runs/25344773693
---

# Summary

Observed green GitHub Actions workflows on `origin/main` for c10loom04 implementation commit `8d47587d5485dadab67bc39008aea8f73c159241`.

# Procedure

Observed at: 2026-05-04T21:47:54Z

Source state: `origin/main` at commit `8d47587d5485dadab67bc39008aea8f73c159241`.

Procedure: pushed commit `8d47587d5485dadab67bc39008aea8f73c159241` to `origin/main`, watched push workflows with `gh run watch --exit-status`, then watched the follow-up Release workflow.

Expected result when applicable: Labeler, lint, Tests, and Release workflows complete successfully for the pushed commit.

Actual observed result: all four workflows completed with `conclusion: success`.

Procedure verdict / exit code: pass.

# Artifacts

GitHub Actions run summaries observed with `gh run view --json databaseId,name,status,conclusion,event,headSha,url,createdAt,updatedAt`:

```json
{"conclusion":"success","createdAt":"2026-05-04T21:23:01Z","databaseId":25344274925,"event":"push","headSha":"8d47587d5485dadab67bc39008aea8f73c159241","name":"Labeler","status":"completed","updatedAt":"2026-05-04T21:23:17Z","url":"https://github.com/z3z1ma/dbt-osmosis/actions/runs/25344274925"}
```

```json
{"conclusion":"success","createdAt":"2026-05-04T21:23:01Z","databaseId":25344274907,"event":"push","headSha":"8d47587d5485dadab67bc39008aea8f73c159241","name":"lint","status":"completed","updatedAt":"2026-05-04T21:24:08Z","url":"https://github.com/z3z1ma/dbt-osmosis/actions/runs/25344274907"}
```

```json
{"conclusion":"success","createdAt":"2026-05-04T21:23:01Z","databaseId":25344274906,"event":"push","headSha":"8d47587d5485dadab67bc39008aea8f73c159241","name":"Tests","status":"completed","updatedAt":"2026-05-04T21:34:25Z","url":"https://github.com/z3z1ma/dbt-osmosis/actions/runs/25344274906"}
```

```json
{"conclusion":"success","createdAt":"2026-05-04T21:34:27Z","databaseId":25344773693,"event":"workflow_run","headSha":"8d47587d5485dadab67bc39008aea8f73c159241","name":"Release","status":"completed","updatedAt":"2026-05-04T21:47:33Z","url":"https://github.com/z3z1ma/dbt-osmosis/actions/runs/25344773693"}
```

Release annotations included GitHub cache warnings and Node.js 20 deprecation warnings, but the workflow conclusion remained `success`.

# Supports Claims

- ticket:c10loom04#ACC-001 through ticket:c10loom04#ACC-005: the implementation commit passed remote push workflows and Release validation on `origin/main`.
- initiative:dbt-110-111-hardening#OBJ-001: remote validation passed for a dbt 1.10/1.11 compatibility fix.
- initiative:dbt-110-111-hardening#OBJ-007: CI evidence records the hardening validation result.

# Challenges Claims

None observed.

# Environment

Commit: `8d47587d5485dadab67bc39008aea8f73c159241`

Branch: `main` on `origin`

Runtime: GitHub Actions hosted runners

OS: GitHub Actions matrix and Release runners

Relevant config: repository workflows at commit `8d47587d5485dadab67bc39008aea8f73c159241`

External service / harness / data source when applicable: GitHub Actions via `gh` CLI

# Validity

Valid for: remote CI state for commit `8d47587d5485dadab67bc39008aea8f73c159241` on `origin/main`.

Fresh enough for: ticket:c10loom04 acceptance and closure.

Recheck when: another commit supersedes this work, workflows are rerun with different results, or release validation criteria change.

Invalidated by: failed rerun of the cited workflows or a superseding commit that changes the c10loom04 implementation.

Supersedes / superseded by: none.

# Limitations

This evidence does not run a separate real dbt-loom package integration scenario beyond the repository's existing CI and tests. It records CI success for the implementation commit, not a guarantee that future dbt-loom manifest shapes will remain compatible.

# Result

Commit `8d47587d5485dadab67bc39008aea8f73c159241` passed Labeler, lint, Tests, and Release workflows on `origin/main`.

# Interpretation

This evidence supports ticket closure when combined with local red/green evidence and final critique. It does not remove the residual risk that dbt-loom could change its manifest shape later.

# Related Records

- ticket:c10loom04
- evidence:c10loom04-dbt-loom-parser-validation
- critique:c10loom04-dbt-loom-parser-review
- packet:ralph-ticket-c10loom04-20260504T210814Z
- packet:critique-ticket-c10loom04-20260504T211528Z
