---
id: evidence:c10feed26-main-ci-success
kind: evidence
status: recorded
created_at: 2026-05-04T23:01:41Z
updated_at: 2026-05-04T23:01:41Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  ticket:
    - ticket:c10feed26
  evidence:
    - evidence:c10feed26-workbench-feed-hardening-validation
  critique:
    - critique:c10feed26-workbench-feed-hardening-review
    - critique:c10feed26-workbench-feed-hardening-final-review
external_refs:
  implementation_commit: https://github.com/z3z1ma/dbt-osmosis/commit/1d3a0cc6bb1f4ec1255b632e97057af4f8808d7c
  github_issue: https://github.com/z3z1ma/dbt-osmosis/issues/377
  labeler_run: https://github.com/z3z1ma/dbt-osmosis/actions/runs/25347312537
  lint_run: https://github.com/z3z1ma/dbt-osmosis/actions/runs/25347312548
  tests_run: https://github.com/z3z1ma/dbt-osmosis/actions/runs/25347312544
  release_run: https://github.com/z3z1ma/dbt-osmosis/actions/runs/25347738128
---

# Summary

Observed green GitHub Actions workflows on `origin/main` for c10feed26 implementation commit `1d3a0cc6bb1f4ec1255b632e97057af4f8808d7c`.

# Procedure

Observed at: 2026-05-04T23:01:41Z

Source state: `origin/main` at commit `1d3a0cc6bb1f4ec1255b632e97057af4f8808d7c`.

Procedure: pushed commit `1d3a0cc6bb1f4ec1255b632e97057af4f8808d7c` to `origin/main`, queried push workflows with `gh run view --json databaseId,name,status,conclusion,event,headSha,url,createdAt,updatedAt`, and watched the follow-up Release workflow with `gh run watch 25347738128 --exit-status`.

Expected result when applicable: Labeler, lint, Tests, and Release workflows complete successfully for the pushed implementation commit.

Actual observed result: all four workflows completed with `conclusion: success` for head SHA `1d3a0cc6bb1f4ec1255b632e97057af4f8808d7c`.

Procedure verdict / exit code: pass; the watched Release workflow exited successfully.

# Artifacts

GitHub Actions run summaries observed with `gh run view --json databaseId,name,status,conclusion,event,headSha,url,createdAt,updatedAt`:

```json
{"conclusion":"success","createdAt":"2026-05-04T22:36:21Z","databaseId":25347312537,"event":"push","headSha":"1d3a0cc6bb1f4ec1255b632e97057af4f8808d7c","name":"Labeler","status":"completed","updatedAt":"2026-05-04T22:36:30Z","url":"https://github.com/z3z1ma/dbt-osmosis/actions/runs/25347312537"}
```

```json
{"conclusion":"success","createdAt":"2026-05-04T22:36:21Z","databaseId":25347312548,"event":"push","headSha":"1d3a0cc6bb1f4ec1255b632e97057af4f8808d7c","name":"lint","status":"completed","updatedAt":"2026-05-04T22:37:29Z","url":"https://github.com/z3z1ma/dbt-osmosis/actions/runs/25347312548"}
```

```json
{"conclusion":"success","createdAt":"2026-05-04T22:36:21Z","databaseId":25347312544,"event":"push","headSha":"1d3a0cc6bb1f4ec1255b632e97057af4f8808d7c","name":"Tests","status":"completed","updatedAt":"2026-05-04T22:47:45Z","url":"https://github.com/z3z1ma/dbt-osmosis/actions/runs/25347312544"}
```

```json
{"conclusion":"success","createdAt":"2026-05-04T22:47:47Z","databaseId":25347738128,"event":"workflow_run","headSha":"1d3a0cc6bb1f4ec1255b632e97057af4f8808d7c","name":"Release","status":"completed","updatedAt":"2026-05-04T23:01:02Z","url":"https://github.com/z3z1ma/dbt-osmosis/actions/runs/25347738128"}
```

Release validation included Ruff, basedpyright, demo manifest parse, pytest, docs dependency install, docs build, package build, package metadata validation, wheel smoke test, and artifact upload. The run emitted non-blocking Node.js 20 deprecation and GitHub cache warnings; the workflow conclusion remained `success`.

# Supports Claims

- ticket:c10feed26#ACC-001 through ticket:c10feed26#ACC-006: the implementation commit passed remote push workflows and Release validation on `origin/main`.
- initiative:dbt-110-111-hardening#OBJ-006: remote validation passed for the workbench external feed hardening change.
- initiative:dbt-110-111-hardening#OBJ-007: CI evidence records the hardening validation result.

# Challenges Claims

None observed.

# Environment

Commit: `1d3a0cc6bb1f4ec1255b632e97057af4f8808d7c`

Branch: `main` on `origin`

Runtime: GitHub Actions hosted runners

OS: GitHub Actions matrix and Release runners

Relevant config: repository workflows at commit `1d3a0cc6bb1f4ec1255b632e97057af4f8808d7c`

External service / harness / data source when applicable: GitHub Actions via `gh` CLI

# Validity

Valid for: remote CI state for commit `1d3a0cc6bb1f4ec1255b632e97057af4f8808d7c` on `origin/main`.

Fresh enough for: ticket:c10feed26 acceptance and closure.

Recheck when: another commit supersedes this work, workflows are rerun with different results, or release validation criteria change.

Invalidated by: failed rerun of the cited workflows or a superseding commit that changes the workbench external feed implementation.

Supersedes / superseded by: none.

# Limitations

This evidence records remote CI success for the implementation commit. It does not add browser-level Streamlit rendering coverage, a real Hacker News RSS fetch, or stronger guarantees about future Streamlit component behavior beyond the repository's test and release workflow coverage.

# Result

Commit `1d3a0cc6bb1f4ec1255b632e97057af4f8808d7c` passed Labeler, lint, Tests, and Release workflows on `origin/main`.

# Interpretation

This evidence supports ticket closure when combined with local red/green evidence and final security/code/test critique. It does not remove the accepted residual risk around browser-level rendering behavior or real external RSS feed behavior.

# Related Records

- ticket:c10feed26
- evidence:c10feed26-workbench-feed-hardening-validation
- critique:c10feed26-workbench-feed-hardening-review
- critique:c10feed26-workbench-feed-hardening-final-review
- packet:ralph-ticket-c10feed26-20260504T221607Z
- packet:ralph-ticket-c10feed26-20260504T222745Z
- packet:critique-ticket-c10feed26-20260504T222303Z
- packet:critique-ticket-c10feed26-20260504T223138Z
- initiative:dbt-110-111-hardening
