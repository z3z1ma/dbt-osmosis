---
id: evidence:c10yag28-main-ci-success
kind: evidence
status: recorded
created_at: 2026-05-05T05:31:00Z
updated_at: 2026-05-05T05:31:00Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  ticket:
    - ticket:c10yag28
  evidence:
    - evidence:c10yag28-lean-cleanup-validation
external_refs:
  implementation_commit: https://github.com/z3z1ma/dbt-osmosis/commit/5d009a9ac60713ae7c071cfc30557f0e4799571b
  github_issue: https://github.com/z3z1ma/dbt-osmosis/issues/379
  labeler_run: https://github.com/z3z1ma/dbt-osmosis/actions/runs/25358876988
  lint_run: https://github.com/z3z1ma/dbt-osmosis/actions/runs/25358876972
  tests_run: https://github.com/z3z1ma/dbt-osmosis/actions/runs/25358876978
  release_run: https://github.com/z3z1ma/dbt-osmosis/actions/runs/25359230032
---

# Summary

Observed green GitHub Actions workflows on `origin/main` for c10yag28 implementation commit `5d009a9ac60713ae7c071cfc30557f0e4799571b`.

# Procedure

Observed at: 2026-05-05T05:31:00Z

Source state: `origin/main` at commit `5d009a9ac60713ae7c071cfc30557f0e4799571b`.

Procedure: pushed commit `5d009a9ac60713ae7c071cfc30557f0e4799571b` to `origin/main`, watched the push-triggered Labeler, lint, and Tests workflows with `gh run watch --exit-status`, queried run summaries with `gh run view --json databaseId,name,status,conclusion,event,headSha,url,createdAt,updatedAt`, then watched the follow-up Release workflow.

Expected result when applicable: Labeler, lint, Tests, and Release workflows complete successfully for the pushed implementation commit.

Actual observed result: Labeler, lint, Tests, and Release completed with `conclusion: success` for head SHA `5d009a9ac60713ae7c071cfc30557f0e4799571b`.

Procedure verdict / exit code: pass; watched workflows exited successfully.

# Artifacts

GitHub Actions run summaries observed with `gh run view --json databaseId,name,status,conclusion,event,headSha,url,createdAt,updatedAt`:

```json
{"conclusion":"success","createdAt":"2026-05-05T05:05:26Z","databaseId":25358876988,"event":"push","headSha":"5d009a9ac60713ae7c071cfc30557f0e4799571b","name":"Labeler","status":"completed","updatedAt":"2026-05-05T05:05:36Z","url":"https://github.com/z3z1ma/dbt-osmosis/actions/runs/25358876988"}
```

```json
{"conclusion":"success","createdAt":"2026-05-05T05:05:26Z","databaseId":25358876972,"event":"push","headSha":"5d009a9ac60713ae7c071cfc30557f0e4799571b","name":"lint","status":"completed","updatedAt":"2026-05-05T05:06:38Z","url":"https://github.com/z3z1ma/dbt-osmosis/actions/runs/25358876972"}
```

```json
{"conclusion":"success","createdAt":"2026-05-05T05:05:26Z","databaseId":25358876978,"event":"push","headSha":"5d009a9ac60713ae7c071cfc30557f0e4799571b","name":"Tests","status":"completed","updatedAt":"2026-05-05T05:17:47Z","url":"https://github.com/z3z1ma/dbt-osmosis/actions/runs/25358876978"}
```

```json
{"conclusion":"success","createdAt":"2026-05-05T05:17:49Z","databaseId":25359230032,"event":"workflow_run","headSha":"5d009a9ac60713ae7c071cfc30557f0e4799571b","name":"Release","status":"completed","updatedAt":"2026-05-05T05:30:48Z","url":"https://github.com/z3z1ma/dbt-osmosis/actions/runs/25359230032"}
```

The Tests workflow completed lockfile, docs builds, latest dbt 1.10/1.11 compatibility jobs, pip install smoke, and Python/dbt matrix pytest/integration jobs. The scheduled/manual latest-dbt canary was skipped on push. Release validation included Ruff, basedpyright, demo manifest parse, pytest, docs build, package build, package metadata validation, wheel smoke test, and artifact upload. GitHub Actions emitted non-blocking Node.js 20 deprecation and cache warnings; workflow conclusions remained `success`.

# Supports Claims

- `ticket:c10yag28#ACC-001` through `ticket:c10yag28#ACC-005`: the implementation commit passed remote push workflows and Release validation on `origin/main`.
- `initiative:dbt-110-111-hardening#OBJ-008`: remote CI and Release validation support the lean cleanup work.

# Challenges Claims

None observed for the implementation commit.

# Environment

Commit: `5d009a9ac60713ae7c071cfc30557f0e4799571b`

Branch: `main` on `origin`

Runtime: GitHub Actions hosted runners

OS: GitHub Actions matrix and Release runners

Relevant config: repository workflows at commit `5d009a9ac60713ae7c071cfc30557f0e4799571b`

External service / harness / data source when applicable: GitHub Actions via `gh` CLI

# Validity

Valid for: remote CI state for commit `5d009a9ac60713ae7c071cfc30557f0e4799571b` on `origin/main`.

Fresh enough for: ticket:c10yag28 acceptance and closure.

Recheck when: another commit supersedes this work, workflows are rerun with different results, executor lifecycle changes, PropertyAccessor source policy changes, or pyproject tooling config changes.

Invalidated by: failed rerun of the cited workflows or a superseding commit that changes executor construction, database source behavior, formatter tooling configuration, tests, or workflow validation.

Supersedes / superseded by: none.

# Limitations

This evidence records remote CI success for the implementation commit. It does not prove all external callers have migrated away from `PropertySource.DATABASE`, and it does not remove compatibility `_get_setting_for_node()` exports.

# Result

Commit `5d009a9ac60713ae7c071cfc30557f0e4799571b` passed Labeler, lint, Tests, and Release workflows on `origin/main`.

# Interpretation

This evidence supports ticket closure when combined with local test-first evidence and the ticket-owned optional-critique not-required disposition.

# Related Records

- ticket:c10yag28
- evidence:c10yag28-lean-cleanup-validation
- packet:ralph-ticket-c10yag28-20260505T045616Z
- initiative:dbt-110-111-hardening
