---
id: evidence:c10anchors30-main-ci-success
kind: evidence
status: recorded
created_at: 2026-05-05T04:38:22Z
updated_at: 2026-05-05T04:38:22Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  ticket:
    - ticket:c10anchors30
  evidence:
    - evidence:c10anchors30-yaml-anchor-validation
  critique:
    - critique:c10anchors30-yaml-anchor-review
external_refs:
  implementation_commit: https://github.com/z3z1ma/dbt-osmosis/commit/af229e597fceb7a4a7a2902c6bb508944dc58e4e
  github_issue: https://github.com/z3z1ma/dbt-osmosis/issues/381
  labeler_run: https://github.com/z3z1ma/dbt-osmosis/actions/runs/25357462268
  lint_run: https://github.com/z3z1ma/dbt-osmosis/actions/runs/25357462257
  tests_run: https://github.com/z3z1ma/dbt-osmosis/actions/runs/25357462260
  release_run: https://github.com/z3z1ma/dbt-osmosis/actions/runs/25357773183
---

# Summary

Observed green GitHub Actions workflows on `origin/main` for c10anchors30 implementation commit `af229e597fceb7a4a7a2902c6bb508944dc58e4e`.

# Procedure

Observed at: 2026-05-05T04:38:22Z

Source state: `origin/main` at commit `af229e597fceb7a4a7a2902c6bb508944dc58e4e`.

Procedure: pushed commit `af229e597fceb7a4a7a2902c6bb508944dc58e4e` to `origin/main`, queried the push-triggered Labeler, lint, and Tests workflows with `gh run view --json databaseId,name,status,conclusion,event,headSha,url,createdAt,updatedAt`, then watched the follow-up Release workflow with `gh run watch --exit-status` and queried its run summary.

Expected result when applicable: Labeler, lint, Tests, and Release workflows complete successfully for the pushed implementation commit.

Actual observed result: Labeler, lint, Tests, and Release completed with `conclusion: success` for head SHA `af229e597fceb7a4a7a2902c6bb508944dc58e4e`.

Procedure verdict / exit code: pass; watched Release workflow exited successfully.

# Artifacts

GitHub Actions run summaries observed with `gh run view --json databaseId,name,status,conclusion,event,headSha,url,createdAt,updatedAt`:

```json
{"conclusion":"success","createdAt":"2026-05-05T04:13:22Z","databaseId":25357462268,"event":"push","headSha":"af229e597fceb7a4a7a2902c6bb508944dc58e4e","name":"Labeler","status":"completed","updatedAt":"2026-05-05T04:13:38Z","url":"https://github.com/z3z1ma/dbt-osmosis/actions/runs/25357462268"}
```

```json
{"conclusion":"success","createdAt":"2026-05-05T04:13:22Z","databaseId":25357462257,"event":"push","headSha":"af229e597fceb7a4a7a2902c6bb508944dc58e4e","name":"lint","status":"completed","updatedAt":"2026-05-05T04:14:23Z","url":"https://github.com/z3z1ma/dbt-osmosis/actions/runs/25357462257"}
```

```json
{"conclusion":"success","createdAt":"2026-05-05T04:13:22Z","databaseId":25357462260,"event":"push","headSha":"af229e597fceb7a4a7a2902c6bb508944dc58e4e","name":"Tests","status":"completed","updatedAt":"2026-05-05T04:25:00Z","url":"https://github.com/z3z1ma/dbt-osmosis/actions/runs/25357462260"}
```

```json
{"conclusion":"success","createdAt":"2026-05-05T04:25:02Z","databaseId":25357773183,"event":"workflow_run","headSha":"af229e597fceb7a4a7a2902c6bb508944dc58e4e","name":"Release","status":"completed","updatedAt":"2026-05-05T04:38:03Z","url":"https://github.com/z3z1ma/dbt-osmosis/actions/runs/25357773183"}
```

Release validation included Ruff, basedpyright, demo manifest parse, pytest, docs build, package build, package metadata validation, wheel smoke test, and artifact upload. The tag/publish job completed successfully; PyPI publish and release-note publish steps were skipped as expected for this run. GitHub Actions emitted non-blocking Node.js 20 deprecation and cache warnings; workflow conclusions remained `success`.

# Supports Claims

- `ticket:c10anchors30#ACC-001` through `ticket:c10anchors30#ACC-005`: the implementation commit passed remote push workflows and Release validation on `origin/main`.
- `initiative:dbt-110-111-hardening#OBJ-003`: remote CI and Release validation support the schema YAML anchor/alias preservation hardening work.

# Challenges Claims

None observed for the implementation commit.

# Environment

Commit: `af229e597fceb7a4a7a2902c6bb508944dc58e4e`

Branch: `main` on `origin`

Runtime: GitHub Actions hosted runners

OS: GitHub Actions matrix and Release runners

Relevant config: repository workflows at commit `af229e597fceb7a4a7a2902c6bb508944dc58e4e`

External service / harness / data source when applicable: GitHub Actions via `gh` CLI

# Validity

Valid for: remote CI state for commit `af229e597fceb7a4a7a2902c6bb508944dc58e4e` on `origin/main`.

Fresh enough for: ticket:c10anchors30 acceptance and closure.

Recheck when: another commit supersedes this work, workflows are rerun with different results, schema YAML reader/writer behavior changes, or a critique finding challenges this validation.

Invalidated by: failed rerun of the cited workflows or a superseding commit that changes schema YAML anchor/alias preservation, managed/unmanaged formatting behavior, tests, or workflow validation.

Supersedes / superseded by: none.

# Limitations

This evidence records remote CI success for the implementation commit. It does not prove every possible YAML anchor topology, does not automate dbt parse compatibility for the temporary anchor fixture across the full dbt matrix, and does not remove the accepted shared-node mutation behavior for alias-bearing managed subtrees.

# Result

Commit `af229e597fceb7a4a7a2902c6bb508944dc58e4e` passed Labeler, lint, Tests, and Release workflows on `origin/main`.

# Interpretation

This evidence supports ticket closure when combined with local red/green evidence, dbt parse fixture evidence, and recommended critique with ticket-owned accepted-risk dispositions. It does not broaden the support policy beyond normal `_read_yaml()` / `_write_yaml()` flows that preserve ruamel object identity.

# Related Records

- ticket:c10anchors30
- evidence:c10anchors30-yaml-anchor-validation
- critique:c10anchors30-yaml-anchor-review
- packet:ralph-ticket-c10anchors30-20260505T015001Z
- packet:ralph-ticket-c10anchors30-20260505T020517Z
- packet:ralph-ticket-c10anchors30-20260505T021703Z
- packet:ralph-ticket-c10anchors30-20260505T022457Z
- packet:ralph-ticket-c10anchors30-20260505T023255Z
- packet:ralph-ticket-c10anchors30-20260505T024405Z
- initiative:dbt-110-111-hardening
