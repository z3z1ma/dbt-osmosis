---
id: evidence:c10lint24-main-ci-success
kind: evidence
status: recorded
created_at: 2026-05-04T19:33:25Z
updated_at: 2026-05-04T19:33:25Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  initiative:
    - initiative:dbt-110-111-hardening
  ticket:
    - ticket:c10lint24
  evidence:
    - evidence:c10lint24-lint-diff-cli-validation
external_refs:
  github_issue: https://github.com/z3z1ma/dbt-osmosis/issues/375
  lint_run: https://github.com/z3z1ma/dbt-osmosis/actions/runs/25338008487
  tests_run: https://github.com/z3z1ma/dbt-osmosis/actions/runs/25338008545
  release_run: https://github.com/z3z1ma/dbt-osmosis/actions/runs/25338519017
---

# Summary

Observed GitHub Actions success on `origin/main` for the `ticket:c10lint24` commit `b9dc86279d5074397306663631b7d47f3e824be0` and closed the linked GitHub issue.

# Procedure

Observed at: 2026-05-04T19:33:25Z

Source state: `origin/main` at `b9dc86279d5074397306663631b7d47f3e824be0` after guarded push from branch `loom/dbt-110-111-hardening`.

Procedure: watched GitHub Actions runs triggered by the push, then commented on and closed GitHub issue #375.

Expected result when applicable: lint, Tests, Labeler, and Release workflows should complete successfully for the pushed commit before treating remote validation as complete.

Actual observed result: lint, Tests, Labeler, and Release completed successfully for `b9dc86279d5074397306663631b7d47f3e824be0`. Issue #375 was commented and closed.

Procedure verdict / exit code: pass; `gh run watch --exit-status` returned success for lint, Tests, Labeler, and Release.

# Artifacts

- Commit: `b9dc86279d5074397306663631b7d47f3e824be0` (`fix: honor lint and diff CLI selection rules`).
- Guarded push: `git fetch origin main && git merge-base --is-ancestor origin/main HEAD && git push origin HEAD:main` advanced `origin/main` from `b2e4e98` to `b9dc862`.
- CI lint run `25338008487` -> success.
- CI Tests run `25338008545` -> success.
- CI Labeler run `25338008503` -> success.
- CI Release run `25338519017` -> success; validation/build/package checks passed, publish/release-note steps were skipped.
- GitHub issue #375 comment: https://github.com/z3z1ma/dbt-osmosis/issues/375#issuecomment-4373911288
- GitHub issue #375 final state: closed.
- Non-blocking annotations observed: Node.js 20 action deprecation warnings and transient GitHub cache save/restore warnings on Release validation.

# Supports Claims

- `ticket:c10lint24#ACC-001` through `ticket:c10lint24#ACC-006` — supports that the implementation commit containing the accepted local validation also passed the repository's remote validation workflows.
- `initiative:dbt-110-111-hardening#OBJ-006` — supports that the user-facing CLI fix passed the main-branch validation surface.

# Challenges Claims

None - no remote validation failure was observed for this commit.

# Environment

Commit: `b9dc86279d5074397306663631b7d47f3e824be0`

Branch: `origin/main`

Runtime: GitHub Actions plus local `gh` CLI observation

OS: GitHub-hosted runners as configured by workflows; local observation from darwin

Relevant config: `.github/workflows/tests.yml`, `.github/workflows/lint.yml`, `.github/workflows/release.yml`

External service / harness / data source when applicable: GitHub Actions and GitHub Issues

# Validity

Valid for: remote validation of commit `b9dc86279d5074397306663631b7d47f3e824be0` on `origin/main`.

Fresh enough for: closure support for `ticket:c10lint24` and issue #375.

Recheck when: a later commit changes the implementation, workflows, dependencies, or branch head used for acceptance.

Invalidated by: a later failing rerun for the same required workflows, or a new commit superseding `b9dc862` for this ticket's implementation.

Supersedes / superseded by: complements `evidence:c10lint24-lint-diff-cli-validation`.

# Limitations

- This evidence observes CI success; it does not add new behavioral scenarios beyond the workflow suites.
- Node.js 20 deprecation and cache warnings were non-blocking annotations, not failures.
- It does not change the low red-evidence limitation recorded for `ticket:c10lint24#ACC-004`.

# Result

Remote validation completed successfully for the pushed `ticket:c10lint24` commit and the linked GitHub issue was closed.

# Interpretation

This evidence removes the prior remote-CI residual risk for `ticket:c10lint24`. It does not supersede the local validation evidence or the critique limitation around strict red evidence for one sub-claim.

# Related Records

- ticket:c10lint24
- evidence:c10lint24-lint-diff-cli-validation
- critique:c10lint24-lint-diff-cli-review
- packet:ralph-ticket-c10lint24-20260504T184608Z
