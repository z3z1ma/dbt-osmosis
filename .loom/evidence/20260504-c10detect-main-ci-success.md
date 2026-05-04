---
id: evidence:c10detect-main-ci-success
kind: evidence
status: recorded
created_at: 2026-05-04T20:38:29Z
updated_at: 2026-05-04T20:38:29Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  initiative:
    - initiative:dbt-110-111-hardening
  ticket:
    - ticket:c10detect
  evidence:
    - evidence:c10detect-fusion-manifest-detection-validation
  critique:
    - critique:c10detect-fusion-manifest-detection-review
external_refs:
  implementation_commit: https://github.com/z3z1ma/dbt-osmosis/commit/2df9ba5f4353716a6051760affc2499044e6b54d
  lint_run: https://github.com/z3z1ma/dbt-osmosis/actions/runs/25341191710
  tests_run: https://github.com/z3z1ma/dbt-osmosis/actions/runs/25341191679
  labeler_run: https://github.com/z3z1ma/dbt-osmosis/actions/runs/25341191692
  release_run: https://github.com/z3z1ma/dbt-osmosis/actions/runs/25341712789
  github_issue: https://github.com/z3z1ma/dbt-osmosis/issues/356
  github_issue_comment: https://github.com/z3z1ma/dbt-osmosis/issues/356#issuecomment-4374318745
---

# Summary

Observed GitHub Actions success on `origin/main` for implementation commit `2df9ba5f4353716a6051760affc2499044e6b54d`, which contains the `ticket:c10detect` Fusion/future-manifest detection implementation and local Loom evidence/critique records.

# Procedure

Observed at: 2026-05-04T20:38:29Z

Source state: `origin/main` and local `HEAD` both resolved to `2df9ba5f4353716a6051760affc2499044e6b54d` after guarded push from branch `loom/dbt-110-111-hardening`.

Procedure: Queried GitHub Actions with `gh run list --branch main` and watched the push-triggered `lint` and `Tests` workflows plus the workflow-run-triggered `Release` workflow with `gh run watch --exit-status`.

Expected result when applicable: all required CI workflows for the pushed implementation commit complete successfully before ticket closure.

Actual observed result: Labeler, lint, Tests, and Release completed successfully for head `2df9ba5f4353716a6051760affc2499044e6b54d`.

Procedure verdict / exit code: pass; watched workflows exited successfully.

# Artifacts

- Commit: `2df9ba5f4353716a6051760affc2499044e6b54d` (`fix: make Fusion manifest detection fail closed`).
- Labeler run `25341191692`: success.
- lint run `25341191710`: success.
- Tests run `25341191679`: success. Included uv lockfile check, docs builds on Node 18 and Node 24, pip install smoke, latest dbt 1.10/1.11 compatibility checks, and Python/dbt matrix pytest/integration jobs.
- Release run `25341712789`: success. Included release candidate validation, Ruff, basedpyright, demo manifest parse, pytest, docs build, package build/metadata validation, wheel smoke test, artifact upload, and skipped publish/release-note steps where appropriate.
- GitHub Actions annotations included Node.js 20 deprecation warnings and transient cache save/restore warnings, but the relevant jobs completed successfully.
- GitHub issue #356 comment `4374318745` was posted with implementation, validation, critique, CI, and evidence summary; issue #356 was then closed.

# Supports Claims

- ticket:c10detect#ACC-001 — remote CI retained the dbt-core v12 non-Fusion behavior tested locally.
- ticket:c10detect#ACC-002 — remote CI covered the v13 future-manifest regression path added by the ticket.
- ticket:c10detect#ACC-003 — remote CI retained known Fusion v20 detection coverage.
- ticket:c10detect#ACC-004 — remote CI validated the changed tests/docs/help with Ruff, basedpyright, docs builds, and test suites.
- ticket:c10detect#ACC-005 — remote CI validated docs/help and override-adjacent tests after the implementation landed on `origin/main`.
- initiative:dbt-110-111-hardening#OBJ-002 — supports that the main branch now handles future manifest/Fusion detection honestly in the CI matrix.

# Challenges Claims

None - observed workflows for the implementation commit completed successfully.

# Environment

Commit: `2df9ba5f4353716a6051760affc2499044e6b54d`

Branch: `main` on `origin`; local branch `loom/dbt-110-111-hardening`

Runtime: GitHub Actions via `gh` CLI observation

OS: GitHub-hosted runners for CI; local observation from darwin

Relevant config: `.github/workflows/tests.yml`, `.github/workflows/lint.yml`, `.github/workflows/release.yml`, `.github/workflows/labeler.yml`

External service / harness / data source when applicable: GitHub Actions and GitHub issue #356

# Validity

Valid for: CI status of implementation commit `2df9ba5f4353716a6051760affc2499044e6b54d` on `origin/main`.

Fresh enough for: closing `ticket:c10detect` after ticket-owned acceptance review, with local evidence and critique also recorded.

Recheck when: source code, docs, tests, workflows, dependencies, or ticket acceptance scope change after this commit.

Invalidated by: a later commit that changes Fusion/future-manifest detection or a later CI failure on the same acceptance surface.

Supersedes / superseded by: none.

# Limitations

- This evidence records CI for the implementation commit, not every future dbt/Fusion artifact shape.
- CI emitted non-blocking Node.js 20 deprecation and cache warnings.
- This evidence does not establish package publication success because publish/release-note steps were skipped by workflow conditions.

# Result

The implementation commit for `ticket:c10detect` reached `origin/main` and all observed required CI workflows completed successfully.

# Interpretation

This evidence supports ticket closure when combined with `evidence:c10detect-fusion-manifest-detection-validation` and `critique:c10detect-fusion-manifest-detection-review`. It does not broaden the project's Fusion support claim beyond the ticket's intentionally conservative detection behavior.

# Related Records

- ticket:c10detect
- evidence:c10detect-fusion-manifest-detection-validation
- critique:c10detect-fusion-manifest-detection-review
- packet:ralph-ticket-c10detect-20260504T200316Z
- initiative:dbt-110-111-hardening
