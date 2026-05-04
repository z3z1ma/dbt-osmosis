---
id: evidence:c10rel08-main-release-success
kind: evidence
status: recorded
created_at: 2026-05-04T02:40:27Z
updated_at: 2026-05-04T02:40:27Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  tickets:
    - ticket:c10rel08
  critique:
    - critique:c10rel08-release-workflow-hardening
  previous_evidence:
    - evidence:c10rel08-local-release-workflow-validation
    - evidence:c10rel08-main-release-detached-head-failure
external_refs:
  github_tests_run: https://github.com/z3z1ma/dbt-osmosis/actions/runs/25297885314
  github_lint_run: https://github.com/z3z1ma/dbt-osmosis/actions/runs/25297885293
  github_release_run: https://github.com/z3z1ma/dbt-osmosis/actions/runs/25298118037
  github_issue: https://github.com/z3z1ma/dbt-osmosis/issues/359
---

# Summary

Observed the final `ticket:c10rel08` follow-up commit `8058f91d3faf90300ad571db11522bee7e02c45d` pass `lint`, `Tests`, and the new `workflow_run` Release workflow on `main`.

# Procedure

Observed at: 2026-05-04T02:40:27Z.

Source state: `8058f91d3faf90300ad571db11522bee7e02c45d` on `origin/main`.

Procedure:

- Pushed follow-up commit `8058f91d3faf90300ad571db11522bee7e02c45d` to `origin/main`.
- Observed push-triggered `lint` run `25297885293` complete successfully.
- Observed push-triggered `Tests` run `25297885314` complete successfully.
- Observed `workflow_run` Release run `25298118037` trigger for the same commit only after `Tests` succeeded.
- Inspected Release run job and step status.

Expected result when applicable: Release should validate the tested commit before tag detection, then complete no-tag behavior without publishing PyPI or GitHub release notes.

Actual observed result: Release completed successfully. The read-only `Validate release candidate` job passed checkout, Python/Node setup, pip/uv install, `uv lock --check`, dependency install, ruff, basedpyright, dbt parse, pytest, docs install/build, package build, `twine check`, wheel install smoke, release-note declaration, and artifact upload. The `Tag and publish release` job passed checkout, local `main` branch attach, setup, artifact download, parent check, tag detection, developmental version bump, and developmental package build. PyPI publish and Release Drafter publish were skipped because no new version tag was detected.

Procedure verdict / exit code: pass.

# Artifacts

- Commit: `8058f91d3faf90300ad571db11522bee7e02c45d`.
- `lint` run `25297885293` - success.
- `Tests` run `25297885314` - success.
- Release run `25298118037` - success.
- Release `Validate release candidate` job `74160202205` - success.
- Release `Tag and publish release` job `74160818441` - success.
- `Attach tested commit to main branch` step - success.
- `Detect and tag new version` step - success.
- `Publish package on PyPI` step - skipped in the no-tag path.
- `Publish the release notes` step - skipped in the no-tag path.

# Supports Claims

- `ticket:c10rel08#ACC-001` - Release built package artifacts and validated metadata in the `validate` job before tag detection in the release job.
- `ticket:c10rel08#ACC-002` - Release wheel smoke verified `dbt-osmosis --help`, `python -m dbt_osmosis --help`, import of `dbt_osmosis.cli.main`, and dependency consistency.
- `ticket:c10rel08#ACC-003` - `Tests` passed before Release triggered, and Release validation passed tests, docs build, package build, metadata validation, and wheel smoke before PyPI publish could run.
- `ticket:c10rel08#ACC-004` - no tag/publish side effect occurred before validation; no-tag Release completed with PyPI and GitHub release-note publishing skipped.
- `ticket:c10rel08#ACC-005` - final workflow used read-only validation permissions and reserved write permissions for the post-validation release job.
- `ticket:c10rel08#ACC-006` - Release declares Release Drafter as the GitHub release notes source and only invokes it for detected tags.

# Challenges Claims

None for the ticket acceptance criteria.

# Environment

Commit: `8058f91d3faf90300ad571db11522bee7e02c45d`.

Branch: `main`.

Runtime: GitHub Actions `ubuntu-latest` / Ubuntu 24.04 runner.

Relevant config: `.github/workflows/release.yml`, `.github/workflows/tests.yml`, `.github/workflows/constraints.txt`, `docs/package-lock.json`, `pyproject.toml`, `uv.lock`.

# Validity

Valid for: accepting `ticket:c10rel08` release workflow hardening.

Fresh enough for: closure of `ticket:c10rel08` at commit `8058f91d3faf90300ad571db11522bee7e02c45d`.

Recheck when: release workflow validation ordering, tag detection action, PyPI publishing auth, Release Drafter integration, package build metadata, or docs/test gates change.

Invalidated by: a later failing Release run caused by this workflow shape, or a later workflow change without new release evidence.

Supersedes / superseded by: supersedes `evidence:c10rel08-main-release-detached-head-failure` for acceptance while preserving it as red/green diagnostic history.

# Limitations

- The observed run exercised no-tag behavior; it did not publish to PyPI or create a GitHub release.
- PyPI token scope and project restriction remain external to repository evidence.
- Tag creation, PyPI publishing, and GitHub release publishing are not atomic if a future tagged release publish fails after tag creation.

# Result

The c10rel08 release workflow hardening passed on `main`.

# Interpretation

The workflow now demonstrates the intended safety gate: successful full `Tests` plus release-local validation must complete before tag detection and any publish path can run.

# Related Records

- `ticket:c10rel08`
- `critique:c10rel08-release-workflow-hardening`
- `evidence:c10rel08-local-release-workflow-validation`
- `evidence:c10rel08-main-release-detached-head-failure`
