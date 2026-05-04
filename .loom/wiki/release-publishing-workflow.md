---
id: wiki:release-publishing-workflow
kind: wiki
page_type: workflow
status: active
created_at: 2026-05-04T02:40:27Z
updated_at: 2026-05-04T04:58:20Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  tickets:
    - ticket:c10rel08
    - ticket:c10docs09
    - ticket:c10pkg10
  evidence:
    - evidence:c10rel08-main-release-success
    - evidence:c10rel08-main-release-detached-head-failure
    - evidence:c10docs09-main-docs-ci-success
    - evidence:c10pkg10-main-ci-release-success
  critique:
    - critique:c10rel08-release-workflow-hardening
    - critique:c10docs09-docs-ci-hardening
    - critique:c10pkg10-release-packaging-review
---

# Summary

The Release workflow is a post-`Tests` publishing gate. It should not tag, publish to PyPI, or publish GitHub release notes until the tested commit has passed the full `Tests` workflow and the release-local validation job.

# When To Use It

Use this page when changing `.github/workflows/release.yml`, package build or metadata checks, PyPI publishing, Release Drafter behavior, or the release-note source of truth.

# Current Shape

Release is triggered by `workflow_run` for the `Tests` workflow completing on `main`. Both Release jobs additionally guard that the upstream run was successful, came from a same-repository push event, and used `head_branch == 'main'`. Since `ticket:c10docs09`, the upstream `Tests` workflow includes Node 18 and Node 24 docs builds, so Release is indirectly gated by docs install, dependency-tree validation, and build success before the release workflow starts.

The `validate` job has read-only repository permission and checks out `github.event.workflow_run.head_sha` with `persist-credentials: false`. It runs lock freshness, Python checks, dbt parse, pytest, docs build, package build, metadata validation, and clean wheel install smoke before uploading the validated `dist/*` artifacts. Since `ticket:c10pkg10`, successful Release validation is also packaging evidence for optional-extra and built-wheel metadata changes after the `Tests` workflow's independent pip smokes have passed.

The `release` job depends on `validate`, has `contents: write` and `pull-requests: read`, checks out the same tested SHA, attaches that SHA to a local `main` branch, downloads the validated artifacts, and then runs tag detection. The local branch attach is required because `salsify/action-detect-and-tag-new-version@v2.0.3` expects a local `refs/heads/main` when invoked from the `workflow_run` checkout.

If no new version tag is detected, the workflow bumps a developmental version and builds a developmental package only. PyPI publishing and Release Drafter publishing are skipped in the no-tag path.

If a new version tag is detected, the workflow publishes the already validated package artifacts to PyPI and publishes GitHub release notes through Release Drafter. Release Drafter is the GitHub release notes source for this workflow.

# Failure Modes

- Do not move tag detection before the `validate` job. That can leave irreversible tags behind after broken package, docs, or test output.
- Do not give validation checkout persisted write credentials. Validation should not be able to push tags or mutate repository state.
- Do not let `workflow_run` accept scheduled or pull-request `Tests` runs for publishing. Keep same-repository push and `main` branch guards.
- Do not remove the local `main` branch attach unless the tag-detection action is replaced or proven not to require `refs/heads/main`.
- Treat PyPI token scope as external configuration; repository evidence cannot prove the token is restricted to the intended project.
- Tag creation, PyPI publish, and GitHub release publish are not atomic; a future publish failure after tag creation may still require operator cleanup.
- Release can validate package build, metadata, and built-wheel install for the committed package state, but it does not replace the `Tests` workflow's per-extra independent pip smoke coverage.

# Sources

- `ticket:c10rel08`
- `critique:c10rel08-release-workflow-hardening`
- `evidence:c10rel08-local-release-workflow-validation`
- `evidence:c10rel08-main-release-detached-head-failure`
- `evidence:c10rel08-main-release-success`
- `ticket:c10docs09`
- `evidence:c10docs09-main-docs-ci-success`
- `critique:c10docs09-docs-ci-hardening`
- `ticket:c10pkg10`
- `evidence:c10pkg10-main-ci-release-success`
- `critique:c10pkg10-release-packaging-review`
- `.github/workflows/release.yml`

# Related Pages

- `wiki:ci-compatibility-matrix`
