---
id: evidence:c10rel08-main-release-detached-head-failure
kind: evidence
status: recorded
created_at: 2026-05-04T02:15:26Z
updated_at: 2026-05-04T02:15:26Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  tickets:
    - ticket:c10rel08
  critique:
    - critique:c10rel08-release-workflow-hardening
external_refs:
  github_run: https://github.com/z3z1ma/dbt-osmosis/actions/runs/25297463535
  github_issue: https://github.com/z3z1ma/dbt-osmosis/issues/359
---

# Summary

Observed the first `main` Release run after `bdc0965` validated successfully, then failed in the post-validation tag/publish job because the tested commit was checked out as detached `HEAD` and `salsify/action-detect-and-tag-new-version@v2.0.3` expected a local `refs/heads/main`.

# Procedure

Observed at: 2026-05-04T02:15:26Z.

Source state: `bdc0965ba793b7f13c4c7350bdcabebacb94d46b` on `origin/main`.

Procedure:

- Pushed `bdc0965` to `origin/main`.
- Observed `Tests` run `25297246058` complete successfully for the commit.
- Observed Release run `25297463535` trigger from `workflow_run` for the same commit.
- Inspected failed Release job log for `Tag and publish release` job `74159248059`.

Expected result when applicable: after validation passes, the tag/publish job should detect whether the tested commit needs a version tag without failing on checkout shape.

Actual observed result: `Validate release candidate` passed all validation steps, including pytest, docs build, package build, `twine check`, and wheel install smoke. `Tag and publish release` failed at `Detect and tag new version` with `git checkout refs/heads/main` because no local `main` branch existed in the detached-SHA checkout. PyPI publish and release-note steps were skipped.

Procedure verdict / exit code: red for the overall Release workflow; validation gate passed, post-validation tag detection failed.

# Artifacts

- `Tests` run `25297246058` - success.
- Release run `25297463535` - failure.
- `Validate release candidate` job `74158598853` - success.
- `Tag and publish release` job `74159248059` - failure.
- Failed step: `Detect and tag new version`.
- Error: `Command failed with exit code 1: git checkout refs/heads/main`; `error: pathspec 'refs/heads/main' did not match any file(s) known to git`.
- Skipped side-effect steps after failure: developmental build, PyPI publish, Release Drafter publish.

# Supports Claims

- `ticket:c10rel08#ACC-003` - `Tests` passed before Release triggered, and the Release validation job passed pytest, docs build, package build, metadata validation, and wheel smoke before tag detection ran.
- `ticket:c10rel08#ACC-004` - the failed post-validation tag detection did not publish PyPI or GitHub release notes.

# Challenges Claims

- `ticket:c10rel08#ACC-004` remains unaccepted because the overall Release workflow did not complete successfully on `main`.

# Environment

Commit: `bdc0965ba793b7f13c4c7350bdcabebacb94d46b`.

Branch: `main`.

Runtime: GitHub Actions `ubuntu-latest` / Ubuntu 24.04 runner.

Relevant config: `.github/workflows/release.yml` at `bdc0965`.

# Validity

Valid for: diagnosing the first live c10rel08 Release run failure.

Fresh enough for: the immediate follow-up fix that attaches the tested commit to a local `main` branch before invoking the tag-detection action.

Recheck when: the release job checkout or tag-detection action changes.

Invalidated by: a later successful Release run for a descendant commit.

# Limitations

- This is red evidence for the overall Release workflow, not acceptance evidence.
- It does not exercise PyPI publishing because the tag-detection failure skipped publish steps.

# Result

The validation-before-tag gate worked, but the post-validation tag job needs a local `main` branch for `action-detect-and-tag-new-version`.

# Interpretation

The minimal follow-up is to keep checking out the tested SHA for provenance, then create a local `main` branch at that SHA before running tag detection.

# Related Records

- `ticket:c10rel08`
- `critique:c10rel08-release-workflow-hardening`
