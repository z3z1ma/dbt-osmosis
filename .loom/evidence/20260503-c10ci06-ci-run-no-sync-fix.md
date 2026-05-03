---
id: evidence:c10ci06-ci-run-no-sync-fix
kind: evidence
status: recorded
created_at: 2026-05-03T22:45:13Z
updated_at: 2026-05-03T22:45:13Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  tickets:
    - ticket:c10ci06
  evidence:
    - evidence:c10ci06-ci-gate-local-verification
external_refs:
  branch_run: https://github.com/z3z1ma/dbt-osmosis/actions/runs/25292770214
  github_issue: https://github.com/z3z1ma/dbt-osmosis/issues/357
---

# Summary

Observed branch CI run `25292770214` failing matrix jobs at installed-version assertions because `uv run` resynced the project environment back to locked `dbt-core 1.10.20` after the matrix runtime install. Recorded the workflow fix that sets `UV_NO_SYNC=1` for matrix jobs and locally verified the no-sync overlay behavior.

# Procedure

Observed at: 2026-05-03T22:45:13Z

Source state: branch `loom/dbt-110-111-hardening` at commit `25e22d3` plus uncommitted workflow/ticket/evidence updates for the no-sync fix.

Procedure:

- Watched GitHub Actions run `25292770214` after commit `25e22d3`.
- Inspected failed logs with `gh run view 25292770214 --log-failed`.
- Added `UV_FROZEN=1`, `UV_NO_PROGRESS=1`, and `UV_NO_SYNC=1` to the `tests` job in `.github/workflows/tests.yml`.
- Ran a ruamel YAML assertion script confirming the workflow env and dbt 1.11 matrix mapping.
- Explicitly installed `dbt-core~=1.11.0` and `dbt-duckdb~=1.10.1` into the sibling worktree `.venv`, then ran `env -u VIRTUAL_ENV UV_NO_SYNC=1 uv run python` to assert that `uv run` reported `dbt-core: 1.11.8` and `dbt-duckdb: 1.10.1`.

Expected result when applicable: matrix jobs should retain the dbt runtime installed by `uv pip install` instead of being resynced to the locked project dependency before assertions/tests.

Actual observed result: branch CI logs showed failures such as `AssertionError: dbt-core 1.10.20 does not match 1.11.x` after `uv run` uninstalled/reinstalled packages. The local no-sync overlay check reported `dbt-core: 1.11.8` and `dbt-duckdb: 1.10.1`.

Procedure verdict / exit code: branch CI failed before the fix; local structural and no-sync overlay checks passed after the fix.

# Artifacts

- GitHub Actions run `25292770214` passed both `Validate latest dbt compatibility` jobs, but failed multiple `Run pytest` matrix jobs at `Assert installed versions`.
- Failed log excerpt pattern: `AssertionError: dbt-core 1.10.20 does not match 1.11.x` or the requested 1.8/1.9 prefix.
- Workflow assertion output: `workflow uv/no-sync and matrix assertions ok`.
- Local overlay assertion output: `dbt-core: 1.11.8` and `dbt-duckdb: 1.10.1`.

# Supports Claims

- `ticket:c10ci06#ACC-002` - supports that installed-version assertions detected incorrect runtime behavior and that the workflow now preserves the overlaid runtime for `uv run`.
- `ticket:c10ci06#ACC-003` - partially supports that matrix jobs can execute against the installed dbt runtime; full support still requires a passing GitHub Actions run after the fix lands on `main`.
- `ticket:c10ci06#ACC-004` - supports matching CI behavior to the existing Taskfile `UV_NO_SYNC=1` local compatibility posture.

# Challenges Claims

- `ticket:c10ci06#ACC-003` - branch run `25292770214` challenged full CI support before the `UV_NO_SYNC=1` workflow fix.

# Environment

Commit: `25e22d3` plus uncommitted no-sync fix.

Branch: `loom/dbt-110-111-hardening` during observation; intended delivery target is `main`.

Runtime: GitHub Actions Ubuntu runners; local macOS/darwin harness with uv and sibling worktree `.venv`.

OS: GitHub Actions Ubuntu for failed run; macOS/darwin for local fix verification.

Relevant config: `.github/workflows/tests.yml`, `Taskfile.yml`.

External service / harness / data source when applicable: GitHub Actions run `25292770214`.

# Validity

Valid for: diagnosing and fixing the CI matrix environment resync failure observed after commit `25e22d3`.

Fresh enough for: committing the workflow no-sync fix and rerunning CI from `main`.

Recheck when: `.github/workflows/tests.yml`, uv behavior, lockfile policy, matrix runtime installation, or Taskfile compatibility commands change.

Invalidated by: a later workflow change that removes `UV_NO_SYNC=1` or changes matrix runtime installation behavior.

Supersedes / superseded by: should be superseded by a passing `main` CI run after the no-sync fix lands.

# Limitations

- This evidence does not prove the full matrix passes after the fix; it records the prior failure and the local no-sync behavior that addresses it.
- This evidence does not resolve broader lockfile determinism work owned by `ticket:c10lock07`.
- This evidence does not prove scheduled canary behavior beyond workflow structure.

# Result

The branch CI failure was caused by `uv run` resyncing matrix jobs to the locked project dependency set after the dbt runtime overlay. Adding `UV_NO_SYNC=1` to the workflow matrix job aligns CI with the local Taskfile and keeps `uv run` on the installed matrix runtime.

# Interpretation

The no-sync fix is within `ticket:c10ci06` scope because installed-version assertions exposed that CI was not actually testing the requested dbt versions. The ticket still needs fresh `main` CI evidence before acceptance or closure.

# Related Records

- `ticket:c10ci06`
- `evidence:c10ci06-ci-gate-local-verification`
- `critique:c10ci06-dbt-111-ci-gate`
