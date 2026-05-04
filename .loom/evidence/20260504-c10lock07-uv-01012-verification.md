---
id: evidence:c10lock07-uv-01012-verification
kind: evidence
status: recorded
created_at: 2026-05-04T00:57:09Z
updated_at: 2026-05-04T00:57:09Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  tickets:
    - ticket:c10lock07
  critique:
    - critique:c10lock07-uv-01012-follow-up
    - critique:c10lock07-adapter-constraint-follow-up
external_refs:
  github_issue: https://github.com/z3z1ma/dbt-osmosis/issues/358
---

# Summary

Observed the local follow-up requested by the operator to use `uv==0.10.12` in CI. The CI constraints file now pins uv 0.10.12, and a clean dbt 1.8 matrix-style install with the existing adapter floor passed dependency and targeted test verification under that exact uv version.

# Procedure

Observed at: 2026-05-04T00:57:09Z

Source state: local worktree at `b228b0a1d3f18ee418f242d85a79760a3c393d39` plus uncommitted change to `.github/workflows/constraints.txt` and Loom reconciliation records.

Procedure:

- Updated `.github/workflows/constraints.txt` from `uv==0.5.13` to `uv==0.10.12`.
- Created a fresh local tool environment and installed from `.github/workflows/constraints.txt`.
- Created a clean Python 3.12 matrix environment with exact `uv==0.10.12`.
- Installed `-e ".[dev,openai]"`, `dbt-core~=1.8.0`, `dbt-duckdb~=1.8.0`, and `dbt-adapters>=1.16.3,<2.0` using `uv --no-config pip install`.
- Asserted installed dbt and adapter versions.
- Ran `uv --no-config pip check` against the matrix environment.
- Ran `pytest tests/core/test_sync_operations.py::test_sync_node_to_yaml_all_versions_share_one_truthful_write` in the matrix environment.
- Ran `uv lock --check` and `pre-commit run --files .github/workflows/constraints.txt .github/workflows/tests.yml Taskfile.yml`.
- Sent the uncommitted uv pin diff through fresh release-packaging/test-coverage/operator-clarity critique.

Expected result when applicable: CI should install uv 0.10.12 or newer-equivalent pinned behavior through the constraints file, and the prior dbt 1.8 failure mode should still be covered by local clean matrix verification before push.

Actual observed result: the local tool environment installed exact `uv 0.10.12`; the clean dbt 1.8 matrix install selected `dbt-adapters==1.16.3` and `mashumaro==3.14`; `pip check` passed; the previously failing targeted sync-operation test passed; lock/pre-commit checks passed; fresh critique returned `pass`.

Procedure verdict / exit code: pass for local uv 0.10.12 follow-up verification; final GitHub Actions evidence remains pending after commit/push.

# Artifacts

Exact uv toolchain and targeted dbt 1.8 verification output included:

- `uv 0.10.12 (00d72dac7 2026-03-19 aarch64-apple-darwin)`.
- `dbt-core: 1.8.9`.
- `dbt-duckdb: 1.8.4`.
- `dbt-adapters: 1.16.3`.
- `mashumaro: 3.14`.
- `dbt-osmosis: 1.3.0`.
- `uv --no-config pip check` reported all installed packages compatible.
- `tests/core/test_sync_operations.py::test_sync_node_to_yaml_all_versions_share_one_truthful_write` passed.

Repository checks after the uv pin patch:

- `uv lock --check` - passed, exit 0; output included `Resolved 156 packages in 8ms`.
- `pre-commit run --files .github/workflows/constraints.txt .github/workflows/tests.yml Taskfile.yml` - passed, exit 0, including YAML checks and actionlint.

Fresh critique output:

- `critique:c10lock07-uv-01012-follow-up` returned `pass` with no open findings.
- The reviewer checked that `tests.yml`, `lint.yml`, and `release.yml` install uv through `.github/workflows/constraints.txt`, so they inherit the new pin.

# Supports Claims

- `ticket:c10lock07#ACC-001` - lock freshness checks still pass after the CI uv pin update.
- `ticket:c10lock07#ACC-002` - exact uv 0.10.12 local clean matrix verification passed for the prior dbt 1.8 failure mode with the adapter floor.
- `ticket:c10lock07#ACC-003` - `uv --no-config pip check` passed under exact uv 0.10.12.
- `ticket:c10lock07#ACC-004` - the exact uv version is logged locally; CI installs uv through the shared constraints file and prints `uv --version` in matrix/latest jobs.

# Challenges Claims

- `ticket:c10lock07#ACC-001` through `ticket:c10lock07#ACC-005` remain pending final GitHub Actions evidence from `main` because this evidence is local plus critique-backed.

# Environment

Commit: `b228b0a1d3f18ee418f242d85a79760a3c393d39` plus uncommitted uv 0.10.12 follow-up changes.

Branch: local worktree branch `loom/dbt-110-111-hardening`; delivery target remains `main`.

Runtime: local macOS/darwin; exact local uv tool environment using `.github/workflows/constraints.txt`.

OS: macOS/darwin.

Relevant config: `.github/workflows/constraints.txt`, `.github/workflows/tests.yml`, `.github/workflows/lint.yml`, `.github/workflows/release.yml`, `Taskfile.yml`, `pyproject.toml`, `uv.lock`.

External service / harness / data source when applicable: none for this local evidence; GitHub Actions evidence is still pending.

# Validity

Valid for: supporting a c10lock07 follow-up commit/push that updates the CI uv toolchain to 0.10.12 while preserving local coverage of the prior dbt 1.8 matrix failure mode.

Fresh enough for: review of the current uncommitted uv 0.10.12 follow-up diff.

Recheck when: CI tool constraints, workflow uv installation steps, adapter constraints, package metadata, dbt support policy, or uv behavior changes.

Invalidated by: a failing GitHub Actions run for these changes, removal of the shared uv constraints path, removal of matrix dependency checks, or changing the adapter floor without new evidence.

Supersedes / superseded by: should be superseded by final `main` CI evidence after this follow-up is pushed.

# Limitations

- This evidence covers a local exact-uv dbt 1.8 targeted path; it does not replace full GitHub Actions matrix evidence.
- This evidence does not prove local Taskfile users have uv 0.10.12 installed; the operator request was scoped to CI.
- Broader package metadata and extras cleanup remains with `ticket:c10pkg10`.

# Result

The uv 0.10.12 follow-up passed local verification and fresh critique found no blocker.

# Interpretation

Pinning CI uv to 0.10.12 through the shared constraints file satisfies the operator-requested toolchain update while preserving deterministic CI installation behavior for c10lock07. The ticket still needs final `main` CI evidence before acceptance.

# Related Records

- `ticket:c10lock07`
- `evidence:c10lock07-adapter-bound-verification`
- `critique:c10lock07-adapter-constraint-follow-up`
- `critique:c10lock07-uv-01012-follow-up`
- `ticket:c10pkg10`
