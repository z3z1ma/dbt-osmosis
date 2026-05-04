---
id: evidence:c10lock07-adapter-bound-verification
kind: evidence
status: recorded
created_at: 2026-05-04T00:47:15Z
updated_at: 2026-05-04T00:47:15Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  tickets:
    - ticket:c10lock07
  critique:
    - critique:c10lock07-adapter-constraint-follow-up
external_refs:
  github_issue: https://github.com/z3z1ma/dbt-osmosis/issues/358
---

# Summary

Observed the local follow-up fix for the `ticket:c10lock07` main CI failure in dbt 1.8 rows. The failure mode was reproduced under the CI-pinned `uv==0.5.13`; adding an explicit `dbt-adapters>=1.16.3,<2.0` matrix constraint selected a compatible adapter/runtime set and drove the failing targeted test green.

# Procedure

Observed at: 2026-05-04T00:47:15Z

Source state: local worktree at `d72c3b805dacfb16d236b2c251f63c0c95cf68c8` plus uncommitted follow-up changes to `.github/workflows/tests.yml`, `Taskfile.yml`, and Loom reconciliation records.

Procedure:

- Created a fresh local tool environment with `.github/workflows/constraints.txt`, which pins `uv==0.5.13`.
- Created a clean Python 3.12 matrix environment with pinned uv.
- Installed `-e ".[dev,openai]"`, `dbt-core~=1.8.0`, `dbt-duckdb~=1.8.0`, and `dbt-adapters>=1.16.3,<2.0` using `uv --no-config pip install`.
- Asserted installed versions, including `dbt-adapters >= 1.16.3`.
- Ran `uv --no-config pip check` against the matrix environment.
- Ran `pytest tests/core/test_sync_operations.py::test_sync_node_to_yaml_all_versions_share_one_truthful_write` in the matrix environment.
- Ran repository checks after patching the workflow/Taskfile: `uv lock --check`, `pre-commit run --files .github/workflows/tests.yml Taskfile.yml`, and `task pip-install-smoke`.
- Sent the follow-up diff through fresh critique; the first pass found that the workflow variable was job-scoped to `lockfile`, parent moved it to workflow-level `env`, reran `uv lock --check` and pre-commit, and the second critique pass returned `pass`.

Expected result when applicable: the explicit adapter bound should prevent CI-pinned uv from selecting the previously observed incompatible `dbt-adapters==1.10.2` / `mashumaro==3.20` combination for dbt 1.8 matrix rows; local checks should pass before committing and pushing for final GitHub Actions evidence.

Actual observed result: pinned uv selected `dbt-adapters==1.16.3` and `mashumaro==3.14`; `pip check` passed; the previously failing targeted sync-operation test passed; workflow hooks passed after the env-scope fix; the plain pip smoke still passed and remained unconstrained.

Procedure verdict / exit code: pass for local follow-up verification; final GitHub Actions evidence remains pending after commit/push.

# Artifacts

Pinned-uv dbt 1.8 targeted verification output included:

- `uv 0.5.13 (c456bae5e 2024-12-27)`.
- `dbt-core: 1.8.9`.
- `dbt-duckdb: 1.8.4`.
- `dbt-adapters: 1.16.3`.
- `mashumaro: 3.14`.
- `dbt-osmosis: 1.3.0`.
- `uv --no-config pip check` reported all installed packages compatible.
- `tests/core/test_sync_operations.py::test_sync_node_to_yaml_all_versions_share_one_truthful_write` passed.

Repository checks after the follow-up patch:

- `uv lock --check` - passed, exit 0; output included `Resolved 156 packages in 7ms` after the env-scope fix.
- `pre-commit run --files .github/workflows/tests.yml Taskfile.yml` - passed, exit 0, including YAML checks and actionlint after the env-scope fix.
- `task pip-install-smoke` - passed, exit 0; `pip check` reported `No broken requirements found`, `dbt-core: 1.11.8`, `dbt-duckdb: 1.10.1`, `dbt-adapters: 1.22.10`, `dbt-osmosis: 1.3.0`, `dbt-osmosis --help` succeeded, and `import dbt_osmosis.cli.main` succeeded.

Fresh critique output:

- Initial follow-up critique verdict was `changes_required` because `DBT_ADAPTERS_CONSTRAINT` was scoped to the `lockfile` job and not inherited by the primary `tests` matrix job.
- Parent moved `DBT_ADAPTERS_CONSTRAINT` to workflow-level `env` and removed redundant job-local definitions.
- Follow-up critique verdict after that fix was `pass` with no remaining findings.

# Supports Claims

- `ticket:c10lock07#ACC-002` - local pinned-uv reproduction shows the matrix install can be made deterministic for the dbt 1.8 failure mode by explicitly constraining `dbt-adapters`; workflow-level env now makes the constraint available to matrix jobs.
- `ticket:c10lock07#ACC-003` - local pinned-uv `pip check` passed after the constrained install, and the workflow/Taskfile still run dependency consistency checks.
- `ticket:c10lock07#ACC-004` - workflow/Taskfile version assertions now include `dbt-adapters` and assert the adapter floor for uv-resolved matrix/latest rows.
- `ticket:c10lock07#ACC-005` - `task pip-install-smoke` still passes with a plain pip install and logs `dbt-adapters`; the smoke remains unconstrained to preserve package metadata coverage.

# Challenges Claims

- `ticket:c10lock07#ACC-001` through `ticket:c10lock07#ACC-005` remain pending final GitHub Actions evidence from `main` because this evidence is local plus critique-backed.

# Environment

Commit: `d72c3b805dacfb16d236b2c251f63c0c95cf68c8` plus uncommitted c10lock07 follow-up changes.

Branch: local worktree branch `loom/dbt-110-111-hardening`; delivery target remains `main`.

Runtime: local macOS/darwin; pinned local uv tool environment using `.github/workflows/constraints.txt`.

OS: macOS/darwin.

Relevant config: `.github/workflows/tests.yml`, `.github/workflows/constraints.txt`, `Taskfile.yml`, `pyproject.toml`, `uv.lock`.

External service / harness / data source when applicable: none for this local evidence; GitHub Actions evidence is still pending.

# Validity

Valid for: supporting a c10lock07 follow-up commit/push that addresses the observed dbt 1.8 main CI failure under CI-pinned uv.

Fresh enough for: review of the current uncommitted c10lock07 follow-up diff.

Recheck when: workflow dependency installation steps, Taskfile matrix/latest tasks, CI uv pin, dbt adapter constraints, package metadata, or dbt support policy changes.

Invalidated by: a failing GitHub Actions run for these changes, removal of the workflow-level adapter constraint, removal of dependency consistency checks, or evidence that the adapter floor hides a package metadata break owned by this ticket rather than `ticket:c10pkg10`.

Supersedes / superseded by: should be superseded by final `main` CI evidence after this follow-up is pushed.

# Limitations

- This evidence covers the dbt 1.8 targeted failure mode locally; it does not replace full GitHub Actions matrix evidence.
- This evidence does not prove package metadata should or should not include a direct `dbt-adapters` dependency; broader package metadata cleanup remains with `ticket:c10pkg10`.
- The plain pip smoke intentionally does not assert the adapter floor, so this record must not be used to claim every install path enforces `dbt-adapters>=1.16.3`.

# Result

The local adapter-bound follow-up passed, and fresh critique found no remaining blocker after the workflow-level env fix.

# Interpretation

The explicit adapter constraint is a CI resolver stabilizer for the observed pinned-uv/dbt 1.8 failure mode. It supports committing and pushing for final `main` CI evidence, but it is not a closure decision and does not replace the pending GitHub Actions acceptance evidence.

# Related Records

- `ticket:c10lock07`
- `evidence:c10lock07-local-dependency-resolution-verification`
- `critique:c10lock07-dependency-resolution`
- `critique:c10lock07-integration-path-follow-up`
- `critique:c10lock07-adapter-constraint-follow-up`
- `ticket:c10pkg10`
