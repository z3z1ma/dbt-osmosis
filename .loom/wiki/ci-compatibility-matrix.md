---
id: wiki:ci-compatibility-matrix
kind: wiki
page_type: workflow
status: active
created_at: 2026-05-03T23:36:45Z
updated_at: 2026-05-03T23:36:45Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  tickets:
    - ticket:c10ci06
  evidence:
    - evidence:c10ci06-ci-gate-local-verification
    - evidence:c10ci06-ci-run-no-sync-fix
    - evidence:c10ci06-main-ci-dbt18-test-fix
    - evidence:c10ci06-main-ci-success
  critique:
    - critique:c10ci06-dbt-111-ci-gate
---

# Summary

The dbt compatibility workflow proves the supported dbt/Python matrix without letting the repository lockfile silently override the matrix runtime. The key rule is: when CI overlays a specific `dbt-core` and `dbt-duckdb` version after `uv sync`, every later `uv run` step must preserve that overlay with `UV_NO_SYNC=1` and assert the installed versions before running meaningful checks.

# When To Use It

Use this page when changing `.github/workflows/tests.yml`, `Taskfile.yml`, dbt version support, Python version support, adapter mappings, or tests that depend on dbt manifest shape.

# Inputs

- `.github/workflows/tests.yml` owns the CI matrix and latest-core compatibility job.
- `Taskfile.yml` mirrors the local compatibility workflow operators should run before changing support claims.
- `pyproject.toml` owns package Python support metadata and dependency constraints.
- `ticket:c10ci06` owns the live execution and acceptance history for promoting dbt 1.11 into CI.
- `evidence:c10ci06-main-ci-success` is the accepted final observation for the dbt 1.11 gate at commit `b3470bff42566dfb475a8a21ec19e45cc7faaf0d`.

# Procedure

The primary `Tests` matrix uses Python `3.10`, `3.11`, `3.12`, and `3.13` with dbt `1.8`, `1.9`, `1.10`, and `1.11`, excluding dbt 1.8/1.9 on Python 3.13 where that support is not expected.

The current adapter mapping is:

- dbt 1.8 -> `dbt-duckdb~=1.8.0`
- dbt 1.9 -> `dbt-duckdb~=1.9.0`
- dbt 1.10 -> `dbt-duckdb~=1.10.0`
- dbt 1.11 -> `dbt-duckdb~=1.10.1`

Each matrix job performs this sequence:

1. Install repository dependencies with `uv sync --extra dev --extra openai`.
2. Overlay the matrix dbt runtime with `uv pip install "dbt-core~=<matrix>" "dbt-duckdb~=<mapped>"`.
3. Preserve the overlay by running later `uv run` commands with workflow-level `UV_NO_SYNC=1`.
4. Assert the installed Python, `dbt-core`, `dbt-duckdb`, and `dbt-osmosis` versions before tests.
5. Run basedpyright, `dbt parse`, full pytest, and the DuckDB integration script.

The `latest-core-compat` job is also scheduled weekly and checks latest patch drift for dbt 1.10 and 1.11 on Python 3.13. It creates an isolated smoke environment, installs the package plus the requested dbt runtime, asserts versions, runs basedpyright, `dbt parse`, `dbt-osmosis --help`, import smoke, and pytest.

# Outputs

- GitHub Actions run records showing matrix and canary job status.
- Evidence records for acceptance-relevant observations.
- Ticket claim-matrix updates that distinguish supported claims from residual risks.

# Failure Modes

- Missing `UV_NO_SYNC=1` can make `uv run` resync the environment back to the lockfile after the matrix dbt runtime was installed. This was observed in branch CI run `25292770214` and fixed before acceptance.
- Version-specific manifest assumptions can break older dbt rows. dbt 1.8 lacks `ColumnInfo.config`, so tests for column config behavior must be version-aware while keeping dbt 1.10/1.11 config coverage active.
- Import smoke can expose unrelated optional dependency issues. Rich 15 required importing `Console` from `rich.console` instead of dereferencing `rich.console` from the package namespace.
- The dbt 1.11 adapter boundary is currently `dbt-duckdb~=1.10.1`. Do not widen that boundary without new evidence.
- Passing a push-triggered latest compatibility job does not prove the first future cron event has run; treat the cron event as a maintenance recheck, not as a separate policy owner.

# Sources

- `ticket:c10ci06`
- `evidence:c10ci06-ci-gate-local-verification`
- `evidence:c10ci06-ci-run-no-sync-fix`
- `evidence:c10ci06-main-ci-dbt18-test-fix`
- `evidence:c10ci06-main-ci-success`
- `critique:c10ci06-dbt-111-ci-gate`
- `critique:c10ci06-no-sync-follow-up`
- `critique:c10ci06-dbt18-test-fix`
- `.github/workflows/tests.yml`
- `Taskfile.yml`

# Related Pages

- `wiki:repository-atlas`
