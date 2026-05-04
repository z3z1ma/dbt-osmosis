---
id: evidence:c10lock07-main-ci-success
kind: evidence
status: recorded
created_at: 2026-05-04T01:10:42Z
updated_at: 2026-05-04T01:10:42Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  tickets:
    - ticket:c10lock07
  wiki:
    - wiki:ci-compatibility-matrix
  evidence:
    - evidence:c10lock07-local-dependency-resolution-verification
    - evidence:c10lock07-adapter-bound-verification
    - evidence:c10lock07-uv-01012-verification
  critique:
    - critique:c10lock07-dependency-resolution
    - critique:c10lock07-integration-path-follow-up
    - critique:c10lock07-adapter-constraint-follow-up
    - critique:c10lock07-uv-01012-follow-up
external_refs:
  github_issue: https://github.com/z3z1ma/dbt-osmosis/issues/358
  tests_run: https://github.com/z3z1ma/dbt-osmosis/actions/runs/25295956433
  lint_run: https://github.com/z3z1ma/dbt-osmosis/actions/runs/25295956438
  release_run: https://github.com/z3z1ma/dbt-osmosis/actions/runs/25295956432
---

# Summary

Observed final `main` GitHub Actions success for `ticket:c10lock07` after commit `19ef3a4bd3d6e6e3f5437e634a51fc11edaa23ba`. The final run used CI `uv==0.10.12`, clean matrix environments, explicit `dbt-adapters>=1.16.3,<2.0` for uv-resolved matrix/latest installs, dependency consistency checks, version logging, and the plain pip install smoke.

# Procedure

Observed at: 2026-05-04T01:10:42Z

Source state: `main` at `19ef3a4bd3d6e6e3f5437e634a51fc11edaa23ba`.

Procedure:

- Inspected GitHub Actions runs for commit `19ef3a4bd3d6e6e3f5437e634a51fc11edaa23ba`.
- Confirmed `Tests`, `lint`, `Release`, and `Labeler` completed successfully.
- Inspected the `Run pytest (3.10, 1.8.0)` job logs for exact uv and dbt package versions.
- Inspected the `Pip install smoke` job logs for exact pip-resolved dbt package versions.

Expected result when applicable: final `main` CI should pass all c10lock07 acceptance-relevant jobs after the dependency-resolution hardening, adapter floor, and uv 0.10.12 toolchain follow-up.

Actual observed result: all observed commit runs completed successfully. The prior dbt 1.8 matrix failure rows completed successfully, and the version logs show the intended uv and adapter floor in the matrix runtime.

Procedure verdict / exit code: pass for GitHub Actions run conclusions.

# Artifacts

GitHub Actions runs for commit `19ef3a4bd3d6e6e3f5437e634a51fc11edaa23ba`:

- `Tests` run `25295956433` - success, completed 2026-05-04T01:08:44Z.
- `lint` run `25295956438` - success.
- `Release` run `25295956432` - success.
- `Labeler` run `25295956434` - success.

Acceptance-relevant `Tests` jobs included:

- `Check uv lockfile` job `74154706013` - success.
- `Pip install smoke` job `74154718711` - success.
- `Validate latest dbt compatibility (1.10.0)` job `74154718721` - success.
- `Validate latest dbt compatibility (1.11.0)` job `74154718715` - success.
- Full pytest matrix jobs for Python 3.10, 3.11, 3.12, and 3.13 across supported dbt rows - success.
- Previously failing dbt 1.8 rows now passed: `Run pytest (3.10, 1.8.0)` job `74154718799`, `Run pytest (3.11, 1.8.0)` job `74154718800`, and `Run pytest (3.12, 1.8.0)` job `74154718807`.

Extracted CI version logs from `Run pytest (3.10, 1.8.0)` job `74154718799`:

- `uv 0.10.12 (x86_64-unknown-linux-gnu)`.
- `dbt-core: 1.8.9`.
- `dbt-duckdb: 1.8.4`.
- `dbt-adapters: 1.16.3`.
- `dbt-osmosis: 1.3.0`.

Extracted CI version logs from `Pip install smoke` job `74154718711`:

- `dbt-core: 1.11.8`.
- `dbt-duckdb: 1.10.1`.
- `dbt-adapters: 1.22.10`.
- `dbt-osmosis: 1.3.0`.

# Supports Claims

- `ticket:c10lock07#ACC-001` - `Check uv lockfile` passed on `main`.
- `ticket:c10lock07#ACC-002` - all matrix rows used clean matrix environments and passed; the previously failing dbt 1.8 rows passed on `main`.
- `ticket:c10lock07#ACC-003` - matrix/latest jobs include `uv --no-config pip check`; all `Tests` jobs passed.
- `ticket:c10lock07#ACC-004` - CI logs exact uv, Python, dbt, adapter, and dbt-osmosis versions; extracted dbt 1.8 logs show `uv 0.10.12` and `dbt-adapters: 1.16.3`.
- `ticket:c10lock07#ACC-005` - `Pip install smoke` passed and logged pip-resolved package versions outside uv resolution.

# Challenges Claims

None - observed GitHub Actions results support the c10lock07 acceptance claims.

# Environment

Commit: `19ef3a4bd3d6e6e3f5437e634a51fc11edaa23ba`.

Branch: `main`.

Runtime: GitHub Actions Ubuntu runners.

OS: Ubuntu/Linux for CI; local observation via `gh` CLI.

Relevant config: `.github/workflows/tests.yml`, `.github/workflows/constraints.txt`, `.github/workflows/lint.yml`, `.github/workflows/release.yml`, `Taskfile.yml`, `pyproject.toml`, `uv.lock`.

External service / harness / data source when applicable: GitHub Actions.

# Validity

Valid for: accepting `ticket:c10lock07` as of commit `19ef3a4bd3d6e6e3f5437e634a51fc11edaa23ba`.

Fresh enough for: ticket acceptance and closure reconciliation.

Recheck when: CI workflows, constraints file, package metadata, adapter bounds, dbt support matrix, or Taskfile parity changes.

Invalidated by: a later failing required `main` CI run for the same acceptance surface, removal of clean matrix environments, removal of dependency consistency checks, or package metadata changes that alter resolver behavior without new evidence.

Supersedes / superseded by: supersedes local-only c10lock07 evidence for final acceptance purposes; future scheduled CI runs may provide maintenance evidence but do not replace this closure evidence unless they challenge it.

# Limitations

- This evidence does not close broader package metadata cleanup; `ticket:c10pkg10` still owns extras/package metadata follow-up.
- This evidence does not make `demo_duckdb/integration_tests.sh` safe for matrix CI if its `uv run` calls are reintroduced unchanged.
- This evidence does not prove future dependency releases will remain compatible; the scheduled latest compatibility job remains the maintenance signal.

# Result

The final c10lock07 `main` CI run passed.

# Interpretation

The c10lock07 implementation and follow-ups satisfy the ticket's CI dependency-resolution acceptance criteria at the observed source state. Residual package metadata cleanup remains explicitly out of scope.

# Related Records

- `ticket:c10lock07`
- `wiki:ci-compatibility-matrix`
- `evidence:c10lock07-local-dependency-resolution-verification`
- `evidence:c10lock07-adapter-bound-verification`
- `evidence:c10lock07-uv-01012-verification`
- `critique:c10lock07-dependency-resolution`
- `critique:c10lock07-integration-path-follow-up`
- `critique:c10lock07-adapter-constraint-follow-up`
- `critique:c10lock07-uv-01012-follow-up`
- `ticket:c10pkg10`
