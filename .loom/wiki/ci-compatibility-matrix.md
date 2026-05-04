---
id: wiki:ci-compatibility-matrix
kind: wiki
page_type: workflow
status: active
created_at: 2026-05-03T23:36:45Z
updated_at: 2026-05-04T04:58:20Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  tickets:
    - ticket:c10ci06
    - ticket:c10lock07
    - ticket:c10docs09
    - ticket:c10pkg10
  evidence:
    - evidence:c10ci06-ci-gate-local-verification
    - evidence:c10ci06-ci-run-no-sync-fix
    - evidence:c10ci06-main-ci-dbt18-test-fix
    - evidence:c10ci06-main-ci-success
    - evidence:c10lock07-local-dependency-resolution-verification
    - evidence:c10lock07-adapter-bound-verification
    - evidence:c10lock07-uv-01012-verification
    - evidence:c10lock07-main-ci-success
    - evidence:c10docs09-main-docs-ci-success
    - evidence:c10pkg10-package-metadata-smoke
    - evidence:c10pkg10-main-ci-release-success
  critique:
    - critique:c10ci06-dbt-111-ci-gate
    - critique:c10lock07-dependency-resolution
    - critique:c10lock07-integration-path-follow-up
    - critique:c10lock07-adapter-constraint-follow-up
    - critique:c10lock07-uv-01012-follow-up
    - critique:c10docs09-docs-ci-hardening
    - critique:c10pkg10-release-packaging-review
---

# Summary

The dbt compatibility workflow proves the supported dbt/Python matrix without letting the repository lockfile or `uv run` silently override the matrix runtime. The key rule is: each matrix row creates a clean virtual environment, installs its dbt runtime with `uv --no-config pip install`, checks dependency consistency with `uv --no-config pip check`, and invokes installed commands directly from that environment. The same `Tests` workflow also validates the Docusaurus docs build on the documented Node floor and current LTS, plus independent pip install smokes for base and supported optional extras, so release, migration, and package metadata documentation cannot silently rot.

# When To Use It

Use this page when changing `.github/workflows/tests.yml`, `Taskfile.yml`, dbt version support, Python version support, adapter mappings, docs build tooling, package extras, install smoke behavior, or tests that depend on dbt manifest shape.

# Inputs

- `.github/workflows/tests.yml` owns the CI matrix and latest-core compatibility job.
- `.github/workflows/constraints.txt` pins CI tooling, including `uv==0.10.12` at the time of the c10lock07 closure evidence.
- `Taskfile.yml` mirrors the local compatibility workflow operators should run before changing support claims.
- `pyproject.toml` owns package Python support metadata and dependency constraints.
- `ticket:c10ci06` owns the live execution and acceptance history for promoting dbt 1.11 into CI.
- `evidence:c10ci06-main-ci-success` is the accepted final observation for the dbt 1.11 gate at commit `b3470bff42566dfb475a8a21ec19e45cc7faaf0d`.
- `ticket:c10lock07` owns the live execution and acceptance history for deterministic dependency resolution and clean matrix environments.
- `evidence:c10lock07-main-ci-success` is the accepted final observation for clean matrix dependency resolution at commit `19ef3a4bd3d6e6e3f5437e634a51fc11edaa23ba`.
- `ticket:c10docs09` owns the live execution and acceptance history for adding docs build coverage to the `Tests` workflow.
- `evidence:c10docs09-main-docs-ci-success` is the accepted final observation for docs build CI at commit `12e9dfee122db41ddb8f85072e1904ecd079dd00`.
- `ticket:c10pkg10` owns the live execution and acceptance history for package extras, direct dependencies, dev dependency canonicalization, and independent pip install smokes.
- `evidence:c10pkg10-main-ci-release-success` is the accepted final observation for package metadata/install-smoke CI and release validation at commit `f4e475314bbb412fea927577f99f8e78a7258f80`.

# Procedure

The primary `Tests` matrix uses Python `3.10`, `3.11`, `3.12`, and `3.13` with dbt `1.8`, `1.9`, `1.10`, and `1.11`, excluding dbt 1.8/1.9 on Python 3.13 where that support is not expected.

The current adapter mapping is:

- dbt 1.8 -> `dbt-duckdb~=1.8.0`
- dbt 1.9 -> `dbt-duckdb~=1.9.0`
- dbt 1.10 -> `dbt-duckdb~=1.10.0`
- dbt 1.11 -> `dbt-duckdb~=1.10.1`

Each matrix job performs this sequence:

1. Install CI uv through `.github/workflows/constraints.txt`.
2. Create a clean virtual environment for the matrix row.
3. Install `dbt-osmosis`, dev/openai/duckdb extras, the mapped `dbt-core` and `dbt-duckdb` runtime, and `DBT_ADAPTERS_CONSTRAINT` with `uv --no-config pip install`.
4. Assert the installed Python, `dbt-core`, `dbt-duckdb`, `dbt-adapters`, and `dbt-osmosis` versions before tests. uv-resolved matrix/latest rows require `dbt-adapters>=1.16.3,<2.0`.
5. Run `uv --no-config pip check` for dependency consistency.
6. Run basedpyright, `dbt parse`, full pytest, and direct `dbt-osmosis` integration commands from the matrix environment.

The `latest-core-compat` job is also scheduled weekly and checks latest patch drift for dbt 1.10 and 1.11 on Python 3.13. It creates an isolated smoke environment, installs the package plus the requested dbt runtime with the same adapter floor, asserts versions, runs `uv --no-config pip check`, basedpyright, `dbt parse`, `dbt-osmosis --help`, import smoke, and pytest.

The `pip-install-smoke` job intentionally uses plain `pip` instead of uv and does not force the matrix adapter floor. It exists to exercise published package metadata outside uv resolution and should log, but not necessarily constrain, the resolved adapter version. Since `ticket:c10pkg10`, it runs independent environments for base install and each supported optional extra: `openai`, `azure`, `workbench`, `duckdb`, and `proxy`. The base smoke asserts `mysql_mimic` is absent, the workbench smoke asserts `dbt.adapters.duckdb` is absent, and the DuckDB smoke proves `.[duckdb]` supplies `dbt-duckdb` without an explicit adapter install argument.

The `docs-build` job runs on Node 18 and Node 24. Each row installs from `docs/package-lock.json` with `npm --prefix docs ci`, validates the dependency tree with `npm --prefix docs ls`, and builds the site with `npm --prefix docs run build`. Because Release waits for a successful `Tests` workflow on `main`, this docs job is part of the pre-release gate before release-local validation runs.

# Outputs

- GitHub Actions run records showing matrix and canary job status.
- Evidence records for acceptance-relevant observations.
- Ticket claim-matrix updates that distinguish supported claims from residual risks.

# Failure Modes

- Reintroducing `uv run` into a matrix row can resync or otherwise switch away from the clean matrix environment. Prefer direct commands from the matrix environment. `demo_duckdb/integration_tests.sh` still contains `uv run` and should not be called from matrix CI unless that path is made sync-safe first.
- Old resolver behavior can select a dbt adapter/runtime combination that passes installation but fails manifest/YAML behavior. Under `uv==0.5.13`, dbt 1.8 rows selected `dbt-adapters==1.10.2` and `mashumaro==3.20`, which failed `tests/core/test_sync_operations.py::test_sync_node_to_yaml_all_versions_share_one_truthful_write`. The current CI pin is `uv==0.10.12`, and uv-resolved matrix/latest installs also pass `dbt-adapters>=1.16.3,<2.0`.
- Version-specific manifest assumptions can break older dbt rows. dbt 1.8 lacks `ColumnInfo.config`, so tests for column config behavior must be version-aware while keeping dbt 1.10/1.11 config coverage active.
- Import smoke can expose unrelated optional dependency issues. Rich 15 required importing `Console` from `rich.console` instead of dereferencing `rich.console` from the package namespace.
- The dbt 1.11 adapter boundary is currently `dbt-duckdb~=1.10.1`. Do not widen that boundary without new evidence.
- Passing a push-triggered latest compatibility job does not prove the first future cron event has run; treat the cron event as a maintenance recheck, not as a separate policy owner.
- Docs dependency alignment can pass on a newer local Node while failing the documented support floor. Keep Node 18 plus current LTS coverage unless the docs tooling policy changes with new evidence.
- Optional-extra smoke can be accidentally masked if a smoke installs the same package explicitly outside the extra under test. Keep each user-facing extra in its own environment and avoid extra positional packages unless the claim explicitly requires them.

# Sources

- `ticket:c10ci06`
- `evidence:c10ci06-ci-gate-local-verification`
- `evidence:c10ci06-ci-run-no-sync-fix`
- `evidence:c10ci06-main-ci-dbt18-test-fix`
- `evidence:c10ci06-main-ci-success`
- `critique:c10ci06-dbt-111-ci-gate`
- `critique:c10ci06-no-sync-follow-up`
- `critique:c10ci06-dbt18-test-fix`
- `ticket:c10lock07`
- `evidence:c10lock07-main-ci-success`
- `critique:c10lock07-dependency-resolution`
- `critique:c10lock07-integration-path-follow-up`
- `critique:c10lock07-adapter-constraint-follow-up`
- `critique:c10lock07-uv-01012-follow-up`
- `ticket:c10docs09`
- `evidence:c10docs09-local-docs-ci-validation`
- `evidence:c10docs09-main-docs-ci-success`
- `critique:c10docs09-docs-ci-hardening`
- `ticket:c10pkg10`
- `evidence:c10pkg10-package-metadata-smoke`
- `evidence:c10pkg10-main-ci-release-success`
- `critique:c10pkg10-release-packaging-review`
- `.github/workflows/tests.yml`
- `.github/workflows/constraints.txt`
- `Taskfile.yml`

# Related Pages

- `wiki:repository-atlas`
