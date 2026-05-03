---
id: evidence:c10ci06-ci-gate-local-verification
kind: evidence
status: recorded
created_at: 2026-05-03T22:30:07Z
updated_at: 2026-05-03T22:30:07Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  tickets:
    - ticket:c10ci06
  packets:
    - packet:ralph-ticket-c10ci06-20260503T222300Z
external_refs:
  github_issue: https://github.com/z3z1ma/dbt-osmosis/issues/357
  failing_run: https://github.com/z3z1ma/dbt-osmosis/actions/runs/25291931743
---

# Summary

Observed local validation for the dbt 1.11 CI-gate change on `ticket:c10ci06`, including workflow/Taskfile structural checks, CLI import smoke, and an isolated `dbt-core~=1.11.0` plus `dbt-duckdb~=1.10.1` parse/import/CLI smoke.

# Procedure

Observed at: 2026-05-03T22:30:07Z

Source state: branch `loom/dbt-110-111-hardening` at commit `063b0a03077841ec91765937fda9e360d8ae4473` plus uncommitted `ticket:c10ci06` diff touching `.github/workflows/tests.yml`, `Taskfile.yml`, `src/dbt_osmosis/core/logger.py`, this packet, and this evidence record.

Procedure:

- Ran `uv run ruff check src/dbt_osmosis/core/logger.py`.
- Ran `uv run python -c "import dbt_osmosis; import dbt_osmosis.cli.main; print('import smoke ok')"`.
- Ran `uv run dbt-osmosis --help`.
- Ran a ruamel YAML assertion script over `.github/workflows/tests.yml` and `Taskfile.yml` to check the weekly schedule, dbt 1.11 matrix entry, DuckDB adapter mapping, latest 1.10/1.11 canary matrix, and local Taskfile parity.
- Ran `task --list`.
- Created a fresh uv venv and installed `dbt-core~=1.11.0`, `dbt-duckdb~=1.10.1`, and `.`; then ran installed-version assertions, `dbt --version`, `dbt parse --project-dir demo_duckdb --profiles-dir demo_duckdb -t test`, `dbt-osmosis --help`, and `python -c "import dbt_osmosis; import dbt_osmosis.cli.main; print('dbt 1.11 import smoke ok')"`.

Expected result when applicable: lint/import/CLI checks pass; workflow and Taskfile encode the declared dbt 1.11 support boundary; the isolated dbt 1.11 runtime imports the package, parses the demo fixture, and shows CLI help without the prior Rich import failure.

Actual observed result: all commands completed successfully. The isolated smoke installed `dbt-core: 1.11.8`, `dbt-duckdb: 1.10.1`, `dbt-osmosis: 1.3.0`, and `rich: 15.0.0`; `dbt parse` completed with only existing generic-test deprecation warnings; `dbt-osmosis --help` and the import smoke succeeded.

Procedure verdict / exit code: pass; final chained command exited 0.

# Artifacts

- `uv run ruff check src/dbt_osmosis/core/logger.py` output: `All checks passed!`.
- Import smoke output: `import smoke ok`.
- CLI smoke output: `Usage: dbt-osmosis [OPTIONS] COMMAND [ARGS]...` with command tree shown.
- Workflow/Taskfile assertion output: `workflow/taskfile assertions ok`.
- `task --list` output included `test` and `test-latest-core-compat` with the expected dbt 1.10/1.11 compatibility description.
- Isolated smoke output showed `dbt-core: 1.11.8`, `dbt-duckdb: 1.10.1`, `rich: 15.0.0`, successful `dbt parse`, successful `dbt-osmosis --help`, and `dbt 1.11 import smoke ok`.

# Supports Claims

- `ticket:c10ci06#ACC-001` - supports that the workflow now includes an explicit dbt 1.11 matrix gate with `dbt-duckdb` mapped to `1.10.1`.
- `ticket:c10ci06#ACC-002` - supports that workflow and Taskfile commands contain installed-version assertions for Python, `dbt-core`, `dbt-duckdb`, and `dbt-osmosis`.
- `ticket:c10ci06#ACC-003` - partially supports parse/import/CLI smoke under dbt 1.11 through the isolated runtime; full pytest support is delegated to the new CI gate and not established by this local evidence.
- `ticket:c10ci06#ACC-004` - supports Taskfile parity for the local dbt 1.10/1.11 support story.
- `ticket:c10ci06#ACC-005` - supports that the workflow contains a weekly scheduled latest-patch canary for dbt 1.10 and 1.11.

# Challenges Claims

None - observed checks passed. Limitations below identify untested matrix coverage.

# Environment

Commit: `063b0a03077841ec91765937fda9e360d8ae4473` plus uncommitted `ticket:c10ci06` diff.

Branch: `loom/dbt-110-111-hardening`.

Runtime: local uv project environment used Python `3.13.9`, `dbt-core 1.10.20`, `dbt-duckdb 1.10.0`, `rich 13.9.4`; isolated smoke used Python `3.13.9`, `dbt-core 1.11.8`, `dbt-duckdb 1.10.1`, `rich 15.0.0`.

OS: macOS/darwin local harness.

Relevant config: `.github/workflows/tests.yml`, `Taskfile.yml`, `src/dbt_osmosis/core/logger.py`, `demo_duckdb` fixture.

External service / harness / data source when applicable: GitHub Actions failure run `25291931743` was the before-state observation; no new remote CI run existed yet for the uncommitted diff.

# Validity

Valid for: local structural validation of the workflow/Taskfile shape, the Rich import fix, and an isolated dbt 1.11 parse/import/CLI smoke for the current diff.

Fresh enough for: mandatory critique of `ticket:c10ci06` and ticket evidence disposition before pushing the implementation commit.

Recheck when: `.github/workflows/tests.yml`, `Taskfile.yml`, `src/dbt_osmosis/core/logger.py`, package dependencies, GitHub Actions behavior, Python version support, dbt-core, or dbt-duckdb change.

Invalidated by: a later diff changing the CI matrix, Taskfile task commands, logger Rich imports, or dependency support boundary before commit.

Supersedes / superseded by: supersedes the local child-output-only observation in `packet:ralph-ticket-c10ci06-20260503T222300Z`; should be superseded by branch CI evidence after push.

# Limitations

- This evidence does not prove the full expanded GitHub Actions matrix passes.
- This evidence does not run full pytest under dbt 1.11 locally.
- This evidence does not resolve dependency lock/reproducibility concerns owned by `ticket:c10lock07`.
- This evidence does not establish release metadata or documentation updates outside `ticket:c10ci06` scope.

# Result

The current diff locally fixes the Rich import failure seen under `rich 15.0.0`, structurally promotes dbt 1.11 into the main and latest-patch compatibility gates, mirrors the support story in `Taskfile.yml`, and passes an isolated dbt 1.11 parse/import/CLI smoke.

# Interpretation

The observed results support proceeding to mandatory critique for `ticket:c10ci06`. They do not justify closing the ticket until critique disposition, acceptance disposition, and post-push CI evidence are recorded by the ticket.

# Related Records

- `ticket:c10ci06`
- `packet:ralph-ticket-c10ci06-20260503T222300Z`
- `initiative:dbt-110-111-hardening`
