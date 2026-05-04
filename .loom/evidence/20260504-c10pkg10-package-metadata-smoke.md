---
id: evidence:c10pkg10-package-metadata-smoke
kind: evidence
status: recorded
created_at: 2026-05-04T04:18:06Z
updated_at: 2026-05-04T04:18:06Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  tickets:
    - ticket:c10pkg10
  critique:
    - critique:c10pkg10-release-packaging-review
  packets:
    - packet:ralph-ticket-c10pkg10-20260504T033410Z
external_refs: {}
---

# Summary

Observed package metadata regression tests, lock checks, Python 3.10 compatibility checks, independent pip install smokes, and targeted hooks for `ticket:c10pkg10`. This evidence supports the ticket's package metadata and install-smoke claims; it does not decide ticket acceptance or closure.

# Procedure

Observed at: 2026-05-04T04:18:06Z
Source state: branch `loom/dbt-110-111-hardening`, base commit `83ee46a57a7aaf9349edb9ac5acacb02f8dee236`, with the `ticket:c10pkg10` working-tree diff applied.
Procedure: Ran and inspected focused metadata tests, lock checks, Python 3.10 focused tests, independent pip install smokes, `pre-commit run --files` for changed files, and `git diff --check`.
Expected result when applicable: Red metadata tests should fail before implementation on real ticket gaps; after implementation, metadata tests, lock checks, hooks, and independent base/extra pip smokes should pass.
Actual observed result: Red tests failed for expected metadata gaps before fixes. Green tests, lock checks, independent pip smokes, targeted pre-commit, and whitespace checks passed after fixes.
Procedure verdict / exit code: mixed red/green sequence. Final green checks observed exit 0 for the focused tests, lock check, independent pip smokes, targeted pre-commit, and `git diff --check`.

# Artifacts

Red observations recorded in `packet:ralph-ticket-c10pkg10-20260504T033410Z`:

- `uv run pytest tests/test_package_metadata.py` initially failed with `4 failed in 0.15s` for missing direct `sqlglot`, stale workbench requirements `==1.1.5`, Ruff drift, and `.python-version` Taskfile source drift.
- A follow-up red check failed with `1 failed, 3 passed in 0.13s` after adding the dev-surface equality assertion, exposing divergence between `project.optional-dependencies.dev` and `[dependency-groups].dev`.
- A critique-driven red check failed with `3 failed, 1 passed in 0.20s`, covering missing direct `PyYAML`, missing Python 3.10 `tomli`, and a DuckDB smoke that still installed `dbt-duckdb` explicitly.
- Strengthened workbench smoke initially failed with `ModuleNotFoundError: No module named 'pkg_resources'` from `ydata_profiling` when `setuptools-82.0.1` resolved.

Final green observations:

- `uv lock --check` returned `Resolved 165 packages in 6ms`.
- `uv run pytest tests/test_package_metadata.py` returned `5 passed in 0.12s`.
- `uv run --python 3.10 pytest tests/test_package_metadata.py` returned `5 passed in 0.53s`, exercising the `tomllib` / `tomli` fallback path.
- `task pip-install-smoke` passed across independent Python 3.13 environments for `base`, `openai`, `azure`, `workbench`, `duckdb`, and `proxy`.
- The pip smokes reported `No broken requirements found.` for each independent environment.
- The base smoke printed `smoke: base`, `dbt-osmosis: 1.3.0`, `dbt-core: 1.11.8`, `dbt-adapters: 1.22.10`, and asserted `mysql_mimic` was absent.
- The OpenAI smoke printed `smoke: openai`, `dbt-osmosis: 1.3.0`, and `dbt-core: 1.11.8`, and imported `openai`.
- The Azure smoke printed `smoke: azure`, `dbt-osmosis: 1.3.0`, and `dbt-core: 1.11.8`, and imported `azure.identity`.
- The workbench smoke printed `smoke: workbench`, `dbt-osmosis: 1.3.0`, and `dbt-core: 1.11.8`, resolved `setuptools-80.10.2`, imported the representative Streamlit/workbench module set, and asserted `dbt.adapters.duckdb` was absent from the workbench-only install. Streamlit emitted bare-mode warnings but the smoke completed successfully.
- The DuckDB smoke printed `smoke: duckdb`, `dbt-osmosis: 1.3.0`, `dbt-core: 1.11.8`, and `dbt-duckdb: 1.10.1`, without an explicit adapter install argument outside `.[duckdb]`.
- The proxy smoke printed `smoke: proxy`, `dbt-osmosis: 1.3.0`, and `dbt-core: 1.11.8`, and imported `mysql_mimic`.
- Targeted `pre-commit run --files pyproject.toml uv.lock Taskfile.yml .pre-commit-config.yaml .github/workflows/tests.yml src/dbt_osmosis/workbench/requirements.txt README.md docs/docs/intro.md docs/docs/reference/cli.md docs/docs/tutorial-basics/installation.md docs/docs/tutorial-yaml/synthesize.md tests/test_package_metadata.py .loom/packets/ralph/20260504T033410Z-ticket-c10pkg10-iter-01.md` passed check-ast, YAML/TOML checks, end-of-file, trailing whitespace, private-key detection, debug-statements, ruff-format, ruff, gitleaks, and actionlint.
- `git diff --check` produced no output.

# Supports Claims

- ticket:c10pkg10#ACC-001: Base package metadata removes `mysql-mimic` from base dependencies while base smoke verifies base CLI imports and absence of `mysql_mimic`.
- ticket:c10pkg10#ACC-002: Direct imports are backed by base dependencies (`PyYAML`, `sqlglot`) or guarded behind documented extras (`openai`, `azure`, `workbench`, `duckdb`, `proxy`); independent smokes import the expected modules.
- ticket:c10pkg10#ACC-003: `src/dbt_osmosis/workbench/requirements.txt` references `dbt-osmosis[workbench,duckdb]==1.3.0` and `setuptools>=70,<81`; metadata tests check the current version and supported extras.
- ticket:c10pkg10#ACC-004: `project.optional-dependencies.dev`, `[dependency-groups].dev`, Taskfile Ruff/pre-commit commands, and pre-commit Ruff pin are asserted by metadata tests and targeted hooks.
- ticket:c10pkg10#ACC-005: Metadata tests assert `.python-version` is not a Taskfile venv source, and the packet records `task --list` rendering successfully without relying on `.python-version`.
- ticket:c10pkg10#ACC-006: Independent pip smokes cover base install and each supported optional extra: `.[openai]`, `.[azure]`, `.[workbench]`, `.[duckdb]`, and `.[proxy]`.

# Challenges Claims

None - the final observed checks matched the expected post-fix results for the cited claims.

# Environment

Commit: base `83ee46a57a7aaf9349edb9ac5acacb02f8dee236` plus uncommitted `ticket:c10pkg10` diff.
Branch: `loom/dbt-110-111-hardening`
Runtime: Python 3.10 and Python 3.13 for focused tests and pip smokes; local default Python also used for hooks.
OS: macOS Darwin.
Relevant config: `pyproject.toml`, `uv.lock`, `Taskfile.yml`, `.github/workflows/tests.yml`, `.pre-commit-config.yaml`, and workbench requirements from the reviewed diff.
External service / harness / data source when applicable: Python package index resolution through pip/uv; no production service exercised.

# Validity

Valid for: package metadata, optional extras, dev dependency canonicalization, Taskfile/CI pip smoke scripts, and install-smoke behavior in the observed source state.
Fresh enough for: `ticket:c10pkg10` local acceptance review and critique of release-packaging changes before post-commit CI.
Recheck when: package metadata, optional extras, Taskfile smoke logic, CI smoke logic, workbench requirements, dependency constraints, Python support range, or dbt support range changes.
Invalidated by: source changes after this evidence that alter package metadata or smoke procedures, failed post-commit CI for the same claims, or dependency resolver changes that make the smokes resolve differently.
Supersedes / superseded by: Supersedes `evidence:oracle-backlog-scan` for the implemented package metadata claims; should be supplemented by post-commit CI evidence before final closure.

# Limitations

- This evidence does not launch the full Streamlit workbench or exercise interactive dashboard flows.
- This evidence does not validate SQL proxy runtime behavior; it only validates dependency routing and importability for `mysql_mimic`.
- This evidence does not prove release publishing behavior.
- This evidence was gathered before the implementation commit existed, so final closure should also cite post-commit CI or confirm the committed diff matches the observed source state.
- Package index state can change; future dependency resolution may need rechecking.

# Result

The observed checks showed that the package metadata cleanup has red/green regression coverage, a fresh lockfile, canonical dev dependency surfaces, independent pip install smokes for base and each optional extra, targeted hook coverage, and no whitespace errors in the reviewed working tree.

# Interpretation

The evidence supports the ticket's local package metadata and install-smoke claims. It does not by itself accept the ticket, close the ticket, establish workbench runtime compatibility, or make the experimental proxy supported.

# Related Records

- ticket:c10pkg10
- critique:c10pkg10-release-packaging-review
- packet:ralph-ticket-c10pkg10-20260504T033410Z
- evidence:oracle-backlog-scan
