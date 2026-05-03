---
id: evidence:oracle-backlog-scan
kind: evidence
status: recorded
created_at: 2026-05-03T21:10:43Z
updated_at: 2026-05-03T21:10:43Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  initiative:
    - initiative:dbt-110-111-hardening
  research:
    - research:dbt-110-111-api-surfaces
external_refs:
  dbt_meta_docs: https://docs.getdbt.com/reference/resource-configs/meta
  dbt_columns_docs: https://docs.getdbt.com/reference/resource-properties/columns
  dbt_deprecations_docs: https://docs.getdbt.com/reference/deprecations
---

# Summary

Observed a read-only multi-agent backlog audit across dbt compatibility, CI/build, core YAML/config architecture, CLI/SQL/workbench surfaces, tests/fixtures, and targeted dbt 1.10/1.11 API research. This evidence records the scan inputs and high-level observed findings so tickets can cite a durable source of the initial backlog discovery.

# Procedure

Observed at: 2026-05-03T21:10:43Z

Source state: branch `main`, commit `f1fe50c`; existing Loom workspace files were untracked before this backlog pass.

Procedure: launched five read-only oracle subagents with separate prompts for dbt compatibility, CI/build, core architecture, CLI/SQL/workbench, and tests/fixtures. Fetched dbt docs for `meta`, `columns`, and deprecations. Fetched dbt-core v1.10.0 and v1.11.0 source for column components, model parser, SQL parser, and SQL task internals. Read local source files and workflow files for line-level confirmation of representative findings.

Expected result when applicable: identify compatibility risks, bugs, missing tests, CI/build failures, and cleanup candidates without modifying source code.

Actual observed result: the scan identified high-priority compatibility bugs around `ColumnInfo.config`, dbt 1.10+ `config.meta`/`config.tags`, dbt-loom parser usage, dbt 1.11 CI coverage, fixture isolation, YAML sync races, unsafe YAML dedupe, SQL compile behavior, workbench target/server behavior, direct YAML writes, LLM optional dependency inconsistencies, lint/diff CLI mismatches, proxy support ambiguity, docs build risk, release workflow ordering, and dependency/tooling drift.

Procedure verdict / exit code: mixed observation pass. Tool calls and web fetches succeeded for the cited docs/source pages. No runtime tests or dbt matrix executions were run during this evidence pass.

# Artifacts

- Oracle dbt compatibility audit reported concrete findings for `src/dbt_osmosis/core/transforms.py:356-370`, `src/dbt_osmosis/core/introspection.py:720-802`, `src/dbt_osmosis/core/config.py:498-541`, `src/dbt_osmosis/core/sql_operations.py:27-56`, and `pyproject.toml:25-34`.
- Oracle CI/build audit reported release-before-validation risk in `.github/workflows/release.yml:39-58`, missing full dbt 1.11 matrix in `.github/workflows/tests.yml:15-16` and `91-104`, unlocked `uv sync` in `.github/workflows/tests.yml:43-48`, ignored `.python-version` in `.gitignore:83-85`, docs build issues in `docs/docusaurus.config.js:4`, `34`, `39`, `125`, and stale workbench requirements.
- Oracle core architecture audit reported YAML sync races in `sync_operations.py:664-731`, shared mutable YAML cache behavior in `schema/reader.py`, fixed temp paths in `schema/writer.py:131` and `238`, versioned model YAML access gaps in `inheritance.py:211-222`, silent duplicate deletion in `sync_operations.py:470-519`, and dry-run stale cache concerns in `schema/writer.py:125-176` and `232-284`.
- Oracle CLI/SQL/workbench audit reported plain SQL compile returning blank/`None`, incorrect Streamlit `--browser.*` args for server binding, workbench target switching without context rebuild, LLM gate inconsistencies, Azure AD OpenAI wiring risk, direct PyYAML/plain YAML writes in `cli/main.py:719-751`, `843-845`, and `931-933`, and lint/diff command mismatches.
- Oracle tests/fixtures audit reported repo-root DuckDB path leakage, source-tree manifest reuse, dbt 1.11 canary-only coverage, mocked dbt config-shape behavior, integration script source mutation, generated artifact copying, and cache reset type replacement.
- Web fetch of dbt docs confirmed column `meta` moved to `config.meta` in v1.10 and was backported to 1.9, and column properties support `config.tags` and `config.meta`.
- Raw source fetches from dbt-core v1.10.0 and v1.11.0 confirmed `ColumnInfo.config: ColumnConfig`, `ColumnConfig.meta`, `ColumnConfig.tags`, `ModelParser.parse_from_dict(self, dct, validate=True)`, `SqlBlockParser.parse_remote(self, sql, name)`, and `SqlCompileRunner.compile(self, manifest)` shapes.

# Supports Claims

- initiative:dbt-110-111-hardening#OBJ-001 - partial support that current CI lacks a full dbt 1.11 compatibility gate.
- initiative:dbt-110-111-hardening#OBJ-002 - partial support that dbt 1.10+ config shapes need first-class handling.
- initiative:dbt-110-111-hardening#OBJ-003 - partial support that YAML sync has race and data-preservation risks.
- initiative:dbt-110-111-hardening#OBJ-004 - partial support that fixtures and caches can mask behavior.
- initiative:dbt-110-111-hardening#OBJ-005 - partial support that release/docs/package workflows need hardening.
- initiative:dbt-110-111-hardening#OBJ-006 - partial support that several user-facing command surfaces do not match advertised behavior.
- initiative:dbt-110-111-hardening#OBJ-007 - partial support that proxy/optional extras need explicit boundaries.
- initiative:dbt-110-111-hardening#OBJ-008 - partial support that cleanup work exists.

# Challenges Claims

- Challenges any current claim that dbt 1.11.x is fully covered by the existing CI matrix. The observed workflow has one latest-core compatibility job on Python 3.13 with `dbt-core~=1.11.0` and `dbt-duckdb~=1.10.1`, not the full advertised Python/dbt matrix.
- Challenges any current claim that all YAML writes flow through the schema helpers. Generate/NL paths directly write YAML content.
- Challenges any current claim that optional workbench/AI paths fail clearly. Several paths can surface raw missing executable/module/env errors.

# Environment

Commit: `f1fe50c`

Branch: `main`

Runtime: OpenCode tool session with read-only oracle subagents and local filesystem/web fetch access

OS: darwin

Relevant config: `pyproject.toml`, `Taskfile.yml`, `.github/workflows/tests.yml`, `.github/workflows/release.yml`, `docs/package.json`, `docs/docusaurus.config.js`, `demo_duckdb/profiles.yml`, `demo_duckdb/integration_tests.sh`

External service / harness / data source when applicable: dbt documentation pages, GitHub raw dbt-core source, oracle subagents

# Validity

Valid for: backlog discovery and issue triage at commit `f1fe50c`, plus dbt docs/source observations fetched on 2026-05-03.

Fresh enough for: creating initial initiative and tickets for dbt 1.10/1.11 hardening, CI/build cleanup, fixture reliability, and user-facing command audit work.

Recheck when: source files change, dbt-core/dbt-duckdb release new patch/minor versions, CI workflows change, docs dependencies update, or a ticket reaches implementation/acceptance.

Invalidated by: material code changes that remove or rework the cited paths, or newer runtime evidence that contradicts the static audit.

Supersedes / superseded by: none.

# Limitations

- This evidence does not prove runtime failures because no pytest, dbt parse, docs build, package build, or matrix execution was run during the backlog scan.
- Oracle outputs are generated analysis and need implementation-time reproduction or focused tests before closure claims.
- External docs and raw source fetches represent observed versions/pages at one point in time and should be rechecked for exact latest patch compatibility.

# Result

The scan found enough concrete bugs, compatibility gaps, and maintenance risks to justify a dedicated dbt 1.10/1.11 hardening initiative and a detailed backlog of bounded tickets.

# Interpretation

The evidence supports opening the backlog. It does not by itself establish final root cause for every ticket, prove user impact in all environments, or satisfy ticket acceptance criteria.

# Related Records

- initiative:dbt-110-111-hardening
- research:dbt-110-111-api-surfaces
- evidence:repository-structure-scan
