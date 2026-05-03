---
id: evidence:repository-structure-scan
kind: evidence
status: recorded
created_at: 2026-05-03T20:46:40Z
updated_at: 2026-05-03T20:46:40Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  wiki:
    - wiki:repository-atlas
external_refs: {}
---

# Summary

Observed the repository root, tracked file distribution, source package directories, tests, docs, demo fixture, and key configuration files to ground the initial repository atlas.

# Procedure

Observed at: 2026-05-03T20:46:40Z

Source state: branch `main`, commit `f1fe50c`; no tracked working-tree changes were reported before Loom records were written.

Procedure: confirmed repository root with `git rev-parse --show-toplevel`; listed root and major directories with workspace file reads; searched source, tests, docs, and demo paths with glob scans; summarized tracked file distribution with a Python aggregation over `git ls-files`; inspected `README.md`, `AGENTS.md`, `src/dbt_osmosis/core/AGENTS.md`, `pyproject.toml`, `Taskfile.yml`, `demo_duckdb/dbt_project.yml`, `demo_duckdb/dbt-osmosis.yml`, `specs/001-unified-config-resolution/plan.md`, and `specs/001-unified-config-resolution/spec.md`.

Expected result when applicable: repository structure is observable enough to write a wiki atlas without relying on transcript memory.

Actual observed result: the repository is a Python package with `src/dbt_osmosis/` split into `cli`, `core`, `sql`, and `workbench`; tests live under `tests/` with a dense `tests/core/` mirror; docs live under a Docusaurus 3 site in `docs/`; `demo_duckdb/` is the canonical dbt fixture; `specs/001-unified-config-resolution/` contains behavior and implementation planning records for configuration resolution work.

Procedure verdict / exit code: observed pass for structural scans; command exits were successful.

# Artifacts

- `git rev-parse --show-toplevel` returned `/Users/alexanderbutler/code_projects/personal/dbt-osmosis`.
- Git source snapshot: branch `main`, short commit `f1fe50c`.
- Tracked file distribution from `git ls-files`: 299 tracked files total.
- Top tracked path counts included `src: 48`, `tests: 44`, `docs: 57`, `demo_duckdb: 29`, `specs: 8`, `.changes: 53`, `.github: 8`, and `screenshots: 6`.
- `src/dbt_osmosis/` contains `__init__.py`, `__main__.py`, `cli/`, `core/`, `sql/`, and `workbench/`.
- `src/dbt_osmosis/core/` contains configuration, introspection, transforms, inheritance, schema, path management, SQL operations, diff, lint, generator, LLM, plugin, validation, and support modules.
- `src/dbt_osmosis/core/schema/` contains `parser.py`, `reader.py`, `writer.py`, and `validation.py` for round-trip YAML work.
- `src/dbt_osmosis/workbench/` contains `app.py`, `requirements.txt`, and dashboard component modules.
- `tests/core/` contains 37 tracked core test files; root tests cover YAML context, inheritance, and knowledge graph behavior; `tests/workbench/` has workbench-specific smoke coverage.
- `docs/package.json` uses Docusaurus 3.7.0 and Node `>=18.0`.
- `Taskfile.yml` defines `format`, `lint`, `dev`, `test`, latest-core compatibility, and mutation test tasks.

# Supports Claims

- wiki:repository-atlas - supports the atlas claim that the repository has the observed top-level, source, test, docs, demo, and validation layout at commit `f1fe50c`.

# Challenges Claims

None - the scan did not target a disputed claim.

# Environment

Commit: `f1fe50c`

Branch: `main`

Runtime: OpenCode tool session with local filesystem access

OS: darwin

Relevant config: `pyproject.toml`, `Taskfile.yml`, `AGENTS.md`, `src/dbt_osmosis/core/AGENTS.md`, `demo_duckdb/dbt_project.yml`, `demo_duckdb/dbt-osmosis.yml`, `docs/package.json`

External service / harness / data source when applicable: none

# Validity

Valid for: repository structure and file placement observed at commit `f1fe50c`.

Fresh enough for: initial Loom repository atlas and future packet orientation that needs the broad module map.

Recheck when: source directories are reorganized, CLI entrypoints move, test layout changes, docs tooling changes, fixture ownership changes, or later evidence supersedes this scan.

Invalidated by: material repository restructure after commit `f1fe50c` that is not reflected in a newer atlas or evidence record.

Supersedes / superseded by: none.

# Limitations

- This evidence records structure and selected source inspections; it does not validate runtime behavior, CLI command correctness, or test pass/fail status.
- Generated and disposable directories are visible in the working tree, but tracked-file counts come from Git and exclude ignored/generated artifacts.
- This evidence does not decide whether documentation, tests, or code are complete.

# Result

The scan showed a coherent repository layout centered on a Python package under `src/dbt_osmosis/`, a mirrored pytest suite under `tests/`, a Docusaurus docs site under `docs/`, and a DuckDB dbt fixture under `demo_duckdb/`.

# Interpretation

The observed structure is sufficient to create an initial repository atlas. The atlas should remain an orientation page and should not become the owner of behavior contracts, execution state, or policy.

# Related Records

- constitution:main
- wiki:repository-atlas
