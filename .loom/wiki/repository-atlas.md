---
id: wiki:repository-atlas
kind: wiki
page_type: atlas
status: active
created_at: 2026-05-03T20:46:40Z
updated_at: 2026-05-03T23:36:45Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  constitution:
    - constitution:main
  evidence:
    - evidence:repository-structure-scan
---

# Summary

This atlas maps the `dbt-osmosis` repository at `repo:root`. The project is a Python CLI/package for dbt development workflows, with primary surfaces for schema YAML management, column documentation inheritance, ad-hoc SQL helpers, generation/lint/diff helpers, and an optional Streamlit workbench.

# Major Modules

- `src/dbt_osmosis/cli/` owns the Click command surface. `main.py` defines top-level command families including `yaml`, `sql`, `workbench`, `generate`, `nl`, `test`, `test-llm`, `diff`, and `lint`.
- `src/dbt_osmosis/core/` owns dbt project bootstrap, configuration resolution, node filtering, transforms, inheritance, path management, schema I/O, SQL operations, diffing, linting, generation, plugins, and optional LLM helpers.
- `src/dbt_osmosis/core/schema/` owns round-trip YAML parsing, cached reading, atomic writing, and validation. This is the safe boundary for schema YAML mutation.
- `src/dbt_osmosis/sql/` contains the MySQL proxy experiment that reuses shared SQL compile and execute helpers.
- `src/dbt_osmosis/workbench/` owns the optional Streamlit app. `app.py` wires dbt context, SQL compile/run actions, profiling, and dashboard state; `components/` contains dashboard items such as editor, preview, profiler, renderer, RSS feed, and AI assistant.
- `tests/` owns pytest coverage. `tests/core/` mirrors core modules; root-level YAML tests exercise higher-level inheritance and knowledge graph behavior; `tests/workbench/` covers workbench smoke behavior.
- `demo_duckdb/` is the canonical dbt fixture project. It carries dbt project config, `dbt-osmosis.yml`, models, seeds, profiles, integration smoke script, and generated `target/` artifacts when present.
- `docs/` owns the Docusaurus documentation site. Actual content lives under `docs/docs/`; tooling lives in `docs/package.json`, `docs/docusaurus.config.js`, and related Docusaurus files.
- `specs/001-unified-config-resolution/` preserves the feature spec and implementation plan for centralized settings and property access work.
- `_deps/`, generated `target/`, `logs/`, database files, docs build output, and local caches are dependency, generated, or disposable surfaces rather than primary source modules.

# Important Entry Points

- `src/dbt_osmosis/cli/main.py` is the package console-script entrypoint target and the complete CLI command surface.
- `src/dbt_osmosis/__main__.py` invokes `dbt_osmosis.cli.main:cli` for module execution.
- `src/dbt_osmosis/core/config.py` creates dbt project context, loads adapters, and manages manifest bootstrap.
- `src/dbt_osmosis/core/settings.py` defines YAML refactor context and formatter/catalog settings used by YAML commands.
- `src/dbt_osmosis/core/introspection.py` contains `SettingsResolver`, `PropertyAccessor`, configuration sources, catalog loading, and column metadata access.
- `src/dbt_osmosis/core/transforms.py` contains `TransformPipeline` and YAML metadata mutation operations.
- `src/dbt_osmosis/core/inheritance.py` builds the column knowledge graph used for documentation inheritance.
- `src/dbt_osmosis/core/path_management.py` owns YAML routing and project-root safety checks.
- `src/dbt_osmosis/core/schema/parser.py`, `reader.py`, and `writer.py` form the YAML round-trip pipeline.
- `src/dbt_osmosis/core/sql_operations.py` is the shared SQL compile/execute path used outside only the CLI.
- `src/dbt_osmosis/workbench/app.py` is the Streamlit workbench bootstrap.
- `pyproject.toml` owns package metadata, Python support, dependencies, console script, Ruff, pytest, pyright, and coverage settings.
- `Taskfile.yml` owns the canonical local workflow for formatting, linting, testing, development environment setup, compatibility smoke, and mutation testing.
- `AGENTS.md` and `src/dbt_osmosis/core/AGENTS.md` provide path-scoped operator guidance. Treat them as instruction/context surfaces, not Loom truth owners.

# Test Surface

- `tests/core/` is the main unit-test mirror for `src/dbt_osmosis/core/` and CLI behavior.
- Root tests such as `tests/test_yaml_inheritance.py`, `tests/test_yaml_knowledge_graph.py`, and `tests/test_yaml_context.py` exercise YAML behavior against dbt fixture context.
- `tests/conftest.py` builds shared dbt fixture context and temporary DuckDB-backed projects.
- `tests/core/conftest.py` ensures `demo_duckdb/target/manifest.json` exists before core tests that need it.
- `tests/workbench/test_ai_assistant.py` provides current workbench-specific test coverage.
- `demo_duckdb/integration_tests.sh` is an integration smoke script, but it performs destructive fixture cleanup and should not be run casually on a dirty tree.

# Build / Validation Commands

Treat these commands as operational references, not blind instructions.

- `task format` runs Ruff import fixing and formatting.
- `task lint` runs Ruff checks.
- `task test` runs basedpyright, dbt parse, and pytest across the supported Python/dbt matrix.
- `uv run dbt parse --project-dir demo_duckdb --profiles-dir demo_duckdb -t test` refreshes the DuckDB fixture manifest for focused tests.
- `uv run pytest` runs the pytest suite.
- `uv run pytest tests/core/test_cli.py` is a focused CLI test example.
- `npm --prefix docs run build` validates the Docusaurus docs site.

# Data / Persistence Boundaries

- Schema YAML changes should flow through `src/dbt_osmosis/core/schema/` helpers so ruamel round-trip formatting, preserved sections, caches, and atomic writes stay intact.
- YAML routing and project-root safety checks belong in `src/dbt_osmosis/core/path_management.py`.
- Configuration resolution should flow through `SettingsResolver.resolve()` and `PropertyAccessor` rather than ad hoc source lookups.
- Shared caches include `_COLUMN_LIST_CACHE` in `introspection.py` and `_YAML_BUFFER_CACHE` in `schema/reader.py`; do not bypass cache locking expectations.
- Generated dbt artifacts live under paths such as `demo_duckdb/target/`, root `target/`, and `logs/` and should be treated as disposable unless a task explicitly targets generated output.
- DuckDB database files such as `demo_duckdb/jaffle_shop.duckdb`, `demo_duckdb/test.db`, `catalog.sqlite`, and root `test.db` are runtime artifacts, not source-of-truth code.

# Risky Areas

- YAML schema editing is fragile if code bypasses parser/reader/writer helpers or introduces PyYAML-style mutation.
- Configuration precedence is cross-version sensitive. New logic should stay centralized in `introspection.py` and follow the documented precedence model.
- Documentation inheritance depends on manifest shape, upstream lineage, column-name matching, and knowledge graph behavior; tests can be brittle across dbt versions.
- Caches are shared and lock-sensitive. Tests that mutate cwd, manifests, or caches need isolation.
- Workbench behavior has less test coverage than core YAML paths and includes optional dependencies.
- LLM paths are optional and should fail clearly when extras or credentials are absent.
- `demo_duckdb/integration_tests.sh` resets fixture paths with Git commands and can destroy local fixture edits.

# Source Records

- evidence:repository-structure-scan
- constitution:main
- Inspected source/context files: `README.md`, `AGENTS.md`, `src/dbt_osmosis/core/AGENTS.md`, `pyproject.toml`, `Taskfile.yml`, `docs/package.json`, `demo_duckdb/dbt_project.yml`, `demo_duckdb/dbt-osmosis.yml`, `specs/001-unified-config-resolution/plan.md`, and `specs/001-unified-config-resolution/spec.md`.

# Last Verified

2026-05-03 at commit `f1fe50c`, grounded by evidence:repository-structure-scan.

# Related Pages

- `wiki:ci-compatibility-matrix` explains the dbt/Python compatibility CI workflow, adapter mapping, `UV_NO_SYNC=1` overlay rule, and known matrix failure modes.
- Future useful pages may include a core module atlas, YAML schema pipeline reference, configuration resolution concept page, and workbench atlas if those areas keep requiring repeated orientation.
