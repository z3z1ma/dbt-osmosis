# Repository Guidelines

MANDATORY: Use loom.

## Project Overview

`dbt-osmosis` is a Python CLI and package for dbt development workflows. The repo centers on four surfaces:

- schema YAML management (`yaml organize`, `yaml document`, `yaml refactor`)
- column-level documentation inheritance across dbt lineage
- ad-hoc SQL compile/run helpers
- an optional Streamlit workbench for interactive dbt SQL development

Other CLI families (`diff`, `lint`, `test`, `generate`, `nl`, `test-llm`) reuse the same project/bootstrap spine rather than defining separate runtimes.

Primary entrypoint: `src/dbt_osmosis/cli/main.py`
Package entrypoint: `src/dbt_osmosis/__main__.py`

## Architecture & Data Flow

### Main execution spine
1. Click commands in `src/dbt_osmosis/cli/main.py` parse flags and build `DbtConfiguration`.
2. `src/dbt_osmosis/core/config.py:create_dbt_project_context()` loads the dbt project, adapter, and manifest.
3. YAML commands create `YamlRefactorContext` from `src/dbt_osmosis/core/settings.py`.
4. Candidate nodes are filtered in `src/dbt_osmosis/core/node_filters.py`.
5. Transform chains in `src/dbt_osmosis/core/transforms.py` mutate model/source metadata.
6. YAML is read and written through `src/dbt_osmosis/core/schema/reader.py` and `writer.py`.
7. `src/dbt_osmosis/core/sync_operations.py` merges manifest-backed truth back into schema files.

### Key architectural boundaries
- `src/dbt_osmosis/core/introspection.py` is the configuration and property-resolution center. Prefer `SettingsResolver` and `PropertyAccessor` over ad hoc config lookups.
- `src/dbt_osmosis/core/path_management.py` owns YAML routing and project-root safety checks.
- `src/dbt_osmosis/core/inheritance.py` builds the column knowledge graph used for documentation inheritance.
- `src/dbt_osmosis/core/sql_operations.py` is the shared SQL compile/execute path used by CLI, workbench, and proxy code.
- `src/dbt_osmosis/core/schema/parser.py`, `reader.py`, and `writer.py` split YAML concerns deliberately: filter dbt-osmosis-owned sections, cache reads, then restore preserved sections on atomic write.
- `src/dbt_osmosis/workbench/app.py` reuses the same dbt context but owns Streamlit state and dashboard composition.

### Public vs. internal surfaces
- `src/dbt_osmosis/core/osmosis.py` is the compatibility/public facade. `src/dbt_osmosis/core/__init__.py` is no longer a re-export surface; internal code should import concrete submodules directly.
- Deep edits under `src/dbt_osmosis/core/` must also follow `src/dbt_osmosis/core/AGENTS.md`.

## Key Directories

- `src/dbt_osmosis/cli/` — Click command groups and user-facing entrypoints
- `src/dbt_osmosis/core/` — dbt context setup, config resolution, transforms, YAML I/O, inheritance, plugins
- `src/dbt_osmosis/core/schema/` — round-trip YAML parsing, caching, writing, validation
- `src/dbt_osmosis/sql/` — SQL proxy and related helpers
- `src/dbt_osmosis/workbench/` — Streamlit workbench and dashboard components
- `tests/` — pytest suite; `tests/core/` mirrors core modules, root tests cover higher-level YAML behavior
- `demo_duckdb/` — canonical dbt fixture project used by tests and examples
- `docs/` — Docusaurus docs site; actual content lives under `docs/docs/`
- `specs/001-unified-config-resolution/` — detailed spec/plan/quickstart for config-resolution work
- `_deps/` — vendored dbt packages; avoid editing unless the task explicitly targets vendored code

## Important Files

| Path | Why it matters |
| --- | --- |
| `pyproject.toml` | Source of truth for Python support, dependencies, console script, Ruff, pytest, pyright |
| `Taskfile.yml` | Canonical developer workflow (`task format`, `task lint`, `task test`, `task dev`) |
| `.pre-commit-config.yaml` / `.pre-commit-hooks.yaml` | Repo hygiene policy plus packaged `dbt-osmosis yaml refactor -C` pre-commit hook contract |
| `src/dbt_osmosis/cli/main.py` | Complete CLI surface: `yaml`, `sql`, `workbench`, `generate`, `nl`, `test`, `test-llm`, `lint`, `diff` |
| `docs/package.json` / `docs/docusaurus.config.js` | Source of truth for docs-site tooling and Docusaurus 3 configuration |
| `demo_duckdb/dbt_project.yml` / `demo_duckdb/dbt-osmosis.yml` | Best concrete examples of routing rules, config precedence, and YAML formatting defaults |
| `src/dbt_osmosis/core/config.py` | dbt project/bootstrap and manifest loading |
| `src/dbt_osmosis/core/settings.py` | `YamlRefactorContext`, formatter settings, catalog handling |
| `src/dbt_osmosis/core/introspection.py` | `SettingsResolver`, `PropertyAccessor`, caches, config precedence |
| `src/dbt_osmosis/core/schema/parser.py` / `reader.py` / `writer.py` | Canonical round-trip YAML filter/cache/preserve pipeline |
| `src/dbt_osmosis/core/transforms.py` | `TransformPipeline` and main YAML mutation operations |
| `src/dbt_osmosis/core/inheritance.py` | column lineage and inheritance logic |
| `src/dbt_osmosis/core/sql_operations.py` | Shared SQL compile/execute helpers used outside just the CLI |
| `src/dbt_osmosis/core/path_management.py` | YAML routing, source YAML bootstrapping, root-path validation |
| `src/dbt_osmosis/workbench/app.py` | Streamlit workbench bootstrap and state initialization |
| `tests/conftest.py` | expensive shared dbt fixture builders and `yaml_context` |
| `tests/core/conftest.py` | ensures `demo_duckdb/target/manifest.json` exists before core tests |
| `demo_duckdb/integration_tests.sh` | integration smoke sequence; resets fixture files with `git checkout`/`git clean` |

## Development Commands

Prefer `task` and `uv`; avoid ad hoc environment management.

```bash
# Setup / full local flow
task dev
task

# Formatting and linting
task format
task lint
pre-commit run --all-files

# Tests
uv run dbt parse --project-dir demo_duckdb --profiles-dir demo_duckdb -t test
uv run pytest
task test

# Focused test runs
uv run pytest tests/core/test_cli.py
uv run pytest tests/test_yaml_inheritance.py

# CLI examples
uv run dbt-osmosis yaml refactor --project-dir demo_duckdb --profiles-dir demo_duckdb
uv run dbt-osmosis sql compile "select 1"
uv run dbt-osmosis workbench --project-dir demo_duckdb --profiles-dir demo_duckdb
```

Docs site commands use the separate Node toolchain in `docs/`:

```bash
npm --prefix docs run start
npm --prefix docs run build
npm --prefix docs run serve
```

## Runtime & Tooling Preferences

- Python: `>=3.10,<3.14`; local default is `.python-version` = `3.12`
- Package manager / venv: `uv`
- Build backend: `hatchling`
- Formatter/linter/import sorter: Ruff is canonical, even though Black/isort config still exists in `pyproject.toml`
- Test runner: `pytest`
- Type checking: pyright only covers `src/dbt_osmosis/core` and `src/dbt_osmosis/cli`
- Docs toolchain: Docusaurus 3 in `docs/`, Node `>=18`
- Streamlit config exists in both `config.toml` and `.streamlit/config.toml`; check both before documenting runtime behavior

Important nuance: `task` is not a pure verification command; it formats, lints, tests, and defers `task dev`.

## Code Conventions & Common Patterns

### YAML and schema handling
- Use `ruamel.yaml` round-trip machinery in `src/dbt_osmosis/core/schema/`. Do not introduce new PyYAML-based schema editing.
- Read/write schema files through the schema helpers, not manual file I/O. The reader/writer preserve non-osmosis sections and clear caches safely.
- Atomic write behavior and preserved sections are part of the contract; bypassing them can silently lose YAML content.
- Keep the parser/reader/writer split intact: parsing filters owned top-level sections, reads cache both filtered and original content, and writes merge preserved sections back before atomic replace.

### Configuration resolution
- New config logic should flow through `SettingsResolver.resolve()`.
- Do not add new call sites for deprecated `_get_setting_for_node()`.
- Respect the established precedence model documented in code and demo config:
  - column meta
  - node meta / dbt-osmosis options
  - `config.extra`
  - supplementary `dbt-osmosis.yml`
  - `vars`
  - fallback defaults

### Transform and inheritance flow
- YAML refactor behavior is pipeline-based; compose operations with `TransformPipeline` and the `>>` operator.
- Column documentation inheritance belongs in `core/inheritance.py` and `core/transforms.py`, not in CLI glue.
- Node selection and ordering should stay in `core/node_filters.py`, not scattered across callers.

### Caching and concurrency
- `_COLUMN_LIST_CACHE` and `_YAML_BUFFER_CACHE` are shared caches with lock/ownership expectations.
- Do not bypass cache helpers or mutate cache state casually in production code.
- Tests explicitly reset caches; keep new tests isolated when touching cache-sensitive code.

### Workbench patterns
- Workbench state belongs under `st.session_state.app`, not arbitrary top-level session keys.
- Components inherit from `Dashboard.Item` and expose `initial_state()`.
- Prefer `lazy()` for non-critical editor updates and `sync()`/explicit actions for compile-run flows.
- Do not add raw Streamlit layout patterns when the existing dashboard system already covers the feature.

### Optional AI paths
- OpenAI-backed features are optional extras. Missing dependencies should fail clearly, not silently degrade.
- The workbench AI assistant is still partially stubbed; do not assume it already performs real writeback or generation.

## Testing & QA

### Test layout
- `tests/core/` mirrors `src/dbt_osmosis/core/` for focused unit coverage.
- Root-level `tests/test_yaml_*.py` files exercise higher-level YAML, manifest, and inheritance behavior against a real dbt fixture.
- CLI tests use `click.testing.CliRunner` and mostly validate command surfaces and help text.
- There is no dedicated `tests/workbench/` suite today; workbench coverage is limited to CLI/smoke-level checks.

### Fixture expectations
- `demo_duckdb/` is the canonical integration fixture.
- Many tests require `demo_duckdb/target/manifest.json`; generate it with `dbt parse` if missing.
- `tests/conftest.py` builds temp DuckDB projects via `dbt seed`, `dbt run`, and `dbt docs generate`.
- The earlier PostgreSQL fixture branch was removed because it was unexercised; test fixture support is DuckDB-only today.
- Some LLM-related paths are also optional and may skip when extras such as `openai` or `azure.identity` are unavailable.

### QA cautions
- dbt-version differences change manifest shape; avoid brittle assertions when adding tests.
- Some tests mutate cwd or shared caches, so they are not automatically parallel-safe.
- `task test` is expensive; for iterative work prefer targeted pytest runs after ensuring the manifest exists.
- CI covers a broader dbt matrix than the local Taskfile.

## Documentation & Demo Surfaces

- Root `README.md` is a lightweight landing page, not the full reference.
- Canonical CLI/config docs live in `docs/docs/`, especially `docs/docs/reference/cli.md` and the YAML workflow/configuration guides.
- `docs/README.md` is boilerplate and currently stale; it still references Docusaurus 2 even though the site runs on Docusaurus 3.
- The README intentionally omits some newer CLI families; use the Docusaurus CLI reference for `generate`, `nl`, and `test-llm` details.
- `screenshots/` is illustrative only.
- Generated/disposable artifacts include `docs/build/`, `demo_duckdb/target/`, `logs/`, and DuckDB database outputs.

## Common Pitfalls

- Do not document Black or isort as the active formatter; use Ruff.
- Do not edit YAML files with plain string manipulation when schema helpers already exist.
- Do not copy the few remaining PyYAML-style legacy paths into new schema-mutating code; round-trip YAML work belongs in `core/schema/`.
- Do not bypass project-root path validation in `path_management.py`.
- Do not run `demo_duckdb/integration_tests.sh` on a dirty tree you care about; it restores fixture paths with destructive git commands.
- Do not assume README command coverage is complete; newer CLI families are documented in the Docusaurus reference.
- Do not treat `core/osmosis.py` re-exports as the best place to implement new behavior.
