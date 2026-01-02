# CLAUDE.md

## Issue Tracking

This project uses **bd (beads)** for issue tracking.
Run `bd prime` for workflow context, or install hooks (`bd hooks install`) for auto-injection.

**Quick reference:**
- `bd ready` - Find unblocked work
- `bd create "Title" --type task --priority 2` - Create issue
- `bd close <id>` - Complete work
- `bd sync` - Sync with git (run at session end)

For full workflow details: `bd prime`

## Repository Overview

**dbt-osmosis** is a CLI tool that enhances the dbt developer experience through automated YAML schema management, column-level documentation inheritance, and a Streamlit-based workbench for interactive dbt SQL development. The tool operates as both a dbt utility and standalone Python package.

## Development Commands

### Environment Setup
```bash
# Install task runner (see https://taskfile.dev/installation/)
# Then run default task (format, lint, dev setup, test)
task

# Setup dev environment only
task dev

# Create virtual environment
task venv
```

### Code Quality
```bash
# Format code (auto-fix imports + ruff format)
task format

# Lint code
task lint

# Manual ruff commands
uvx ruff check
uvx ruff format --preview
uvx ruff check --fix --select I  # Fix imports only
```

### Testing
```bash
# Run full test matrix (Python 3.10-3.12 × dbt 1.8-1.9)
task test

# Run tests for current environment
uv run pytest

# Run specific test file
uv run pytest tests/core/test_introspection.py

# Parse demo project (useful for debugging)
uv run dbt parse --project-dir demo_duckdb --profiles-dir demo_duckdb -t test
```

### Package Management
This project uses **uv** for dependency management:
```bash
# Sync dependencies
uv sync

# Sync with extras
uv sync --extra dev
uv sync --extra workbench

# Install package in editable mode
uv pip install -e .

# Add dependency
uv add <package>
```

### Running dbt-osmosis Commands
```bash
# Main YAML refactor command (organize + document)
uv run dbt-osmosis yaml refactor --project-dir <path> --profiles-dir <path>

# Organize YAML files only (no documentation)
uv run dbt-osmosis yaml organize --project-dir <path> --profiles-dir <path>

# Document models only (inherit upstream docs)
uv run dbt-osmosis yaml document --project-dir <path> --profiles-dir <path>

# Start workbench (requires workbench extra)
uv run dbt-osmosis workbench --project-dir <path> --profiles-dir <path>

# Compile SQL
uv run dbt-osmosis sql compile "SELECT * FROM {{ ref('my_model') }}"

# Execute SQL
uv run dbt-osmosis sql run "SELECT 1"
```

### Demo Project
The `demo_duckdb/` directory contains a test dbt project based on jaffle_shop:
```bash
cd demo_duckdb
dbt run --profiles-dir . --target test
dbt test --profiles-dir . --target test
```

## Code Architecture

### Entry Points
- **CLI**: `src/dbt_osmosis/cli/main.py` - Click-based CLI with subcommands for `yaml`, `sql`, and `workbench`
- **Core API**: `src/dbt_osmosis/core/osmosis.py` - Re-exports all public APIs for backwards compatibility

### Core Module Structure (`src/dbt_osmosis/core/`)

The core functionality is split into specialized modules:

- **config.py**: dbt project initialization, manifest loading, profiles/project discovery
- **settings.py**: `YamlRefactorSettings` and `YamlRefactorContext` dataclasses that configure behavior
- **osmosis.py**: Main API re-export layer for backwards compatibility with imports

#### YAML Management Pipeline
1. **path_management.py**: Determines where YAML files should live based on `dbt_project.yml` configuration (e.g., `+dbt-osmosis: "{node.schema}/{node.name}.yml"`)
2. **restructuring.py**: Creates move/delete plans for YAML reorganization
3. **schema/reader.py**: Reads and caches YAML files
4. **schema/parser.py**: Parses YAML using ruamel.yaml with custom formatting
5. **schema/writer.py**: Writes YAML back to disk with formatting preservation

#### Documentation Inheritance Pipeline
1. **introspection.py**: Queries database for column schema, caches results
2. **inheritance.py**: Builds column knowledge graph from upstream models
3. **transforms.py**: Pipeline of transforms that can be composed with `>>` operator:
   - `inject_missing_columns`: Adds columns from database not in YAML
   - `remove_columns_not_in_database`: Removes stale columns
   - `inherit_upstream_column_knowledge`: Propagates docs/tags/meta from upstream
   - `sort_columns_as_configured`: Orders columns
   - `synchronize_data_types`: Updates data types from database
   - `synthesize_missing_documentation_with_openai`: AI-generated docs (requires OpenAI)

#### Other Key Modules
- **node_filters.py**: Filters dbt nodes by FQN/path, topological sorting
- **sync_operations.py**: Syncs individual nodes to YAML
- **sql_operations.py**: Compiles and executes dbt SQL via dbt's internal APIs
- **llm.py**: OpenAI integration for AI-generated documentation
- **plugins.py**: Pluggy-based plugin system for fuzzy matching (FuzzyCaseMatching, FuzzyPrefixMatching)

### Transform Pipeline Pattern

dbt-osmosis uses a functional pipeline pattern with the `>>` operator:
```python
transform = (
    inject_missing_columns
    >> remove_columns_not_in_database
    >> inherit_upstream_column_knowledge
    >> sort_columns_as_configured
    >> synchronize_data_types
)
result = transform(context=context)
```

Each transform function takes a `YamlRefactorContext` and returns it for chaining.

### Workbench (`src/dbt_osmosis/workbench/`)
- **app.py**: Main Streamlit app
- **components/**: Modular UI components (editor, preview, profiler, dashboard, feed)
- Provides real-time dbt compilation, query execution, and pandas profiling

### Configuration in dbt_project.yml
Users configure YAML organization via node properties:
```yaml
models:
  my_project:
    +dbt-osmosis: "{node.schema}/{node.name}.yml"  # Template for YAML paths

vars:
  dbt-osmosis:
    yaml_settings:
      map_indent: 2
      sequence_indent: 4
      brace_single_entry_mapping_in_flow_sequence: true
      explicit_start: true
```

### Testing Approach
- Tests live in `tests/core/` mirroring `src/dbt_osmosis/core/`
- Uses pytest with demo_duckdb project as test fixture
- Test matrix covers Python 3.10-3.12 and dbt-core 1.8-1.9
- Run `dbt parse` before tests to generate manifest.json

## Important Implementation Details

### dbt Integration
- dbt-osmosis loads dbt projects via `dbt.cli.main.dbtRunner` and `dbt.cli.main.dbtRunnerResult`
- Accesses parsed manifest at `target/manifest.json` via `dbt.contracts.graph.manifest.Manifest`
- Uses dbt's internal SQL compilation via `dbt.task.sql.SqlCompileRunner`

### YAML Formatting
- Uses `ruamel.yaml` (NOT PyYAML) to preserve formatting, comments, and anchors
- YAML settings in `dbt_project.yml` under `vars.dbt-osmosis.yaml_settings` control output formatting
- Custom `create_yaml_instance()` in `schema/parser.py` configures ruamel.yaml

### Column Knowledge Graph
- `inheritance.py` builds a directed graph of column lineage across models
- Documentation/tags/meta inherit from nearest documented upstream column
- Handles multiple inheritance paths (chooses first documented source)
- Uses `_build_node_ancestor_tree()` for topological traversal

### Caching Strategy
- Column lists cached in `introspection._COLUMN_LIST_CACHE` (thread-safe)
- YAML buffers cached in `schema/reader._YAML_BUFFER_CACHE`
- Manifest reloaded via `config._reload_manifest()` when YAML changes

### Plugin System
- Uses `pluggy` for plugin discovery
- Built-in plugins: `FuzzyCaseMatching`, `FuzzyPrefixMatching`
- Hooks defined in `plugins.py` via `dbt_osmosis_hookspec`

### Pre-commit Integration
Users can add dbt-osmosis as a pre-commit hook:
```yaml
repos:
  - repo: https://github.com/z3z1ma/dbt-osmosis
    rev: v1.1.17
    hooks:
      - id: dbt-osmosis
        files: ^models/
        args: [--target=prod]
        additional_dependencies: [dbt-duckdb]
```

## Code Style

- **Formatter**: Ruff with `--preview` mode
- **Line Length**: 100 characters
- **Python Version**: 3.10+ (uses modern typing)
- **Import Style**: Auto-sorted with ruff's isort rules
- Type hints: Uses `from __future__ import annotations` for forward references
- Pyright: Some modules have pyright suppressions (see `# pyright: reportX=false`)

## Key Files Reference

- **CLI Entry**: src/dbt_osmosis/cli/main.py:48 (`cli()` function)
- **Transform Pipeline**: src/dbt_osmosis/core/transforms.py
- **YAML Path Logic**: src/dbt_osmosis/core/path_management.py:45 (`get_target_yaml_path()`)
- **Column Inheritance**: src/dbt_osmosis/core/inheritance.py:22 (`_build_column_knowledge_graph()`)
- **Database Introspection**: src/dbt_osmosis/core/introspection.py:33 (`get_columns()`)

## Documentation and Resources

- **Official Docs**: https://z3z1ma.github.io/dbt-osmosis/
- **Migration Guide**: https://z3z1ma.github.io/dbt-osmosis/docs/migrating (for 0.x.x → 1.x.x)
- **Workbench Demo**: https://dbt-osmosis-playground.streamlit.app/

## Landing the Plane (Session Completion)

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd sync
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
