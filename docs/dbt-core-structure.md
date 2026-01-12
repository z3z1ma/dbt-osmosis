# dbt-core Repository Structure

**Generated**: 2026-01-11
**Source**: https://github.com/dbt-labs/dbt-core
**Branch**: `main`

## High-Level Layout

```
dbt-core/
├── core/dbt/           # Main Python source (17k+ lines)
├── tests/              # Unit + functional tests
├── docs/               # Documentation site
├── schemas/            # JSON validation schemas
├── docker/             # Release Dockerfiles
├── scripts/            # Maintainer utilities
└── .github/            # CI/CD workflows
```

## Core Modules (`core/dbt/`)

| Module | Purpose |
|--------|---------|
| `cli/` | CLI entry point (`main.py`), options, params |
| `task/` | Command implementations (run, compile, test, seed, etc.) |
| `runners/` | Execution engines for different resource types |
| `parser/` | Project file parsing (models, macros, tests, etc.) |
| `graph/` | DAG construction with NetworkX, resource selection |
| `config/` | Configuration reconciliation (project, profile, runtime) |
| `context/` | Jinja2 context building |
| `contracts/` | Pydantic-based data contracts |
| `events/` | Event-driven logging system |
| `materializations/` | Table/view materialization strategies |
| `plugins/` | Plugin system for adapters |
| `clients/` | External service interfaces (agate, jinja) |
| `deps/` | Package dependency management |
| `utils/` | Utility functions |
| `docs/` | Documentation generation |
| `include/` | Starter project scaffold |
| `compilation.py` | SQL compilation logic |
| `exceptions.py` | Exception definitions |
| `flags.py` | Feature flags |
| `version.py` | Version information |
| `tracking.py` | Telemetry/tracking |

## Detailed Module Breakdown

### 1. CLI (`cli/`)
- `main.py` - CLI entry point
- `params.py` - Parameter definitions
- `options.py` - Option handling
- `resolvers.py` - Dependency resolution for CLI

### 2. Tasks (`task/`)
Maps to dbt commands:
- `run.py` - `dbt run` command
- `compile.py` - `dbt compile` command
- `test.py` - `dbt test` command
- `seed.py` - `dbt seed` command
- `snapshot.py` - `dbt snapshot` command
- `build.py` - `dbt build` command
- `debug.py` - `dbt debug` command
- `deps.py` - `dbt deps` command
- `list.py` - `dbt list` command
- `freshness.py` - Source freshness checking

### 3. Runners (`runners/`)
- `no_op_runner.py` - No-op runner for dry runs
- `exposure_runner.py` - Exposure validation runner
- `saved_query_runner.py` - Saved query execution

### 4. Parsers (`parser/`)
- `models.py` - Model parsing
- `macros.py` - Macro parsing
- `analysis.py` - Analysis parsing
- `hooks.py` - Hook parsing
- `generic_test.py` - Test parsing
- `manifest.py` - Manifest construction

### 5. Graph (`graph/`)
- `graph.py` - NetworkX DAG creation
- `selector.py` - Resource selection logic
- `selector_methods.py` - Selection criteria implementations
- `queue.py` - Execution queue management
- `thread_pool.py` - Parallel execution management

### 6. Configuration (`config/`)
- `project.py` - Project configuration
- `profile.py` - Profile configuration
- `runtime.py` - Runtime configuration
- `renderer.py` - Configuration rendering
- `selection.py` - Resource selection criteria

### 7. Context (`context/`)
- `context.py` - Main context builder
- `requires.py` - Context requirements
- `resolvers.py` - Context resolver implementations

### 8. Contracts (`contracts/`)
- `files.py` - File contract definitions
- `graph/` - Graph-related contracts
- `project.py` - Project contract
- `results.py` - Result contracts
- `state.py` - State contracts

### 9. Events (`events/`)
- `logging.py` - Event logging setup
- `base_types.py` - Base event types
- `types.py` - Event type definitions

### 10. Materializations (`materializations/`)
- `incremental/` - Incremental model logic
- `microbatch.py` - Micro-batch processing

### 11. Plugins (`plugins/`)
- `manager.py` - Plugin lifecycle management
- `contracts.py` - Plugin contracts
- `manifest.py` - Plugin manifest handling

## Testing Structure

```
tests/
├── unit/                    # Unit tests (mocked dependencies)
│   ├── README.md
│   ├── cli/                 # CLI unit tests
│   ├── config/              # Configuration unit tests
│   ├── graph/               # Graph unit tests
│   ├── parser/              # Parser unit tests
│   └── contracts/           # Contract validation tests
│
├── functional/              # Integration tests (real adapters)
│   ├── README.md
│   ├── access/              # Access testing
│   ├── analysis/            # Analysis testing
│   ├── basic/               # Basic functionality tests
│   ├── build_command/       # Build command tests
│   ├── catalogs/            # Catalog generation tests
│   └── ...
│
├── fixtures/                # Test fixtures and data
├── data/                    # Sample data for tests
└── conftest.py             # Pytest configuration
```

## Configuration Files

- **`pyproject.toml`** - Build backend, dependencies, project metadata
- **`pytest.ini`** - Pytest configuration and test discovery
- **`Makefile`** - Common build and test commands
- **`CONTRIBUTING.md`** - Development setup and contribution guidelines
- **`ARCHITECTURE.md`** - Technical architecture documentation
- **`.pre-commit-config.yaml`** - Pre-commit hook configuration
- **`hatch.toml`** - Hatch environment management

## GitHub & CI/CD (`.github/`)

```
.github/
├── workflows/               # GitHub Actions workflows
│   ├── main.yml            # Main CI pipeline
│   └── ...
├── ISSUE_TEMPLATE/         # Bug/feature issue templates
├── CODEOWNERS              # File ownership rules
├── dependabot.yml          # Dependency update automation
└── pull_request_template.md
```

## Key Architectural Patterns

### 1. Plugin Architecture
- **Adapters** are separate Python packages that inherit from base classes in `core/dbt/adapters/`
- Each adapter includes:
  - `dbt/include/[name]/` - Jinja macros with database-specific SQL
  - `dbt/adapters/[name]/` - Python adapter classes
  - `pyproject.toml` - Package configuration

### 2. Task-Runner Pattern
- **Tasks** (`core/dbt/task/`) map to top-level dbt commands
- **Runners** execute specific resource types in parallel
- Parallelism managed via thread pool in `GraphRunnableTask`

### 3. DAG-Based Execution
- **Graph construction** (`core/dbt/graph/`) creates NetworkX DAG from project resources
- **Resource selection** (`core/dbt/config/selection.py`) filters resources for execution
- **Dependency resolution** ensures proper execution order

### 4. Configuration Precedence
Configuration flows from: CLI args → Environment variables → Profile → Project → Defaults
- **SettingsResolver** (in dbt-osmosis) mirrors this pattern

### 5. Jinja2 Templating
- SQL templates in `core/dbt/include/` provide base functionality
- Adapters override macros for database-specific implementations
- Context builders (`core/dbt/context/`) expose dbt functions to templates

## Entry Points for Major Functionality

1. **CLI Entry**: `core/dbt/cli/main.py`
2. **Task Execution**: `core/dbt/task/` (specific command files)
3. **Parsing**: `core/dbt/parser/` (file → Python objects)
4. **Graph Building**: `core/dbt/graph/graph.py` (Python objects → DAG)
5. **Execution**: `core/dbt/runners/` (DAG → database operations)
6. **Configuration**: `core/dbt/config/runtime.py` (YAML → resolved settings)

## Notable Conventions

- **Pydantic contracts** for data validation (`contracts/`)
- **Event-driven logging** (`events/`)
- **Thread-based parallelism** for task execution
- **Manifest-based caching** of parsed project state
- **Plugin system** using Python entry points
- **Strict separation** of core vs adapter functionality
- **Comprehensive test coverage** with unit + functional test structure

## Relevance to dbt-osmosis

This modular architecture provides clear extension points for tools like dbt-osmosis:

- **Configuration resolution**: Understanding `config/runtime.py` helps integrate with dbt's config system
- **Parsing pipeline**: `parser/` modules show how dbt reads project files and builds manifests
- **Graph construction**: `graph/` modules reveal dependency tracking and resource selection
- **Plugin system**: `plugins/` demonstrates how to extend dbt with additional functionality
- **Manifest structure**: `contracts/` and `parser/manifest.py` define the data structures dbt-osmosis works with
