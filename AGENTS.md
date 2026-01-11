# PROJECT KNOWLEDGE BASE

**Generated:** 2026-01-11 14:44:11
**Commit:** (update on commit)
**Branch:** (current branch)

## OVERVIEW
dbt-osmosis is a CLI tool for automated YAML schema management, column-level documentation inheritance, and interactive dbt SQL development via Streamlit workbench. Operates as both dbt utility and standalone Python package.

## STRUCTURE
```
./
├── src/dbt_osmosis/    # Main package (core, workbench, cli, sql)
├── tests/                 # Pytest test suite (mirrors src structure)
├── demo_duckdb/           # Demo dbt project (test fixture)
└── docs/                  # Documentation site source
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| CLI entry | src/dbt_osmosis/cli/main.py | Click-based, yaml/sql/nl/workbench commands |
| Core transforms | src/dbt_osmosis/core/transforms.py | Pipeline pattern with `>>` operator |
| YAML handling | src/dbt_osmosis/core/schema/ | ruamel.yaml parsing/reading/writing |
| Config resolution | src/dbt_osmosis/core/introspection.py:576 | SettingsResolver, PropertyAccessor |
| Column inheritance | src/dbt_osmosis/core/inheritance.py:30 | Knowledge graph builder |
| Workbench | src/dbt_osmosis/workbench/app.py | Streamlit app with modular components |
| Public API | src/dbt_osmosis/core/osmosis.py | Re-exports for backwards compatibility |

## CODE MAP
| Symbol | Type | Location | Refs | Role |
|--------|------|----------|------|------|
| cli | Function | cli/main.py:48 | - | Entry point |
| TransformPipeline | Class | core/transforms.py:86 | High | Operation chaining |
| SettingsResolver | Class | core/introspection.py:576 | High | Config resolution |
| PropertyAccessor | Class | core/introspection.py:1302 | High | Property access |
| inherit_upstream_column_knowledge | Function | core/transforms.py:203 | High | Column docs propagation |
| inject_missing_columns | Function | core/transforms.py:301 | High | Column injection |
| _build_column_knowledge_graph | Function | core/inheritance.py:475 | High | Lineage graph |

## CONVENTIONS
- **Transform pipeline**: Chain with `>>` operator (e.g., `inject_missing >> inherit_docs`)
- **ruamel.yaml**: Always use (NOT PyYAML) for formatting preservation
- **Type hints**: `from __future__ import annotations` for forward references
- **Pyright**: Some modules have suppressions (`# pyright: reportX=false`)
- **Config precedence**: Column meta > Node meta > config.extra > config.meta > vars > fallback

## ANTI-PATTERNS (THIS PROJECT)
- **NEVER** use PyYAML - use ruamel.yaml for formatting preservation
- **NEVER** suppress type errors with `as any` - prefer proper types or suppressions
- **NEVER** delete test files to "pass" - fix root cause
- **NEVER** leave code in broken state - revert before continuing
- **NEVER** use `_get_setting_for_node()` - deprecated, use SettingsResolver.resolve()

## UNIQUE STYLES
- **Unified config resolution**: SettingsResolver with 8-source precedence chain
- **Column knowledge graph**: Directed graph for documentation lineage
- **Transform pipeline**: Functional composition with operator overloading
- **PropertyAccessor**: Unified interface for manifest/YAML properties with unrendered jinja support
- **Thread-safe caching**: Global caches with dedicated locks (_COLUMN_LIST_CACHE, _YAML_BUFFER_CACHE)

## COMMANDS
```bash
# Development
task                    # Format, lint, dev setup, test
task format               # Ruff format + import sort
task lint                 # Ruff lint
task test                 # Full test matrix

# dbt-osmosis
uv run dbt-osmosis yaml refactor --project-dir <path> --profiles-dir <path>
uv run dbt-osmosis yaml organize --project-dir <path> --profiles-dir <path>
uv run dbt-osmosis yaml document --project-dir <path> --profiles-dir <path>
uv run dbt-osmosis workbench --project-dir <path> --profiles-dir <path>
uv run dbt-osmosis sql compile "SELECT..."
uv run dbt-osmosis sql run "SELECT..."
uv run dbt-osmosis nl query "Show me..."  # Natural language
uv run dbt-osmosis nl generate "Model name..."  # Generate from NL
```

## NOTES
- **Large files**: introspection.py (58k), llm.py (63k), transforms.py (40k), config.py (24k), inheritance.py (23k)
- **Caching**: Column lists in `_COLUMN_LIST_CACHE`, YAML buffers in `_YAML_BUFFER_CACHE` (both thread-safe)
- **dbt integration**: Loads via `dbt.cli.main.dbtRunner`, accesses manifest at `target/manifest.json`
- **Plugin system**: Pluggy-based with FuzzyCaseMatching, FuzzyPrefixMatching
- **Test fixture**: demo_duckdb/ project, run `dbt parse` before tests
- **Vendored deps**: _deps/ contains dbt packages (240MB, unusual for PyPI lib)
- **Task runner**: Uses task (Taskfile.yml) instead of Make/nox
- **uv.lock**: Modern package manager lockfile (uv-based)

---

## WORKBENCH (`src/dbt_osmosis/workbench/`)

### OVERVIEW
Streamlit-based interactive dbt development workbench with real-time compilation, query execution, and pandas profiling.

### STRUCTURE
```
components/
├── dashboard.py   # Dashboard.Item base class with drag-drop grid
├── editor.py       # Monaco editor with SQL/YAML tabs
├── renderer.py     # Read-only compiled SQL viewer
├── preview.py      # Query results with DataGrid
├── profiler.py     # ydata_profiling integration
├── feed.py         # Hacker News RSS feed
└── ai_assistant.py # AI documentation generation
```

### WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Entry point | app.py:296 | main() with state initialization |
| State mgmt | app.py:300-320 | st.session_state.app SimpleNamespace |
| Component init | app.py:303-314 | Dashboard items with grid positions |
| Dashboard base | components/dashboard.py:24 | Dashboard.Item ABC |
| Editor logic | components/editor.py:23 | Monaco with tab switching |
| Query execution | app.py:248-273 | run_query() mutation state |
| CLI launch | cli/main.py:824-876 | subprocess: streamlit run |
| Hotkeys | app.py:376-379 | Ctrl+Enter (compile), Ctrl+Shift+Enter (run) |

### CONVENTIONS (workbench-specific)
- **Component inheritance**: All inherit from `Dashboard.Item`, implement `__call__()`
- **State initialization**: Static `initial_state()` method for component state
- **Grid positioning**: Components use (x, y, w, h) coordinates on dashboard grid
- **Action callbacks**: Passed via constructor (compile_action, query_action, prof_action)
- **Theme switching**: Each component has `_dark_mode` property, toggles via title bar
- **Streamlit Elements**: Uses streamlit-elements-fluence (mui components) + dashboard.Grid
- **Monaco editor**: Multiple tabs (SQL, YAML), language-aware syntax highlighting
- **Hotkey bindings**: event.Hotkey() for keyboard shortcuts (Ctrl+Enter, Ctrl+Shift+Enter)
- **Extra dependencies**: streamlit>=1.20.0,<1.42.0, streamlit-elements-fluence>=0.1.4, ydata-profiling~=4.13.0, feedparser~=6.0.12

### ANTI-PATTERNS (workbench-specific)
- **NEVER** use st.session_state for component state - use st.session_state.app
- **NEVER** use raw Streamlit widgets - use streamlit-elements-fluence dashboard
- **NEVER** trigger unnecessary reruns - use lazy() for non-critical on_change handlers
- **NEVER** create manual layouts - use Dashboard.Item with grid positioning
- **NEVER** call st.rerun() in hotkey callbacks - use sync() or action lambdas
