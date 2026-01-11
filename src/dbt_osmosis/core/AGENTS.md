# CORE MODULE KNOWLEDGE BASE

**Generated:** 2026-01-11
**Scope:** src/dbt_osmosis/core/

## OVERVIEW
Core transformation engine with config resolution, YAML handling, column inheritance via knowledge graphs.

## STRUCTURE
```
core/
├── schema/           # YAML parsing (ruamel.yaml), reading, writing, validation
├── formats/          # Empty (reserved for future YAML format variants)
├── transforms.py     # TransformPipeline with >> operator, 1059 lines
├── introspection.py  # SettingsResolver, PropertyAccessor, 1647 lines
├── inheritance.py    # Column knowledge graph builder, 624 lines
├── llm.py           # LLM integration for documentation, 1863 lines
├── config.py        # Configuration management, 618 lines
├── exceptions.py    # OsmosisError hierarchy (base class)
└── plugins.py       # Pluggy system (FuzzyCaseMatching, FuzzyPrefixMatching)
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Transform pipeline | transforms.py:86 | TransformPipeline with `>>` operator |
| Config resolution | introspection.py:576 | SettingsResolver class, 8-source precedence |
| Property access | introspection.py:1302 | PropertyAccessor (manifest/YAML/auto sources) |
| Column inheritance | inheritance.py:475 | _build_column_knowledge_graph() |
| Column caching | introspection.py:970 | _COLUMN_LIST_CACHE with lock |
| YAML caching | schema/reader.py:117 | _YAML_BUFFER_CACHE with lock |
| Plugin system | plugins.py:66 | get_plugin_manager() with @cache |
| YAML handling | schema/ | ruamel.yaml (NOT PyYAML) |

## CODE MAP
| Symbol | Type | Location | Refs | Role |
|--------|------|----------|------|------|
| TransformPipeline | Class | transforms.py:86 | High | Operation chaining |
| SettingsResolver | Class | introspection.py:576 | High | Config resolution |
| PropertyAccessor | Class | introspection.py:1302 | High | Manifest/YAML access |
| _COLUMN_LIST_CACHE | Dict | introspection.py:970 | High | Thread-safe column cache |
| _YAML_BUFFER_CACHE | Class | schema/reader.py:117 | High | Thread-safe YAML cache |
| _build_column_knowledge_graph | Function | inheritance.py:475 | High | Lineage graph |
| get_plugin_manager | Function | plugins.py:66 | High | Plugin loading |
| OsmosisError | Class | exceptions.py:27 | - | Base exception |

## CONVENTIONS
- **Transform chaining**: Use `>>` operator (e.g., `inject_missing >> inherit_docs`)
- **ruamel.yaml**: Always use for formatting preservation (NEVER PyYAML)
- **Config precedence**: Column meta > Node meta > config.extra > config.meta > vars > fallback
- **Thread safety**: All cache access must acquire locks (_COLUMN_LIST_CACHE_LOCK, _YAML_BUFFER_CACHE_LOCK)
- **PropertyAccessor**: Source selection (manifest=YAML/rendered, yaml=unrendered, auto=detect)

## ANTI-PATTERNS (THIS MODULE)
- **NEVER** use `_get_setting_for_node()` - deprecated, use SettingsResolver.resolve()
- **NEVER** bypass cache locks - always use `with _CACHE_LOCK:` for concurrent access
- **NEVER** use PyYAML - ruamel.yaml only for YAML formatting preservation
- **NEVER** delete test files - fix root cause instead
- **NEVER** use `type: ignore` - use proper types or `# pyright: reportX=false` suppressions

## UNIQUE STYLES
- **TransformPipeline**: Functional composition with operator overloading (`>>`)
- **SettingsResolver**: 8-source precedence chain with kebab/snake case conversion
- **Column knowledge graph**: Directed graph for documentation lineage across ancestors
- **PropertyAccessor**: Unified interface for manifest/YAML with unrendered jinja support
- **Plugin system**: Pluggy with @hookimpl decorator, @cache decorated manager
- **Thread-safe caches**: Global dicts with dedicated locks for concurrent YAML/DB access

## NOTES
- **Large files**: llm.py (1863 lines), introspection.py (1647 lines), transforms.py (1059 lines)
- **Caching**: _COLUMN_LIST_CACHE (column metadata), _YAML_BUFFER_CACHE (LRUCache, maxsize=256)
- **Plugin hooks**: get_candidates() for column name matching (case variants, prefix removal)
- **Exception hierarchy**: OsmosisError base with 12 subclasses (ConfigurationError, LLMError, etc.)
- **dbt integration**: Loads via `dbt.cli.main.dbtRunner`, accesses manifest at target/manifest.json
