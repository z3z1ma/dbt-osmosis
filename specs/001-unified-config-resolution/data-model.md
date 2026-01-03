# Data Model: Unified Configuration Resolution System

**Feature**: 001-unified-config-resolution
**Date**: 2026-01-02

## Overview

This document defines the data entities for the unified configuration resolution system. The system consists of two main interfaces: `ConfigResolver` for configuration settings and `PropertyAccessor` for model properties.

## Core Entities

### ConfigResolver

The main configuration resolution interface. Resolves settings from multiple sources with defined precedence.

**Attributes**: None (stateless class)

**Methods**:
- `resolve(setting_name: str, node: ResultNode | None, column_name: str | None, *, fallback: Any) -> Any`
- `has(setting_name: str, node: ResultNode | None, column_name: str | None) -> bool`

**Behavior**:
- Checks sources in precedence order
- Returns first non-None value found
- Returns fallback if no source has the key
- Logs which source provided the value (debug level)

**Validation Rules**:
- Setting names are case-insensitive (kebab-case/snake_case both supported)
- Prefixes (`dbt-osmosis-`, `dbt_osmosis_`) are automatically stripped
- Nested keys (`dbt-osmosis-options.*`) are resolved via dict access

---

### PropertyAccessor

Unified interface for accessing model properties from manifest or YAML sources.

**Attributes**:
- `context: YamlRefactorContext` - Reference to dbt project context
- `yaml_cache: dict` - Reference to YAML buffer cache

**Methods**:
- `get(property_key: str, node: ResultNode, column: str | None, source: Literal['manifest', 'yaml', 'auto']) -> Any`
- `get_description(node: ResultNode, column: str | None, source: Literal['manifest', 'yaml', 'auto']) -> str | None`
- `get_meta(node: ResultNode, key: str, source: Literal['manifest', 'yaml', 'auto']) -> Any`
- `has_property(property_key: str, node: ResultNode, column: str | None) -> bool`

**Behavior**:
- `source='manifest'`: Read from parsed manifest (rendered values)
- `source='yaml'`: Read from raw YAML files (unrendered values)
- `source='auto'`: Prefer YAML if available and contains unrendered templates, else manifest
- Returns None if property not found in specified source

**Validation Rules**:
- Property keys must be valid node or column attributes
- YAML source falls back to manifest if file not found (with warning)
- Unrendered jinja templates are preserved when source='yaml'

---

### ConfigurationSource

Abstract base for configuration sources. Each source knows how to extract values from a specific location.

**Concrete Implementations**:

#### ColumnMetaSource
- **Location**: `column.meta` dict
- **Priority**: 1 (highest)
- **Supported Keys**: All, with or without prefix
- **Variants**: `<key>`, `dbt-osmosis-<key>`, `dbt_osmosis_<key>`, `dbt-osmosis-options.<key>`

#### NodeMetaSource
- **Location**: `node.meta` dict
- **Priority**: 2
- **Supported Keys**: All, with or without prefix
- **Variants**: Same as ColumnMetaSource

#### ConfigExtraSource
- **Location**: `node.config.extra` dict
- **Priority**: 3
- **Supported Keys**: Prefixed variants only (no direct keys)
- **Variants**: `dbt-osmosis-<key>`, `dbt_osmosis_<key>`, `dbt-osmosis-options.<key>`

#### ConfigMetaSource
- **Location**: `node.config.meta` dict
- **Priority**: 4
- **Supported Keys**: All, with or without prefix
- **Version Requirement**: dbt 1.10+ (gracefully skipped if missing)
- **Variants**: Same as ColumnMetaSource

#### UnrenderedConfigSource
- **Location**: `node.unrendered_config` dict
- **Priority**: 5
- **Supported Keys**: Prefixed variants only
- **Version Requirement**: dbt 1.10+ (gracefully skipped if missing)
- **Variants**: Same as ConfigExtraSource

#### ProjectVarsSource
- **Location**: `context.project.runtime_cfg.vars` dict
- **Priority**: 6
- **Supported Keys**: All, with or without prefix
- **Special**: Checks both `dbt-osmosis` and `dbt_osmosis` top-level keys

#### SupplementaryFileSource
- **Location**: `dbt-osmosis.yml` in project root
- **Priority**: 7
- **Supported Keys**: All, with or without prefix
- **Special**: Optional file, cached in `_YAML_BUFFER_CACHE`

---

### PropertySource

Enum representing where a model property is stored.

**Values**:
- `MANIFEST`: Parsed manifest.json (rendered jinja)
- `YAML`: Raw model/schema YAML files (unrendered jinja)
- `DATABASE`: Warehouse metadata (via introspection)

**Usage**: Passed to `PropertyAccessor.get()` to specify source preference

---

### ConfigurationKey

Represents a setting name with optional prefix.

**Fields**:
- `base_name: str` - The key without prefix (e.g., `skip-add-tags`)
- `prefixed_names: list[str]` - All valid prefixed variants
- `is_python_identifier: bool` - Whether `dbt_osmosis_*` variant is valid

**Normalization**:
- Kebab-case ↔ snake_case conversion
- Prefix stripping (`dbt-osmosis-`, `dbt_osmosis_`)
- Nested key resolution (`dbt-osmosis-options.key` → `options['key']`)

**Example**:
```python
key = ConfigurationKey("skip-add-tags")
# base_name: "skip-add-tags"
# prefixed_names: [
#   "skip-add-tags",
#   "skip_add_tags",
#   "dbt-osmosis-skip-add-tags",
#   "dbt_osmosis_skip_add_tags"
# ]
```

---

### PropertyKey

Represents a model or column property identifier.

**Supported Properties**:
- `description`: Text documentation (may contain jinja in YAML)
- `tags`: List of string tags
- `meta`: Arbitrary metadata dict
- `data_type`: String data type name
- `name`: Column or model name

**Access Pattern**:
```python
# Direct access
accessor.get("description", node, column="user_id")

# Convenience method
accessor.get_description(node, column="user_id")
```

---

### ResolutionContext

Contains all context needed for configuration and property resolution.

**Fields**:
- `project: DbtProjectContext` - dbt project with manifest and runtime config
- `settings: YamlRefactorSettings` - dbt-osmosis settings
- `yaml_cache: dict` - Reference to `_YAML_BUFFER_CACHE`
- `manifest: Manifest` - Shortcut to project.manifest
- `runtime_cfg: RuntimeConfig` - Shortcut to project.runtime_cfg

**Lifecycle**: Created once per dbt-osmosis invocation, passed to resolvers

---

### PrecedenceRule

Defines the order in which sources are checked for configuration values.

**Representation**: Ordered list of `ConfigurationSource` implementations

**Execution**:
```python
for source in precedence_rules:
    if value := source.get(setting_name, node, column):
        logger.debug(f"Resolved from {source.name}")
        return value
return fallback
```

**Column-specific**: When column specified, `ColumnMetaSource` is prepended to rules

---

## Entity Relationships

```
┌─────────────────┐
│ ConfigResolver  │
│  (stateless)    │
└────────┬────────┘
         │ uses
         ▼
┌─────────────────────────────────────────────────────────┐
│                   ConfigurationSource                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │ColumnMetaSource│ │NodeMetaSource│ │ConfigExtraSource│ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │ConfigMetaSource│ │UnrenderedConfigSource│ │ProjectVarsSource│ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
│  ┌──────────────────────────────────┐                 │
│  │   SupplementaryFileSource        │                 │
│  └──────────────────────────────────┘                 │
└─────────────────────────────────────────────────────────┘
         │ ordered by
         ▼
┌─────────────────┐
│ PrecedenceRule  │
└─────────────────┘

┌─────────────────┐         ┌─────────────────┐
│PropertyAccessor │         │ ResolutionContext│
│  ─────────────  │   uses   │ ─────────────── │
│ • context       │◄─────────│ • project       │
│ • yaml_cache    │         │ • settings      │
└────────┬────────┘         │ • yaml_cache    │
         │                  └─────────────────┘
         │ reads
         ▼
┌─────────────────┐
│  PropertySource │
│  ─────────────  │
│ • MANIFEST      │
│ • YAML          │
│ • DATABASE      │
└─────────────────┘
```

## State Transitions

### Configuration Resolution

```
┌─────────────┐
│   Start     │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────┐
│ Build precedence chain              │
│ (add ColumnMetaSource if column)    │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│ Iterate sources in order            │
└──────┬──────────────────────────────┘
       │
       ▼
  ┌─────────┐
  │Found?   │
  └──┬───┬──┘
     │   │
  Yes│   │No
     │   │
     ▼   ▼
┌─────────┐ ┌─────────────────┐
│Return   │ │Next source      │
│value    │ └────────┬────────┘
│+log     │          │
└─────────┘          ▼
              ┌─────────┐
              │More    │
              │sources?│
              └──┬───┬──┘
                 │   │
               Yes│   │No
                  │   │
                  ▼   ▼
              ┌─────────┐
              │Return   │
              │fallback │
              └─────────┘
```

### Property Access

```
┌─────────────┐
│   Start     │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────┐
│ Check source parameter               │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│ source='manifest'                    │
│ → Read from node.column.description  │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│ source='yaml'                        │
│ → Read via _get_node_yaml()         │
│ → Use _YAML_BUFFER_CACHE             │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│ source='auto'                        │
│ → Check YAML for unrendered jinja   │
│ → If found, use YAML                │
│ → Else, use manifest                │
└─────────────────────────────────────┘
```

## Validation Rules

### ConfigResolver

1. **Setting name validation**:
   - Must be non-empty string
   - Kebab-case and snake_case treated as equivalent
   - Prefixes are optional and automatically handled

2. **Node validation**:
   - If None, return fallback immediately
   - Must be instance of `ResultNode` (model, source, seed, etc.)

3. **Column validation**:
   - If specified, must exist in `node.columns`
   - If not found, fall back to node-level sources

4. **Return value**:
   - Returns first non-None value from precedence chain
   - Returns fallback if no source has the key
   - Never returns None unless fallback is None

### PropertyAccessor

1. **Property key validation**:
   - Must be valid attribute name
   - Supported keys: description, tags, meta, data_type, name

2. **Source validation**:
   - Must be one of: 'manifest', 'yaml', 'auto'
   - Invalid sources raise ValueError

3. **YAML file handling**:
   - Missing files log warning and fall back to manifest
   - Parse errors raise ConfigurationError

4. **Unrendered jinja detection**:
   - Patterns: `{{ doc(`, `{% docs %}`, `{% enddocs %}`
   - If detected in YAML, prefer YAML for 'auto' source

## Data Structures

### Configuration Precedence Chain

```python
# Type alias for readability
PrecedenceChain = list[Callable[[str, ResultNode, str | None], Any | None]]

# Build chain dynamically
def build_precedence_chain(
    node: ResultNode,
    column: str | None,
    context: ResolutionContext
) -> PrecedenceChain:
    chain = []

    # Column level (if specified)
    if column:
        chain.append(ColumnMetaSource(node, column))

    # Node level
    chain.extend([
        NodeMetaSource(node),
        ConfigExtraSource(node),
    ])

    # dbt 1.10+ sources
    if hasattr(node, 'config') and hasattr(node.config, 'meta'):
        chain.append(ConfigMetaSource(node))
    if hasattr(node, 'unrendered_config'):
        chain.append(UnrenderedConfigSource(node))

    # Project level
    chain.append(ProjectVarsSource(context))
    chain.append(SupplementaryFileSource(context))

    return chain
```

### Property Cache Entry

```python
@dataclass
class PropertyCacheEntry:
    """Cached property access result."""
    source: PropertySource
    value: Any
    timestamp: float
    node_id: str
    property_key: str
    column_name: str | None = None
```

## Enums and Constants

```python
class PropertySource(Enum):
    """Source for model property values."""
    MANIFEST = "manifest"  # Parsed manifest.json
    YAML = "yaml"          # Raw YAML files
    DATABASE = "database"  # Warehouse introspection

class ConfigSourceName(Enum):
    """Names for logging purposes."""
    COLUMN_META = "column_meta"
    NODE_META = "node_meta"
    CONFIG_EXTRA = "config_extra"
    CONFIG_META = "config_meta"
    UNRENDERED_CONFIG = "unrendered_config"
    PROJECT_VARS = "project_vars"
    SUPPLEMENTARY_FILE = "supplementary_file"
    FALLBACK = "fallback"

# Supported prefixes
CONFIG_PREFIXES = ["dbt-osmosis-", "dbt_osmosis_"]

# Options dict key
OPTIONS_KEY = "dbt-osmosis-options"
OPTIONS_KEY_ALT = "dbt_osmosis_options"
```

## Type Definitions

```python
from typing import Any, Literal, Protocol, Callable

# Configuration resolution
ConfigGetter = Callable[[str, ResultNode | None, str | None], Any]
ConfigChecker = Callable[[str, ResultNode | None, str | None], bool]

# Property access
PropertySource = Literal["manifest", "yaml", "auto"]
PropertyValue = str | list[str] | dict[str, Any] | None

# Node and column references
NodeRef = ResultNode
ColumnRef = str

# Context objects
ContextProto = Protocol:
    project: DbtProjectContext
    settings: YamlRefactorSettings
```

## Migration Notes

### Existing Data Structures

**Current `SettingsResolver`** (being extended):
- Already implements precedence logic
- Already handles kebab-case/snake_case
- Already supports column-level overrides

**Current YAML cache** (being reused):
- `_YAML_BUFFER_CACHE`: LRU cache for parsed YAML
- Thread-safe with `_YAML_BUFFER_CACHE_LOCK`
- Returns empty dict for non-existent files

### New Data Structures

**`PropertyAccessor`**: Completely new
- No existing equivalent
- Will use existing `_get_node_yaml()` internally
- Will use existing `_YAML_BUFFER_CACHE`

**`SupplementaryFileSource`**: New source type
- Extends source pattern
- Uses existing YAML reading infrastructure
