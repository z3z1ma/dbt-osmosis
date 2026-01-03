# Developer Quickstart: Unified Configuration Resolution

**Feature**: 001-unified-config-resolution
**For**: dbt-osmosis contributors implementing this feature

## Overview

This guide helps developers implement the unified configuration resolution system. The feature consists of two main components:

1. **ConfigResolver**: Resolves settings from multiple sources with precedence
2. **PropertyAccessor**: Unified interface for model properties (manifest vs YAML)

## Prerequisites

- Read the [feature specification](spec.md)
- Read the [research document](research.md)
- Read the [data model](data-model.md)
- Read the [API contract](contracts/config-resolver.py)
- Familiarity with existing codebase: `src/dbt_osmosis/core/introspection.py`

## Implementation Checklist

### Phase 1: Extend ConfigResolver

**File**: `src/dbt_osmosis/core/introspection.py`

- [ ] Add `SupplementaryFileSource` class
  - Read `dbt-osmosis.yml` from project root
  - Use existing `_YAML_BUFFER_CACHE`
  - Handle missing file gracefully (return empty dict)

- [ ] Add `ProjectVarsSource` class
  - Access `context.project.runtime_cfg.vars`
  - Check both `dbt-osmosis` and `dbt_osmosis` top-level keys

- [ ] Add `ConfigMetaSource` class
  - Access `node.config.meta` (dbt 1.10+)
  - Check for field existence with `hasattr()`

- [ ] Add `UnrenderedConfigSource` class
  - Access `node.unrendered_config` (dbt 1.10+)
  - Check for field existence with `hasattr()`

- [ ] Update `SettingsResolver.resolve()` method
  - Insert new sources into precedence chain
  - Add debug logging for source that provided value (FR-010)

- [ ] Add `SettingsResolver.has()` method (FR-012)
  - Check all sources for key existence
  - Return boolean without fallback

- [ ] Add `SettingsResolver.get_precedence_chain()` method
  - Return list of source names for debugging
  - Include column source if column specified

**Tests**: `tests/core/test_config_resolution.py`

### Phase 2: Create PropertyAccessor

**File**: `src/dbt_osmosis/core/introspection.py`

- [ ] Create `PropertyAccessor` class
  - Accept `YamlRefactorContext` in constructor
  - Store reference to YAML cache

- [ ] Implement `PropertyAccessor.get()` method
  - Handle `source='manifest'`: Read from node attributes
  - Handle `source='yaml'`: Use `_get_node_yaml()` from inheritance.py
  - Handle `source='auto'`: Detect unrendered jinja, prefer YAML if found

- [ ] Implement `PropertyAccessor.get_description()` convenience method
  - Call `get()` with `property_key="description"`

- [ ] Implement `PropertyAccessor.get_meta()` convenience method
  - Call `get()` with `property_key="meta"` and additional key lookup

- [ ] Implement `PropertyAccessor.has_property()` method
  - Check both manifest and YAML for property existence

- [ ] Add unrendered jinja detection helper
  - Patterns: `{{ doc(`, `{% docs %}`, `{% enddocs %}`

- [ ] Handle missing YAML files gracefully
  - Fall back to manifest with warning
  - Log missing file path

**Tests**: `tests/core/test_property_accessor.py`

### Phase 3: Update Exports

**File**: `src/dbt_osmosis/core/osmosis.py`

- [ ] Add `ConfigResolver` to `__all__`
- [ ] Add `PropertyAccessor` to `__all__`
- [ ] Keep `_get_setting_for_node` in `__all__` (backward compatibility)

### Phase 4: Documentation

**File**: `CLAUDE.md`

- [ ] Document ConfigResolver usage
- [ ] Document PropertyAccessor usage
- [ ] Document configuration precedence rules
- [ ] Add examples for common use cases

## Code Patterns

### Creating a New Configuration Source

```python
class MyCustomSource:
    """Configuration source example."""

    def __init__(self, node: ResultNode, context: YamlRefactorContext):
        self.node = node
        self.context = context
        self.data = self._load_data()

    def _load_data(self) -> dict:
        """Load configuration data."""
        # Implementation here
        return {}

    def get(self, key: str) -> Any | None:
        """Get value for key."""
        # Check both kebab and snake variants
        kebab_key = key.replace("_", "-")
        snake_key = key.replace("-", "_")

        # Check with prefixes
        for prefix in ("dbt-osmosis-", "dbt_osmosis_"):
            if prefix + kebab_key in self.data:
                return self.data[prefix + kebab_key]
            if prefix + snake_key in self.data:
                return self.data[prefix + snake_key]

        # Check direct key
        if kebab_key in self.data:
            return self.data[kebab_key]
        if snake_key in self.data:
            return self.data[snake_key]

        return None
```

### Updating SettingsResolver Precedence

```python
def resolve(
    self,
    setting_name: str,
    node: ResultNode | None,
    column_name: str | None,
    *,
    fallback: Any = None,
) -> Any:
    """Resolve setting from sources in precedence order."""
    if node is None:
        return fallback

    # Normalize key
    kebab_key = setting_name.replace("_", "-")
    snake_key = setting_name.replace("-", "_")

    # Build sources list
    sources = []

    # Column level
    if column_name and (column := node.columns.get(column_name)):
        sources.append(ColumnMetaSource(column))

    # Node level
    sources.extend([
        NodeMetaSource(node),
        ConfigExtraSource(node),
    ])

    # dbt 1.10+ sources (check existence)
    if hasattr(node, 'config') and hasattr(node.config, 'meta'):
        sources.append(ConfigMetaSource(node))
    if hasattr(node, 'unrendered_config'):
        sources.append(UnrenderedConfigSource(node))

    # Project level
    sources.extend([
        ProjectVarsSource(self.context),
        SupplementaryFileSource(self.context),
    ])

    # Check each source
    for source in sources:
        if value := source.get(setting_name):
            logger.debug(
                ":mag: Resolved '%s' from %s for node '%s'",
                setting_name,
                source.name,
                node.name
            )
            return value

    return fallback
```

### Property Access Pattern

```python
class PropertyAccessor:
    def get(
        self,
        property_key: str,
        node: ResultNode,
        column_name: str | None = None,
        source: Literal["manifest", "yaml", "auto"] = "auto",
    ) -> Any:
        # Manifest source
        if source == "manifest":
            return self._get_from_manifest(property_key, node, column_name)

        # YAML source
        if source == "yaml":
            return self._get_from_yaml(property_key, node, column_name)

        # Auto source
        yaml_value = self._get_from_yaml(property_key, node, column_name)
        if yaml_value and self._has_unrendered_jinja(str(yaml_value)):
            return yaml_value
        return self._get_from_manifest(property_key, node, column_name)
```

## Testing Patterns

### Unit Test Example

```python
import pytest
from dbt_osmosis.core.introspection import ConfigResolver

class TestConfigResolver:
    def test_column_meta_takes_precedence(self):
        """Column-level config should override node-level."""
        node = MockNode(
            meta={"output-to-lower": False},
            columns={
                "user_id": MockColumn(
                    meta={"output-to-lower": True}
                )
            }
        )
        resolver = ConfigResolver(context)

        # Column-level should win
        assert resolver.resolve(
            "output-to-lower",
            node,
            column_name="user_id"
        ) is True

        # Node-level for other columns
        assert resolver.resolve(
            "output-to-lower",
            node,
            column_name="email"
        ) is False
```

### Integration Test Example

```python
def test_dbt_osmosis_yaml_precedence(demo_project):
    """Test dbt-osmosis.yml is read correctly."""
    # Create dbt-osmosis.yml
    config_file = demo_project.project_root / "dbt-osmosis.yml"
    config_file.write_text("""
skip-add-tags: true
use-unrendered-descriptions: true
""")

    # Load project
    context = load_context(demo_project.project_root)
    resolver = ConfigResolver(context)

    # Should read from file
    assert resolver.resolve("skip-add-tags", None) is True
```

## Common Pitfalls

### 1. Forgetting to Check Field Existence

**Problem**: Accessing `node.config.meta` on dbt 1.8 raises AttributeError.

**Solution**: Always use `hasattr()` checks for version-specific fields.

```python
# ❌ Wrong
value = node.config.meta.get("key")

# ✅ Correct
if hasattr(node, 'config') and hasattr(node.config, 'meta'):
    value = node.config.meta.get("key")
```

### 2. Not Handling Missing YAML Files

**Problem**: Calling `_get_node_yaml()` on ephemeral models may fail.

**Solution**: Fall back to manifest with warning.

```python
# ❌ Wrong
yaml_data = _get_node_yaml(node)

# ✅ Correct
try:
    yaml_data = _get_node_yaml(node)
except FileNotFoundError:
    logger.warning(":warning: No YAML file for node %s, using manifest", node.name)
    yaml_data = {}
```

### 3. Ignoring Thread Safety

**Problem**: YAML buffer cache is not thread-safe without locks.

**Solution**: Always use existing lock when accessing cache.

```python
# ❌ Wrong
yaml_data = _YAML_BUFFER_CACHE.get(file_path)

# ✅ Correct
with _YAML_BUFFER_CACHE_LOCK:
    yaml_data = _YAML_BUFFER_CACHE.get(file_path)
```

## Debugging Tips

### Enable Debug Logging

```bash
# Run with verbose flag
dbt-osmosis yaml organize --project-dir ./my-project --verbose

# Check which source provided a value
# Logs will show:
# :mag: Resolved 'skip-add-tags' from node_meta for node 'my_model'
```

### Inspect Precedence Chain

```python
resolver = ConfigResolver(context)
chain = resolver.get_precedence_chain(node, column_name)
print(chain)
# ['column_meta', 'node_meta', 'config_extra', 'config_meta', ...]
```

### Test Specific Source

```python
# Test only supplementary file
source = SupplementaryFileSource(context)
value = source.get("my-setting")
print(f"Supplementary file value: {value}")
```

## Next Steps

After implementation:

1. Run full test suite: `task test`
2. Run linting: `task lint`
3. Run formatting: `task format`
4. Update documentation in `CLAUDE.md`
5. Create migration guide for users (if needed)

## Questions?

- Check the [research document](research.md) for technical decisions
- Check the [data model](data-model.md) for entity relationships
- Check the [API contract](contracts/config-resolver.py) for interface details
- Ask in project discussions or review existing PRs for similar changes
