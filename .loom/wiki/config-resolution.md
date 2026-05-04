---
id: wiki:config-resolution
kind: wiki
page_type: reference
status: active
created_at: 2026-05-04T09:45:38Z
updated_at: 2026-05-04T09:45:38Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  tickets:
    - ticket:c10res14
  evidence:
    - evidence:c10res14-context-aware-resolver-validation
  critique:
    - critique:c10res14-context-aware-settings-resolver-review
---

# Summary

dbt-osmosis setting lookups should flow through `SettingsResolver` or the context-aware `resolve_setting(context, setting_name, node, col=None, fallback=...)` helper. Production code that has a YAML refactor context should not call `_get_setting_for_node()` directly.

# Precedence

The accepted resolver order is:

1. Column meta and column `config.meta` dbt-osmosis settings.
2. Node meta dbt-osmosis settings.
3. Node `config.extra` dbt-osmosis settings.
4. dbt 1.10+ node `config.meta` and `unrendered_config` dbt-osmosis settings.
5. Explicit non-default `context.settings` values when the call passes that value as fallback.
6. Supplementary project-root `dbt-osmosis.yml`.
7. Project vars under `dbt-osmosis`, `dbt_osmosis`, prefixed top-level vars, direct top-level vars, or supported options objects.
8. Caller fallback.

# Rules

- Use a missing sentinel internally so `False`, `0`, and `""` remain valid resolved settings.
- Treat explicit `None` as missing unless a future spec changes source-class semantics.
- Support kebab-case and snake_case setting names, including nested `dbt-osmosis-options` and `dbt_osmosis_options` inner keys.
- Pass the full `YamlRefactorContext` to resolver-aware helpers when project-root supplementary files or project vars should be visible.
- Do not pass `context.project` to plugin hooks or resolver paths that need supplementary file or project vars access.
- In all-node fan-out functions, avoid global project-level shortcuts when higher-precedence per-node overrides may need to win.
- Keep `_get_setting_for_node()` only as a compatibility wrapper or public facade surface; new context-bearing production behavior should use `resolve_setting()`.

# Debugging

Use `SettingsResolver.has(..., context=context)` to check whether any node or context source supplies a setting. Use `SettingsResolver.get_precedence_chain(..., context=context)` to inspect the source chain, including `CONTEXT_SETTINGS`, supplementary file, project vars, and fallback.

When a context setting should outrank project config, production call sites should pass the current setting as fallback:

```python
resolve_setting(
    context,
    "skip-add-tags",
    node,
    column_name,
    fallback=context.settings.skip_add_tags,
)
```

# Boundaries

- This page explains accepted resolver behavior. Ticket closure, evidence sufficiency, and critique verdicts remain owned by their respective Loom layers.
- Full CI matrix validation for the broader dbt 1.10/1.11 hardening initiative is still handled at initiative-level final validation.
- Third-party plugins should tolerate receiving full `YamlRefactorContext` in hook `context` parameters.

# Sources

- ticket:c10res14
- evidence:c10res14-context-aware-resolver-validation
- critique:c10res14-context-aware-settings-resolver-review
- `src/dbt_osmosis/core/introspection.py`
- `src/dbt_osmosis/core/transforms.py`
- `src/dbt_osmosis/core/inheritance.py`
- `src/dbt_osmosis/core/sync_operations.py`
- `src/dbt_osmosis/core/plugins.py`
- `tests/core/test_config_resolution.py`
- `tests/core/test_transforms.py`
- `tests/core/test_sync_operations.py`
- `tests/core/test_introspection.py`
- `tests/core/test_inheritance_behavior.py`

# Related Pages

- wiki:repository-atlas
