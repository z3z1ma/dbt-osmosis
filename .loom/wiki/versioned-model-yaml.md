---
id: wiki:versioned-model-yaml
kind: wiki
page_type: reference
status: active
created_at: 2026-05-04T11:20:10Z
updated_at: 2026-05-04T11:20:10Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  tickets:
    - ticket:c10ver15
  evidence:
    - evidence:c10ver15-versioned-yaml-access-validation
  critique:
    - critique:c10ver15-versioned-yaml-access-review
---

# Summary

dbt versioned models store version-specific YAML under `models[].versions[]`. dbt-osmosis treats the selected version block as the YAML view for a versioned `ModelNode`, while preserving explicit top-level model fallback behavior and dbt selector controls.

# Accepted Rules

- Versioned `ModelNode` YAML access selects the matching `models[].versions[].v` entry before reading node or column properties.
- Version-level `columns` own column YAML for that version. dbt-osmosis does not merge top-level model columns into a selected version view.
- Top-level model `name`, `description`, `meta`, and `tags` remain available as node-level fallback metadata when the selected version block does not provide them. Blank version `description` also falls back to the top-level model description, matching dbt patch behavior.
- Version identity matching tries exact raw identity first. Restricted numeric fallback may bridge common manifest/YAML skew such as `2` and `"2"`, but dbt-distinct string identities such as `"1.1"`, `"1.10"`, and `"01"` must not be collapsed.
- `versions[].columns[]` include/exclude controls are selector entries, not columns. They have no `name`, should be skipped by column lookup, and should be preserved during sync/refactor.
- Selector validation follows dbt's `IncludeExclude` shape: `include` is required, string include is only `all` or `*`, list include must contain strings, `exclude` must be a list of strings, `exclude` is valid only when include is `all` or `*`, and a version can have at most one selector entry.
- Versioned sync/refactor must use the same version identity rules as YAML access so it updates the intended version block instead of appending duplicate blocks for int/string skew.

# Implementation Pointers

- Version view and identity helpers live in `src/dbt_osmosis/core/inheritance.py`.
- YAML property access flows through `PropertyAccessor(source="yaml")`, which reads the selected view from `_get_node_yaml()`.
- Version validation lives in `ModelValidator` in `src/dbt_osmosis/core/schema/validation.py`.
- Versioned sync and selector preservation live in `src/dbt_osmosis/core/sync_operations.py`.
- Regression coverage lives in `tests/core/test_inheritance_behavior.py`, `tests/core/test_property_accessor.py`, `tests/core/test_validation.py`, and `tests/core/test_sync_operations.py`.

# Boundaries

- This page explains accepted dbt-osmosis behavior; dbt itself remains the authority on full dbt schema semantics.
- `ModelValidator` is intentionally focused and does not replace dbt parse validation for every version-level field.
- `_get_node_yaml()` returns a shallow read-only mapping. Nested YAML data remains mutable through existing cache behavior.

# Sources

- ticket:c10ver15
- evidence:c10ver15-versioned-yaml-access-validation
- critique:c10ver15-versioned-yaml-access-review
- packet:ralph-ticket-c10ver15-20260504T095044Z
- implementation commit `ef1d5409bcceee40c3403c06d75bf5cbe4cc4bb1`

# Related Pages

- wiki:yaml-sync-safety
- wiki:config-resolution
