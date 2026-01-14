---
sidebar_position: 2
---

# Settings resolution

dbt-osmosis resolves settings from multiple sources, prioritizing the most specific configuration.

## Resolution order

For per-column settings, precedence is:

1. Column `meta` and column `dbt-osmosis-options`
2. Node `meta` and node `dbt_osmosis_options`
3. Node `config.extra` and `dbt_osmosis_options`
4. CLI defaults / fallback values

This means column-level overrides always win, while CLI flags act as the baseline.

## Key formats supported

Most settings can be expressed in multiple ways:

- `skip-add-tags`
- `skip_add_tags`
- `dbt-osmosis-skip-add-tags`
- `dbt_osmosis_skip_add_tags`
- `dbt-osmosis-options: { skip-add-tags: true }`
- `dbt_osmosis_options: { skip_add_tags: true }`

The resolver normalizes kebab-case and snake_case, so choose the style that best matches your existing dbt config.

## Why this matters

Settings are evaluated for every model and column during YAML sync. Understanding precedence helps you avoid surprises when a folder-level rule is overridden by a node or column override.
