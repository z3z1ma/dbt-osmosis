---
sidebar_position: 1
---

# Configuration

This page describes how dbt-osmosis discovers file paths and behavior settings.

## Routing YAML files

### Models and seeds

Provide a `+dbt-osmosis` template under each folder you want managed:

```yaml title="dbt_project.yml"
models:
  <your_project_name>:
    +dbt-osmosis: "_{model}.yml"

    staging:
      +dbt-osmosis: "{parent}.yml"

    intermediate:
      +dbt-osmosis: "{node.config[materialized]}/{model}.yml"

seeds:
  <your_project_name>:
    +dbt-osmosis: "_schema.yml"
```

If a node does not have a `+dbt-osmosis` rule, dbt-osmosis can fall back to `vars.dbt_osmosis_default_path`:

```yaml title="dbt_project.yml"
vars:
  dbt_osmosis_default_path: "_{model}.yml"
```

### Sources

Configure managed sources under `vars.dbt-osmosis.sources`:

```yaml title="dbt_project.yml"
vars:
  dbt-osmosis:
    sources:
      salesforce:
        path: "staging/salesforce/source.yml"
        schema: "salesforce_v2"
      marketo: "staging/customer/marketo.yml"

    column_ignore_patterns:
      - "_FIVETRAN_SYNCED"
      - ".*__key__.namespace"
```

## Behavior settings

Use CLI flags for global defaults and override them in config when needed.

## YAML writer settings

You can tune the underlying `ruamel.yaml` serializer via `vars.dbt-osmosis.yaml_settings`:

```yaml title="dbt_project.yml"
vars:
  dbt-osmosis:
    yaml_settings:
      width: 120
      preserve_quotes: true
```

### CLI defaults

```bash
dbt-osmosis yaml refactor \
  --skip-add-columns \
  --skip-add-data-types \
  --skip-merge-meta \
  --skip-add-tags \
  --numeric-precision-and-scale \
  --string-length \
  --force-inherit-descriptions \
  --output-to-lower \
  --add-progenitor-to-meta \
  --strip-eof-blank-lines
```

### Folder-level overrides

```yaml title="dbt_project.yml"
models:
  my_project:
    staging:
      +dbt-osmosis: "{parent}.yml"
      +dbt-osmosis-options:
        skip-add-columns: true
        sort-by: "alphabetical"

    intermediate:
      +dbt-osmosis: "{node.config[materialized]}/{model}.yml"
      +dbt-osmosis-options:
        skip-add-tags: true
        output-to-lower: true
```

### Node-level overrides

```jinja title="models/intermediate/some_model.sql"
{{ config(
    materialized='incremental',
    dbt_osmosis_options={
      "skip-add-data-types": true,
      "sort-by": "alphabetical"
    }
) }}
```

### Column-level overrides

```yaml
tables:
  - name: some_model
    columns:
      - name: tricky_column
        meta:
          dbt-osmosis-skip-add-data-types: true
          dbt_osmosis_options:
            skip-add-tags: true
```

## Setting precedence (most specific wins)

1. Column `meta` and column `dbt-osmosis-options`
2. Node `meta` and `dbt_osmosis_options`
3. Node `config.extra` and `dbt_osmosis_options`
4. CLI defaults / fallback settings

## Common options (excerpt)

See the [settings reference](../reference/settings) for the full list of options and defaults.
