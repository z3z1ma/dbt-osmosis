---
sidebar_position: 2
---

# Manage source YAML files

Use dbt-osmosis to create and keep source YAML in sync with your warehouse.

## 1. Declare sources in `dbt_project.yml`

```yaml title="dbt_project.yml"
vars:
  dbt-osmosis:
    sources:
      salesforce:
        path: "staging/salesforce/source.yml"
        schema: "salesforce_v2"
      marketo: "staging/customer/marketo.yml"
```

## 2. Bootstrap source YAML

```bash
dbt-osmosis yaml organize
```

This creates missing source YAML files at the paths you declared.

## 3. Sync columns

```bash
dbt-osmosis yaml document
```

Use `--skip-add-source-columns` if you want to manage source columns manually.

To persist that behavior for all sources, set a project variable:

```yaml title="dbt_project.yml"
vars:
  dbt-osmosis:
    skip-add-source-columns: true
```

Settings under `models:` apply to model nodes. A `skip-add-source-columns` option
inside `models.<project>.+meta` therefore does not configure sources, even when
the source YAML lives in the same folder as those models.

For source-scoped behavior, put the option under `sources:` instead:

```yaml title="dbt_project.yml"
sources:
  my_project:
    +meta:
      dbt-osmosis-options:
        skip-add-source-columns: true
```

You can also set `dbt-osmosis-options` in an individual source table's `meta`
(`config.meta` on newer dbt versions). A table-level value of `false` overrides
the project variable and allows columns to be added for that table. These
settings apply to both `yaml document` and `yaml refactor`; model column
injection remains enabled.

## 4. Ignore columns globally

```yaml title="dbt_project.yml"
vars:
  dbt-osmosis:
    column_ignore_patterns:
      - "_FIVETRAN_SYNCED"
      - "_AIRBYTE_EMITTED_AT"
```

These regex patterns prevent ephemeral/system columns from being injected.
