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
dbt-osmosis yaml document --skip-add-source-columns=false
```

Use `--skip-add-source-columns` if you want to manage source columns manually.

## 4. Ignore columns globally

```yaml title="dbt_project.yml"
vars:
  dbt-osmosis:
    column_ignore_patterns:
      - "_FIVETRAN_SYNCED"
      - "_AIRBYTE_EMITTED_AT"
```

These regex patterns prevent ephemeral/system columns from being injected.
