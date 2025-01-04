---
sidebar_position: 1
---

# Configuration

## Configuring dbt-osmosis

### Models

At minimum, each **folder** (or subfolder) of models in your dbt project must specify the `+dbt-osmosis` directive so that dbt-osmosis knows **where** to create or move the YAML files.

```yaml title="dbt_project.yml"
models:
  <your_project_name>:
    +dbt-osmosis: "_{model}.yml"   # Default for entire project

    staging:
      +dbt-osmosis: "{parent}.yml" # Each subfolder lumps docs by folder name

    intermediate:
      # Example of using node.config or node.tags
      +dbt-osmosis: "{node.config[materialized]}/{model}.yml"

    marts:
      # A single schema file for all models in 'marts'
      +dbt-osmosis: "prod.yml"

seeds:
  <your_project_name>:
    +dbt-osmosis: "_schema.yml"
```

### Sources

You can optionally configure dbt-osmosis to manage sources automatically. In your `dbt_project.yml`:

```yaml title="dbt_project.yml"
vars:
  dbt-osmosis:
    sources:
      salesforce:
        path: "staging/salesforce/source.yml"
        schema: "salesforce_v2"

      marketo: "staging/customer/marketo.yml"
      jira: "staging/project_mgmt/{parent}.yml"
      github: "all_sources/github.yml"

  # Columns matching these patterns will be ignored (like ephemeral system columns)
  column_ignore_patterns:
    - "_FIVETRAN_SYNCED"
    - ".*__key__.namespace"
```

**Key points:**

- `vars: dbt-osmosis: sources: <source_name>` sets where the source YAML file should live.
- If the source does not actually exist yet, dbt-osmosis can bootstrap it.
- If you omit `schema`, dbt-osmosis infers it is the same as your source name.
