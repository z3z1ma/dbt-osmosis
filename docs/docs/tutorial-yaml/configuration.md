---
sidebar_position: 1
---
# Configuration

## Configuring dbt-osmosis

### Models

At a minimum, each **folder** (or subfolder) of models in your dbt project must specify **where** dbt-osmosis should place the YAML files, using the `+dbt-osmosis` directive:

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
```

You can also apply it to **seeds** exactly the same way:

```yaml title="dbt_project.yml"
seeds:
  <your_project_name>:
    +dbt-osmosis: "_schema.yml"
```

This ensures seeds also end up with automatically created YAML schemas.

---

### Sources

Optionally, you can configure dbt-osmosis to manage **sources** by specifying an entry under `vars.dbt-osmosis.sources`. For each source you want managed:

```yaml title="dbt_project.yml"
vars:
  dbt-osmosis:
    sources:
      salesforce:
        path: "staging/salesforce/source.yml"
        schema: "salesforce_v2"  # If omitted, defaults to the source name

      marketo: "staging/customer/marketo.yml"
      jira: "staging/project_mgmt/schema.yml"
      github: "all_sources/github.yml"

  # (Optional) columns that match these patterns will be ignored
  column_ignore_patterns:
    - "_FIVETRAN_SYNCED"
    - ".*__key__.namespace"
```

**Key Points**:

- `vars: dbt-osmosis: sources: <source_name>` sets **where** the source YAML file lives.
- If the source doesn't exist yet, dbt-osmosis can **bootstrap** that YAML automatically when you run `yaml organize` or `yaml refactor`.
- `schema: salesforce_v2` overrides the default schema name if desired. If you omit it, dbt-osmosis assumes your source name is the schema name.
- Patterns in `column_ignore_patterns` let you skip ephemeral or system columns across your entire project.

---

## Fine-Grained Control Over Behavior

Beyond **where** to place files, dbt-osmosis provides many **tunable options** for how it handles column injection, data types, inheritance, etc. You can specify these in **multiple levels**—globally, folder-level, node-level, or even per-column. dbt-osmosis merges them in a chain, so the most specific setting “wins.”

### 1. Global Options via Command Line Flags

You can declare project-wide defaults using command line flags when running the dbt-osmosis CLI:

```sh
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
  --sort-by=database
```

These **global** settings apply to **all** models and sources unless overridden at a lower level.

### 2. Folder-Level +dbt-osmosis-options

This is the canonical approach in 1.1 forward. Inside `dbt_project.yml`, you can attach `+dbt-osmosis-options` to a subfolder:

```yaml title="dbt_project.yml"
models:
  my_project:
    # Blanket rule for entire project
    +dbt-osmosis: "_{model}.yml"

    staging:
      +dbt-osmosis: "{parent}.yml"
      +dbt-osmosis-options:
        skip-add-columns: true
        skip-add-data-types: false
        # Reorder columns alphabetically
        sort-by: "alphabetical"

    intermediate:
      +dbt-osmosis: "{node.config[materialized]}/{model}.yml"
      +dbt-osmosis-options:
        skip-add-tags: true
        output-to-lower: true
      +dbt-osmosis-sort-by: "alphabetical" # Flat keys work too
```

This means everything in the `staging` folder will skip adding **new** columns from the database, reorder existing columns alphabetically, but **won’t** skip data types (the default from the global level stands). Meanwhile, `intermediate` models skip adding tags and convert all columns/data types to lowercase.

### 3. Node-Level Config in the SQL File

You can also specify **node-level** overrides in the `.sql` file via dbt’s `config(...)`:

```jinja
-- models/intermediate/some_model.sql
{{ config(
    materialized='incremental',
    dbt_osmosis_options={
      "skip-add-data-types": True,
      "sort-by": "alphabetical"
    }
) }}

SELECT * FROM ...
```

Here, we’re telling dbt-osmosis that for **this** model specifically, skip adding data types and sort columns alphabetically. This merges on top of any folder-level or global-level config.

### 4. Per-Column Meta

If you want to override dbt-osmosis behavior for a **specific column** only, you can do so in your schema YAML:

```yaml
models:
  - name: some_model
    columns:
      - name: tricky_column
        description: "This column is weird, do not reorder me"
        meta:
          dbt-osmosis-skip-add-data-types: true
          dbt_osmosis_options:
            skip-add-tags: true
```

Or in your node’s dictionary-based definition. dbt-osmosis checks:

1. `column.meta["dbt-osmosis-skip-add-data-types"]` or `column.meta["dbt_osmosis_skip_add_data_types"]`
2. `column.meta["dbt-osmosis-options"]` or `dbt_osmosis_options`
3. Then your **node** meta/config
4. Then folder-level
5. Finally global project-level

At each level, dbt-osmosis merges or overrides as needed.

---

## Examples of Commonly Used dbt-osmosis Options

Below is a reference table of some popular flags or options you can set at **any** of the levels (global, folder, node, column). Many of these are also available as CLI flags, but when set in your configuration, they become “defaults.”

| Option Name                       | Purpose                                                                                                                    |
| -------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `skip-add-columns`               | If `true`, dbt-osmosis won’t inject columns that exist in the warehouse but are missing in your YAML.                      |
| `skip-add-source-columns`        | If `true`, skip column injection **specifically** on sources. Useful if sources have wide schemas and you only want columns for models. |
| `skip-add-data-types`            | If `true`, dbt-osmosis won’t populate the `data_type` field for columns.                                                  |
| `skip-merge-meta`                | If `true`, dbt-osmosis won’t inherit or merge `meta` fields from upstream models.                                         |
| `skip-add-tags`                  | If `true`, dbt-osmosis won’t inherit or merge `tags` from upstream models.                                                |
| `numeric-precision-and-scale`    | If `true`, numeric columns will keep precision/scale in their type (like `NUMBER(38, 8)` vs. `NUMBER`).                    |
| `string-length`                  | If `true`, string columns will keep length in their type (like `VARCHAR(256)` vs. `VARCHAR`).                              |
| `force-inherit-descriptions`     | If `true`, a child model’s columns will always accept upstream descriptions if the child’s description is **empty** or a placeholder. |
| `output-to-lower`                | If `true`, all column names and data types in the YAML become lowercase.                                                  |
| `sort-by`                        | `database` or `alphabetical`. Tells dbt-osmosis how to reorder columns.                                                   |
| `prefix`                         | A special string used by the **fuzzy** matching plugin. If you consistently prefix columns in staging, dbt-osmosis can strip it when matching. |
| `add-inheritance-for-specified-keys` | Provide a list of **additional** keys (e.g., `["policy_tags"]`) that should also be inherited from upstream.             |

And much more. Many flags also exist as **command-line** arguments (`--skip-add-tags`, `--skip-merge-meta`, `--force-inherit-descriptions`, etc.), which can override or complement your config settings in `dbt_project.yml`.

---

## Summary

**dbt-osmosis** configuration is highly **modular**. You:

1. **Always** specify a `+dbt-osmosis: "<some_path>.yml"` directive per folder (so osmosis knows where to place YAML).
2. Set **options** (like skipping columns, adding data types, etc.) **globally** via either cli flags, a more granular **folder-level** with `+dbt-osmosis-options`, **node-level** in `.sql`, or **column-level** in metadata.
3. Let dbt-osmosis handle the merging and logic so that the final outcome respects your most **specific** settings.

With this approach, you can achieve everything from a simple one-YAML-per-model style to a more advanced structure that merges doc from multiple upstream sources while selectively skipping columns or data types.
