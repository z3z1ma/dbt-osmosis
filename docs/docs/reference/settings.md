---
sidebar_position: 2
---

# Settings reference

These settings map to `YamlRefactorSettings`, the YAML-routing configuration surface, or the supplementary repo-level `dbt-osmosis.yml` file.

## Where settings live

The configuration surface is split across a few places:

- `+dbt-osmosis` in `dbt_project.yml` chooses where a node's YAML should live.
- `+dbt-osmosis-options` in `dbt_project.yml` sets folder-level YAML behavior.
- `dbt-osmosis-options` / `dbt_osmosis_options` in model config or meta can override behavior at node level.
- column `meta` can override some inheritance behavior for a specific column.
- `dbt-osmosis.yml` currently provides supplementary repo-level settings such as the external `formatter` command.
- CLI flags always apply to the current invocation.

If you set options inside a SQL `config(...)` block, use Python identifiers such as `dbt_osmosis_options={...}` rather than hyphenated keys.

## Routing

| Setting | Default | Description |
| --- | --- | --- |
| `+dbt-osmosis` | none | YAML routing template for a folder or node. Example: `"_{model}.yml"`, `"{parent}.yml"`, or `"{node.config[materialized]}/{model}.yml"`. |
| `vars.dbt_osmosis_default_path` | unset | Fallback YAML path template when no `+dbt-osmosis` rule applies. |

## Fusion compatibility

> `dbt-osmosis` still runs on dbt Core. `fusion-compat` controls YAML output shape for projects that want Fusion-compatible `config` blocks.

| Setting | Default | Description |
| --- | --- | --- |
| `fusion-compat` | `null` (auto) | Output Fusion-compatible YAML (`meta` and `tags` nested inside `config`). When `null`, dbt-osmosis auto-detects from Fusion manifest evidence first, then falls back to dbt Core version detection (`>= 1.9.6`). |

## Core behavior

| Setting | Default | Description |
| --- | --- | --- |
| `skip-merge-meta` | `false` | Skip inheriting and merging upstream `meta` fields. |
| `skip-add-columns` | `false` | Skip injecting missing model columns from the warehouse or catalog. |
| `skip-add-source-columns` | `false` | Skip injecting missing source columns. |
| `skip-add-tags` | `false` | Skip inheriting upstream tags. |
| `skip-add-data-types` | `false` | Skip populating `data_type`. |
| `force-inherit-descriptions` | `false` | Overwrite child descriptions with upstream descriptions when available. |
| `use-unrendered-descriptions` | `false` | Preserve unrendered `{{ doc(...) }}` descriptions instead of rendered manifest text. |
| `prefer-yaml-values` | `false` | Preserve YAML values as authored across supported fields, including unrendered Jinja such as `{{ var(...) }}` or `{{ env_var(...) }}`. |
| `add-progenitor-to-meta` | `false` | Add progenitor metadata so downstream columns can point back to their inherited origin. |
| `add-inheritance-for-specified-keys` | `[]` | Additional keys to inherit, supplied as a repeatable CLI flag or list-valued option. |

## Output formatting

| Setting | Default | Description |
| --- | --- | --- |
| `output-to-lower` | `false` | Force column names and data types to lowercase when possible. |
| `output-to-upper` | `false` | Force column names and data types to uppercase when possible. |
| `numeric-precision-and-scale` | `false` | Preserve numeric precision and scale in rendered data types. |
| `string-length` | `false` | Preserve string length in rendered data types. |
| `sort-by` | `database` | Column ordering mode (`database` or `alphabetical`). |
| `strip-eof-blank-lines` | `false` | Remove trailing blank lines at EOF when writing YAML. |
| `scaffold-empty-configs` | `false` | Emit empty or placeholder YAML fields instead of omitting them. |
| `formatter` | `null` | External formatter command to run once after dbt-osmosis writes YAML files. Set via `--formatter` or `dbt-osmosis.yml` (for example `prettier --write` or `yamlfmt`). |

## Catalog and introspection

| Setting | Default | Description |
| --- | --- | --- |
| `catalog-path` | `null` | Use a specific `catalog.json` instead of live warehouse introspection for column/type lookup. |
| `create-catalog-if-not-exists` | `false` | Generate a catalog when one is missing and then use it for introspection. |
| `disable-introspection` | `false` | Allow YAML commands to run without a live database connection. Pair this with `catalog-path` when you still want column/type information. |
| `include-external` | `false` | Include models and sources from external dbt packages. |

## Execution

| Setting | Default | Description |
| --- | --- | --- |
| `dry-run` | `false` | Compute changes without writing them to disk. |
| `check` | `false` | Exit non-zero if files changed or would have changed. This is a CLI behavior flag rather than a persisted config setting. |
| `auto-apply` | `false` | Apply the restructure plan without interactive confirmation. This is available on commands that move files. |

## Practical examples

Folder-level routing and behavior:

```yaml title="dbt_project.yml"
models:
  my_project:
    staging:
      +dbt-osmosis: "{parent}.yml"
      +dbt-osmosis-options:
        skip-add-columns: true
        sort-by: alphabetical
```

Repo-level formatter:

```yaml title="dbt-osmosis.yml"
formatter: "prettier --write"
```

Model-level SQL config:

```sql
{{
  config(
    dbt_osmosis_options={
      "string_length": true,
      "numeric_precision_and_scale": true,
    }
  )
}}
```
