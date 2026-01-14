---
sidebar_position: 2
---

# Settings reference

These settings map to `YamlRefactorSettings` and can be set via CLI flags or config keys.

## Core behavior

| Setting | Default | Description |
| --- | --- | --- |
| `skip-merge-meta` | `false` | Skip inheriting/merging upstream `meta` fields. |
| `skip-add-columns` | `false` | Skip injecting missing model columns from the warehouse. |
| `skip-add-source-columns` | `false` | Skip injecting missing source columns. |
| `skip-add-tags` | `false` | Skip inheriting upstream tags. |
| `skip-add-data-types` | `false` | Skip populating `data_type`. |
| `force-inherit-descriptions` | `false` | Overwrite child descriptions with upstream descriptions. |
| `use-unrendered-descriptions` | `false` | Preserve unrendered `{{ doc(...) }}` descriptions. |
| `prefer-yaml-values` | `false` | Preserve unrendered Jinja values across all fields. |
| `add-progenitor-to-meta` | `false` | Add `meta.osmosis_progenitor` to inherited columns. |
| `add-inheritance-for-specified-keys` | `[]` | Additional keys to inherit (repeatable). |

## Output formatting

| Setting | Default | Description |
| --- | --- | --- |
| `output-to-lower` | `false` | Force column names and data types to lowercase. |
| `output-to-upper` | `false` | Force column names and data types to uppercase. |
| `numeric-precision-and-scale` | `false` | Preserve numeric precision and scale in types. |
| `string-length` | `false` | Preserve string length in types. |
| `strip-eof-blank-lines` | `false` | Remove trailing blank lines at EOF when writing YAML. |
| `sort-by` | `database` | Column ordering (`database` or `alphabetical`). |

## Catalog and introspection

| Setting | Default | Description |
| --- | --- | --- |
| `catalog-path` | `null` | Use a specific `catalog.json` for types instead of live introspection. |
| `create-catalog-if-not-exists` | `false` | Build a catalog if one is missing. |

## Execution

| Setting | Default | Description |
| --- | --- | --- |
| `dry-run` | `false` | Do not write changes to disk. |
| `include-external` | `false` | Include nodes from external packages. |
| `scaffold-empty-configs` | `false` | Emit empty/placeholder YAML fields. |

## Where to set these

- CLI flags (global defaults)
- Folder-level `+dbt-osmosis-options`
- Node-level `dbt_osmosis_options` in `config(...)`
- Column-level `meta` overrides
