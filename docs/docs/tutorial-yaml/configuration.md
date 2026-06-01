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

### Fusion-Compatible Routing via `vars`

If you use **dbt-fusion**, the `+dbt-osmosis` config keys in `dbt_project.yml` will cause parse errors because fusion's strict parser rejects unknown `+` prefixed keys. As an alternative, you can specify per-folder routing under `vars.dbt-osmosis.models`:

```yaml title="dbt_project.yml"
vars:
  dbt-osmosis:
    models:
      staging: "_stg_{parent}__models.yml"
      intermediate: "_int_{parent}__models.yml"
      marts: "_marts_{parent}__models.yml"
    seeds: "_seeds__models.yml"
```

This is functionally equivalent to `+dbt-osmosis` config keys but uses `vars:`, which both dbt-core and dbt-fusion accept. Routing matches against the node's FQN folder path -- a model in `models/staging/oem_raw/` matches the `staging` key. For nested folders, use dot notation (`staging.oem_raw`) -- the most specific match wins.

Seeds can be a single string (applies to all seeds) or a dict with per-folder keys like models.

**Precedence**: `+dbt-osmosis` config keys (if present) take priority over vars routing. This means existing dbt-core projects keep working unchanged -- vars routing is only used when the config key is absent.

### Placing vars in `vars.yml` (dbt-core 1.12+)

dbt-core 1.12 and dbt-core 2.0 (Fusion) support an external `vars.yml` file at the project root as an alternative to the `vars:` key in `dbt_project.yml`. dbt-osmosis reads from both locations transparently -- no configuration change is needed in osmosis itself.

To use `vars.yml`, create the file and move the `dbt-osmosis` vars block there:

```yaml title="vars.yml"
vars:
  dbt-osmosis:
    models:
      staging: "_stg_{parent}__models.yml"
      intermediate: "_int_{parent}__models.yml"
      marts: "_marts_{parent}__models.yml"
    seeds: "_seeds__models.yml"
```

Then remove the `vars:` block from `dbt_project.yml`. The two locations are mutually exclusive -- having `vars:` defined in both files raises a `DbtProjectError` at parse time.

Note that `+meta: {dbt-osmosis: ...}` and `+dbt-osmosis:` model config keys are separate from the `vars:` block and are not affected by this rule. Those keys live in `dbt_project.yml` regardless of where vars are declared.

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

## Fusion compatibility

:::caution dbt-osmosis requires dbt-core

dbt-osmosis **does not run on dbt Fusion**. It depends on dbt-core for manifest parsing, database introspection, and SQL compilation. If your team uses dbt Fusion, you will need a **hybrid setup** with two virtual environments: one running dbt-core (for osmosis) and one running dbt Fusion (for your normal development workflow). Both engines can share the same project directory.

:::

dbt-osmosis can produce **Fusion-compatible YAML** where `meta` and `tags` are nested inside `config` blocks instead of at the top level. This output format is required for dbt >= 1.9.6 and is the only format recognized by dbt Fusion.

### Auto-detection

By default (`--fusion-compat` not specified), dbt-osmosis auto-detects whether to produce Fusion-compatible output:

1. **Fusion manifest** — if `target/manifest.json` has a schema version > v12 (Fusion produces v20), fusion-compat is enabled. This check reads the manifest before osmosis re-parses the project, since parsing via dbt-core overwrites it with a v12 manifest.
2. **Fusion binary on PATH** — if no manifest exists but `dbt-fusion` or `dbtf` is found on `PATH`, fusion-compat is enabled.
3. **dbt-core version** — if dbt-core >= 1.9.6 is installed, fusion-compat is enabled (these versions natively support the `config` block format).

### Explicit override

```bash
# Force Fusion-compatible output
dbt-osmosis yaml refactor --fusion-compat

# Force legacy output (even on dbt >= 1.9.6)
dbt-osmosis yaml refactor --no-fusion-compat
```

### Hybrid workflow for Fusion projects

If your team is testing dbt Fusion alongside dbt-core:

1. Maintain **two virtual environments** — one with `dbt-core` + `dbt-osmosis`, another with `dbt-fusion`.
2. Run dbt Fusion for compilation and execution in your normal workflow.
3. Run dbt-osmosis from the dbt-core environment to manage YAML schema files. Osmosis will detect the Fusion manifest and automatically produce compatible output.
4. Both environments can share the same `dbt_project.yml` and model files.

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
  --strip-eof-blank-lines \
  --fusion-compat
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
