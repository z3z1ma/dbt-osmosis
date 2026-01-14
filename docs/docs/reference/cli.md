---
sidebar_position: 1
---

# CLI reference

This page documents the current `dbt-osmosis` CLI surface area.

## Conventions

- Most commands accept `--project-dir` and `--profiles-dir` (dbt project discovery defaults to the current directory and its parents; profiles default to `~/.dbt`).
- YAML commands support positional selectors (`models/`, `models/foo.sql`, or `model_name`) and `--fqn` filters.

## `dbt-osmosis yaml`

YAML commands manage schema files and column-level documentation inheritance.

### Common YAML options

- Positional selectors: `dbt-osmosis yaml <command> [<selector> ...]`
- `-f, --fqn` (repeatable)
- `-d, --dry-run`
- `-C, --check`
- `--catalog-path`
- `--create-catalog-if-not-exists`
- `--disable-introspection`
- `--include-external`
- `--profile`
- `--vars` (YAML string)
- `--scaffold-empty-configs/--no-scaffold-empty-configs`
- `--strip-eof-blank-lines/--keep-eof-blank-lines`

### `dbt-osmosis yaml refactor`

Runs `organize` (file placement) and then `document` (inherit docs) in one command.

Common behavior flags:

- `--auto-apply` (skip confirmation for file moves)
- `-F, --force-inherit-descriptions`
- `--use-unrendered-descriptions` (propagate `{{ doc(...) }}` descriptions)
- `--prefer-yaml-values` (preserve unrendered templates for all fields)
- `--skip-merge-meta`
- `--skip-add-tags`
- `--skip-add-columns`
- `--skip-add-source-columns`
- `--skip-add-data-types`
- `--add-progenitor-to-meta`
- `--add-inheritance-for-specified-keys <key>` (repeatable)
- `--numeric-precision-and-scale`
- `--string-length`
- `--output-to-lower`
- `--output-to-upper`
- `--include-external`

Experimental:

- `--synthesize` (requires installing `dbt-osmosis[openai]` and configuring `LLM_PROVIDER` + provider-specific environment variables)

Example:

```bash
dbt-osmosis yaml refactor models/staging --dry-run
```

### `dbt-osmosis yaml organize`

Ensures YAML files exist and are placed according to your `+dbt-osmosis:` rules.

- `--auto-apply`

### `dbt-osmosis yaml document`

Applies column documentation inheritance and optional column injection/sorting.

Supports the same inheritance and output flags as `yaml refactor` (except `--auto-apply`).

## `dbt-osmosis sql`

Executes or compiles ad-hoc SQL (including dbt Jinja) against your project.

- `dbt-osmosis sql run "select ..."`
- `dbt-osmosis sql compile "select ..."`

## `dbt-osmosis workbench`

Runs the Streamlit workbench.

- `--host` (default `localhost`)
- `--port` (default `8501`)

Tip: `dbt-osmosis workbench --options` prints Streamlit runner help.

## `dbt-osmosis test-llm`

Validates LLM client configuration for synthesis features.

- Requires `LLM_PROVIDER` to be set.
- Prints the configured provider and model engine when successful.

## `dbt-osmosis generate`

Generates dbt artifacts (introspection- and/or LLM-driven).

LLM-assisted subcommands require `dbt-osmosis[openai]` and a configured `LLM_PROVIDER`.

- `dbt-osmosis generate sources` (generate dbt source YAML from database introspection)
- `dbt-osmosis generate staging <source_name> <table_name>` (generate a staging model; add `--ai` for LLM-assisted logic)
- `dbt-osmosis generate model "<description>"` (generate a dbt model from a natural language description)
- `dbt-osmosis generate query "<question>"` (generate SQL from a natural language question; add `--execute` to run it)

## `dbt-osmosis nl`

Natural-language helpers.

- `dbt-osmosis nl query "<question>"` (generate SQL; add `--execute` to run it)
- `dbt-osmosis nl generate "<description>"` is deprecated; use `dbt-osmosis generate model` instead.
