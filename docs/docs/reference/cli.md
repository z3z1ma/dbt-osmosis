---
sidebar_position: 1
---

# CLI reference

This page documents the current `dbt-osmosis` CLI surface area as exposed by `dbt-osmosis --help`.

## Global command groups

`dbt-osmosis` currently exposes these top-level commands:

- `yaml` — schema YAML management and documentation inheritance
- `sql` — compile or run ad-hoc SQL in dbt context
- `workbench` — launch the Streamlit workbench
- `generate` — generate sources, staging models, models, and SQL
- `nl` — natural-language helpers
- `test` — suggest dbt tests
- `test-llm` — validate LLM configuration
- `diff` — compare YAML definitions with live database schema
- `lint` — lint SQL strings, models, or a whole project

## Shared dbt options

Most project-aware commands accept some or all of the following:

- `--project-dir`
- `--profiles-dir`
- `-t, --target`
- `--threads`
- `--profile`
- `--vars`
- `--log-level`

Project discovery defaults to the current directory and its parents. Profiles default to `DBT_PROFILES_DIR`, the current directory, the discovered project root, or `~/.dbt`.

## `dbt-osmosis yaml`

YAML commands manage schema files and column-level documentation inheritance.

### Common YAML options

- positional selectors: `dbt-osmosis yaml <command> [<selector> ...]`
- `-f, --fqn` (repeatable)
- `-d, --dry-run`
- `-C, --check`
- `--catalog-path`
- `--disable-introspection`
- `--include-external`
- `--scaffold-empty-configs/--no-scaffold-empty-configs`
- `--strip-eof-blank-lines/--keep-eof-blank-lines`
- `--fusion-compat/--no-fusion-compat`
- `--formatter <command>`

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

Fusion compatibility:

- `--fusion-compat/--no-fusion-compat` outputs Fusion-compatible YAML with `meta` and `tags` nested under `config`. If unspecified, dbt-osmosis auto-detects from Fusion manifest evidence or dbt Core >= 1.9.6.

External formatting:

- `--formatter "prettier --write"` or another CLI formatter command. dbt-osmosis appends written file paths and runs the formatter once after successful writes.

Experimental:

- `--synthesize` requires installing `dbt-osmosis[openai]` and configuring provider-specific environment variables. If `LLM_PROVIDER` is unset, dbt-osmosis uses the OpenAI provider by default. Azure AD authentication also requires `dbt-osmosis[azure]`.

Example:

```bash
dbt-osmosis yaml refactor models/staging --dry-run --check
```

### `dbt-osmosis yaml organize`

Ensures YAML files exist and are placed according to your `+dbt-osmosis` routing rules.

Additional flag:

- `--auto-apply`

### `dbt-osmosis yaml document`

Applies column documentation inheritance and optional column injection/sorting.

It supports the same inheritance, output, and formatting flags as `yaml refactor`, except `--auto-apply`.

## `dbt-osmosis sql`

Executes or compiles ad-hoc SQL, including dbt Jinja, against your project.

- `dbt-osmosis sql run "select ..."`
- `dbt-osmosis sql compile "select ..."`

## `dbt-osmosis workbench`

Runs the Streamlit workbench.

Core options:

- `--project-dir`
- `--profiles-dir`
- `--host` (default `localhost`)
- `--port` (default `8501`)
- `--options` passes through to Streamlit help output
- `--config` prints Streamlit configuration

The workbench runtime requires installing `dbt-osmosis[workbench]` plus a dbt adapter for your project, such as `dbt-osmosis[duckdb]` for the demo DuckDB project.

## `dbt-osmosis generate`

Generates dbt artifacts.

- `dbt-osmosis generate model "<description>"`
  - options: `--model-name`, `--output-path`, `--schema-yml`, `--dry-run`
- `dbt-osmosis generate sources`
  - options: `--source-name`, `--schema-name`, `--exclude-schemas`, `--exclude-tables`, `--quote-identifiers`, `--output-path`, `--dry-run`
- `dbt-osmosis generate staging <source_name> <table_name>`
  - options: `--ai`, `--staging-path`, `--dry-run`
- `dbt-osmosis generate query "<question>"`
  - option: `--execute`

LLM-assisted subcommands require `dbt-osmosis[openai]` and provider-specific environment variables. If `LLM_PROVIDER` is unset, dbt-osmosis uses the OpenAI provider by default. Azure AD authentication also requires `dbt-osmosis[azure]`.

## `dbt-osmosis nl`

Natural-language helpers.

- `dbt-osmosis nl query "<question>"` optionally adds `--execute`
- `dbt-osmosis nl generate "<description>"` remains available but is deprecated in favor of `dbt-osmosis generate model`

## `dbt-osmosis test`

Test suggestion helpers.

Currently exposed subcommand:

- `dbt-osmosis test suggest [<model> ...]`

Important options:

- `-f, --fqn`
- `--use-ai`
- `--pattern-only`
- `--temperature`
- `-o, --output`
- `--format [json|yaml|table]`

AI suggestions are enabled by default and require the OpenAI extra plus provider-specific environment variables. If AI configuration or generation fails, dbt-osmosis reports that fallback and uses pattern-based suggestions. `--pattern-only` disables AI and keeps suggestions deterministic.

## `dbt-osmosis test-llm`

Validates LLM client configuration.

- Reads `LLM_PROVIDER`, defaulting to `openai` when it is unset
- Verifies the provider-specific environment variables are present
- Reports missing optional packages and invalid providers as friendly CLI errors
- Prints the configured provider and model engine on success

## `dbt-osmosis diff`

Schema diff helpers.

Currently exposed subcommand:

- `dbt-osmosis diff schema`

Important options:

- YAML selection flags (`[MODELS]`, `-f/--fqn`, `--include-external`)
- `--output-format [text|json|markdown]`
- `--severity [safe|moderate|breaking|all]`
- `--fuzzy-match-threshold`
- `--detect-column-renames/--no-detect-column-renames`

This command compares YAML definitions with live database schema and reports additions, removals, type changes, and fuzzy-matched renames.

## `dbt-osmosis lint`

SQL lint helpers.

Exposed subcommands:

- `dbt-osmosis lint file <sql-or-path>`
- `dbt-osmosis lint model <model_name>`
- `dbt-osmosis lint project`

Important options across lint commands:

- `--rules`
- `--disable-rules`
- `--dialect`
- `-f, --fqn` on `lint project`

When both `--rules` and `--disable-rules` are supplied, linting starts from the
enabled rule set and then removes disabled rules; disabled rules win on overlap.
`lint model` and `lint project` use the same project-owned, non-ephemeral FQN
matching semantics as YAML selection and exclude external package models by
default.

These commands exit non-zero when errors or warnings are reported.
