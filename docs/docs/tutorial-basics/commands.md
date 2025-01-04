---
sidebar_position: 2
---

# CLI Overview

Below is a high-level overview of the commands currently provided by dbt-osmosis. Each command also supports additional options such as:

- `--dry-run` to prevent writing changes to disk
- `--check` to exit with a non-zero code if changes would have been made
- `--fqn` to filter nodes by [dbt's FQN](https://docs.getdbt.com/reference/node-selection/syntax#the-fqn-method) segments
- `--disable-introspection` to run without querying the warehouse (helpful if you are offline), often paired with `--catalog-path`
- `--catalog-path` to read columns from a prebuilt `catalog.json`

Other helpful flags are described in each command below.

## YAML Management

**All of the following commands live under** `dbt-osmosis yaml <command>`.

### Organize

Restructures your schema YAML files based on the **declarative** configuration in `dbt_project.yml`. Specifically, it:

- Bootstraps missing YAML files for any undocumented models or sources
- Moves or merges existing YAML files according to your configured rules (the `+dbt-osmosis:` keys)

```bash
dbt-osmosis yaml organize [--project-dir] [--profiles-dir] [--target] [--fqn ...] [--dry-run] [--check]
```

Options often used:

- `--auto-apply` to apply all file location changes without asking for confirmation
- `--disable-introspection` + `--catalog-path=/path/to/catalog.json` if not connected to a warehouse

### Document

Passes down column-level documentation from upstream nodes to downstream nodes (a deep inheritance). Specifically, it can:

- Add columns that are present in the database (or `catalog.json`) but missing from your YAML
- Remove columns missing from your database (optional, if used with other steps)
- Reorder columns (optional, if combined with your sorting preference—see below)
- Inherit tags, descriptions, and meta fields from upstream models

```bash
dbt-osmosis yaml document [--project-dir] [--profiles-dir] [--target] [--fqn ...] [--dry-run] [--check]
```

Options often used:

- `--force-inherit-descriptions` to override *existing* descriptions if they are placeholders
- `--use-unrendered-descriptions` so that you can propagate Jinja-based docs (like `{{ doc(...) }}`)
- `--skip-add-columns`, `--skip-add-data-types`, `--skip-merge-meta`, `--skip-add-tags`, etc., if you want to limit changes
- `--synthesize` to autogenerate missing documentation with ChatGPT/OpenAI (see *Synthesis* below)

### Refactor

The **combination** of both `organize` and `document` in the correct order. Typically the recommended command to run:

- Creates or moves YAML files to match your `dbt_project.yml` rules
- Ensures columns are up to date with warehouse or catalog
- Inherits descriptions and metadata
- Reorders columns if desired

```bash
dbt-osmosis yaml refactor [--project-dir] [--profiles-dir] [--target] [--fqn ...] [--dry-run] [--check]
```

Options often used:

- `--auto-apply`
- `--force-inherit-descriptions`, `--use-unrendered-descriptions`
- `--skip-add-data-types`, `--skip-add-columns`, etc.
- `--synthesize` to autogenerate missing documentation with ChatGPT/OpenAI

### Commonly Used Flags in YAML Commands

- `--fqn=staging.some_subfolder` to limit to a particular subfolder or results of dbt ls
- `--check` to fail your CI if dbt-osmosis *would* make changes
- `--dry-run` to preview changes without writing them to disk
- `--catalog-path=target/catalog.json` to avoid live queries
- `--disable-introspection` to skip warehouse queries entirely
- `--auto-apply` to skip manual confirmation for file moves

## SQL

These commands let you compile or run SQL snippets (including Jinja) directly:

### Run

Runs a SQL statement or a dbt Jinja-based query.

```bash
dbt-osmosis sql run "select * from {{ ref('my_model') }} limit 50"
```

Returns results in tabular format to stdout. Use `--threads` to run multiple queries in parallel (though typically you’d run one statement at a time).

### Compile

Compiles a SQL statement (including Jinja) but doesn’t run it. Useful for quickly validating macros, refs, or Jinja logic:

```bash
dbt-osmosis sql compile "select * from {{ ref('my_model') }}"
```

Prints the compiled SQL to stdout.

## Workbench

Launches a [Streamlit](https://streamlit.io/) application that:

- Lets you explore and run queries against your dbt models in a REPL-like environment
- Provides side-by-side compiled SQL
- Offers real-time iteration on queries

```bash
dbt-osmosis workbench [--project-dir] [--profiles-dir] [--host] [--port]
```
