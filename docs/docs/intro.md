---
sidebar_position: 1
---

# dbt-osmosis Intro

Let's discover **dbt-osmosis** in less than 5 minutes.

## Getting Started

Get started by **running dbt-osmosis**.

### What you'll need

- [Python](https://www.python.org/downloads/) (3.9+)
- [dbt](https://docs.getdbt.com/docs/core/installation) (1.8.0+)
- or [uv](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer)
- An existing dbt project (or you can play with it using [jaffle shop](https://github.com/dbt-labs/jaffle_shop_duckdb))

## Configure dbt-osmosis

Add the following to your `dbt_project.yml` file. This example configuration tells dbt-osmosis that for every model in your project, there should exist a YAML file in the same directory with the same name as the model prefixed with an underscore. For example, if you have a model named `my_model` then there should exist a YAML file named `_my_model.yml` in the same directory as the model. The configuration is extremely flexible and can be used to declaratively organize your YAML files in any way you want as you will see later.

```yaml title="dbt_project.yml"
models:
  your_project_name:
    +dbt-osmosis: "_{model}.yml"
seeds:
  your_project_name:
    +dbt-osmosis: "_schema.yml"
```

## Run dbt-osmosis

If using uv(x):

```bash
uvx --with='dbt-<adapter>==1.9.0' dbt-osmosis yaml refactor
```

Or, if installed in your Python environment:

```bash
dbt-osmosis yaml refactor
```

Run this command from the root of your dbt project. Ensure your git repository is clean before running. Replace `<adapter>` with the name of your dbt adapter (e.g. `snowflake`, `bigquery`, `redshift`, `postgres`, `athena`, `spark`, `trino`, `sqlite`, `duckdb`, `oracle`, `sqlserver`).

Watch the magic unfold. ✨
