---
sidebar_position: 1
---

# dbt-osmosis Intro

Let's discover **dbt-osmosis in less than 5 minutes**.

## Getting Started

Get started by **running dbt-osmosis**.

### What you'll need

- [Python](https://www.python.org/downloads/) (3.8+)
- [dbt](https://docs.getdbt.com/docs/core/installation) (1.0.0+)
- [pipx](https://pypa.github.io/pipx/installation/)
- An existing dbt project (or you can play with it using [jaffle shop](https://github.com/dbt-labs/jaffle_shop_duckdb))

## Configure dbt-osmosis

Add the following to your `dbt_project.yml` file. This example configuration tells dbt-osmosis that for every model in your project, there should exist a YAML file in the same directory with the same name as the model prefixed with an underscore. For example, if you have a model named `my_model` then there should exist a YAML file named `_my_model.yml` in the same directory as the model. The configuration is extremely flexible and can be used to declaratively organize your YAML files in any way you want as you will see later.

```yaml title="dbt_project.yml"
models:
  your_project_name:
    +dbt-osmosis: "_{model}.yml"
```

## Run dbt-osmosis

Run dbt-osmosis with the following command to automatically perform a refactoring of your dbt project YAML files. Run this command from the root of your dbt project. Ensure your git repository is clean before running this command. Replace `<adapter>` with the name of your dbt adapter (e.g. `snowflake`, `bigquery`, `redshift`, `postgres`, `athena`, `spark`, `trino`, `sqlite`, `duckdb`, `oracle`, `sqlserver`).

```bash
pipx run --pip-args="dbt-<adapter>" dbt-osmosis yaml refactor
```

Watch the magic unfold. ✨
