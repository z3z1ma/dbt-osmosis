---
sidebar_position: 2
---

# CLI Overview

This section describes the commands available in dbt-osmosis.

## YAML Management

These commands are used to manage the YAML files in your dbt project. Please read the [YAML configuration](/docs/tutorial-yaml/configuration) section to understand the minimum required configuration to use these commands.

### Document

This command will document your dbt project YAML files. Specifically it will:

- Reorder columns in your YAML files to match the order of the columns in your database
- Add columns to your YAML files that are present in your database
- Remove columns from your YAML files that are missing from your database
- Pass down column level documentation from upstream models to downstream models (if the downstream model does not have documentation for that column)

```bash
dbt-osmosis yaml document [--project-dir] [--profiles-dir] [--target]
```

### Organize

This command will organize your dbt project YAML files. Specifically it will:

- Bootstrap sources if they do not exist based on the `dbt-osmosis` **var** in your `dbt_project.yml` file.
- Migrate your YAML files based on the dbt-osmosis **config** (ideally) set in your `dbt_project.yml` file.
- Ensures that your project matches a declarative specification (i.e. your YAML files are in the correct location and have the correct name).

```bash
dbt-osmosis yaml organize [--project-dir] [--profiles-dir] [--target]
```

### Refactor

This command will refactor your dbt project YAML files. Specifically it will:

- Bootstrap sources if they do not exist based on the `dbt-osmosis` **var** in your `dbt_project.yml` file.
- Migrate your YAML files based on the dbt-osmosis **config** (ideally) set in your `dbt_project.yml` file.
- Ensures that your project matches a declarative specification (i.e. your YAML files are in the correct location and have the correct name).
- Reorder columns in your YAML files to match the order of the columns in your database
- Add columns to your YAML files that are present in your database
- Remove columns from your YAML files that are missing from your database
- Pass down column level documentation from upstream models to downstream models (if the downstream model does not have documentation for that column)

This command is a combination of the `document` and `organize` commands run in the correct order.

```bash
dbt-osmosis yaml refactor [--project-dir] [--profiles-dir] [--target]
```

## Server

dbt-osmosis ships with a server that can be used to drive 3rd party tools. This server is a zero dependency WSGI server powered by [bottle](https://bottlepy.org/docs/dev/). It provides high performance endpoints that leverage the plumbing in dbt-osmosis to provide a fast and reliable API. The server is "multi-tenant" in that it can serve multiple dbt projects at once. The server is not intended to be run on a public facing network. dbt-osmosis is essentially providing a thin CLI wrapper over dbt-core-interface where the server is actually implemented.

### Serve

This command will start the dbt-osmosis server. The server will be available at `http://localhost:8581` by default.

```bash
dbt-osmosis server serve [--host] [--port]
```

### Register Project

This command will register a dbt project with the dbt-osmosis server.

```bash
dbt-osmosis server register-project --project-dir /path/to/dbt/project
```

### Unregister Project

This command will unregister a dbt project with the dbt-osmosis server.

```bash
dbt-osmosis server unregister-project --project-dir /path/to/dbt/project
```

## SQL

These commands provide two unique and interesting ways to interact with dbt models. Both of these commands support stdin as an input source. This allows you to pipe a SQL query into the command or `cat` a dbt model into the command.

### Run

This command will run a dbt model and return the results as a JSON object. This command is useful for testing dbt models in a REPL environment.

```bash
dbt-osmosis sql run [--project-dir] [--profiles-dir] [--target] "select * from {{ ref('my_model') }}"
```

### Compile

This command will compile a dbt model and return the results as a JSON object. This command is useful for testing dbt models in a REPL environment.

```bash
dbt-osmosis sql compile [--project-dir] [--profiles-dir] [--target] "select * from {{ ref('my_model') }}"
```

## Workbench

This command starts a [streamlit](https://streamlit.io/) workbench. The workbench is a REPL environment that allows you to run dbt models, provides realtime side by side compilation, and lets you explore the results.

```bash
dbt-osmosis workbench [--project-dir] [--profiles-dir] [--target] [--host] [--port]
```

## Diff

This command will diff a dbt model across git commits. This command is useful for understanding how a model has changed over time. Currently this feature is under development. 🚧
