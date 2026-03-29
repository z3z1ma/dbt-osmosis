---
sidebar_position: 1
---

# dbt-osmosis intro

dbt-osmosis automates dbt YAML management: file placement, column documentation inheritance, and optional LLM-assisted synthesis.

## What you'll do

- Install `dbt-osmosis` with your dbt adapter
- Configure YAML routing in `dbt_project.yml`
- Run a safe dry run and apply updates

## Prerequisites

- Python 3.10-3.13
- dbt Core 1.8-1.10 and a matching dbt adapter package
- A dbt project with models (and optionally sources)
- A clean git working tree (recommended)

## 1. Install

dbt-osmosis is exercised against dbt Core 1.8-1.10. Install a dbt adapter from the same supported minor line as your project.

```bash
uv tool install --with="dbt-<adapter>>=1.8,<1.11" dbt-osmosis
```

Or with pip:

```bash
pip install "dbt-osmosis" "dbt-<adapter>>=1.8,<1.11"
```

## 2. Configure YAML routing

Add a `+dbt-osmosis` rule for models (and seeds if you use them):

```yaml title="dbt_project.yml"
models:
  your_project_name:
    +dbt-osmosis: "_{model}.yml"
seeds:
  your_project_name:
    +dbt-osmosis: "_schema.yml"
```

## 3. Run a dry run

```bash
dbt-osmosis yaml refactor --dry-run --check
```

- `--dry-run` prevents writes.
- `--check` exits non-zero if changes would be made.

## 4. Apply changes

```bash
dbt-osmosis yaml refactor --auto-apply
```

Review the diff and commit the generated YAML changes with your models.
