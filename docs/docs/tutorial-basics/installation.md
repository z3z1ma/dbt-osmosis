---
sidebar_position: 1
---

# Installation

Pick the installation approach that matches your workflow. All examples assume you install a dbt adapter alongside dbt-osmosis.

## Install with `uv` (recommended)

```bash
uv tool install --with="dbt-<adapter>~=1.9.0" dbt-osmosis
```

This creates an isolated tool environment and exposes the `dbt-osmosis` command globally.

### Run with `uvx` (ephemeral)

```bash
uvx --with="dbt-<adapter>~=1.9.0" dbt-osmosis --help
```

Use `uvx` when you want a one-off run without installing the tool globally.

## Install with `pip`

```bash
pip install dbt-osmosis dbt-<adapter>
```

This installs into the active Python environment (virtualenv, venv, or system Python).

## Verify

```bash
dbt-osmosis --help
```

Replace `<adapter>` with your dbt adapter (for example: `snowflake`, `bigquery`, `postgres`, `redshift`, `duckdb`).
