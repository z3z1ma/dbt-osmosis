---
sidebar_position: 1
---
# Installation

## Install with uv

```bash
uv tool install --with="dbt-<adapter>~=1.9.0" dbt-osmosis
```

This will install `dbt-osmosis` and its dependencies in a virtual environment, and make it available as a command-line tool via `dbt-osmosis`. You can also use `uvx` like in the intro to run it directly in a more ephemeral way.

## Install with pip

```bash
pip install dbt-osmosis dbt-<adapter>
```

(This installs `dbt-osmosis` into your current Python environment.)
