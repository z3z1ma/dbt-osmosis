# dbt-osmosis

![PyPI](https://img.shields.io/pypi/v/dbt-osmosis)
[![Downloads](https://static.pepy.tech/badge/dbt-osmosis)](https://pepy.tech/project/dbt-osmosis)
![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://dbt-osmosis-playground.streamlit.app/)

`dbt-osmosis` is a Python CLI and package for dbt development workflows.

It centers on four primary surfaces:

- schema YAML management (`yaml organize`, `yaml document`, `yaml refactor`)
- column-level documentation inheritance across dbt lineage
- ad-hoc SQL compile/run helpers
- an optional Streamlit workbench for interactive dbt SQL development

The repository also ships additional command families for generation, natural-language helpers, schema diffing, SQL linting, and test suggestions.

The Docusaurus site is the canonical reference for the current CLI, configuration model, support matrix, and workflow guides:

- Docs site: https://z3z1ma.github.io/dbt-osmosis/
- CLI reference: https://z3z1ma.github.io/dbt-osmosis/docs/reference/cli
- Configuration guide: https://z3z1ma.github.io/dbt-osmosis/docs/tutorial-yaml/configuration
- Migration guide: https://z3z1ma.github.io/dbt-osmosis/docs/migrating

[![dbt-osmosis](/screenshots/docs_site.png)](https://z3z1ma.github.io/dbt-osmosis/)

## Supported runtime

`dbt-osmosis` currently targets:

- Python 3.10-3.13
- dbt Core 1.8+
- a dbt adapter version compatible with the dbt Core runtime in that environment

Repository-managed DuckDB fixture coverage is explicitly exercised through the published DuckDB-backed matrix in CI today (1.8-1.10) plus a latest-core compatibility job that runs basedpyright, `dbt parse`, and the full pytest suite under `dbt-core` 1.11 with the latest published `dbt-duckdb` adapter (currently 1.10.1). Package metadata and install paths are not capped at dbt Core 1.10.

Optional extras:

- `dbt-osmosis[workbench]` for the Streamlit workbench and related UI dependencies
- `dbt-osmosis[duckdb]` for the DuckDB adapter used by the demo project and local fixture workflows
- `dbt-osmosis[openai]` for LLM-assisted synthesis and natural-language generation features
- `dbt-osmosis[azure]` for Azure AD authentication used with Azure OpenAI
- `dbt-osmosis[proxy]` only installs dependencies for the experimental opt-in SQL proxy runtime; it does not expand the supported product surface, start a proxy server, configure authentication, TLS, or listen/bind settings, or make comment middleware durable. The proxy module entrypoint is a local-only experiment with `mysql-mimic` defaults, not a hardened user-facing server; do not expose it to untrusted networks. The proxy comment middleware is in-memory only, and `ticket:c10proxy25` owns proxy support semantics.

## Install

With `uv`:

```bash
uv tool install --with="dbt-<adapter>" dbt-osmosis
```

With `pip`:

```bash
pip install "dbt-osmosis" "dbt-<adapter>"
```

Replace `<adapter>` with your dbt adapter package, for example `duckdb`, `snowflake`, `bigquery`, `postgres`, or `redshift`.

## Quick start

1. Configure YAML routing in `dbt_project.yml`:

```yaml title="dbt_project.yml"
models:
  your_project_name:
    +dbt-osmosis: "_{model}.yml"
```

2. Optionally set per-folder behavior with `+dbt-osmosis-options` and a repo-level YAML formatter in `dbt-osmosis.yml`:

```yaml title="dbt-osmosis.yml"
formatter: "prettier --write"
```

3. Preview changes safely:

```bash
dbt-osmosis yaml refactor --dry-run --check
```

4. Apply the update once the diff looks right:

```bash
dbt-osmosis yaml refactor --auto-apply
```

## CLI surface

Top-level commands currently exposed by `dbt-osmosis --help`:

- `yaml` — manage schema YAML files and documentation inheritance
- `sql` — compile or run ad-hoc SQL in dbt context
- `workbench` — launch the Streamlit workbench
- `generate` — generate sources, staging models, models, and SQL
- `nl` — natural-language query/model helpers
- `test` — suggest dbt tests
- `test-llm` — validate LLM client configuration
- `diff` — report schema drift between YAML and the database
- `lint` — lint SQL strings, models, or a whole project

For command-by-command flags and examples, use the docs-site CLI reference rather than relying on this landing page.

## Developer tooling

Local development in this repository is built around `uv`, `task`, and Ruff.

Common workflows:

```bash
task format
task lint
task test
```

Notes:

- Ruff is the active formatter, linter, and import sorter.
- `task` is not just verification; the default task formats, lints, runs tests, and then ensures the dev environment is synced.
- Repository test fixtures are DuckDB-only today; contributor examples use `demo_duckdb`, and targeted core tests may need `uv run dbt parse --project-dir demo_duckdb --profiles-dir demo_duckdb -t test` to refresh `demo_duckdb/target/manifest.json`.
- Docs-site commands use the Node toolchain under `docs/`:

```bash
npm --prefix docs run start
npm --prefix docs run build
npm --prefix docs run serve
```

## Workbench

The optional workbench is a Streamlit app for interactive dbt SQL development.

Install the extra and launch it with:

```bash
pip install "dbt-osmosis[workbench]" "dbt-<adapter>"
dbt-osmosis workbench
```

The hosted demo is linked from the badge at the top of this README.

## Pre-commit hook

You can run `dbt-osmosis yaml refactor -C` as a pre-commit hook:

```yaml title=".pre-commit-config.yaml"
repos:
  - repo: https://github.com/z3z1ma/dbt-osmosis
    rev: v1.3.0
    hooks:
      - id: dbt-osmosis
        files: ^models/
        args: [--target=prod]
        additional_dependencies: [dbt-<adapter>]
```

That hook keeps schema YAML changes visible in the commit that introduced them.
