---
sidebar_position: 1
---

# Installation

Pick the installation approach that matches your workflow. All examples assume you install a dbt adapter alongside dbt-osmosis.

dbt-osmosis keeps its package support open for dbt Core 1.8+.

The repository's DuckDB-backed fixture matrix is still explicitly exercised through the currently published DuckDB-compatible lines in CI, and a separate latest-core compatibility job runs basedpyright, `dbt parse`, and the full pytest suite under `dbt-core` 1.11 with the latest published `dbt-duckdb` adapter (currently 1.10.1). The package metadata and install path are not capped at dbt Core 1.10. Install an adapter version that is compatible with the dbt Core runtime in that environment.

## Install with `uv` (recommended)

```bash
uv tool install --with="dbt-<adapter>" dbt-osmosis
```

This creates an isolated tool environment and exposes the `dbt-osmosis` command globally.

## Run with `uvx` (ephemeral)

```bash
uvx --with="dbt-<adapter>" dbt-osmosis --help
```

Use `uvx` when you want a one-off run without installing the tool globally.

## Install with `pip`

```bash
pip install "dbt-osmosis" "dbt-<adapter>"
```

This installs into the active Python environment (virtualenv, venv, or system Python).

## Optional extras

- `dbt-osmosis[workbench]` installs the Streamlit workbench UI dependencies.
- `dbt-osmosis[duckdb]` installs the DuckDB adapter used by the demo project and fixture workflows.
- `dbt-osmosis[openai]` installs the OpenAI SDK for synthesis, natural-language generation, `test-llm`, and AI test suggestions. If `LLM_PROVIDER` is unset, these commands default to the OpenAI provider.
- `dbt-osmosis[azure]` installs Azure AD authentication support for Azure OpenAI; use it together with `dbt-osmosis[openai]` for `LLM_PROVIDER=azure-openai-ad`.
- `dbt-osmosis[proxy]` only installs dependencies for the experimental opt-in SQL proxy runtime; it does not imply production support, start a proxy server, configure authentication, TLS, or listen/bind settings, or make comment middleware durable. The proxy module entrypoint is a local-only experiment with `mysql-mimic` defaults, not a hardened user-facing server; do not expose it to untrusted networks. The proxy comment middleware is in-memory only, and `ticket:c10proxy25` owns proxy support semantics.

## Verify

```bash
dbt-osmosis --help
```

Replace `<adapter>` with your dbt adapter (for example: `snowflake`, `bigquery`, `postgres`, `redshift`, `duckdb`).
