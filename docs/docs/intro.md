---
sidebar_position: 1
---

# dbt-osmosis intro

`dbt-osmosis` is a dbt developer workflow tool. Its primary job is schema YAML management, but the package also ships SQL helpers, a Streamlit workbench, schema diffing, SQL linting, test suggestions, and optional LLM-assisted generation paths.

Use this page as the shortest truthful path from install to a safe first refactor. For the full command surface and detailed configuration behavior, follow the reference links at the end.

## What you'll do

- install `dbt-osmosis` with a matching dbt adapter
- configure YAML routing and folder-level options
- run a safe dry run before applying changes

## Prerequisites

- Python 3.10-3.13
- dbt Core 1.8+ package resolution and a dbt adapter version compatible with that runtime
- a dbt project with models (and optionally sources)
- a clean git working tree for reviewable YAML diffs

## 1. Install

`dbt-osmosis` keeps its package support open for dbt Core 1.8+.

Audited blocking support covers dbt Core 1.8.x through 1.11.x in CI. The package metadata intentionally remains `dbt-core>=1.8` without an upper bound so installers can resolve newer dbt releases. Future dbt Core minors are canary-only until explicitly audited; scheduled/manual canary CI uses unpinned latest `dbt-core` and `dbt-duckdb` to make upstream breakage visible without redefining audited support. Install a dbt adapter version that is compatible with the dbt Core runtime in your environment; adapter compatibility is owned by the adapter and dbt Core pairing, not by dbt-osmosis extras.

```bash
uv tool install --with="dbt-<adapter>" dbt-osmosis
```

Or with pip:

```bash
pip install "dbt-osmosis" "dbt-<adapter>"
```

Optional extras:

- `dbt-osmosis[workbench]` enables the Streamlit workbench
- `dbt-osmosis[duckdb]` installs the DuckDB adapter used by the demo project and fixture workflows
- `dbt-osmosis[openai]` enables synthesis and natural-language generation features
- `dbt-osmosis[azure]` installs Azure AD authentication support for Azure OpenAI
- `dbt-osmosis[proxy]` only installs dependencies for the experimental opt-in SQL proxy runtime; it does not start a proxy server, configure authentication, TLS, or listen/bind settings, or make comment middleware durable. The proxy module entrypoint is a local-only experiment with `mysql-mimic` defaults, not a hardened user-facing server; do not expose it to untrusted networks. The proxy comment middleware is in-memory only, and `ticket:c10proxy25` owns proxy support semantics.

## 2. Configure YAML routing

Add a `+dbt-osmosis` rule under each folder you want dbt-osmosis to manage:

```yaml title="dbt_project.yml"
models:
  your_project_name:
    +dbt-osmosis: "_{model}.yml"
seeds:
  your_project_name:
    +dbt-osmosis: "_schema.yml"
```

Add folder-level behavior with `+dbt-osmosis-options` when you need to tune inheritance or output:

```yaml title="dbt_project.yml"
models:
  your_project_name:
    staging:
      +dbt-osmosis: "{parent}.yml"
      +dbt-osmosis-options:
        skip-add-columns: true
        sort-by: alphabetical
```

If you want dbt-osmosis to run an external YAML formatter after writes, set it in the supplementary repo-level file:

```yaml title="dbt-osmosis.yml"
formatter: "prettier --write"
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

Review the generated diff before committing.

## 5. Explore the rest of the CLI

The current top-level command groups are:

- `yaml`
- `sql`
- `workbench`
- `generate`
- `nl`
- `test`
- `test-llm`
- `diff`
- `lint`

## Next reads

- [CLI reference](./reference/cli)
- [YAML configuration and options](./tutorial-yaml/configuration)
- [Settings reference](./reference/settings)
- [Migration guide](./migrating)
