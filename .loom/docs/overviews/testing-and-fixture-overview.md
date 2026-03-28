---
id: testing-and-fixture-overview
title: "Testing and Fixture Overview"
status: active
type: overview
section: overviews
topic-id: testing-and-fixtures
topic-role: owner
publication-status: current-owner
publication-summary: "Current canonical overview for governed topic testing-and-fixtures."
recommended-action: update-current-owner
current-owner: testing-and-fixture-overview
active-owners:
  - testing-and-fixture-overview
audience:
  - ai
  - human
source: workspace:dbt-osmosis
verified-at: 2026-03-28
verification-source: "Derived from pytest config, shared fixture modules, Taskfile test matrix, demo fixture assets, and integration test script review."
successor: null
successor-title: null
predecessors: []
retirement-reason: null
topics: []
outputs:
  - https-github-com-vdfaller-dbt-osmosis-git:tests/conftest.py
  - https-github-com-vdfaller-dbt-osmosis-git:tests/core/conftest.py
  - https-github-com-vdfaller-dbt-osmosis-git:demo_duckdb/integration_tests.sh
upstream-path: tests/conftest.py
---

# Testing and Fixture Overview

## Test shape

The repository uses pytest with a layered strategy: `tests/core/` mirrors implementation modules for focused coverage, while root-level YAML tests exercise real dbt/manifests against the demo fixture project.

## Shared fixture model

`demo_duckdb/` is the canonical default fixture project. Session-scoped fixtures copy it into temporary directories, run `dbt seed`, `dbt run`, and `dbt docs generate`, and then reuse the resulting template for function-scoped isolation. Core tests also bootstrap `demo_duckdb/target/manifest.json` with `dbt parse` if it is missing.

## Optional paths

- PostgreSQL-backed tests are available when `POSTGRES_URL` and related env vars are provided.
- Some LLM-related tests skip when optional dependencies such as `openai` or `azure.identity` are absent.

## Validation workflow

Use `task test` for the full supported matrix, but prefer targeted `uv run pytest ...` plus a manifest parse for iteration. CI runs a broader dbt-version matrix than the local Taskfile.

## Operational cautions

- Some tests mutate cwd and shared caches; they are not automatically parallel-safe.
- `demo_duckdb/integration_tests.sh` performs destructive git restore/clean operations and should only run in a disposable or clean working tree.
- Fixture teardown and database state management are part of the test contract; updates here should be made carefully and verified with real runs.
