---
id: dbt-osmosis-overview
title: "dbt-osmosis Overview"
status: active
type: overview
section: overviews
topic-id: dbt-osmosis-overview
topic-role: owner
publication-status: current-owner
publication-summary: "Current canonical overview for governed topic dbt-osmosis-overview."
recommended-action: update-current-owner
current-owner: dbt-osmosis-overview
active-owners:
  - dbt-osmosis-overview
audience:
  - ai
  - human
source: workspace:dbt-osmosis
verified-at: 2026-03-28
verification-source: "Derived from the refreshed README.md, docs intro, pyproject metadata, and CLI help verified via `uv run dbt-osmosis --help`."
successor: null
successor-title: null
predecessors: []
retirement-reason: null
topics: []
outputs:
  - https-github-com-vdfaller-dbt-osmosis-git:README.md
  - https-github-com-vdfaller-dbt-osmosis-git:docs/docs/intro.md
upstream-path: README.md
---

# dbt-osmosis Overview

## Purpose

`dbt-osmosis` is a Python CLI and package that improves dbt development by automating schema YAML maintenance, inheriting column documentation across lineage, supporting ad-hoc dbt SQL compilation/execution, and offering an optional Streamlit workbench for interactive model development.

## Primary surfaces

- `dbt-osmosis yaml ...` manages schema files and inheritance workflows.
- `dbt-osmosis sql ...` compiles or runs dbt/Jinja SQL.
- `dbt-osmosis workbench` launches the interactive Streamlit UI.
- `dbt-osmosis generate`, `nl`, `test`, `lint`, `diff`, and `test-llm` extend the same runtime rather than introducing separate subsystems.

## Packaging and runtime

The project ships as a Python package with the console script `dbt-osmosis`. Core runtime dependencies live in `pyproject.toml`; optional extras enable the workbench (`[workbench]`) and OpenAI-backed features (`[openai]`). The default development workflow uses `uv` and `task`.

## Documentation map

- `README.md` is the landing page and quickstart.
- `docs/` is the canonical published docs site for CLI, configuration, migration, and workflow topics, including newer command families.
- `CLAUDE.md` and `AGENTS.md` are contributor-facing operational references.
- Governed docs memory captures high-value summaries and stable context for AI and human collaborators.

## Boundaries

This overview is product-facing. Deep implementation details such as transform composition, config resolution internals, schema caching, and fixture architecture belong in the architecture, configuration, and testing topic docs.
