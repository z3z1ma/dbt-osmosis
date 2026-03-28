---
id: workbench-overview
title: "Workbench Overview"
status: active
type: overview
section: overviews
topic-id: workbench
topic-role: owner
publication-status: current-owner
publication-summary: "Current canonical overview for governed topic workbench."
recommended-action: update-current-owner
current-owner: workbench-overview
active-owners:
  - workbench-overview
audience:
  - ai
  - human
source: workspace:dbt-osmosis
verified-at: 2026-03-28
verification-source: "Derived from README workbench sections, CLI reference, workbench app/component structure, and repository guidance about Streamlit state conventions."
successor: null
successor-title: null
predecessors: []
retirement-reason: null
topics: []
outputs:
  - https-github-com-vdfaller-dbt-osmosis-git:README.md
  - https-github-com-vdfaller-dbt-osmosis-git:docs/docs/reference/cli.md
  - https-github-com-vdfaller-dbt-osmosis-git:src/dbt_osmosis/workbench/app.py
upstream-path: README.md
---

# Workbench Overview

## Purpose

The workbench is an optional Streamlit-based interface for interactive dbt SQL development. It lets users compile dbt/Jinja SQL, run queries, inspect results, and profile datasets without leaving the dbt-osmosis runtime.

## Dependency model

The workbench is not part of the minimal install. It requires the `[workbench]` extra and its Streamlit/UI dependencies.

## Runtime shape

`dbt-osmosis workbench` launches the UI from the same project/bootstrap context used elsewhere in the repository. The workbench then owns its own Streamlit state tree and dashboard component layout.

## UI structure

`workbench/app.py` wires the main dashboard plus editor, renderer, preview, profiler, AI assistant, and feed components. Component state is expected to live under `st.session_state.app`, and dashboard items provide `initial_state()` definitions.

## Current limitations

- Repository tests only provide smoke-level coverage for the workbench command.
- The AI assistant component is still partially stubbed and should not be treated as a fully integrated LLM writeback path.
- Screenshots in the README are illustrative, not a source of implementation truth.

## Maintenance note

When workbench behavior changes materially, update this topic together with the CLI reference and any affected contributor guidance on session-state or component conventions.
