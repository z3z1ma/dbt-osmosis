---
id: llm-features-overview
title: "LLM Features Overview"
status: active
type: overview
section: overviews
topic-id: llm-features
topic-role: owner
publication-status: current-owner
publication-summary: "Current canonical overview for governed topic llm-features."
recommended-action: update-current-owner
current-owner: llm-features-overview
active-owners:
  - llm-features-overview
audience:
  - ai
  - human
source: workspace:dbt-osmosis
verified-at: 2026-03-28
verification-source: "Derived from optional dependency metadata, CLI reference, contributor docs, and source review of OpenAI gating plus the workbench AI assistant component."
successor: null
successor-title: null
predecessors: []
retirement-reason: null
topics: []
outputs:
  - https-github-com-vdfaller-dbt-osmosis-git:pyproject.toml
  - https-github-com-vdfaller-dbt-osmosis-git:docs/docs/reference/cli.md
  - https-github-com-vdfaller-dbt-osmosis-git:src/dbt_osmosis/core/osmosis.py
upstream-path: pyproject.toml
---

# LLM Features Overview

## Scope

LLM-assisted behavior in dbt-osmosis is optional. It includes synthesis-oriented YAML/documentation features, natural-language SQL/model generation commands, and connection validation via `test-llm`.

## Dependency model

These paths require the `[openai]` extra and provider-specific environment configuration. Missing dependencies are expected to fail clearly rather than silently pretending the feature is available.

## CLI surfaces

Published docs currently expose LLM-related behavior through:

- `dbt-osmosis test-llm`
- `dbt-osmosis generate model ...`
- `dbt-osmosis generate query ...`
- `dbt-osmosis nl query ...`
- experimental synthesis flags on YAML refactor flows

## Implementation boundaries

`core/osmosis.py` gates some LLM helpers behind optional dependency checks. The workbench AI assistant exists as a UI component but is still partially stubbed and should not be documented as a fully realized generation workflow.

## Maintenance note

Because setup details are provider-sensitive and evolve quickly, this overview should stay high-level while deeper provider or troubleshooting guides are added over time.
