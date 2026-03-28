---
id: configuration-system-overview
title: "Configuration System Overview"
status: active
type: overview
section: overviews
topic-id: configuration-system
topic-role: owner
publication-status: current-owner
publication-summary: "Current canonical overview for governed topic configuration-system."
recommended-action: update-current-owner
current-owner: configuration-system-overview
active-owners:
  - configuration-system-overview
audience:
  - ai
  - human
source: workspace:dbt-osmosis
verified-at: 2026-03-28
verification-source: "Derived from the published configuration and settings docs, demo project config files, and the `SettingsResolver`/`PropertyAccessor` implementation guidance in CLAUDE.md and core source review."
successor: null
successor-title: null
predecessors: []
retirement-reason: null
topics: []
outputs:
  - https-github-com-vdfaller-dbt-osmosis-git:docs/docs/tutorial-yaml/configuration.md
  - https-github-com-vdfaller-dbt-osmosis-git:docs/docs/reference/settings.md
  - https-github-com-vdfaller-dbt-osmosis-git:demo_duckdb/dbt_project.yml
  - https-github-com-vdfaller-dbt-osmosis-git:demo_duckdb/dbt-osmosis.yml
upstream-path: docs/docs/tutorial-yaml/configuration.md
---

# Configuration System Overview

## Scope

dbt-osmosis configuration spans two related concerns:

- where YAML files should live
- how refactor/documentation behavior should be controlled

Both are surfaced in published docs and implemented in the core resolver/path-management stack.

## Routing model

YAML placement is driven primarily by `+dbt-osmosis` rules in `dbt_project.yml`, with fallback behavior available through project vars. Source management can also be configured through `vars.dbt-osmosis.sources`. The demo project provides the best concrete examples of these patterns.

## Behavior settings and precedence

Behavior flags can come from CLI defaults, folder-level config, node-level config, column metadata, project vars, and the supplementary `dbt-osmosis.yml` file. The implementation detail to preserve is that new work should flow through `SettingsResolver` rather than ad hoc config lookups or the legacy `_get_setting_for_node()` helper.

## Technical access layer

`PropertyAccessor` complements `SettingsResolver` by giving callers a consistent way to read rendered manifest values, raw YAML values, or auto-selected values when unrendered Jinja matters.

## Fusion compatibility

Fusion support is output-compatibility logic, not a separate runtime. dbt-osmosis still depends on dbt-core, but it can auto-detect or explicitly force Fusion-compatible YAML layout when the environment or manifest demands it.

## Maintenance note

Routing examples, precedence explanations, and option catalogs should stay consistent across the published docs, demo project config files, and governed docs memory because this area is both user-facing and implementation-sensitive.
