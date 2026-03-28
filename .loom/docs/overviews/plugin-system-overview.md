---
id: plugin-system-overview
title: "Plugin System Overview"
status: active
type: overview
section: overviews
topic-id: plugin-system
topic-role: owner
publication-status: current-owner
publication-summary: "Current canonical overview for governed topic plugin-system."
recommended-action: update-current-owner
current-owner: plugin-system-overview
active-owners:
  - plugin-system-overview
audience:
  - ai
  - human
source: workspace:dbt-osmosis
verified-at: 2026-03-28
verification-source: "Derived from core plugin-system guidance, CLAUDE.md architecture notes, and current source review of `src/dbt_osmosis/core/plugins.py`."
successor: null
successor-title: null
predecessors: []
retirement-reason: null
topics: []
outputs:
  - https-github-com-vdfaller-dbt-osmosis-git:src/dbt_osmosis/core/plugins.py
  - https-github-com-vdfaller-dbt-osmosis-git:src/dbt_osmosis/core/AGENTS.md
upstream-path: src/dbt_osmosis/core/plugins.py
---

# Plugin System Overview

## Purpose

The plugin system lets dbt-osmosis extend column-name matching behavior without hard-coding every naming convention into the inheritance engine.

## Implementation model

The system uses Pluggy. `src/dbt_osmosis/core/plugins.py` creates the plugin manager, registers built-in implementations, and loads external entry points from the `dbt-osmosis` group.

## Current scope

The primary hook surface is candidate generation for matching columns across lineage when names differ by case, prefixes, or similar patterns. Built-in plugins cover fuzzy case and prefix matching.

## Extension boundary

Plugins should extend matching behavior through the plugin manager rather than by forking inheritance logic or scattering special cases into unrelated modules. Because the plugin system affects documentation inheritance results, changes here should be tested against real fixture scenarios.

## Maintenance note

This topic is intentionally narrow: it documents the extension surface and its architectural role, not every individual matching heuristic.
