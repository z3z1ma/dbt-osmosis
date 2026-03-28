---
id: core-architecture-overview
title: "Core Architecture Overview"
status: active
type: overview
section: overviews
topic-id: core-architecture
topic-role: owner
publication-status: current-owner
publication-summary: "Current canonical overview for governed topic core-architecture."
recommended-action: update-current-owner
current-owner: core-architecture-overview
active-owners:
  - core-architecture-overview
audience:
  - ai
  - human
source: workspace:dbt-osmosis
verified-at: 2026-03-28
verification-source: "Derived from core-source review of `src/dbt_osmosis/cli/main.py`, `src/dbt_osmosis/core/*`, workbench modules, core AGENTS guidance, and current CLI surface."
successor: null
successor-title: null
predecessors: []
retirement-reason: null
topics: []
outputs:
  - https-github-com-vdfaller-dbt-osmosis-git:src/dbt_osmosis/core/AGENTS.md
  - https-github-com-vdfaller-dbt-osmosis-git:CLAUDE.md
upstream-path: src/dbt_osmosis/core/AGENTS.md
---

# Core Architecture Overview

## Execution spine

`src/dbt_osmosis/cli/main.py` is the fan-out point for the product. Commands build a dbt project configuration, create a project context, and then route into shared YAML, SQL, generation, lint/test, or workbench flows.

## Core module boundaries

- `core/config.py` bootstraps the dbt project, adapters, manifests, and Fusion detection.
- `core/settings.py` owns `YamlRefactorContext` and runtime settings.
- `core/introspection.py` centralizes `SettingsResolver`, `PropertyAccessor`, catalog helpers, and schema caches.
- `core/path_management.py` decides where YAML should live and enforces project-root safety.
- `core/transforms.py` composes YAML mutation behavior through `TransformPipeline` and the `>>` operator.
- `core/inheritance.py` builds the lineage-based knowledge graph that powers documentation inheritance.

## YAML schema stack

The YAML layer is intentionally split:

- `core/schema/parser.py` filters dbt-osmosis-owned sections.
- `core/schema/reader.py` caches both filtered and original YAML content.
- `core/schema/writer.py` restores preserved sections and performs atomic replacement.
- `core/schema/validation.py` handles validation and safe structural fixes.

## Reused subsystems

- `core/sql_operations.py` is the shared SQL compile/execute path used by CLI code, the workbench, and SQL proxy surfaces.
- `workbench/app.py` reuses the same dbt context but owns its own Streamlit state and dashboard/component layout.
- `core/osmosis.py` and `core/__init__.py` are compatibility facades, not the preferred home for new implementation logic.

## Architectural cautions

Large files such as `introspection.py`, `transforms.py`, and `llm.py` carry significant behavior. New work should usually extend the existing subsystem boundary rather than bypassing it from CLI glue or re-export layers.
