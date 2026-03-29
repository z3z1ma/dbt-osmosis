---
id: core-public-api-surface
title: "Core public API surface"
status: active
type: concept
section: concepts
topic-id: core-public-api-surface
topic-role: companion
publication-status: governed-without-owner
publication-summary: "Governed doc for core-public-api-surface without an active owner overview."
recommended-action: publish-topic-owner
current-owner: null
active-owners: []
audience:
  - ai
  - human
source: ticket:do-0002
verified-at: 2026-03-29T05:01:00Z
verification-source: "Reviewed after merge commit b7b0627 against AGENTS.md, core/__init__.py, core/osmosis.py, and related internal import callsites."
successor: null
successor-title: null
predecessors: []
retirement-reason: null
topics: []
outputs:
  - https-github-com-vdfaller-dbt-osmosis-git:AGENTS.md
upstream-path: AGENTS.md
---

# Core public API surface

## Purpose
This concept note records the intended import contract for dbt-osmosis core code after the facade cleanup in ticket do-0002.

## Public versus internal surfaces
- `dbt_osmosis.core.osmosis` remains the single compatibility/public facade for callers that need a stable re-export surface.
- `dbt_osmosis.core.__init__` is no longer a broad re-export layer. It exists only as the internal package marker and should not be treated as the public API.
- Internal code should import concrete submodules directly, for example `dbt_osmosis.core.config`, `dbt_osmosis.core.sql_operations`, or `dbt_osmosis.core.transforms`.

## Why the change matters
The previous design exported most of the core package through both `core/__init__.py` and `core/osmosis.py`. That created import cycles, widened optional-dependency boundaries until they were hard to reason about, and let static analysis report impossible types at CLI callsites. Removing the `core/__init__` star re-exports narrows the public surface to one truthful facade and makes internal dependencies explicit.

## Operational expectations
- New internal code should not add fresh re-exports to `core/__init__.py`.
- Optional feature gates belong in the public facade only when that symbol is actually part of the supported compatibility surface.
- Tests that are not explicitly exercising legacy imports should target concrete modules directly.

## Verification signals
The corrected facade contract was checked with:
- `uv run basedpyright --level error src/dbt_osmosis/core/osmosis.py src/dbt_osmosis/core/__init__.py`
- `uv run ruff check src/dbt_osmosis/core/osmosis.py`
- `uv run pytest tests/core/test_legacy.py`
- `uv run pytest tests/core/test_config_resolution.py -k 'osmosis or exported'`

## Non-goals
This note does not promise that every historical import path remains supported forever. It only defines the current compatibility/public facade and the correct internal import discipline.
