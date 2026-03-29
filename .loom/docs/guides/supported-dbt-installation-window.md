---
id: supported-dbt-installation-window
title: "Supported dbt installation window"
status: active
type: guide
section: guides
topic-id: supported-dbt-installation-window
topic-role: companion
publication-status: governed-without-owner
publication-summary: "Governed doc for supported-dbt-installation-window without an active owner overview."
recommended-action: publish-topic-owner
current-owner: null
active-owners: []
audience:
  - ai
  - human
source: ticket:do-0004
verified-at: 2026-03-29T04:52:00Z
verification-source: "Reviewed after merge commit 012492f against pyproject, CI workflow, and installation docs updated by do-0004."
successor: null
successor-title: null
predecessors: []
retirement-reason: null
topics: []
outputs:
  - https-github-com-vdfaller-dbt-osmosis-git:docs/docs/intro.md
  - https-github-com-vdfaller-dbt-osmosis-git:docs/docs/tutorial-basics/installation.md
upstream-path: docs/docs/tutorial-basics/installation.md
---

# Supported dbt installation window

## Purpose
This guide records the truthful dbt runtime window that dbt-osmosis currently supports after the support-matrix correction in ticket do-0004.

## Supported versions
- Python: 3.10 through 3.13
- dbt Core: 1.8.x through 1.10.x
- dbt adapter packages used alongside dbt-osmosis should stay on the same supported minor line as the chosen dbt Core runtime.

## Why the support window is capped
The repository audit found that the installable dependency metadata allowed `dbt-core>=1.8`, which let fresh uv environments resolve dbt Core 1.11.x even though the repository's test matrix and adapter coverage only exercised 1.8.x through 1.10.x. At audit time, a verification attempt against `dbt-duckdb==1.11.2` failed because that package line was not available on PyPI, making the broader implied support contract false.

## Operational expectations
- Package metadata should constrain dbt Core and dbt-duckdb to the supported 1.8-1.10 window.
- Local `uv sync` developer setup should resolve a supported pair by default instead of drifting into 1.11.x.
- CI should exercise the same bounded matrix rather than a broader or different set of versions.
- Installation docs should tell users to install an adapter from the same supported minor line as their dbt Core runtime.

## Verification signals
The contract was re-verified with:
- `uv lock`
- `UV_PYTHON=3.13 uv -q sync --extra dev --extra openai`
- `uv -q pip install --reinstall "dbt-core~=1.10.0" "dbt-duckdb~=1.10.0"`
- `uv run dbt parse --project-dir demo_duckdb --profiles-dir demo_duckdb -t test`
- `uv run pytest tests/core/test_config.py tests/core/test_config_resolution.py`

Those commands succeeded on dbt Core 1.10.20 with dbt-duckdb 1.10.0.

## Non-goals
This guide does not promise support for dbt Core 1.11+ and does not document adapter-specific compatibility beyond the requirement that the adapter line matches a supported dbt Core minor version.
