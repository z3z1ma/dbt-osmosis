---
id: development-workflow-overview
title: "Development Workflow Overview"
status: active
type: overview
section: overviews
topic-id: development-workflow
topic-role: owner
publication-status: current-owner
publication-summary: "Current canonical overview for governed topic development-workflow."
recommended-action: update-current-owner
current-owner: development-workflow-overview
active-owners:
  - development-workflow-overview
audience:
  - ai
  - human
source: workspace:dbt-osmosis
verified-at: 2026-03-28
verification-source: "Derived from the refreshed CLAUDE.md and AGENTS.md plus current Taskfile, pyproject, and pre-commit configuration after normalizing test-matrix and CLI-surface references."
successor: null
successor-title: null
predecessors: []
retirement-reason: null
topics: []
outputs:
  - https-github-com-vdfaller-dbt-osmosis-git:CLAUDE.md
  - https-github-com-vdfaller-dbt-osmosis-git:AGENTS.md
  - https-github-com-vdfaller-dbt-osmosis-git:Taskfile.yml
upstream-path: CLAUDE.md
---

# Development Workflow Overview

## Purpose

This topic defines the default way contributors and AI agents should work in this repository: use `uv` for environment management, `task` for common workflows, Ruff for formatting/linting, and targeted pytest/dbt validation before claiming changes are complete.

## Local setup model

- Local Python defaults to `.python-version` (`3.12` today), while the project supports Python 3.10-3.13.
- `task dev` provisions the virtual environment and installs pre-commit hooks.
- `task` is the broad local workflow, but it is not a pure check command because it also defers `task dev`.
- Docs-site work uses a separate Node toolchain in `docs/`.

## Day-to-day commands

- `task format`, `task lint`, `task test`
- `uv run pytest ...` for focused test execution
- `uv run dbt parse --project-dir demo_duckdb --profiles-dir demo_duckdb -t test` before tests that need a manifest
- `pre-commit run --all-files` for hygiene gates
- `uv run dbt-osmosis test-llm` when validating optional provider-backed LLM paths

## Workflow constraints

- Prefer targeted validation over unnecessarily expensive full-matrix runs.
- Treat generated YAML changes and fixture resets carefully; some integration scripts use destructive git restore/clean commands.
- Optional paths such as workbench extras, OpenAI features, and PostgreSQL-backed tests should fail or skip clearly rather than pretending to be available.

## Session-level guidance

`CLAUDE.md` also records the repository's issue-tracking tool (`bd`/beads) and command conventions. `AGENTS.md` is the sharper operational playbook for architecture, testing, and code patterns. Together they form the contributor-facing operating layer beneath the published docs site.
