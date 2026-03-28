---
id: repository-guidelines-overview
title: "Repository Guidelines Overview"
status: active
type: overview
section: overviews
topic-id: repository-guidelines
topic-role: owner
publication-status: current-owner
publication-summary: "Current canonical overview for governed topic repository-guidelines."
recommended-action: update-current-owner
current-owner: repository-guidelines-overview
active-owners:
  - repository-guidelines-overview
audience:
  - ai
  - human
source: workspace:dbt-osmosis
verified-at: 2026-03-28
verification-source: "Derived from the refreshed AGENTS.md, refreshed CLAUDE.md, refreshed README.md, and the current docs site structure under `docs/`."
successor: null
successor-title: null
predecessors: []
retirement-reason: null
topics: []
outputs:
  - https-github-com-vdfaller-dbt-osmosis-git:AGENTS.md
  - https-github-com-vdfaller-dbt-osmosis-git:CLAUDE.md
  - https-github-com-vdfaller-dbt-osmosis-git:README.md
upstream-path: AGENTS.md
---

# Repository Guidelines Overview

## Purpose

This topic governs the repository-level operating guidance for dbt-osmosis contributors. It explains which documentation surface to trust for which kind of question and anchors the companion `Repository Guidelines` guide stored in docs memory.

## Documentation responsibilities

- `README.md` is the user-facing landing page and quickstart.
- `docs/` is the canonical published product reference for CLI, configuration, migration, and workflows.
- `CLAUDE.md` is the dense contributor reference for commands, architecture details, and session norms.
- `AGENTS.md` is the concise repository playbook for code navigation, conventions, testing expectations, and common pitfalls.

## How to use this topic

Start here when you need to know where authoritative documentation lives. Then drop into the companion `Repository Guidelines` guide for the detailed repository-specific rules, or into the product and architecture topics for domain detail.

## Maintenance rule

When repository conventions, validation expectations, or high-value contributor guidance materially change, update both the raw repository guides and the governed docs topic so the operating layer stays truthful.
