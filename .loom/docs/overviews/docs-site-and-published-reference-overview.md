---
id: docs-site-and-published-reference-overview
title: "Docs Site and Published Reference Overview"
status: active
type: overview
section: overviews
topic-id: docs-site-reference
topic-role: owner
publication-status: current-owner
publication-summary: "Current canonical overview for governed topic docs-site-reference."
recommended-action: update-current-owner
current-owner: docs-site-and-published-reference-overview
active-owners:
  - docs-site-and-published-reference-overview
audience:
  - ai
  - human
source: workspace:dbt-osmosis
verified-at: 2026-03-28
verification-source: "Derived from docs site configuration, package scripts, sidebar taxonomy, and published reference/workflow/config docs."
successor: null
successor-title: null
predecessors: []
retirement-reason: null
topics: []
outputs:
  - https-github-com-vdfaller-dbt-osmosis-git:docs/package.json
  - https-github-com-vdfaller-dbt-osmosis-git:docs/docusaurus.config.js
  - https-github-com-vdfaller-dbt-osmosis-git:docs/sidebars.js
upstream-path: docs/package.json
---

# Docs Site and Published Reference Overview

## Role

The `docs/` directory contains the published Docusaurus documentation site for dbt-osmosis. This is the canonical public reference for CLI behavior, configuration, migration guidance, and day-to-day YAML workflows.

## Structure

The site follows a Diataxis-style split visible in `docs/sidebars.js`:

- Tutorials
- How-to guides
- Reference
- Explanation

This taxonomy matters when deciding where new published material should land.

## Tooling

- `docs/package.json` defines the docs-site scripts (`start`, `build`, `serve`, `deploy`, etc.).
- `docs/docusaurus.config.js` is the source of truth for Docusaurus 3 configuration, GitHub Pages deployment, and navbar/footer wiring.
- `docs/README.md` is not authoritative for current tooling; it still reflects older Docusaurus boilerplate.

## Relationship to other docs

- Use the docs site for product behavior and user-facing workflows.
- Use repository guides such as `AGENTS.md` and `CLAUDE.md` for contributor and implementation guidance.
- Use governed docs memory to preserve high-value summaries, ownership, and cross-surface context.

## Maintenance note

When raw docs files, scripts, or site structure change, update the governed topic so the relationship between public docs and contributor docs stays clear.
