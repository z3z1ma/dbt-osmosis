---
id: ticket:c10pkg10
kind: ticket
status: ready
change_class: release-packaging
risk_class: medium
created_at: 2026-05-03T21:10:43Z
updated_at: 2026-05-03T21:10:43Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  initiative:
    - initiative:dbt-110-111-hardening
  evidence:
    - evidence:oracle-backlog-scan
depends_on: []
---

# Summary

Rationalize packaging extras, direct dependencies, dev dependency sources, `.python-version` handling, and install smoke tests so base and optional installs match what the code actually imports.

# Context

`pyproject.toml:25-34` includes `mysql-mimic` in base dependencies for an undocumented proxy. `src/dbt_osmosis/sql/proxy.py:23` imports `sqlglot` directly without a direct dependency. `pyproject.toml:37-45` puts `dbt-duckdb` inside the `workbench` extra, while `src/dbt_osmosis/workbench/requirements.txt:1` references stale `dbt-osmosis[workbench,duckdb]==1.1.5` even though no `duckdb` extra exists. Dev deps are split between `project.optional-dependencies.dev`, `[dependency-groups].dev`, pre-commit, and unpinned `uvx ruff` in `Taskfile.yml:20`, `25-26`. `.gitignore:83-85` ignores `.python-version` while `Taskfile.yml:4-7` reads it.

# Why Now

dbt 1.10/1.11 compatibility requires predictable installs and optional dependency gates. Leaky base dependencies and stale extras make build failures and user install problems harder to diagnose.

# Scope

- Decide base versus optional extras for workbench, proxy, OpenAI, Azure identity, DuckDB demo adapter, and direct imports.
- Remove or update `src/dbt_osmosis/workbench/requirements.txt` so it references real current extras.
- Add direct dependencies for direct imports or move those imports behind extras.
- Align dev dependency declarations and Ruff/pre-commit versions.
- Track `.python-version` or make Taskfile robust when it is absent.
- Add install smoke tests for base, workbench, openai, and any new proxy/azure/demo extras.

# Out Of Scope

- Removing the SQL proxy entirely; ticket:c10proxy25 owns the support decision.
- Changing CI resolution mechanics covered by ticket:c10lock07.

# Acceptance Criteria

- ACC-001: Base install includes only dependencies needed by documented base CLI behavior.
- ACC-002: Every direct import either has a direct package dependency or is guarded behind a documented optional extra.
- ACC-003: Workbench requirements file is removed or updated to valid current extras.
- ACC-004: Dev dependency versions are canonical across pyproject, dependency groups, Taskfile, and pre-commit.
- ACC-005: `task --list` and common Taskfile commands do not fail in a fresh clone because `.python-version` is missing.
- ACC-006: Install smoke covers base install and each supported optional extra.

# Coverage

Covers:

- ticket:c10pkg10#ACC-001
- ticket:c10pkg10#ACC-002
- ticket:c10pkg10#ACC-003
- ticket:c10pkg10#ACC-004
- ticket:c10pkg10#ACC-005
- ticket:c10pkg10#ACC-006
- initiative:dbt-110-111-hardening#OBJ-005
- initiative:dbt-110-111-hardening#OBJ-007
- initiative:dbt-110-111-hardening#OBJ-008

# Claim Matrix

| Claim | Evidence | Critique | Status |
| --- | --- | --- | --- |
| ticket:c10pkg10#ACC-003 | evidence:oracle-backlog-scan | None | open |
| ticket:c10pkg10#ACC-006 | None - install smoke not added yet | None | open |

# Execution Notes

Keep this ticket focused on packaging truth. If deciding proxy support requires product choice, stop and route to ticket:c10proxy25 or ask the user.

# Blockers

Human decision may be needed if removing base dependencies or adding extras changes public install expectations.

# Evidence

Existing evidence: evidence:oracle-backlog-scan. Missing evidence: install smoke outputs and dependency checks.

# Critique Disposition

Risk class: medium

Critique policy: recommended

Policy rationale: Packaging changes affect user installs but can be validated with smoke tests.

Required critique profiles: release-packaging, operator-clarity

Findings: None - no critique yet.

Disposition status: pending

Deferral / not-required rationale: None.

# Retrospective / Promotion Disposition

Disposition status: pending

Promoted: None - implementation not complete.

Deferred / not-required rationale: Consider docs/wiki updates if extras change.

# Wiki Disposition

N/A - no wiki promotion selected yet.

# Acceptance Decision

Accepted by: Not accepted yet.
Accepted at: N/A.
Basis: Pending install evidence.
Residual risks: Extras changes can surprise existing users if not documented.

# Dependencies

Coordinate with ticket:c10wb22, ticket:c10llm23, and ticket:c10proxy25.

# Journal

- 2026-05-03T21:10:43Z: Created from CI/build and CLI/workbench oracle findings.
