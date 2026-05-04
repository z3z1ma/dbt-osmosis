---
id: ticket:c10pkg10
kind: ticket
status: closed
change_class: release-packaging
risk_class: medium
created_at: 2026-05-03T21:10:43Z
updated_at: 2026-05-04T04:58:20Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  initiative:
    - initiative:dbt-110-111-hardening
  evidence:
    - evidence:oracle-backlog-scan
    - evidence:c10pkg10-package-metadata-smoke
    - evidence:c10pkg10-main-ci-release-success
  critique:
    - critique:c10pkg10-release-packaging-review
  packets:
    - packet:ralph-ticket-c10pkg10-20260504T033410Z
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
| ticket:c10pkg10#ACC-001 | evidence:c10pkg10-package-metadata-smoke; evidence:c10pkg10-main-ci-release-success | critique:c10pkg10-release-packaging-review#FIND-006 resolved | accepted |
| ticket:c10pkg10#ACC-002 | evidence:c10pkg10-package-metadata-smoke; evidence:c10pkg10-main-ci-release-success | critique:c10pkg10-release-packaging-review#FIND-002 and critique:c10pkg10-release-packaging-review#FIND-006 resolved | accepted |
| ticket:c10pkg10#ACC-003 | evidence:c10pkg10-package-metadata-smoke; evidence:c10pkg10-main-ci-release-success | critique:c10pkg10-release-packaging-review#FIND-003 resolved | accepted |
| ticket:c10pkg10#ACC-004 | evidence:c10pkg10-package-metadata-smoke; evidence:c10pkg10-main-ci-release-success | critique:c10pkg10-release-packaging-review#FIND-001 and critique:c10pkg10-release-packaging-review#FIND-005 resolved | accepted |
| ticket:c10pkg10#ACC-005 | evidence:c10pkg10-package-metadata-smoke; evidence:c10pkg10-main-ci-release-success | critique:c10pkg10-release-packaging-review | accepted |
| ticket:c10pkg10#ACC-006 | evidence:c10pkg10-package-metadata-smoke; evidence:c10pkg10-main-ci-release-success | critique:c10pkg10-release-packaging-review#FIND-003 and critique:c10pkg10-release-packaging-review#FIND-004 resolved | accepted |

# Execution Notes

Ralph implementation completed the package metadata cleanup and stopped without changing proxy product semantics. Proxy support/removal remains with ticket:c10proxy25.

# Blockers

None.

# Evidence

Existing evidence:

- evidence:oracle-backlog-scan
- evidence:c10pkg10-package-metadata-smoke
- evidence:c10pkg10-main-ci-release-success

Evidence disposition: sufficient. Local red/green metadata and independent pip-smoke evidence was supplemented by main `Tests`, `lint`, and Release validation for commit `f4e475314bbb412fea927577f99f8e78a7258f80`.

# Critique Disposition

Risk class: medium

Critique policy: recommended

Policy rationale: Packaging changes affect user installs but can be validated with smoke tests.

Required critique profiles: release-packaging, operator-clarity

Findings:

- critique:c10pkg10-release-packaging-review#FIND-001 - `resolved` by adding Python 3.10 `tomli` fallback and canonical dev dependency coverage; supported by evidence:c10pkg10-package-metadata-smoke and evidence:c10pkg10-main-ci-release-success.
- critique:c10pkg10-release-packaging-review#FIND-002 - `resolved` by adding direct `PyYAML>=6.0` and base import smoke; supported by evidence:c10pkg10-package-metadata-smoke and evidence:c10pkg10-main-ci-release-success.
- critique:c10pkg10-release-packaging-review#FIND-003 - `resolved` by strengthening workbench import smoke and bounding `setuptools>=70,<81`; supported by evidence:c10pkg10-package-metadata-smoke and evidence:c10pkg10-main-ci-release-success.
- critique:c10pkg10-release-packaging-review#FIND-004 - `resolved` by making DuckDB smoke rely on `.[duckdb]` without an explicit adapter install argument; supported by evidence:c10pkg10-package-metadata-smoke and evidence:c10pkg10-main-ci-release-success.
- critique:c10pkg10-release-packaging-review#FIND-005 - `resolved` by pinning Taskfile pre-commit tool installation to `pre-commit>3.0.0,<5`; supported by evidence:c10pkg10-package-metadata-smoke and evidence:c10pkg10-main-ci-release-success.
- critique:c10pkg10-release-packaging-review#FIND-006 - `resolved` by documenting `dbt-osmosis[proxy]` as dependency-only and leaving support semantics to ticket:c10proxy25; supported by evidence:c10pkg10-package-metadata-smoke and evidence:c10pkg10-main-ci-release-success.

Disposition status: completed

Deferral / not-required rationale: None.

# Retrospective / Promotion Disposition

Disposition status: completed

Promoted:

- `wiki:ci-compatibility-matrix` now explains independent pip install smokes for base and supported optional extras and the masking failure mode for explicit packages outside the tested extra.
- `wiki:release-publishing-workflow` now records that Release validation is packaging evidence for built package metadata/wheel smoke after Tests pip smokes pass.

Deferred / not-required rationale: No additional research, spec, plan, initiative, constitution, or memory promotion needed. Proxy support semantics remain deferred to ticket:c10proxy25 by scope.

# Wiki Disposition

Completed. Updated `wiki:ci-compatibility-matrix` and `wiki:release-publishing-workflow` with the accepted CI/release packaging-smoke behavior. User-facing install docs were updated in source documentation as part of the implementation.

# Acceptance Decision

Accepted by: OpenCode
Accepted at: 2026-05-04T04:58:20Z
Basis: Implementation commit `f4e475314bbb412fea927577f99f8e78a7258f80`; local evidence:c10pkg10-package-metadata-smoke; final critique:c10pkg10-release-packaging-review; main CI/release evidence:c10pkg10-main-ci-release-success; retrospective promotions completed.
Residual risks: Workbench was import-smoked but not interactively launched; proxy remains dependency-only and experimental under ticket:c10proxy25; package index resolution can drift and should be rechecked when metadata or dependency constraints change.

# Dependencies

Coordinate with ticket:c10wb22, ticket:c10llm23, and ticket:c10proxy25.

# Journal

- 2026-05-03T21:10:43Z: Created from CI/build and CLI/workbench oracle findings.
- 2026-05-04T03:34:10Z: Activated ticket and compiled Ralph packet `packet:ralph-ticket-c10pkg10-20260504T033410Z` for package extras/dev tooling/install-smoke implementation.
- 2026-05-04T04:18:06Z: Reconciled Ralph output, recorded release-packaging critique `critique:c10pkg10-release-packaging-review`, recorded install evidence `evidence:c10pkg10-package-metadata-smoke`, and moved ticket to `complete_pending_acceptance` pending post-commit CI and retrospective disposition.
- 2026-05-04T04:58:20Z: Recorded main CI/release evidence `evidence:c10pkg10-main-ci-release-success`, promoted packaging-smoke lessons to `wiki:ci-compatibility-matrix` and `wiki:release-publishing-workflow`, accepted all scoped claims, and closed ticket.
