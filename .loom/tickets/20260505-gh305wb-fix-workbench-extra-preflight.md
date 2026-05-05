---
id: ticket:gh305wb
kind: ticket
status: ready
change_class: release-packaging
risk_class: medium
created_at: 2026-05-05T06:02:19Z
updated_at: 2026-05-05T06:02:19Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  initiative:
    - initiative:issue-pr-zero
external_refs:
  github_issue: https://github.com/z3z1ma/dbt-osmosis/issues/305
depends_on: []
---

# Summary

Make the `dbt-osmosis[workbench]` extra install and preflight cleanly when `ydata_profiling` requires IPython, and keep OpenAI optional rather than required for workbench startup.

# Context

Issue #305 reports `ModuleNotFoundError: No module named 'IPython'` while starting workbench from an install with `dbt-osmosis[workbench]`. Current `src/dbt_osmosis/workbench/app.py` imports `ydata_profiling` at module import time. The `workbench` extra includes `ydata-profiling` but not `IPython`, and CLI preflight only checks top-level module availability with `find_spec()`, which does not catch transitive import failures. The current AI assistant component no longer imports OpenAI, so OpenAI should not be required for workbench startup.

# Why Now

Workbench is an advertised optional surface. Optional extras should either install everything needed for startup or fail before launching Streamlit with a clear install hint.

# Scope

- Fix workbench optional dependency metadata and/or lazy import behavior so missing IPython does not surface as a Streamlit traceback.
- Improve preflight so transitive workbench import failures are caught with a clear `dbt-osmosis[workbench]` install hint.
- Preserve OpenAI as a separate optional extra unless real workbench AI functionality is deliberately added later.
- Add package/preflight tests for the workbench dependency contract.

# Out Of Scope

- Implementing a real workbench AI assistant.
- Bundling OpenAI into the workbench extra without a separate product decision.
- Broad Streamlit UI redesign.

# Acceptance Criteria

- ACC-001: A fresh install of `dbt-osmosis[workbench]` does not fail with missing `IPython` when starting workbench.
- ACC-002: If a workbench dependency is missing, the CLI fails before Streamlit launch with a clear optional-extra install hint.
- ACC-003: Workbench startup does not require OpenAI.
- ACC-004: Tests cover package metadata or preflight behavior for workbench dependencies.
- ACC-005: Existing workbench CLI smoke behavior remains intact.

# Coverage

Covers:

- ticket:gh305wb#ACC-001
- ticket:gh305wb#ACC-002
- ticket:gh305wb#ACC-003
- ticket:gh305wb#ACC-004
- ticket:gh305wb#ACC-005
- initiative:issue-pr-zero#OBJ-001
- initiative:issue-pr-zero#OBJ-002
- initiative:issue-pr-zero#OBJ-005

# Claim Matrix

| Claim | Evidence | Critique | Status |
| --- | --- | --- | --- |
| ticket:gh305wb#ACC-001 | None yet | None yet | open |
| ticket:gh305wb#ACC-002 | None yet | None yet | open |
| ticket:gh305wb#ACC-003 | None yet | None yet | open |
| ticket:gh305wb#ACC-004 | None yet | None yet | open |
| ticket:gh305wb#ACC-005 | None yet | None yet | open |

# Execution Notes

Coordinate with open Dependabot PRs that touch `pyproject.toml` or `uv.lock`, especially dependency group updates. The implementation should not reintroduce the stale expectation that workbench AI requires OpenAI.

# Blockers

None.

# Evidence

Expected evidence: package metadata/preflight tests, relevant CLI tests, `uv lock --check` or updated lock validation if dependency metadata changes, and remote CI before closure.

# Critique Disposition

Risk class: medium

Critique policy: recommended

Policy rationale: Packaging/extras changes can affect user installs and dependency resolution.

Required critique profiles:

- packaging-extra
- workbench-startup

Findings:

None - no critique yet.

Disposition status: pending

Deferral / not-required rationale: N/A.

# Retrospective / Promotion Disposition

Disposition status: pending

Promoted:

None yet.

Deferred / not-required rationale: N/A.

# Wiki Disposition

N/A - no wiki promotion selected yet.

# Acceptance Decision

Accepted by: Pending implementation and evidence.
Accepted at: N/A.
Basis: N/A.
Residual risks: N/A.

# Dependencies

Coordinate with open dependency PRs that modify `pyproject.toml` or `uv.lock`.

# Journal

- 2026-05-05T06:02:19Z: Created from GitHub issue #305 and Oracle triage as a validated workbench optional dependency/preflight bug.
