---
id: ticket:gh305wb
kind: ticket
status: closed
change_class: release-packaging
risk_class: medium
created_at: 2026-05-05T06:02:19Z
updated_at: 2026-05-05T08:25:59Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  initiative:
    - initiative:issue-pr-zero
  evidence:
    - evidence:gh305wb-workbench-extra-preflight-validation
  critique:
    - critique:gh305wb-workbench-extra-preflight-review
  packets:
    - packet:ralph-ticket-gh305wb-20260505T081714Z
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
| ticket:gh305wb#ACC-001 | evidence:gh305wb-workbench-extra-preflight-validation | critique:gh305wb-workbench-extra-preflight-review | accepted |
| ticket:gh305wb#ACC-002 | evidence:gh305wb-workbench-extra-preflight-validation | critique:gh305wb-workbench-extra-preflight-review | accepted |
| ticket:gh305wb#ACC-003 | evidence:gh305wb-workbench-extra-preflight-validation | critique:gh305wb-workbench-extra-preflight-review | accepted |
| ticket:gh305wb#ACC-004 | evidence:gh305wb-workbench-extra-preflight-validation | critique:gh305wb-workbench-extra-preflight-review | accepted |
| ticket:gh305wb#ACC-005 | evidence:gh305wb-workbench-extra-preflight-validation | critique:gh305wb-workbench-extra-preflight-review | accepted |

# Execution Notes

Ralph added `ipython>=8,<9` to the workbench extra, updated `uv.lock`, and changed workbench preflight to import required modules so transitive `ImportError`s are caught before launching Streamlit. OpenAI remains outside the workbench extra and preflight module list.

# Blockers

None.

# Evidence

Evidence status: local red/green Ralph evidence, parent focused CLI/package pytest, Ruff, format check, `uv lock --check`, and whitespace check support ACC-001 through ACC-005. Remote CI will be checked at the issue-backlog initiative gate per operator direction.

# Critique Disposition

Risk class: medium

Critique policy: recommended

Policy rationale: Packaging/extras changes can affect user installs and dependency resolution.

Required critique profiles:

- packaging-extra
- workbench-startup

Findings:

None - critique:gh305wb-workbench-extra-preflight-review returned `pass` with no findings.

Disposition status: completed

Deferral / not-required rationale: N/A.

# Retrospective / Promotion Disposition

Disposition status: completed

Promoted:

None - retrospective found no durable explanation needing wiki/research/spec promotion beyond this ticket, evidence, and critique.

Deferred / not-required rationale: Behavior is a narrow optional-extra/preflight fix with focused package and CLI tests.

# Wiki Disposition

N/A - no wiki promotion selected yet.

# Acceptance Decision

Accepted by: OpenCode parent agent.
Accepted at: 2026-05-05T08:25:59Z.
Basis: Local red/green implementation evidence, focused validation, lock validation, and Oracle critique support ACC-001 through ACC-005. The ticket is ready for issue closure with remote CI deferred to the issue-backlog initiative gate per operator direction.
Residual risks: Remote CI not yet checked; no isolated fresh `dbt-osmosis[workbench]` install/start smoke was run; preflight catches import failures but not arbitrary workbench runtime failures.

# Dependencies

Coordinate with open dependency PRs that modify `pyproject.toml` or `uv.lock`.

# Journal

- 2026-05-05T06:02:19Z: Created from GitHub issue #305 and Oracle triage as a validated workbench optional dependency/preflight bug.
- 2026-05-05T08:17:14Z: Compiled Ralph packet packet:ralph-ticket-gh305wb-20260505T081714Z and moved ticket to active for the workbench extra/preflight implementation iteration.
- 2026-05-05T08:25:59Z: Ralph implemented the workbench extra/preflight fix. Parent validation passed and Oracle critique accepted with no findings. Retrospective completed with no promotion needed beyond ticket/evidence/critique records. Accepted and closed locally for issue packaging.
