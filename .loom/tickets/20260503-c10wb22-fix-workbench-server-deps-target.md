---
id: ticket:c10wb22
kind: ticket
status: closed
change_class: code-behavior
risk_class: high
created_at: 2026-05-03T21:10:43Z
updated_at: 2026-05-04T14:43:11Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  initiative:
    - initiative:dbt-110-111-hardening
  evidence:
    - evidence:oracle-backlog-scan
    - evidence:c10wb22-workbench-target-validation
  packets:
    - packet:ralph-ticket-c10wb22-20260504T141941Z
  critique:
    - critique:c10wb22-workbench-target-review
depends_on: []
---

# Summary

Fix workbench server binding flags, optional dependency checks, and target switching so users see truthful Streamlit/dbt behavior.

# Context

`src/dbt_osmosis/cli/main.py:1339-1346` passes `--browser.serverAddress` and `--browser.serverPort`, which affect browser URL behavior rather than Streamlit server bind address/port. `workbench` invokes `streamlit` without checking the executable/extra is installed. `src/dbt_osmosis/workbench/app.py:139-148` changes `ctx.runtime_cfg.target_name` and reparses, but does not rebuild credentials or adapter for the selected target.

# Why Now

Workbench is user-facing and shares dbt context paths. Misleading target switching can run queries against the wrong credentials while the UI says otherwise.

# Scope

- Map `--host` and `--port` to Streamlit server bind settings.
- Add clear optional dependency errors for missing Streamlit/workbench packages.
- Rebuild `DbtProjectContext` on target change, including credentials and adapter, and close old connections.
- Preserve pass-through Streamlit args after `--`.
- Add CLI tests with mocked subprocess and workbench target-switch tests with mocked context creation.

# Out Of Scope

- Full workbench UI redesign.
- Adding browser-based E2E tests unless necessary for confidence.

# Acceptance Criteria

- ACC-001: `dbt-osmosis workbench --host 0.0.0.0 --port 8502` passes Streamlit `server.address=0.0.0.0` and `server.port=8502`.
- ACC-002: Missing `streamlit` produces a clear `ClickException` with the install command.
- ACC-003: `--options` and `--config` paths also fail clearly when Streamlit is unavailable.
- ACC-004: Target switching rebuilds dbt context/adapter/credentials for the selected target.
- ACC-005: Old adapter connections are closed or safely released when target changes.
- ACC-006: Failure to switch target leaves the previous target active and displays a clear error.

# Coverage

Covers:

- ticket:c10wb22#ACC-001
- ticket:c10wb22#ACC-002
- ticket:c10wb22#ACC-003
- ticket:c10wb22#ACC-004
- ticket:c10wb22#ACC-005
- ticket:c10wb22#ACC-006
- initiative:dbt-110-111-hardening#OBJ-006

# Claim Matrix

| Claim | Evidence | Critique | Status |
| --- | --- | --- | --- |
| ticket:c10wb22#ACC-001 | evidence:c10wb22-workbench-target-validation | critique:c10wb22-workbench-target-review | accepted |
| ticket:c10wb22#ACC-002 | evidence:c10wb22-workbench-target-validation | critique:c10wb22-workbench-target-review | accepted |
| ticket:c10wb22#ACC-003 | evidence:c10wb22-workbench-target-validation | critique:c10wb22-workbench-target-review | accepted |
| ticket:c10wb22#ACC-004 | evidence:c10wb22-workbench-target-validation | critique:c10wb22-workbench-target-review | accepted |
| ticket:c10wb22#ACC-005 | evidence:c10wb22-workbench-target-validation | critique:c10wb22-workbench-target-review | accepted |
| ticket:c10wb22#ACC-006 | evidence:c10wb22-workbench-target-validation | critique:c10wb22-workbench-target-review | accepted |

# Execution Notes

Keep workbench optional. Do not import Streamlit at CLI module import time. Mocking subprocess/context creation should cover most behavior without launching a browser.

# Blockers

None.

# Evidence

Evidence recorded:

- evidence:oracle-backlog-scan
- evidence:c10wb22-workbench-target-validation

# Critique Disposition

Risk class: high

Critique policy: mandatory

Policy rationale: Target switching can create a user-impacting correctness issue across warehouses/schemas.

Required critique profiles: code-change, operator-clarity, test-coverage

Findings: None in final critique. Earlier critique blockers for stale target widget state and optional dependency preflight were resolved before acceptance.

Disposition status: completed

Deferral / not-required rationale: N/A - mandatory critique completed with no blockers.

# Retrospective / Promotion Disposition

Disposition status: not_required

Promoted: None.

Deferred / not-required rationale: Existing docs already list `dbt-osmosis[workbench]`, `--host`, and `--port`; this ticket changed implementation behavior and test coverage rather than adding a new reusable workflow concept.

# Wiki Disposition

N/A - no wiki promotion selected; behavior is captured by ticket, evidence, critique, and tests.

# Acceptance Decision

Accepted by: OpenCode parent agent.
Accepted at: 2026-05-04T14:43:11Z.
Basis: implementation commit `ab138546acf0e32d7ad9339382d291bac8365fdc`, evidence:c10wb22-workbench-target-validation, and critique:c10wb22-workbench-target-review.
Residual risks: No live Streamlit server launch or real multi-target dbt adapter switch was exercised locally; Streamlit CLI flags can change by version; final repository/CI validation is deferred to initiative closure.

# Dependencies

Coordinate with ticket:c10pkg10 and ticket:c10sql21.

# Journal

- 2026-05-03T21:10:43Z: Created from CLI/SQL/workbench oracle findings.
- 2026-05-04T14:19:41Z: Activated ticket and compiled Ralph packet `packet:ralph-ticket-c10wb22-20260504T141941Z` for test-first workbench server flags, optional dependency checks, and target-switch context rebuild behavior with mandatory critique before acceptance.
- 2026-05-04T14:43:11Z: Ralph iteration consumed. Implementation commit `ab138546acf0e32d7ad9339382d291bac8365fdc` switched workbench launch to Streamlit server bind flags, added clear optional dependency errors, preserved pass-through args, rebuilt dbt context on target changes, closed old context connections, and rolled back failed switches. Local validation passed with `36 passed`; final mandatory critique found no blockers. Accepted and closed with live Streamlit/multi-target validation deferred to final initiative checks.
