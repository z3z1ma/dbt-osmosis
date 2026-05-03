---
id: ticket:c10wb22
kind: ticket
status: ready
change_class: code-behavior
risk_class: high
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
| ticket:c10wb22#ACC-001 | evidence:oracle-backlog-scan | None | open |
| ticket:c10wb22#ACC-004 | evidence:oracle-backlog-scan | None | open |

# Execution Notes

Keep workbench optional. Do not import Streamlit at CLI module import time. Mocking subprocess/context creation should cover most behavior without launching a browser.

# Blockers

None.

# Evidence

Existing evidence: evidence:oracle-backlog-scan. Missing evidence: focused CLI/workbench tests.

# Critique Disposition

Risk class: high

Critique policy: mandatory

Policy rationale: Target switching can create a user-impacting correctness issue across warehouses/schemas.

Required critique profiles: code-change, operator-clarity, test-coverage

Findings: None - no critique yet.

Disposition status: pending

Deferral / not-required rationale: None.

# Retrospective / Promotion Disposition

Disposition status: pending

Promoted: None - implementation not complete.

Deferred / not-required rationale: Docs update likely needed after acceptance.

# Wiki Disposition

N/A - no wiki promotion selected yet.

# Acceptance Decision

Accepted by: Not accepted yet.
Accepted at: N/A.
Basis: Pending tests and critique.
Residual risks: Streamlit CLI flags can change by version.

# Dependencies

Coordinate with ticket:c10pkg10 and ticket:c10sql21.

# Journal

- 2026-05-03T21:10:43Z: Created from CLI/SQL/workbench oracle findings.
