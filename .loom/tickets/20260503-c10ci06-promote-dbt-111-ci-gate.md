---
id: ticket:c10ci06
kind: ticket
status: ready
change_class: release-packaging
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

Promote dbt 1.11.x from a narrow canary job to an explicit, meaningful compatibility gate with installed-version assertions and adapter-backed smoke coverage.

# Context

`.github/workflows/tests.yml:15-16` runs the main matrix for Python 3.10-3.13 against dbt 1.8/1.9/1.10 only. `.github/workflows/tests.yml:91-104` has a separate latest-core job on Python 3.13 with `dbt-core~=1.11.0` and `dbt-duckdb~=1.10.1`. `Taskfile.yml:53-87` mirrors 1.8-1.10 local testing, while `Taskfile.yml:89-110` has the narrow latest-core smoke.

# Why Now

The user explicitly asked to ensure compatibility with dbt 1.10.x and 1.11.x. Current CI cannot truthfully prove 1.11 support across the supported Python range or meaningful YAML/SQL workflows.

# Scope

- Decide and encode whether dbt 1.11.x is fully supported or canary-only.
- Add explicit dbt 1.11 matrix entries where adapter support permits.
- Assert installed dbt-core and adapter versions in every job.
- Run a meaningful parse plus CLI/YAML/SQL smoke under dbt 1.11.
- Add a scheduled latest-patch canary for dbt 1.10.x and 1.11.x.

# Out Of Scope

- Fixing every compatibility failure uncovered by the new matrix; those should become or link to separate implementation tickets.
- Changing package support metadata without corresponding evidence and release notes.

# Acceptance Criteria

- ACC-001: CI includes an explicit dbt 1.11.x gate for each supported Python version where a compatible DuckDB adapter exists, or the repository documents a narrower support contract.
- ACC-002: Every matrix job prints and verifies installed `dbt-core`, adapter, Python, and package versions.
- ACC-003: dbt 1.10.x and 1.11.x jobs run parse/import/CLI smoke at minimum, and full pytest where adapter support allows.
- ACC-004: `Taskfile.yml` local compatibility commands match the CI support story.
- ACC-005: A scheduled canary catches latest patch drift for dbt 1.10.x and 1.11.x.

# Coverage

Covers:

- ticket:c10ci06#ACC-001
- ticket:c10ci06#ACC-002
- ticket:c10ci06#ACC-003
- ticket:c10ci06#ACC-004
- ticket:c10ci06#ACC-005
- initiative:dbt-110-111-hardening#OBJ-001

# Claim Matrix

| Claim | Evidence | Critique | Status |
| --- | --- | --- | --- |
| ticket:c10ci06#ACC-001 | evidence:oracle-backlog-scan | None | open |
| ticket:c10ci06#ACC-003 | None - CI not changed/run yet | None | open |

# Execution Notes

Avoid making the matrix huge without first isolating fixtures. If dbt-duckdb for 1.11 is not available for every Python version, make that limitation explicit in CI names and documentation rather than pretending full support.

# Blockers

Potential blocker: published adapter compatibility for dbt 1.11.x. If unavailable, record the limitation and keep a canary until adapter support lands.

# Evidence

Existing evidence: evidence:oracle-backlog-scan. Missing evidence: CI run logs after matrix changes.

# Critique Disposition

Risk class: high

Critique policy: mandatory

Policy rationale: CI support claims affect release quality and compatibility guarantees.

Required critique profiles: release-packaging, test-coverage, operator-clarity

Findings: None - no critique yet.

Disposition status: pending

Deferral / not-required rationale: None.

# Retrospective / Promotion Disposition

Disposition status: pending

Promoted: None - implementation not complete.

Deferred / not-required rationale: Consider wiki/update docs if support policy changes.

# Wiki Disposition

N/A - no wiki promotion selected yet.

# Acceptance Decision

Accepted by: Not accepted yet.
Accepted at: N/A.
Basis: Pending CI evidence.
Residual risks: Matrix cost and adapter availability.

# Dependencies

Coordinate with ticket:c10lock07, ticket:c10fix11, and ticket:c10cfg12 for reliable matrix execution.

# Journal

- 2026-05-03T21:10:43Z: Created from CI/build and tests/fixtures oracle findings.
