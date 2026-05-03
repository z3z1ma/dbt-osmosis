---
id: ticket:c10ci06
kind: ticket
status: complete_pending_acceptance
change_class: release-packaging
risk_class: high
created_at: 2026-05-03T21:10:43Z
updated_at: 2026-05-03T22:48:24Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  initiative:
    - initiative:dbt-110-111-hardening
  evidence:
    - evidence:oracle-backlog-scan
    - evidence:c10ci06-ci-gate-local-verification
    - evidence:c10ci06-ci-run-no-sync-fix
  critique:
    - critique:c10ci06-dbt-111-ci-gate
    - critique:c10ci06-no-sync-follow-up
  packets:
    - packet:ralph-ticket-c10ci06-20260503T222300Z
depends_on: []
---

# Summary

Promote dbt 1.11.x from a narrow canary job to an explicit, meaningful compatibility gate with installed-version assertions and adapter-backed smoke coverage.

# Context

Original gap: the main CI matrix covered Python 3.10-3.13 against dbt 1.8/1.9/1.10 only, and dbt 1.11 existed only as a narrow latest-core smoke. Current implementation adds dbt 1.11 to the explicit matrix with `dbt-duckdb~=1.10.1`, adds installed-version assertions, widens the latest-patch canary to dbt 1.10 and 1.11, mirrors that story in `Taskfile.yml`, and fixes the Rich 15 CLI import failure observed in GitHub Actions run `25291931743`.

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
| ticket:c10ci06#ACC-001 | evidence:c10ci06-ci-gate-local-verification; evidence:c10ci06-ci-run-no-sync-fix | critique:c10ci06-dbt-111-ci-gate | structurally supported; pending main CI run |
| ticket:c10ci06#ACC-002 | evidence:c10ci06-ci-gate-local-verification; evidence:c10ci06-ci-run-no-sync-fix | critique:c10ci06-dbt-111-ci-gate | structurally supported; pending main CI run |
| ticket:c10ci06#ACC-003 | evidence:c10ci06-ci-gate-local-verification; evidence:c10ci06-ci-run-no-sync-fix | critique:c10ci06-dbt-111-ci-gate | partially supported by isolated dbt 1.11 parse/import/CLI smoke; full matrix/pytest pending main CI |
| ticket:c10ci06#ACC-004 | evidence:c10ci06-ci-gate-local-verification; evidence:c10ci06-ci-run-no-sync-fix | critique:c10ci06-dbt-111-ci-gate | structurally supported; pending main CI run |
| ticket:c10ci06#ACC-005 | evidence:c10ci06-ci-gate-local-verification | critique:c10ci06-dbt-111-ci-gate | structurally supported; pending first scheduled/manual CI observation |

# Execution Notes

Avoid making the matrix huge without first isolating fixtures. If dbt-duckdb for 1.11 is not available for every Python version, make that limitation explicit in CI names and documentation rather than pretending full support.

# Blockers

No active blocker. Remaining acceptance gap: GitHub Actions evidence from the expanded matrix/canary after landing on `main`.

Published adapter boundary remains explicit: dbt 1.11 jobs use `dbt-duckdb~=1.10.1` until upstream adapter support changes.

# Evidence

Existing evidence:

- evidence:oracle-backlog-scan - backlog scan that identified the missing dbt 1.11 gate.
- evidence:c10ci06-ci-gate-local-verification - local structural checks and isolated dbt 1.11 parse/import/CLI smoke for the implementation diff.
- evidence:c10ci06-ci-run-no-sync-fix - branch CI failure showed `uv run` resynced matrix jobs back to locked `dbt-core 1.10.20`; workflow now sets `UV_NO_SYNC=1`, matching `Taskfile.yml`, and local overlay verification confirms `uv run` keeps the requested dbt 1.11 runtime.

Missing evidence:

- GitHub Actions run logs from `main` for the expanded matrix and latest-patch canary job.

# Critique Disposition

Risk class: high

Critique policy: mandatory

Policy rationale: CI support claims affect release quality and compatibility guarantees.

Required critique profiles: release-packaging, test-coverage, operator-clarity

Critique completed:

- critique:c10ci06-dbt-111-ci-gate - mandatory implementation critique returned `pass_with_findings` and the ticket resolved the stale-ledger finding.
- critique:c10ci06-no-sync-follow-up - follow-up critique for the `UV_NO_SYNC=1` workflow fix returned pass with no open findings.

Findings:

- critique:c10ci06-dbt-111-ci-gate#FIND-001 - resolved. The ticket now links the consumed packet, local evidence, critique record, updated claim matrix, and remaining post-push CI evidence gap.

Disposition status: completed

Deferral / not-required rationale: None. Mandatory critique is complete; post-push CI remains an acceptance evidence gap, not a critique blocker.

# Retrospective / Promotion Disposition

Disposition status: pending

Promoted: None yet - retrospective runs after implementation commit/push and CI observation.

Deferred / not-required rationale: Pending post-push evidence; consider wiki/update docs if support policy changes.

# Wiki Disposition

N/A - no wiki promotion selected yet.

# Acceptance Decision

Accepted by: Not accepted yet.
Accepted at: N/A.
Basis: Local evidence and mandatory critique support committing and pushing the implementation for CI trial on `main`; final acceptance is pending GitHub Actions evidence from `main` and retrospective disposition.
Residual risks: Full expanded matrix has not passed on `main` yet; matrix cost and dbt 1.11 adapter boundary remain active risks.

# Dependencies

Coordinate with ticket:c10lock07, ticket:c10fix11, and ticket:c10cfg12 for reliable matrix execution.

# Journal

- 2026-05-03T21:10:43Z: Created from CI/build and tests/fixtures oracle findings.
- 2026-05-03T22:23:00Z: Compiled Ralph packet `packet:ralph-ticket-c10ci06-20260503T222300Z` for CI matrix, Taskfile parity, and Rich import smoke fix.
- 2026-05-03T22:30:07Z: Reconciled Ralph output as consumed and recorded `evidence:c10ci06-ci-gate-local-verification` after focused local checks and isolated dbt 1.11 parse/import/CLI smoke passed.
- 2026-05-03T22:32:44Z: Mandatory critique `critique:c10ci06-dbt-111-ci-gate` returned `pass_with_findings`; ticket-owned disposition resolved the stale-ledger finding and left post-push CI evidence pending.
- 2026-05-03T22:45:13Z: Branch CI run `25292770214` showed matrix jobs were resynced by `uv run` back to locked `dbt-core 1.10.20`; added workflow `UV_NO_SYNC=1` env to preserve the installed dbt runtime and recorded `evidence:c10ci06-ci-run-no-sync-fix`.
- 2026-05-03T22:48:24Z: Follow-up critique `critique:c10ci06-no-sync-follow-up` reviewed the final workflow no-sync fix and found no open findings; main CI evidence remains required before closure.
