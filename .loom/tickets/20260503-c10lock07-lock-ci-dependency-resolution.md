---
id: ticket:c10lock07
kind: ticket
status: closed
change_class: release-packaging
risk_class: high
created_at: 2026-05-03T21:10:43Z
updated_at: 2026-05-04T01:10:42Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  initiative:
    - initiative:dbt-110-111-hardening
  evidence:
    - evidence:oracle-backlog-scan
    - evidence:c10lock07-local-dependency-resolution-verification
    - evidence:c10lock07-adapter-bound-verification
    - evidence:c10lock07-uv-01012-verification
    - evidence:c10lock07-main-ci-success
  wiki:
    - wiki:ci-compatibility-matrix
  packets:
    - packet:ralph-ticket-c10lock07-20260503T234103Z
  critique:
    - critique:c10lock07-dependency-resolution
    - critique:c10lock07-integration-path-follow-up
    - critique:c10lock07-adapter-constraint-follow-up
    - critique:c10lock07-uv-01012-follow-up
depends_on: []
---

# Summary

Make CI dependency resolution reproducible and clean per dbt/Python matrix entry instead of running unlocked `uv sync` and overlaying dbt packages into a pre-synced environment.

# Context

`.github/workflows/tests.yml:43-48` runs `uv sync --extra dev --extra openai` and then `uv pip install "dbt-core~=..." "dbt-duckdb~=..."`. `Taskfile.yml:67-68` and `81-82` do similar environment mutation. `pyproject.toml:63-64` has a uv-only protobuf override that may hide pip behavior.

# Why Now

Compatibility claims are only as strong as the environments that test them. If CI silently updates or mutates dependency state, dbt 1.10/1.11 failures can be hidden or become unreproducible.

# Scope

- Add lock freshness checks such as `uv lock --check`, `uv sync --locked`, or an equivalent frozen mode.
- Ensure each matrix entry starts from a clean environment or deterministic resolution.
- Avoid relying on transitive state from an initial lock when installing matrix dbt versions.
- Add `pip check` and version reporting in matrix jobs.
- Add at least one pip-based install smoke to catch uv-only override gaps.

# Out Of Scope

- Reworking package metadata beyond what is needed for reproducible installs; ticket:c10pkg10 owns broader extras/package cleanup.
- Deciding the full dbt support matrix; ticket:c10ci06 owns support coverage.

# Acceptance Criteria

- ACC-001: CI fails when `pyproject.toml` and `uv.lock` are inconsistent.
- ACC-002: Matrix dbt installs do not inherit incompatible locked dbt transitive state from a different dbt version.
- ACC-003: Every matrix entry runs `pip check` or equivalent dependency consistency validation.
- ACC-004: CI logs exact installed versions for Python, dbt-core, dbt adapter, uv, and dbt-osmosis.
- ACC-005: At least one clean pip install smoke covers published package metadata outside uv resolution.

# Coverage

Covers:

- ticket:c10lock07#ACC-001
- ticket:c10lock07#ACC-002
- ticket:c10lock07#ACC-003
- ticket:c10lock07#ACC-004
- ticket:c10lock07#ACC-005
- initiative:dbt-110-111-hardening#OBJ-001
- initiative:dbt-110-111-hardening#OBJ-005

# Claim Matrix

| Claim | Evidence | Critique | Status |
| --- | --- | --- | --- |
| ticket:c10lock07#ACC-001 | evidence:oracle-backlog-scan; evidence:c10lock07-local-dependency-resolution-verification; evidence:c10lock07-adapter-bound-verification; evidence:c10lock07-uv-01012-verification; evidence:c10lock07-main-ci-success | critique:c10lock07-dependency-resolution; critique:c10lock07-integration-path-follow-up; critique:c10lock07-adapter-constraint-follow-up; critique:c10lock07-uv-01012-follow-up | accepted - main CI lockfile job passed |
| ticket:c10lock07#ACC-002 | evidence:c10lock07-local-dependency-resolution-verification; evidence:c10lock07-adapter-bound-verification; evidence:c10lock07-uv-01012-verification; evidence:c10lock07-main-ci-success | critique:c10lock07-dependency-resolution; critique:c10lock07-integration-path-follow-up; critique:c10lock07-adapter-constraint-follow-up; critique:c10lock07-uv-01012-follow-up | accepted - clean matrix rows passed |
| ticket:c10lock07#ACC-003 | evidence:c10lock07-local-dependency-resolution-verification; evidence:c10lock07-adapter-bound-verification; evidence:c10lock07-uv-01012-verification; evidence:c10lock07-main-ci-success | critique:c10lock07-dependency-resolution; critique:c10lock07-integration-path-follow-up; critique:c10lock07-adapter-constraint-follow-up; critique:c10lock07-uv-01012-follow-up | accepted - dependency checks passed |
| ticket:c10lock07#ACC-004 | evidence:c10lock07-local-dependency-resolution-verification; evidence:c10lock07-adapter-bound-verification; evidence:c10lock07-uv-01012-verification; evidence:c10lock07-main-ci-success | critique:c10lock07-dependency-resolution; critique:c10lock07-integration-path-follow-up; critique:c10lock07-adapter-constraint-follow-up; critique:c10lock07-uv-01012-follow-up | accepted - CI version logs observed |
| ticket:c10lock07#ACC-005 | evidence:c10lock07-local-dependency-resolution-verification; evidence:c10lock07-adapter-bound-verification; evidence:c10lock07-main-ci-success | critique:c10lock07-integration-path-follow-up; critique:c10lock07-adapter-constraint-follow-up | accepted - pip smoke passed |

# Execution Notes

Be careful with `UV_FROZEN` and `UV_NO_SYNC` in `Taskfile.yml`; verify the local workflow still works from a fresh clone. Include a deliberate stale-lock test strategy in PR description or evidence when implementing.

# Blockers

None.

# Evidence

Existing evidence:

- evidence:oracle-backlog-scan - original backlog finding for unreproducible dependency resolution.
- evidence:c10lock07-local-dependency-resolution-verification - local lock, workflow, Taskfile, pip smoke, and child uv matrix smoke evidence after implementation.
- evidence:c10lock07-adapter-bound-verification - local pinned-uv dbt 1.8 follow-up evidence after main CI exposed old adapter selection.
- evidence:c10lock07-uv-01012-verification - local exact uv 0.10.12 follow-up evidence after operator requested a newer CI uv pin.
- evidence:c10lock07-main-ci-success - final `main` GitHub Actions evidence at commit `19ef3a4bd3d6e6e3f5437e634a51fc11edaa23ba`.

Missing evidence:

None - final `main` CI evidence is recorded.

# Critique Disposition

Risk class: high

Critique policy: mandatory

Policy rationale: Dependency resolution determines compatibility evidence trustworthiness.

Required critique profiles: release-packaging, test-coverage

Critique completed:

- critique:c10lock07-dependency-resolution - mandatory implementation critique returned `changes_required` with one high-severity finding.
- critique:c10lock07-integration-path-follow-up - follow-up critique reviewed the direct integration-path fix and returned `pass` with no open findings.
- critique:c10lock07-adapter-constraint-follow-up - follow-up critique reviewed the adapter-bound fix, caught a pre-final env-scope issue, then returned `pass` after parent moved the constraint to workflow-level `env`.
- critique:c10lock07-uv-01012-follow-up - follow-up critique reviewed the operator-requested uv 0.10.12 CI toolchain update and returned `pass` with no open findings.

Findings:

- critique:c10lock07-dependency-resolution#FIND-001 - resolved. The workflow integration step now runs `dbt-osmosis` directly from the matrix environment instead of invoking `demo_duckdb/integration_tests.sh`, and follow-up critique confirmed the prior challenge is resolved.
- critique:c10lock07-adapter-constraint-follow-up pre-final env-scope challenge - resolved before critique finalization. `DBT_ADAPTERS_CONSTRAINT` now lives at workflow-level `env`, so the primary pytest matrix and latest compatibility smoke both receive the same adapter floor.

Disposition status: completed

Deferral / not-required rationale: None. Mandatory critique is complete and final `main` CI evidence is recorded.

# Retrospective / Promotion Disposition

Disposition status: completed

Promoted: Updated `wiki:ci-compatibility-matrix` with the accepted clean-environment matrix workflow, shared CI uv tool constraint, adapter floor, plain pip smoke purpose, and failure modes for `uv run` resync and old adapter selection.

Deferred / not-required rationale: Broader package metadata cleanup remains explicitly deferred to `ticket:c10pkg10`; no spec, plan, constitution, or memory promotion was required for this ticket.

# Wiki Disposition

Completed - `wiki:ci-compatibility-matrix` updated during closure.

# Acceptance Decision

Accepted by: OpenCode
Accepted at: 2026-05-04T01:10:42Z
Basis: Final `main` GitHub Actions evidence passed for commit `19ef3a4bd3d6e6e3f5437e634a51fc11edaa23ba`; mandatory critique completed with all findings resolved; retrospective promotion updated `wiki:ci-compatibility-matrix`.
Residual risks: Resolver differences between uv and pip may remain for optional extras; the adapter floor is currently a CI stabilizer rather than package metadata cleanup; local Taskfile execution still depends on operator-installed uv; `demo_duckdb/integration_tests.sh` still uses `uv run` and should not be reintroduced into matrix CI without a sync-safe change; the existing uv-only protobuf override remains for `ticket:c10pkg10`.

# Dependencies

Coordinate with ticket:c10pkg10 for package metadata and extras cleanup.

# Journal

- 2026-05-03T21:10:43Z: Created from CI/build oracle finding.
- 2026-05-03T23:41:03Z: Activated ticket and compiled Ralph packet `packet:ralph-ticket-c10lock07-20260503T234103Z` for deterministic CI dependency resolution and pip install smoke implementation.
- 2026-05-03T23:50:14Z: Consumed Ralph packet after implementation changed `.github/workflows/tests.yml` and `Taskfile.yml`; recorded `evidence:c10lock07-local-dependency-resolution-verification`; moved to mandatory critique before commit/push acceptance.
- 2026-05-04T00:19:53Z: Mandatory critique `critique:c10lock07-dependency-resolution` found high-severity integration path risk because CI still reached `uv run` through `demo_duckdb/integration_tests.sh`; parent fixed workflow integration coverage to run direct `dbt-osmosis` commands from the matrix environment.
- 2026-05-04T00:19:53Z: Follow-up critique `critique:c10lock07-integration-path-follow-up` passed with no open findings; ticket moved to `complete_pending_acceptance` pending commit/push and final `main` CI evidence.
- 2026-05-04T00:47:15Z: Main CI for pushed commit `d72c3b8` failed dbt 1.8 pytest rows under CI-pinned `uv==0.5.13`; local reproduction showed adding `dbt-adapters>=1.16.3,<2.0` selected `dbt-adapters==1.16.3` and `mashumaro==3.14`, passed `pip check`, and drove the failing sync-operation test green.
- 2026-05-04T00:47:15Z: Added explicit adapter floor to workflow/Taskfile uv-resolved matrix/latest installs, fixed reviewer-caught workflow env scoping by moving `DBT_ADAPTERS_CONSTRAINT` to workflow-level `env`, recorded `evidence:c10lock07-adapter-bound-verification`, and finalized `critique:c10lock07-adapter-constraint-follow-up` with no remaining blocker; final `main` CI evidence remains pending after commit/push.
- 2026-05-04T00:57:09Z: Operator requested CI use `uv==0.10.12` at least; updated `.github/workflows/constraints.txt` to exact `uv==0.10.12`, verified the dbt 1.8 matrix-style path locally with that uv version, recorded `evidence:c10lock07-uv-01012-verification`, and finalized `critique:c10lock07-uv-01012-follow-up` with no open findings; final `main` CI evidence remains pending after commit/push.
- 2026-05-04T01:10:42Z: Final `main` CI for commit `19ef3a4bd3d6e6e3f5437e634a51fc11edaa23ba` passed (`Tests` run `25295956433`, `lint` run `25295956438`, `Release` run `25295956432`); recorded `evidence:c10lock07-main-ci-success`, updated `wiki:ci-compatibility-matrix`, accepted the ticket, and closed it.
