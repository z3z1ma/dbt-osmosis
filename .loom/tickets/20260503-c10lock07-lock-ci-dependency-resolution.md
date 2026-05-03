---
id: ticket:c10lock07
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
| ticket:c10lock07#ACC-001 | evidence:oracle-backlog-scan | None | open |
| ticket:c10lock07#ACC-005 | None - install smoke not added yet | None | open |

# Execution Notes

Be careful with `UV_FROZEN` and `UV_NO_SYNC` in `Taskfile.yml`; verify the local workflow still works from a fresh clone. Include a deliberate stale-lock test strategy in PR description or evidence when implementing.

# Blockers

None.

# Evidence

Existing evidence: evidence:oracle-backlog-scan. Missing evidence: CI/local command output after changes.

# Critique Disposition

Risk class: high

Critique policy: mandatory

Policy rationale: Dependency resolution determines compatibility evidence trustworthiness.

Required critique profiles: release-packaging, test-coverage

Findings: None - no critique yet.

Disposition status: pending

Deferral / not-required rationale: None.

# Retrospective / Promotion Disposition

Disposition status: pending

Promoted: None - implementation not complete.

Deferred / not-required rationale: Not decided.

# Wiki Disposition

N/A - no wiki promotion selected yet.

# Acceptance Decision

Accepted by: Not accepted yet.
Accepted at: N/A.
Basis: Pending CI evidence.
Residual risks: Resolver differences between uv and pip may remain for optional extras.

# Dependencies

Coordinate with ticket:c10pkg10 for package metadata and extras cleanup.

# Journal

- 2026-05-03T21:10:43Z: Created from CI/build oracle finding.
