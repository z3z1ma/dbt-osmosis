---
id: ticket:c10yag28
kind: ticket
status: complete_pending_acceptance
change_class: code-behavior
risk_class: low
created_at: 2026-05-03T21:10:43Z
updated_at: 2026-05-05T05:03:18Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  initiative:
    - initiative:dbt-110-111-hardening
  evidence:
    - evidence:oracle-backlog-scan
    - evidence:c10yag28-lean-cleanup-validation
  packets:
    - packet:ralph-ticket-c10yag28-20260505T045616Z
depends_on:
  - ticket:c10res14
---

# Summary

Clean up low-value leaky abstractions and YAGNI code after higher-priority compatibility fixes land.

# Context

The audit flagged `src/dbt_osmosis/core/settings.py:381-383` mutating private `ThreadPoolExecutor._max_workers`, `src/dbt_osmosis/core/introspection.py:1580-1586` exposing a stub database property source that falls back to manifest, and partially dead resolver source classes until ticket:c10res14 wires or removes them.

# Why Now

The user explicitly asked to ticket trash code, code smells, leaky abstractions, and YAGNI. These should not block high-risk compatibility fixes but should have a durable cleanup owner.

# Scope

- Replace private `ThreadPoolExecutor._max_workers` mutation with construction-time worker count or another supported mechanism.
- Remove or clearly mark unsupported `PropertySource.DATABASE` behavior if it is not real.
- Remove dead resolver/source abstractions after ticket:c10res14 decides which are used.
- Remove stale Black/isort configuration if Ruff is canonical and no tooling consumes them.
- Add lightweight tests only where cleanup changes observable behavior.

# Out Of Scope

- Large refactors before high-risk compatibility tickets are done.
- Removing public APIs or documented behavior without a deprecation plan.

# Acceptance Criteria

- ACC-001: No production code mutates private `ThreadPoolExecutor._max_workers`.
- ACC-002: Unsupported database property access is removed, renamed, or documented so callers cannot mistake it for real database introspection.
- ACC-003: Resolver helper/source classes are either used by the context-aware resolver or removed after ticket:c10res14.
- ACC-004: Obsolete formatter configs are removed only if no active workflow depends on them.
- ACC-005: Cleanup diff is small, behavior-preserving, and covered by existing or focused tests where needed.

# Coverage

Covers:

- ticket:c10yag28#ACC-001
- ticket:c10yag28#ACC-002
- ticket:c10yag28#ACC-003
- ticket:c10yag28#ACC-004
- ticket:c10yag28#ACC-005
- initiative:dbt-110-111-hardening#OBJ-008

# Claim Matrix

| Claim | Evidence | Critique | Status |
| --- | --- | --- | --- |
| ticket:c10yag28#ACC-001 | evidence:c10yag28-lean-cleanup-validation | None | covered |
| ticket:c10yag28#ACC-002 | evidence:c10yag28-lean-cleanup-validation | None | covered |
| ticket:c10yag28#ACC-003 | evidence:c10yag28-lean-cleanup-validation | None | covered |
| ticket:c10yag28#ACC-004 | evidence:c10yag28-lean-cleanup-validation | None | covered |
| ticket:c10yag28#ACC-005 | evidence:c10yag28-lean-cleanup-validation | None | covered |

# Execution Notes

ticket:c10res14 is closed, so this cleanup is unblocked. Prefer deleting misleading code over adding comments that preserve confusion, but do not remove public compatibility exports or documented behavior without a deprecation plan.

# Blockers

None. Hard dependency ticket:c10res14 is closed.

# Evidence

Evidence status: local test-first validation, parent focused pytest, Ruff checks, whitespace check, and basedpyright zero-error validation support ACC-001 through ACC-005 for the uncommitted implementation diff. Missing evidence: remote CI for the eventual implementation commit.

# Critique Disposition

Risk class: low

Critique policy: optional

Policy rationale: Low-risk cleanup once dependencies are resolved.

Required critique profiles: None - optional cleanup review unless diff grows.

Findings: None - no critique performed.

Disposition status: not_required

Deferral / not-required rationale: Optional critique is not required because the final diff remains narrow, low-risk, and covered by focused red/green validation plus parent static checks. Revisit if remote CI or review reveals a broader behavior risk.

# Retrospective / Promotion Disposition

Disposition status: not_required

Promoted: None.

Deferred / not-required rationale: Local cleanup is covered by tests and evidence; no durable explanation promotion is needed.

# Wiki Disposition

N/A - no wiki promotion selected yet.

# Acceptance Decision

Accepted by: Pending remote CI.
Accepted at: N/A.
Basis: Local implementation, red/green evidence, and parent validation are complete. Final acceptance waits for implementation commit packaging and remote CI.
Residual risks: basedpyright warning debt remains but error count is zero; `_get_setting_for_node()` remains a compatibility wrapper rather than being removed; external callers that used `PropertySource.DATABASE` for manifest fallback will now receive an explicit unsupported-source error.

# Dependencies

Hard dependency: ticket:c10res14.

# Journal

- 2026-05-03T21:10:43Z: Created from core architecture oracle cleanup findings.
- 2026-05-05T04:56:16Z: Hard dependency ticket:c10res14 is closed. Promoted through ready into active and compiled packet:ralph-ticket-c10yag28-20260505T045616Z for a narrow test-first cleanup iteration covering executor construction, unsupported database property source behavior, resolver/source YAGNI verification, and stale formatter config removal.
- 2026-05-05T05:03:18Z: Ralph iteration returned stop. Parent diff review and validation passed: focused settings/config-resolution/introspection pytest reported 134 passed, Ruff format/check and `git diff --check` passed, and basedpyright reported zero errors. Recorded evidence:c10yag28-lean-cleanup-validation, marked optional critique and retrospective/promotion not required, and moved to complete_pending_acceptance pending implementation commit packaging and remote CI.
