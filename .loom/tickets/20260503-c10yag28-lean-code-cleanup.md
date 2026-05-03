---
id: ticket:c10yag28
kind: ticket
status: proposed
change_class: code-behavior
risk_class: low
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
| ticket:c10yag28#ACC-001 | evidence:oracle-backlog-scan | None | open |

# Execution Notes

Keep this ticket behind ticket:c10res14 so cleanup does not remove code that the resolver integration decides to use. Prefer deleting misleading code over adding comments that preserve confusion.

# Blockers

Blocked on ticket:c10res14 for resolver cleanup pieces.

# Evidence

Existing evidence: evidence:oracle-backlog-scan. Missing evidence: cleanup diff and focused tests if behavior changes.

# Critique Disposition

Risk class: low

Critique policy: optional

Policy rationale: Low-risk cleanup once dependencies are resolved.

Required critique profiles: None - optional cleanup review unless diff grows.

Findings: None - no critique yet.

Disposition status: pending

Deferral / not-required rationale: Critique can be marked not_required if the final diff remains narrow and covered.

# Retrospective / Promotion Disposition

Disposition status: pending

Promoted: None - implementation not complete.

Deferred / not-required rationale: Not likely wiki-worthy unless cleanup changes operator guidance.

# Wiki Disposition

N/A - no wiki promotion selected yet.

# Acceptance Decision

Accepted by: Not accepted yet.
Accepted at: N/A.
Basis: Pending cleanup.
Residual risks: Cleanup can become risky if scope expands.

# Dependencies

Hard dependency: ticket:c10res14.

# Journal

- 2026-05-03T21:10:43Z: Created from core architecture oracle cleanup findings.
