---
id: ticket:c10proxy25
kind: ticket
status: ready
change_class: code-behavior
risk_class: medium
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

Decide whether the MySQL SQL proxy is supported, experimental, or internal, then align dependencies, docs, SQL preservation, and security posture with that decision.

# Context

`src/dbt_osmosis/sql/proxy.py` is described as an experiment but imports `mysql_mimic` and `sqlglot`; `mysql-mimic` is a base dependency in `pyproject.toml:32`, while `sqlglot` is a direct import at `proxy.py:23`. Proxy execution rewrites SQL through `expression.sql(dialect=self.project.adapter.type())` at `proxy.py:119-123`. Comment middleware mutates only the in-memory manifest at `proxy.py:72-109` while comments imply possible writeback.

# Why Now

Experimental code in base installs creates support and dependency drag. If the proxy is public, it needs tests/docs/security boundaries. If it is not public, it should not shape base dependency resolution.

# Scope

- Make an explicit support decision: supported, experimental opt-in, or internal/removed.
- Move proxy-only dependencies to an optional extra if not base behavior.
- Add direct dependency for `sqlglot` if proxy remains importable.
- Avoid destructive SQL reserialization unless explicitly required and tested.
- Document launch/auth/security/writeback behavior if supported.
- Add tests for adapter-specific SQL preservation and in-memory comment mutation semantics if retained.

# Out Of Scope

- Building a production database proxy beyond the selected support boundary.
- Implementing durable YAML writeback for proxy comments unless explicitly scoped later.

# Acceptance Criteria

- ACC-001: The repository clearly states whether the proxy is supported, experimental opt-in, or internal.
- ACC-002: Base install no longer carries proxy-only dependencies unless the proxy is documented as base behavior.
- ACC-003: All proxy direct imports are declared direct dependencies in the appropriate extra.
- ACC-004: Executed SQL is not lossy-rewritten by SQLGlot unless tests prove the rewrite is intended and safe.
- ACC-005: Comment middleware behavior is documented and tested as in-memory-only, or writeback is implemented through safe YAML helpers.
- ACC-006: Security/auth/listen behavior is documented if the proxy can be run by users.

# Coverage

Covers:

- ticket:c10proxy25#ACC-001
- ticket:c10proxy25#ACC-002
- ticket:c10proxy25#ACC-003
- ticket:c10proxy25#ACC-004
- ticket:c10proxy25#ACC-005
- ticket:c10proxy25#ACC-006
- initiative:dbt-110-111-hardening#OBJ-007
- initiative:dbt-110-111-hardening#OBJ-008

# Claim Matrix

| Claim | Evidence | Critique | Status |
| --- | --- | --- | --- |
| ticket:c10proxy25#ACC-001 | evidence:oracle-backlog-scan | None | open |
| ticket:c10proxy25#ACC-004 | evidence:oracle-backlog-scan | None | open |

# Execution Notes

This may need a human support decision before implementation. If no decision exists, prefer isolating the proxy behind an extra and marking it experimental rather than polishing it into a supported feature by accident.

# Blockers

Potential human decision on support/removal boundary.

# Evidence

Existing evidence: evidence:oracle-backlog-scan. Missing evidence: dependency/install and proxy behavior tests.

# Critique Disposition

Risk class: medium

Critique policy: recommended

Policy rationale: Proxy support has dependency and security implications.

Required critique profiles: code-change, security, operator-clarity

Findings: None - no critique yet.

Disposition status: pending

Deferral / not-required rationale: None.

# Retrospective / Promotion Disposition

Disposition status: pending

Promoted: None - implementation not complete.

Deferred / not-required rationale: Wiki/docs likely required if proxy stays supported or experimental.

# Wiki Disposition

N/A - no wiki promotion selected yet.

# Acceptance Decision

Accepted by: Not accepted yet.
Accepted at: N/A.
Basis: Pending support decision and tests.
Residual risks: SQL proxy can expose security and dialect correctness issues.

# Dependencies

Coordinate with ticket:c10pkg10 and ticket:c10sql21.

# Journal

- 2026-05-03T21:10:43Z: Created from CLI/SQL/workbench oracle finding.
