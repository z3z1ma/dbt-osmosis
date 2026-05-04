---
id: ticket:c10proxy25
kind: ticket
status: closed
change_class: code-behavior
risk_class: medium
created_at: 2026-05-03T21:10:43Z
updated_at: 2026-05-04T16:50:00Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  initiative:
    - initiative:dbt-110-111-hardening
  evidence:
    - evidence:oracle-backlog-scan
    - evidence:c10proxy25-sql-proxy-boundary-validation
  critique:
    - critique:c10proxy25-sql-proxy-boundary-review
  wiki:
    - wiki:sql-proxy-boundary
  packets:
    - packet:ralph-ticket-c10proxy25-20260504T163103Z
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
| ticket:c10proxy25#ACC-001 | evidence:c10proxy25-sql-proxy-boundary-validation | critique:c10proxy25-sql-proxy-boundary-review | accepted |
| ticket:c10proxy25#ACC-002 | evidence:c10proxy25-sql-proxy-boundary-validation | critique:c10proxy25-sql-proxy-boundary-review | accepted |
| ticket:c10proxy25#ACC-003 | evidence:c10proxy25-sql-proxy-boundary-validation | critique:c10proxy25-sql-proxy-boundary-review | accepted |
| ticket:c10proxy25#ACC-004 | evidence:c10proxy25-sql-proxy-boundary-validation | critique:c10proxy25-sql-proxy-boundary-review | accepted |
| ticket:c10proxy25#ACC-005 | evidence:c10proxy25-sql-proxy-boundary-validation | critique:c10proxy25-sql-proxy-boundary-review | accepted |
| ticket:c10proxy25#ACC-006 | evidence:c10proxy25-sql-proxy-boundary-validation | critique:c10proxy25-sql-proxy-boundary-review | accepted |

# Execution Notes

Support decision: keep the proxy as an experimental opt-in local runtime. Do not remove it, promote it to production-supported, add auth/listen/TLS hardening, or implement durable comment writeback in this ticket.

# Blockers

None. The support boundary was resolved as experimental opt-in local runtime for this ticket.

# Evidence

Existing evidence: `evidence:oracle-backlog-scan`.

Validation evidence: `evidence:c10proxy25-sql-proxy-boundary-validation`.

Implementation commit: `99f0ff6d29141cb3c6201da2bae5c81d553f8579` (`fix: harden experimental SQL proxy boundary`).

Observed validation:

- `uv run pytest tests/core/test_sql_proxy.py tests/test_package_metadata.py tests/core/test_sql_operations.py tests/core/test_model_validation.py -q` passed `28 passed`.
- `uv run ruff check ... && git diff --check` passed.
- Targeted `uv run pre-commit run --files ...` passed.
- Local basedpyright summary reported `errorCount: 0`, `warningCount: 1869`.

# Critique Disposition

Risk class: medium

Critique policy: recommended

Policy rationale: Proxy support has dependency and security implications.

Required critique profiles: code-change, security, operator-clarity

Critique record: `critique:c10proxy25-sql-proxy-boundary-review`.

Verdict: `pass`.

Findings: None - no open findings. Initial critique found an ACC-006 documentation gap for the runnable module entrypoint security/listen boundary; the implementation was amended before commit and final re-review passed.

Disposition status: completed

Deferral / not-required rationale: None.

# Retrospective / Promotion Disposition

Disposition status: completed

Promoted: `wiki:sql-proxy-boundary`.

Deferred / not-required rationale: Retrospective found one durable lesson: the SQL proxy is an experimental opt-in local runtime with no dbt-osmosis auth, TLS, bind/listen hardening, or durable comment writeback. That accepted explanation was promoted to wiki and mirrored in README/Docusaurus docs.

# Wiki Disposition

Completed: `wiki:sql-proxy-boundary` preserves the accepted SQL proxy support, dependency, security/listen, SQL preservation, and comment middleware boundaries.

# Acceptance Decision

Accepted by: OpenCode.
Accepted at: 2026-05-04T16:50:00Z.
Basis: `evidence:c10proxy25-sql-proxy-boundary-validation`, `critique:c10proxy25-sql-proxy-boundary-review`, and retrospective promotion `wiki:sql-proxy-boundary` cover all scoped acceptance criteria with no critique blockers.
Residual risks: The proxy remains experimental and real `mysql-mimic` runtime defaults are not exercised by tests; no auth, TLS, bind/listen hardening, production support, or durable comment writeback was added; full GitHub Actions validation after push remains pending outside this ticket's local acceptance.

# Dependencies

Coordinate with ticket:c10pkg10 and ticket:c10sql21.

# Journal

- 2026-05-03T21:10:43Z: Created from CLI/SQL/workbench oracle finding.
- 2026-05-04T16:31:03Z: Activated ticket and compiled Ralph packet `packet:ralph-ticket-c10proxy25-20260504T163103Z` for experimental opt-in proxy boundary, dependency, SQL preservation, docs, and in-memory comment middleware tests.
- 2026-05-04T16:50:00Z: Accepted and closed after implementation commit `99f0ff6d29141cb3c6201da2bae5c81d553f8579`, evidence `evidence:c10proxy25-sql-proxy-boundary-validation`, critique `critique:c10proxy25-sql-proxy-boundary-review`, and retrospective promotion `wiki:sql-proxy-boundary`.
