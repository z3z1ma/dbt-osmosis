---
id: ticket:c10sql21
kind: ticket
status: closed
change_class: code-behavior
risk_class: high
created_at: 2026-05-03T21:10:43Z
updated_at: 2026-05-04T14:13:06Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  initiative:
    - initiative:dbt-110-111-hardening
  research:
    - research:dbt-110-111-api-surfaces
  evidence:
    - evidence:oracle-backlog-scan
    - evidence:c10sql21-sql-compile-contract-validation
  packets:
    - packet:ralph-ticket-c10sql21-20260504T134541Z
  critique:
    - critique:c10sql21-sql-compile-contract-review
external_refs:
  dbt_core_110_sql_parser: https://raw.githubusercontent.com/dbt-labs/dbt-core/v1.10.0/core/dbt/parser/sql.py
  dbt_core_111_sql_parser: https://raw.githubusercontent.com/dbt-labs/dbt-core/v1.11.0/core/dbt/parser/sql.py
  dbt_core_110_sql_task: https://raw.githubusercontent.com/dbt-labs/dbt-core/v1.10.0/core/dbt/task/sql.py
  dbt_core_111_sql_task: https://raw.githubusercontent.com/dbt-labs/dbt-core/v1.11.0/core/dbt/task/sql.py
depends_on: []
---

# Summary

Fix the SQL compile contract so plain SQL returns executable SQL, and add real dbt 1.10/1.11 coverage around private dbt SQL parser/task APIs.

# Context

`src/dbt_osmosis/core/sql_operations.py:27-56` returns a parsed `SqlNode` for plain SQL without setting `compiled_code`. `src/dbt_osmosis/cli/main.py:1407-1409` and `src/dbt_osmosis/workbench/app.py:225-231` consume `compiled_code`, so plain SQL can print `None` or become blank. Tests currently assert the broken behavior. The same module imports private dbt APIs: `process_node`, `SqlCompileRunner`, and `SqlBlockParser.parse_remote` via the project context.

# Why Now

SQL compile/run is user-facing and reused by the workbench and proxy. dbt 1.10/1.11 source shows signatures still match, but only real integration tests can protect the private API boundary.

# Scope

- Define and document `compile_sql_code()` return contract for plain and Jinja SQL.
- Ensure `dbt-osmosis sql compile "select 1"` prints `select 1`, not `None`.
- Ensure workbench compile/run uses raw SQL fallback correctly.
- Add real dbt fixture tests for SQL with and without Jinja/ref under dbt 1.10.x and 1.11.x.
- Verify temporary manifest node cleanup on success and failure.

# Out Of Scope

- Replacing all dbt private SQL APIs if a compatibility shim plus tests is sufficient.
- SQL proxy query rewriting; ticket:c10proxy25 owns proxy behavior.

# Acceptance Criteria

- ACC-001: Plain SQL compile returns an object/value callers can use as executable SQL without special-casing `None`.
- ACC-002: CLI `sql compile "select 1"` emits `select 1`.
- ACC-003: Workbench compile pane and run query work for the default scratch query.
- ACC-004: Real fixture tests compile SQL containing `ref()` under dbt 1.10.x and 1.11.x.
- ACC-005: Real fixture tests execute `select 1` through `execute_sql_code()`.
- ACC-006: Temporary manifest nodes are cleaned up after compile success and failure.

# Coverage

Covers:

- ticket:c10sql21#ACC-001
- ticket:c10sql21#ACC-002
- ticket:c10sql21#ACC-003
- ticket:c10sql21#ACC-004
- ticket:c10sql21#ACC-005
- ticket:c10sql21#ACC-006
- initiative:dbt-110-111-hardening#OBJ-001
- initiative:dbt-110-111-hardening#OBJ-006

# Claim Matrix

| Claim | Evidence | Critique | Status |
| --- | --- | --- | --- |
| ticket:c10sql21#ACC-001 | evidence:c10sql21-sql-compile-contract-validation | critique:c10sql21-sql-compile-contract-review | accepted |
| ticket:c10sql21#ACC-002 | evidence:c10sql21-sql-compile-contract-validation | critique:c10sql21-sql-compile-contract-review | accepted |
| ticket:c10sql21#ACC-003 | evidence:c10sql21-sql-compile-contract-validation | critique:c10sql21-sql-compile-contract-review | accepted |
| ticket:c10sql21#ACC-004 | evidence:c10sql21-sql-compile-contract-validation | critique:c10sql21-sql-compile-contract-review | accepted with final 1.11 CI observation deferred |
| ticket:c10sql21#ACC-005 | evidence:c10sql21-sql-compile-contract-validation | critique:c10sql21-sql-compile-contract-review | accepted |
| ticket:c10sql21#ACC-006 | evidence:c10sql21-sql-compile-contract-validation | critique:c10sql21-sql-compile-contract-review | accepted |

# Execution Notes

Prefer returning raw SQL in `compiled_code` for non-Jinja paths or adding a documented helper for callers to choose executable SQL. Update tests that currently assert `compiled_code is None` because that assertion encodes user-visible breakage.

# Blockers

None.

# Evidence

Evidence recorded:

- evidence:oracle-backlog-scan
- research:dbt-110-111-api-surfaces
- evidence:c10sql21-sql-compile-contract-validation

# Critique Disposition

Risk class: high

Critique policy: mandatory

Policy rationale: User-facing SQL behavior and private dbt API usage both affect compatibility guarantees.

Required critique profiles: code-change, test-coverage, dbt-compatibility

Findings: None in final critique. Earlier critique blocker for incomplete workbench run-query coverage was resolved before acceptance.

Disposition status: completed

Deferral / not-required rationale: N/A - mandatory critique completed with no blockers.

# Retrospective / Promotion Disposition

Disposition status: not_required

Promoted: None.

Deferred / not-required rationale: No SQL compatibility shim was introduced; the accepted contract is captured in tests, evidence, and critique.

# Wiki Disposition

N/A - no wiki promotion selected; no new reusable SQL compatibility shim or workflow page was introduced.

# Acceptance Decision

Accepted by: OpenCode parent agent.
Accepted at: 2026-05-04T14:13:06Z.
Basis: implementation commit `098d4d2188706c32b49b2cc0eba16100018e41df`, evidence:c10sql21-sql-compile-contract-validation, and critique:c10sql21-sql-compile-contract-review.
Residual risks: Local runtime validation used dbt-core 1.10.20 and dbt-duckdb 1.10.0; dbt 1.11 matrix execution is deferred to final initiative validation; workbench coverage uses import-time stubs for optional UI dependencies; private dbt SQL APIs may change after 1.11.

# Dependencies

Coordinate with ticket:c10wb22 and ticket:c10proxy25.

# Journal

- 2026-05-03T21:10:43Z: Created from CLI/SQL/workbench and dbt compatibility oracle findings.
- 2026-05-04T13:45:41Z: Activated ticket and compiled Ralph packet `packet:ralph-ticket-c10sql21-20260504T134541Z` for test-first SQL compile/run contract coverage with local fixture validation and mandatory critique before acceptance.
- 2026-05-04T14:13:06Z: Ralph iteration consumed. Implementation commit `098d4d2188706c32b49b2cc0eba16100018e41df` made plain SQL compile return executable `compiled_code`, added CLI/workbench/default query coverage, added real fixture SQL compile/execute and manifest cleanup tests, and documented the return contract. Local validation passed with `39 passed`; final mandatory critique found no blockers. Accepted and closed with dbt 1.11 runtime observation deferred to final initiative CI.
