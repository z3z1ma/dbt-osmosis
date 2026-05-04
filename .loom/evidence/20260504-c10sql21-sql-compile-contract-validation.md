---
id: evidence:c10sql21-sql-compile-contract-validation
kind: evidence
status: recorded
created_at: 2026-05-04T14:13:06Z
updated_at: 2026-05-04T14:13:06Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  tickets:
    - ticket:c10sql21
  packets:
    - packet:ralph-ticket-c10sql21-20260504T134541Z
  critique:
    - critique:c10sql21-sql-compile-contract-review
external_refs: {}
---

# Summary

Observed test-first local verification for `ticket:c10sql21` SQL compile/run contract behavior. The evidence records red/green behavior, workbench run-query coverage, real dbt fixture SQL operations, post-commit checks, and mandatory critique; it does not decide ticket closure by itself.

# Procedure

Observed at: 2026-05-04T14:13:06Z
Source state: branch `loom/dbt-110-111-hardening`, implementation commit `098d4d2188706c32b49b2cc0eba16100018e41df`, based on prior main commit `34b24c8b8976db24525b2293601f7fa0b77fdbe2`.
Procedure: Reviewed Ralph child output, inspected implementation diff, reran focused SQL/CLI/workbench checks, added parent-side workbench `run_query()` coverage after mandatory critique found ACC-003 incomplete, reran related SQL/CLI/workbench/legacy/model-validation suites, ran Ruff format/check, ran `git diff --check`, ran targeted pre-commit hooks, reran mandatory read-only critique to acceptance, committed the implementation, then reran post-commit validation against the committed source state.
Expected result when applicable: plain SQL compile should return a node whose `compiled_code` is executable SQL; CLI `sql compile "select 1"` should print `select 1`; workbench compile and run-query paths should work for default scratch SQL and default query template; real fixture tests should compile `ref()` SQL and execute `select 1`; temporary manifest SQL operation nodes should be cleaned up after success and failure.
Actual observed result: Initial red tests failed before implementation because plain SQL `compiled_code` was `None`, CLI compile printed `None`, and workbench compile returned blank. Final focused, related, hook, lint, diff, and post-commit checks passed locally. Final mandatory critique reported no findings.
Procedure verdict / exit code: mixed red/green sequence. Final green checks observed exit 0 for focused/related pytest, Ruff, targeted pre-commit, and `git diff --check`.

# Artifacts

Red observations recorded in `packet:ralph-ticket-c10sql21-20260504T134541Z`:

- Targeted red run returned 3 failures before implementation.
- Plain SQL compile returned `compiled_code is None` instead of raw executable SQL.
- CLI `sql compile "select 1"` printed `None`.
- Workbench `compile(default_prompt)` returned `""` for raw scratch SQL.

Final green observations:

- Child focused SQL/CLI/workbench checks returned `25 passed`.
- Parent related validation returned `39 passed in 16.09s` before commit.
- Ruff format/check over changed source/test files passed.
- `git diff --check` passed.
- Targeted `pre-commit run --files ...` over changed source/test/packet/ticket files passed all applicable hooks.
- Final mandatory read-only critique accepted the diff with no findings.
- Local dbt version observation: `dbt-core 1.10.20`, `dbt-duckdb 1.10.0`.
- Implementation commit: `098d4d2188706c32b49b2cc0eba16100018e41df`.
- Post-implementation-commit acceptance validation: `uv run pytest tests/core/test_workbench_app.py tests/core/test_sql_operations.py tests/core/test_cli.py tests/core/test_legacy.py tests/core/test_model_validation.py -q` returned `39 passed in 15.97s`.
- Post-implementation-commit `uv run ruff check src/dbt_osmosis/core/sql_operations.py tests/core/test_sql_operations.py tests/core/test_cli.py tests/core/test_workbench_app.py && git diff --check` passed.
- Post-implementation-commit targeted pre-commit over changed source/test/packet/ticket files passed all applicable hooks.

# Supports Claims

- ticket:c10sql21#ACC-001: plain SQL compile now returns `compiled_code == raw_sql`, so callers can use `compiled_code` without special-casing `None`.
- ticket:c10sql21#ACC-002: CLI regression proves `sql compile "select 1"` emits `select 1`.
- ticket:c10sql21#ACC-003: workbench regressions prove `compile(default_prompt)` preserves raw scratch SQL and `run_query()` succeeds through the `Preview.initial_state()` default query template.
- ticket:c10sql21#ACC-004: real fixture regression compiles SQL containing `ref('customers')`; the test is matrix-capable and will run under the dbt 1.10/1.11 initiative CI, though local observation was dbt 1.10.20 only.
- ticket:c10sql21#ACC-005: real fixture regression executes `select 1 as value` through `execute_sql_code()`.
- ticket:c10sql21#ACC-006: real fixture/regression checks prove temporary SQL operation manifest keys are absent after ref compile success and forced compile failure.
- initiative:dbt-110-111-hardening#OBJ-001: local validation adds behavioral coverage around the private dbt SQL parser/task API path.
- initiative:dbt-110-111-hardening#OBJ-006: local validation supports user-facing SQL CLI/workbench behavior.

# Challenges Claims

None - final observed checks matched the expected post-fix results for the cited claims, with 1.11 runtime observation deferred to initiative CI.

# Environment

Commit: implementation commit `098d4d2188706c32b49b2cc0eba16100018e41df`.
Branch: `loom/dbt-110-111-hardening`
Runtime: local `uv run` environment.
dbt: `dbt-core 1.10.20`, `dbt-duckdb 1.10.0`.
OS: macOS Darwin.
Relevant config: SQL operations, SQL CLI command, workbench compile/run helper, and related core tests from the reviewed source state.
External service / harness / data source when applicable: no production service exercised; tests used local DuckDB fixtures and import-time stubs for optional workbench UI dependencies.

# Validity

Valid for: plain SQL compile contract, CLI compile output, workbench compile/run-query helper behavior, local dbt 1.10.20 private SQL parser/task behavior, `execute_sql_code("select 1")`, and temporary manifest cleanup in the observed implementation commit.
Fresh enough for: critique and local ticket acceptance under the current directive to defer per-ticket GitHub Actions waiting to final initiative validation.
Recheck when: `compile_sql_code()`, `execute_sql_code()`, SQL CLI output, workbench compile/run helpers, or dbt SQL parser/task internals change.
Invalidated by: source changes after this evidence that alter SQL operation compile/run behavior or workbench query execution.
Supersedes / superseded by: Supplements `evidence:oracle-backlog-scan` and `research:dbt-110-111-api-surfaces`; final initiative-level CI should supplement this evidence with dbt 1.11 runtime observation.

# Limitations

- Local runtime validation used dbt-core 1.10.20 and dbt-duckdb 1.10.0; dbt 1.11 matrix execution is deferred to final initiative validation.
- Workbench tests use import-time stubs for optional UI dependencies and validate compile/run-query helpers, not full Streamlit rendering.
- Private dbt SQL APIs remain inherently fragile; this ticket adds behavioral coverage rather than replacing them with a public API.

# Result

The observed checks showed plain SQL compile now provides executable `compiled_code`, CLI compile prints SQL, workbench default scratch query compile/run succeeds, real fixture SQL compile/execute paths pass locally, and temporary manifest SQL operation nodes are cleaned up on success and failure.

# Interpretation

The evidence supports ticket acceptance with residual risks documented and final initiative-level CI still pending outside this ticket. It does not replace broader dbt 1.11 matrix validation.

# Related Records

- ticket:c10sql21
- packet:ralph-ticket-c10sql21-20260504T134541Z
- critique:c10sql21-sql-compile-contract-review
- research:dbt-110-111-api-surfaces
- evidence:oracle-backlog-scan
