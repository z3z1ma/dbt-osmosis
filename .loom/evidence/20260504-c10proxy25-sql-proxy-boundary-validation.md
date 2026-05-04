---
id: evidence:c10proxy25-sql-proxy-boundary-validation
kind: evidence
status: recorded
created_at: 2026-05-04T16:50:00Z
updated_at: 2026-05-04T16:50:00Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  tickets:
    - ticket:c10proxy25
  packets:
    - packet:ralph-ticket-c10proxy25-20260504T163103Z
external_refs: {}
---

# Summary

Observed validation for `ticket:c10proxy25` after implementation commit `99f0ff6d29141cb3c6201da2bae5c81d553f8579`. The checks show the SQL proxy boundary is covered by focused tests, package metadata/docs assertions, recent SQL and validation regressions remain green, targeted hooks pass, and local basedpyright reports zero errors.

# Procedure

Observed at: 2026-05-04T16:50:00Z

Source state: `99f0ff6d29141cb3c6201da2bae5c81d553f8579` on branch `loom/dbt-110-111-hardening`.

Procedure:

- `uv run pytest tests/core/test_sql_proxy.py tests/test_package_metadata.py tests/core/test_sql_operations.py tests/core/test_model_validation.py -q`
- `uv run ruff check src/dbt_osmosis/sql/proxy.py tests/core/test_sql_proxy.py tests/test_package_metadata.py && git diff --check`
- `uv run pre-commit run --files src/dbt_osmosis/sql/proxy.py tests/core/test_sql_proxy.py tests/test_package_metadata.py README.md docs/docs/intro.md docs/docs/tutorial-basics/installation.md .loom/packets/ralph/20260504T163103Z-ticket-c10proxy25-iter-01.md .loom/tickets/20260503-c10proxy25-decide-sql-proxy-support-boundary.md`
- `uv run basedpyright --outputjson` with summary inspected from JSON output

Expected result when applicable: focused proxy/package tests pass, recent SQL and validation regressions remain green, Ruff and whitespace checks pass, targeted pre-commit hooks pass, and basedpyright reports zero errors.

Actual observed result: targeted pytest reported `28 passed`; Ruff reported `All checks passed!`; `git diff --check` produced no output; targeted pre-commit hooks passed; basedpyright summary reported `errorCount: 0`, `warningCount: 1869`.

Procedure verdict / exit code: pass / exit code 0 for pytest, Ruff, whitespace, pre-commit, and basedpyright error gate. basedpyright warnings remain pre-existing/non-blocking under repository CI policy.

# Artifacts

- Red evidence from Ralph packet: pre-fix proxy tests failed because `DbtSession.query()` called `expression.sql(...)` instead of preserving the original client SQL string, and comment middleware wording did not state the in-memory-only boundary.
- Parent critique initially returned `changes_required` for `ticket:c10proxy25#ACC-006` because docs did not explain the runnable module entrypoint security/listen boundary; the implementation was amended before commit.
- Final parent pytest: `28 passed in 14.55s`.
- Final basedpyright summary: `{'filesAnalyzed': 35, 'errorCount': 0, 'warningCount': 1869, 'informationCount': 0, 'timeInSec': 2.947}`.
- Final critique re-review verdict: `pass`, with no findings.

# Supports Claims

- `ticket:c10proxy25#ACC-001`: docs and package metadata assertions state the proxy is an experimental opt-in runtime.
- `ticket:c10proxy25#ACC-002`: `mysql-mimic` remains outside base dependencies and in the `proxy` optional extra; package metadata tests assert this boundary.
- `ticket:c10proxy25#ACC-003`: direct proxy-only dependency `mysql-mimic` is declared in the `proxy` extra; `sqlglot` remains base because SQL linting also imports it.
- `ticket:c10proxy25#ACC-004`: `DbtSession.query()` passes original client SQL text to `execute_sql_code()`, and `tests/core/test_sql_proxy.py` verifies SQLGlot reserialization is not used.
- `ticket:c10proxy25#ACC-005`: comment middleware docstring and tests cover in-memory-only manifest mutation and no durable YAML writeback claim.
- `ticket:c10proxy25#ACC-006`: docs and module docstring state there is no dbt-osmosis auth, TLS, or bind hardening; the module entrypoint is a local-only experiment using `mysql-mimic` defaults and should not be exposed to untrusted networks.
- `initiative:dbt-110-111-hardening#OBJ-007`: optional/dependency boundary stays explicit for the experimental proxy.
- `initiative:dbt-110-111-hardening#OBJ-008`: user-facing docs now preserve the support/security boundary.

# Challenges Claims

None - no observed validation result challenged the scoped claims after the ACC-006 documentation fix.

# Environment

Commit: `99f0ff6d29141cb3c6201da2bae5c81d553f8579`

Branch: `loom/dbt-110-111-hardening`

Runtime: `uv run` project environment; warning noted that an unrelated active `VIRTUAL_ENV` was ignored.

OS: macOS / Darwin.

Relevant config: base local environment does not install `mysql_mimic`; proxy tests use local fake `mysql_mimic` modules and do not open sockets or start a server.

External service / harness / data source when applicable: no network service, real proxy server, database socket, OpenAI, Azure, or GitHub Actions execution was used for this validation.

# Validity

Valid for: `ticket:c10proxy25` implementation at commit `99f0ff6d29141cb3c6201da2bae5c81d553f8579` and the listed local environment.

Fresh enough for: ticket acceptance review and critique disposition.

Recheck when: `src/dbt_osmosis/sql/proxy.py`, proxy optional dependencies, proxy docs, `tests/core/test_sql_proxy.py`, `tests/test_package_metadata.py`, SQL execution helpers, or the proxy support decision changes.

Invalidated by: changes after commit `99f0ff6d29141cb3c6201da2bae5c81d553f8579` that alter proxy runtime behavior, package extras, comment middleware persistence, or docs/security wording.

Supersedes / superseded by: supersedes `evidence:oracle-backlog-scan` for c10proxy25 proxy boundary behavior; not superseded.

# Limitations

- Does not exercise a real `mysql-mimic` server, socket listener, client handshake, auth path, TLS path, or network bind behavior.
- Does not make the proxy production-supported, remove the proxy, or add runtime hardening.
- Does not implement durable YAML writeback for proxy comment middleware.
- Full GitHub Actions validation for the pushed commits remains pending until after guarded push.

# Result

The committed c10proxy25 implementation passed focused tests, lint, targeted pre-commit, and local basedpyright with zero errors.

# Interpretation

The evidence supports accepting the scoped experimental opt-in proxy boundary with explicit residual risk that the proxy remains a local-only, non-hardened experiment and real `mysql-mimic` runtime defaults are untested by this ticket.

# Related Records

- `ticket:c10proxy25`
- `packet:ralph-ticket-c10proxy25-20260504T163103Z`
- `critique:c10proxy25-sql-proxy-boundary-review`
- `wiki:sql-proxy-boundary`
- `initiative:dbt-110-111-hardening`
