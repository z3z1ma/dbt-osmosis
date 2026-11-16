Status: done
Created: 2026-11-16
Updated: 2026-11-16

# Repair dependency smoke and future Core canary

## Scope
Repair CI run 43832908769 on main 3cc00a57 in an isolated checkout. User explicitly authorized implementation, tests, branch push and draft PR.

## Non-goals
No merge, release, dependency narrowing, or expansion of audited Core 1.8–1.11 support.

## Acceptance Criteria
- DuckDB pip smoke validates the published extra requirement; all six independent smokes reach completion including proxy.
- Inheritance assertions accommodate Core 1.12 nullable semantic fields without suppressing non-null metadata regressions.
- Verify preservation and explicitly configured inheritance of semantic metadata through YAML writeback.
- Run focused and full supported/canary checks and record precise results and limits; publish a draft PR.

## References
README.md; pyproject.toml; .github/workflows/tests.yml; tests/test_yaml_inheritance.py; https://github.com/z3z1ma/dbt-osmosis/actions/runs/43832908769

## Assumptions
Source-backed: extra permits dbt-duckdb>=1,<2; future Core is canary-only. User-ratified: fix the contract checks, investigate runtime preservation, publish draft only.

## Journal
- Fresh clone verifies latest main 3cc00a57fa4883050fd5cffeba7adee9d6d0d4d3. Existing personal checkout untouched. No .agents directory exists in either checkout. Read root/core AGENTS, local 10x skill and related durable records.
- Isolated Python 3.13.14 resolves Core 1.12.7, DuckDB adapter 1.11.1, adapters 1.24.5. Original inheritance test reproduced (1 failed); ColumnInfo adds nullable dimension/entity. Runtime uses omit_none=True for graph and YAML serialization, replace preserves other fields.
- Initial sandbox test could not write standard ~/.dbt-osmosis log; reran with approved escalation. No fixture mutation in original checkout.

## Blockers
No implementation blockers. Remote matrix/canary runs remain in progress or queued; draft PR is not merge-ready until its current-head checks finish.

## Evidence
All local checks used Python 3.13.14 on macOS, isolated temporary environments and fixture copies.

| Check | Supported Core 1.11.17 / DuckDB 1.10.1 | Canary Core 1.12.7 / DuckDB 1.11.1 |
| --- | --- | --- |
| `pytest -q tests/test_yaml_inheritance.py` | 15 passed, 19.04s | 15 passed, 12.89s |
| `pytest -q` | 979 passed, 11 skipped, 563.04s | 979 passed, 11 skipped, 555.92s |
| `bash demo_duckdb/integration_tests.sh` | passed | passed |
| `basedpyright --outputjson` | 0 errors, 1980 warnings | 0 errors, 1981 warnings |
| `uv --no-config pip check` | passed | passed |

Both full suites skip five Azure SDK tests (extra absent), three source-node tests (fixture has no sources), and three property-accessor fixture tests. Azure's independent install/import smoke passes; these are not full Azure behavior results. Local full suites cover two Core versions on one Python/platform; the remaining support matrix is delegated to GitHub CI.

Extracted the exact pip-install-smoke shell block from the modified workflow and ran it with Python 3.13 and a temporary RUNNER_TEMP. All six clean pip environments passed install, pip check, CLI help and expected-module imports: base, openai, azure, workbench, duckdb, proxy. DuckDB resolved 1.11.1 and satisfied installed metadata `<2,>=1`; proxy resolved mysql-mimic 3.0.5 and ran after DuckDB. Every smoke used Core 1.11.17. No repository-root test.db or demo_duckdb/target was created.

Repository-wide Ruff, changed-file Ruff formatting, `git diff --check` and `uv lock --check` passed. Independent review passed. No runtime source or dependency declaration changed; README's audited 1.8–1.11 range remains intact.

Draft PR: https://github.com/z3z1ma/dbt-osmosis/pull/453. Code commit: 4ec41bd02fa5a1c109cf9992064bfdf7ddd867cb. GitHub run 43873451309 has passed pip smokes, both docs builds and lockfile check; supported matrix still running at observation. Lint run 43873513276 passed. Manual run 43873531338 was dispatched to include the future canary; remote results were not treated as complete local evidence.

Local raw logs: `/tmp/osmosis-{supported,canary}-{full,targeted,integration}.log` (canary targeted is `/tmp/osmosis-targeted.log`), `/tmp/osmosis-pip-smokes.log`, `/tmp/osmosis-{supported,canary}-types.json`, `/tmp/osmosis-lock.log`.


## Review
Pass. Independent reviewer found no actionable issues. Confirmed installed requirement bounds, strict column comparison, scalar/structured semantic preservation, explicit opt-in inheritance, and cache invalidation before persisted YAML readback. No production behavior or audited-support change.

## Retrospective
Unpinned extra smokes should validate distribution requirements rather than copy a matrix adapter minor. dbt serialization assertions should retain exactness while recording new nullable fields; pair schema-shape fixes with non-null round-trip regressions. Fresh environments exposed both issues without changing dependency contracts. Existing runtime omission of nulls and preservation of extra fields made a production change unnecessary.

### Verification in progress
- Canary targeted inheritance suite: 15 passed in 12.89s, including four scalar/structured preservation and opt-in inheritance cases.
- Ruff check (whole repository), changed-file formatting, diff whitespace and uv lock --check pass.
- basedpyright: supported Core 1.11.17 / DuckDB 1.10.1: 0 errors, 1980 warnings; canary Core 1.12.7 / DuckDB 1.11.1: 0 errors, 1981 warnings. Both use adapters 1.24.5 and pass uv pip check.
- Independent reviewer: pass, no actionable findings. Verified requirement boundary rejects 0.9/2.0 and accepts 1.0/1.10.1/1.11.1; YAML commit invalidates cache before readback; strict equality still detects unexpected non-null fields. Full suites and six pip smokes remain in progress.

### Closure
- Full supported/canary suites and all independent pip smokes completed successfully; no implementation follow-up required.
- Initial push used the active work account and received GitHub 403 without executing. Personal-account push succeeded without changing global account selection.
- Final evidence-only commit updates this record. Current remote check status is reported separately in the PR and handoff; no merge or release performed.
