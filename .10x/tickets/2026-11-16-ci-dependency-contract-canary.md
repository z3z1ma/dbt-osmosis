Status: active
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
None currently; verification pending.

## Evidence
Pending.

## Review
Pending independent review.

## Retrospective
Pending.

### Verification in progress
- Canary targeted inheritance suite: 15 passed in 12.89s, including four scalar/structured preservation and opt-in inheritance cases.
- Ruff check (whole repository), changed-file formatting, diff whitespace and uv lock --check pass.
- basedpyright: supported Core 1.11.17 / DuckDB 1.10.1: 0 errors, 1980 warnings; canary Core 1.12.7 / DuckDB 1.11.1: 0 errors, 1981 warnings. Both use adapters 1.24.5 and pass uv pip check.
- Independent reviewer: pass, no actionable findings. Verified requirement boundary rejects 0.9/2.0 and accepts 1.0/1.10.1/1.11.1; YAML commit invalidates cache before readback; strict equality still detects unexpected non-null fields. Full suites and six pip smokes remain in progress.
