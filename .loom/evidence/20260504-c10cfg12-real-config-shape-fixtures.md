---
id: evidence:c10cfg12-real-config-shape-fixtures
kind: evidence
status: recorded
created_at: 2026-05-04T06:40:06Z
updated_at: 2026-05-04T06:47:21Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  tickets:
    - ticket:c10cfg12
    - ticket:c10col01
    - ticket:c10meta02
  packets:
    - packet:ralph-ticket-c10cfg12-20260504T063203Z
external_refs: {}
---

# Summary

Observed real dbt-parsed config-shape fixture tests for `ticket:c10cfg12`, including node `config.meta`, node `config.extra`, `unrendered_config`, column `config.meta`, column `config.tags`, `SettingsResolver`, `PropertyAccessor`, and an adapter-backed missing-column injection/sync path. This evidence supports the ticket's fixture coverage claims; it does not decide acceptance or closure.

# Procedure

Observed at: 2026-05-04T06:47:21Z
Source state: branch `loom/dbt-110-111-hardening`, base commit `a92a225f0d0f616bc6b7d41788a5542c22e7bc9d`, with the `ticket:c10cfg12` working-tree diff applied.
Procedure: Reviewed Ralph child output, inspected `tests/core/test_real_config_shapes.py`, checked source fixture artifacts before and after tests, ran installed dbt version observation, ran focused real config-shape tests, ran broader config/property/transform tests, ran targeted `pre-commit`, ran `git diff --check`, addressed critique precondition coverage, and reran focused/broader tests plus hooks.
Expected result when applicable: Test-first red check should fail before fixture helper implementation; final tests should pass on a real parsed temp dbt fixture without creating repo-root `test.db` or source `demo_duckdb/target/manifest.json`.
Actual observed result: Packet records the expected red failures. Parent rerun observed focused and broader tests passing, no source artifact leaks, targeted hooks passing, and no whitespace errors.
Procedure verdict / exit code: mixed red/green sequence. Final green checks observed exit 0 for focused pytest, broader pytest, targeted `pre-commit`, and `git diff --check`.

# Artifacts

Red observations recorded in `packet:ralph-ticket-c10cfg12-20260504T063203Z`:

- `uv run pytest tests/core/test_real_config_shapes.py -q` returned `2 failed in 0.19s` before fixture helper implementation.
- Both tests failed with `NameError: name '_parsed_config_shape_fixture' is not defined`, proving the real parsed-fixture tests were written before the fixture support existed.

Final green observations:

- Pre-check artifact guards returned `No files found` for repo-root `test.db` and `demo_duckdb/target/manifest.json`.
- `uv run dbt --version` reported dbt-core `1.10.20` and dbt-duckdb `1.10.0`.
- Critique-driven patch added a precondition assertion that `warehouse_only_col` is absent from the parsed manifest before `inject_missing_columns()` runs, strengthening the adapter-backed missing-column injection claim.
- `uv run pytest tests/core/test_real_config_shapes.py -q` returned `2 passed in 3.65s` after the critique-driven patch.
- `uv run pytest tests/core/test_config_resolution.py tests/core/test_settings_resolver.py tests/core/test_property_accessor.py tests/core/test_transforms.py tests/core/test_real_config_shapes.py -q` returned `102 passed, 3 skipped in 14.88s` after the critique-driven patch.
- The three skips were existing `tests/core/test_property_accessor.py` demo fixture setup skips, not new `c10cfg12` skips.
- Post-test artifact guards again returned `No files found` for repo-root `test.db` and `demo_duckdb/target/manifest.json`.
- `pre-commit run --files tests/core/test_real_config_shapes.py .loom/evidence/20260504-c10cfg12-real-config-shape-fixtures.md .loom/packets/ralph/20260504T063203Z-ticket-c10cfg12-iter-01.md .loom/tickets/20260503-c10cfg12-add-real-dbt-config-shape-fixtures.md` passed check-ast, end-of-file, trailing whitespace, private-key detection, debug-statements, ruff-format, ruff, gitleaks, and applicable hooks.
- `git diff --check` produced no output.

# Supports Claims

- ticket:c10cfg12#ACC-001: real parsed fixture exposes dbt-osmosis options under node `config.meta` and column `config.meta`; focused tests assert the manifest fields and resolver/property access.
- ticket:c10cfg12#ACC-002: focused tests assert actual `node.config.meta`, `node.config.extra`, and `node.unrendered_config` values from dbt parse.
- ticket:c10cfg12#ACC-003: focused tests assert actual column `config.tags` and `config.meta` without mocks.
- ticket:c10cfg12#ACC-004: tests skip only for dbt-core `<1.10`; dbt 1.10 local execution passed and CI matrix will exercise 1.10/1.11 while older rows remain honest skips.
- ticket:c10cfg12#ACC-005: packet records the test-first red state and the final green state.
- ticket:c10cfg12#ACC-006: converted follow-up gaps from `ticket:c10col01#ACC-005` and `ticket:c10meta02#ACC-005` are locally supported by the new fixture tests; dbt 1.11 closure still needs post-commit matrix CI or equivalent evidence.
- ticket:c10col01#ACC-005: local real DuckDB adapter-backed missing-column injection path proves injected columns retain `ColumnInfo.config` and sync-serialize without leaking empty config; dbt 1.11 closure still needs post-commit matrix CI or equivalent evidence.
- ticket:c10meta02#ACC-005: local real dbt-parsed column `config.meta` / `config.tags` shape is asserted and consumed through `SettingsResolver` and `PropertyAccessor`; dbt 1.11 closure still needs post-commit matrix CI or equivalent evidence.
- initiative:dbt-110-111-hardening#OBJ-001, #OBJ-002, #OBJ-004: local fixture coverage now exercises actual parsed dbt config shapes without source-tree generated artifacts.

# Challenges Claims

None - the final observed checks matched the expected post-fix results for the cited claims.

# Environment

Commit: base `a92a225f0d0f616bc6b7d41788a5542c22e7bc9d` plus uncommitted `ticket:c10cfg12` diff.
Branch: `loom/dbt-110-111-hardening`
Runtime: local `uv run` environment; `uv run dbt --version` reported dbt-core `1.10.20` and dbt-duckdb `1.10.0`.
OS: macOS Darwin.
Relevant config: `tests/core/test_real_config_shapes.py`, `tests/support.py`, `demo_duckdb`, `tests/core/test_config_resolution.py`, `tests/core/test_settings_resolver.py`, `tests/core/test_property_accessor.py`, and `tests/core/test_transforms.py` from the reviewed source state.
External service / harness / data source when applicable: no production service exercised; dbt commands ran against isolated local DuckDB temp projects.

# Validity

Valid for: local dbt 1.10 parsed-fixture behavior and the repository's real parsed config-shape tests in the observed source state.
Fresh enough for: mandatory critique and local acceptance review of `ticket:c10cfg12` before post-commit CI.
Recheck when: dbt version matrix, dbt-duckdb adapter mapping, fixture helper behavior, settings/property/inheritance code, missing-column injection, sync serialization, or `demo_duckdb` changes.
Invalidated by: source changes after this evidence that alter real config-shape tests or implementation, failed post-commit CI for the same claims, or dbt/DuckDB behavior changes that make the fixture parse differently.
Supersedes / superseded by: Supplements the exact parsed-fixture gaps recorded in `ticket:c10col01#ACC-005` and `ticket:c10meta02#ACC-005` with local dbt 1.10 evidence; should be supplemented by post-commit CI evidence before final closure.

# Limitations

- Local execution observed dbt-core `1.10.20`, not dbt-core `1.11.x`; final acceptance should cite post-commit matrix CI for dbt 1.11 coverage.
- dbt-core `<1.10` intentionally skips the new shape assertions because the fixture targets the dbt 1.10+ column config namespace.
- The implementation is test-only and does not prove unrelated resolver behavior outside the asserted config-shape surfaces.

# Result

The observed checks showed that `ticket:c10cfg12` has real parsed dbt fixture coverage for dbt 1.10 config-shape fields, resolver/property accessor consumption, and adapter-backed missing-column injection/sync serialization, with no source fixture artifact leaks.

# Interpretation

The evidence supports moving `ticket:c10cfg12` to mandatory critique and post-commit CI validation. It does not by itself close the ticket or replace CI evidence for dbt 1.11 matrix coverage.

# Related Records

- ticket:c10cfg12
- ticket:c10col01
- ticket:c10meta02
- packet:ralph-ticket-c10cfg12-20260504T063203Z
- research:dbt-110-111-api-surfaces
- evidence:oracle-backlog-scan
