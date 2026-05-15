# Honor Output Case Settings In Column Comparisons

ID: ticket:20260515-case-config-column-comparison
Type: Ticket
Status: closed
Created: 2026-05-15
Updated: 2026-05-15
Risk: medium - the change affects shared column comparison behavior in YAML transforms and schema diff reporting.
Priority: high - configured output casing can currently produce false diffs or ineffective database-order sorting.

## Summary

Column comparison code does not consistently honor `output-to-upper` or `output-to-lower`. `inject_missing_columns`, `remove_columns_not_in_database`, and `synchronize_data_types` already handle configured case conversion, but `sort_columns_as_in_database` and `SchemaDiff.compare_node()` compare normalized names without applying the same case-aware matching. The bounded result is that database-order sorting and schema diff treat case-only differences as the same logical column when output case conversion is configured, while still detecting true additions and removals.

This matters for adapters such as Postgres that introspect lowercase names when a user intentionally configures dbt-osmosis to write uppercase YAML names.

## Related Records

- `AGENTS.md` - describes the transform and diff surfaces and warns that node selection and transform behavior should stay in core.
- `src/dbt_osmosis/core/AGENTS.md` - requires `SettingsResolver`/`resolve_setting` patterns and core cache/config discipline.

## Scope

Likely files: `src/dbt_osmosis/core/transforms.py`, `src/dbt_osmosis/core/diff.py`, `tests/core/test_transforms.py`, and `tests/core/test_diff.py`. The fix should use the repo's existing `resolve_setting()` and `normalize_column_name()` patterns rather than inventing a parallel normalization stack.

Do not change Snowflake quoted identifier semantics, do not weaken true add/remove detection, and do not broaden this into a full column matching redesign.

## Acceptance

- ACC-001: `sort_columns_as_in_database()` orders uppercased or lowercased YAML column keys according to database order when the corresponding output case setting is active.
  - Evidence: Add a focused unit test for a Postgres-like context where DB keys are lowercase, YAML keys are uppercase, and database order differs from the current YAML order.
  - Audit: Review that true missing DB positions still sort after known DB columns.

- ACC-002: `SchemaDiff.compare_node()` does not emit `ColumnAdded` and `ColumnRemoved` for case-only differences introduced by configured output case conversion.
  - Evidence: Add a focused diff test where YAML has `ZEBRA`, DB introspection has `zebra`, `output_to_upper=True`, and no changes are reported.
  - Audit: Review that real added and removed columns still produce changes under the same settings.

- ACC-003: Existing case-sensitive behavior remains intact when neither output case setting is active.
  - Evidence: Run targeted transform and diff tests plus a focused regression test for the no-case-conversion path.
  - Audit: Review compatibility with the existing Snowflake normalization behavior in `normalize_column_name()`.

## Current State

Closed. Implementation and focused verification are complete under `packet:20260515T085607Z-column-comparison-semantic-updates`, with parent review and final repo verification complete.

Observed during scan:

- A direct reproduction of `sort_columns_as_in_database()` with Postgres-style lowercase DB keys and uppercase YAML keys left the YAML order as `['B', 'A']` instead of database order `['A', 'B']`.
- A direct reproduction of `SchemaDiff.compare_node()` with YAML `ZEBRA`, DB `zebra`, and `output_to_upper=True` produced `ColumnAdded('zebra')` and `ColumnRemoved('ZEBRA')`.

Verification after the fix:

- `uv run pytest tests/core/test_transforms.py tests/core/test_diff.py` passed with 47 passed and 12 warnings.
- `uv run ruff check src/dbt_osmosis/core/transforms.py src/dbt_osmosis/core/diff.py tests/core/test_transforms.py tests/core/test_diff.py` passed.
- Final repo verification passed: `uv run pytest` with 943 passed and 11 skipped, and `uv run ruff check`.
- Final type-check observation: `uv run basedpyright` reported 0 errors and existing warning-only output.

## Journal

- 2026-05-15: Created ticket with Status `open` from repo bug scan evidence. Automated baseline at creation time: `uv lock --check` passed, `uv run ruff check` passed, `uv run dbt parse --project-dir demo_duckdb --profiles-dir demo_duckdb -t test` passed, `uv run pytest` passed with 933 passed and 11 skipped, and `uv run basedpyright` reported 0 errors with existing warnings.
- 2026-05-15: Activated for Ralph execution via `packet:20260515T085607Z-column-comparison-semantic-updates`.
- 2026-05-15: Added focused tests proving DB-order sorting honors configured output casing, `SchemaDiff.compare_node()` suppresses configured case-only add/remove noise, real add/remove changes still surface, and no-output-case behavior remains case-sensitive. Red evidence: `uv run pytest tests/core/test_transforms.py tests/core/test_diff.py` failed in the new sort and diff tests before implementation.
- 2026-05-15: Implemented case-aware comparison for `sort_columns_as_in_database()` and `SchemaDiff.compare_node()` using existing `resolve_setting()` and `normalize_column_name()` patterns. Green evidence: `uv run pytest tests/core/test_transforms.py tests/core/test_diff.py` passed with 47 passed and 12 warnings; `uv run ruff check src/dbt_osmosis/core/transforms.py src/dbt_osmosis/core/diff.py tests/core/test_transforms.py tests/core/test_diff.py` passed.
- 2026-05-15: Parent review closed the ticket after final verification: `uv run pytest` passed with 943 passed and 11 skipped, `uv run ruff check` passed, and `uv run basedpyright` reported 0 errors with existing warnings.
