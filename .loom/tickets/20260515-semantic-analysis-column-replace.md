# Preserve All Semantic Analysis Column Updates

ID: ticket:20260515-semantic-analysis-column-replace
Type: Ticket
Status: closed
Created: 2026-05-15
Updated: 2026-05-15
Risk: low - the bug is isolated to the optional semantic analysis transform and can be covered with mocked LLM tests.
Priority: medium - optional AI users can silently lose generated descriptions and tags when meta is also returned.

## Summary

`apply_semantic_analysis()` updates a column by calling `column_info.replace(...)` multiple times from the original `column_info` object. If semantic analysis returns a generated description, tags, and meta for the same column, later replacements overwrite earlier updates. The bounded result is that all semantic analysis updates for a column are accumulated before writing back to `node.columns[column_name]`.

This matters because the transform logs that semantic analysis was applied while silently discarding some of the applied fields.

## Related Records

- `AGENTS.md` - notes optional AI paths should fail clearly and should not assume real writeback/generation is already wired everywhere.
- `src/dbt_osmosis/core/AGENTS.md` - constrains transform changes under `src/dbt_osmosis/core/`.

## Scope

Likely files: `src/dbt_osmosis/core/transforms.py` and `tests/core/test_transforms.py`. The fix should stay inside `apply_semantic_analysis()` and its tests unless a tiny local helper improves clarity.

Do not change the LLM prompt contract, project-wide semantic analysis routing, or unrelated documentation suggestion behavior.

## Acceptance

- ACC-001: When semantic analysis returns a new description, tags, and meta for a column, the final `ColumnInfo` keeps all three updates.
  - Evidence: Add a focused unit test with mocked `analyze_column_semantics()`, `generate_semantic_description()`, and column knowledge graph behavior.
  - Audit: Review that the test would fail against the current stale-replace implementation.

- ACC-002: Existing tag and meta merge semantics remain intact: existing tags are not duplicated, and existing meta keys still take precedence over suggested meta.
  - Evidence: Add assertions to the focused test or a second small test for preservation/merge behavior.
  - Audit: Review that only the accumulated writeback is changed, not the precedence rules.

- ACC-003: Per-column failures still do not abort the whole transform.
  - Evidence: Run targeted transform tests, including any existing semantic-analysis failure path if present or a new focused test if needed.
  - Audit: Review that the existing `except Exception` continue behavior remains unchanged.

## Current State

Closed. Implementation and focused verification are complete under `packet:20260515T085607Z-column-comparison-semantic-updates`, with parent review and final repo verification complete.

Observed during scan:

- A direct mocked reproduction returned `{'description': '', 'tags': [], 'meta': {'semantic_type': 'identifier'}}` after semantic analysis supplied `description='new generated description'`, `tags=['primary_key']`, and `meta={'semantic_type': 'identifier'}`. The later meta replacement clobbered the earlier description and tags.

Verification after the fix:

- `uv run pytest tests/core/test_transforms.py tests/core/test_diff.py` passed with 47 passed and 12 warnings.
- `uv run ruff check src/dbt_osmosis/core/transforms.py src/dbt_osmosis/core/diff.py tests/core/test_transforms.py tests/core/test_diff.py` passed.
- Final repo verification passed: `uv run pytest` with 943 passed and 11 skipped, and `uv run ruff check`.
- Final type-check observation: `uv run basedpyright` reported 0 errors and existing warning-only output.

## Journal

- 2026-05-15: Created ticket with Status `open` from repo bug scan evidence. Automated baseline at creation time: `uv lock --check` passed, `uv run ruff check` passed, `uv run dbt parse --project-dir demo_duckdb --profiles-dir demo_duckdb -t test` passed, `uv run pytest` passed with 933 passed and 11 skipped, and `uv run basedpyright` reported 0 errors with existing warnings.
- 2026-05-15: Activated for Ralph execution via `packet:20260515T085607Z-column-comparison-semantic-updates`.
- 2026-05-15: Added focused mocked semantic-analysis tests proving one column keeps generated description, merged tags, and merged meta together, existing tag/meta precedence remains intact, and a per-column semantic-analysis exception does not abort updates to later columns. Red evidence: `uv run pytest tests/core/test_transforms.py tests/core/test_diff.py` failed in the new accumulation test before implementation.
- 2026-05-15: Updated `apply_semantic_analysis()` to accumulate description, tag, and meta replacements in one `ColumnInfo` before final writeback while preserving existing tag/meta merge rules and the per-column `except Exception` continue behavior. Green evidence: `uv run pytest tests/core/test_transforms.py tests/core/test_diff.py` passed with 47 passed and 12 warnings; `uv run ruff check src/dbt_osmosis/core/transforms.py src/dbt_osmosis/core/diff.py tests/core/test_transforms.py tests/core/test_diff.py` passed.
- 2026-05-15: Parent review closed the ticket after final verification: `uv run pytest` passed with 943 passed and 11 skipped, `uv run ruff check` passed, and `uv run basedpyright` reported 0 errors with existing warnings.
