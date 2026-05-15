# Ralph Packet: Column Comparison And Semantic Update Fixes

ID: packet:20260515T085607Z-column-comparison-semantic-updates
Type: Ralph Packet
Status: consumed
Created: 2026-05-15
Updated: 2026-05-15
Branch: main
Context Style: hybrid
Mode: execution
Worker: subagent
Verification Posture: test-first, then focused regression proof

## Tickets

- `ticket:20260515-case-config-column-comparison`
- `ticket:20260515-semantic-analysis-column-replace`

## Objective

Fix two core bugs that overlap in the transform/diff surface:

1. Make database-order sorting and schema diff compare column names case-aware when `output-to-upper` or `output-to-lower` is configured.
2. Make `apply_semantic_analysis()` accumulate description, tag, and meta replacements before writing the final column object.

## Read Scope

- `AGENTS.md`
- `src/dbt_osmosis/core/AGENTS.md`
- `src/dbt_osmosis/core/transforms.py`
- `src/dbt_osmosis/core/diff.py`
- `src/dbt_osmosis/core/introspection.py`
- `tests/core/test_transforms.py`
- `tests/core/test_diff.py`

## Write Scope

- `src/dbt_osmosis/core/transforms.py`
- `src/dbt_osmosis/core/diff.py`
- `tests/core/test_transforms.py`
- `tests/core/test_diff.py`
- `.loom/tickets/20260515-case-config-column-comparison.md`
- `.loom/tickets/20260515-semantic-analysis-column-replace.md`
- this packet

## Constraints

- Workers are not alone in the codebase. Do not revert edits by others; adjust your implementation to fit any concurrent changes.
- Use existing `resolve_setting()` and `normalize_column_name()` patterns.
- Do not introduce PyYAML or bypass schema helpers.
- Do not weaken true add/remove detection in schema diff.
- Preserve tag/meta precedence in semantic analysis: existing tags are not duplicated, and existing meta keys win.
- Keep per-column semantic-analysis failure handling intact.

## Acceptance

- `sort_columns_as_in_database()` respects database order for configured uppercase/lowercase YAML output.
- `SchemaDiff.compare_node()` emits no add/remove changes for configured case-only differences, while still detecting real added/removed columns.
- `apply_semantic_analysis()` preserves generated description, merged tags, and merged meta on the same final `ColumnInfo`.

## Verification Commands

- `uv run pytest tests/core/test_transforms.py tests/core/test_diff.py`
- `uv run ruff check src/dbt_osmosis/core/transforms.py src/dbt_osmosis/core/diff.py tests/core/test_transforms.py tests/core/test_diff.py`

## Stop Conditions

- Stop and report if case-aware comparison requires changing adapter-specific normalization beyond existing `normalize_column_name()` semantics.
- Stop and report if semantic-analysis tests require real network or real OpenAI dependencies.

## Output Contract

Update this packet with a `## Worker Report` section containing changed files, commands run, results, and unresolved risks. Update both ticket journals with concise implementation and verification evidence.

## Worker Report

Outcome: stop

Changed files:

- `src/dbt_osmosis/core/transforms.py`
- `src/dbt_osmosis/core/diff.py`
- `tests/core/test_transforms.py`
- `tests/core/test_diff.py`
- `.loom/tickets/20260515-case-config-column-comparison.md`
- `.loom/tickets/20260515-semantic-analysis-column-replace.md`
- `.loom/packets/ralph/20260515T085607Z-column-comparison-semantic-updates.md`

Implementation:

- `sort_columns_as_in_database()` now resolves `output-to-upper` and `output-to-lower`, uses `normalize_column_name()` for adapter-aware comparison, and lowercases comparison keys only when configured output casing makes matching intentionally case-insensitive. Unknown DB positions still sort after known DB columns.
- `SchemaDiff.compare_node()` now uses configured output casing to compare YAML and database names case-insensitively only for those settings, suppressing configured case-only add/remove noise while preserving true add/remove changes and no-case-conversion sensitivity.
- `apply_semantic_analysis()` now accumulates description, tag, and meta replacements in an `updated_column` before one final writeback, preserving tag de-duplication, existing-meta precedence, and per-column failure handling.

Commands run:

- Red check before implementation: `uv run pytest tests/core/test_transforms.py tests/core/test_diff.py` failed as expected with 5 failing tests in the new coverage: two DB-order sorting case-output tests, one semantic-analysis accumulation test, and two schema-diff case-output tests.
- Green check after implementation: `uv run pytest tests/core/test_transforms.py tests/core/test_diff.py` passed with 47 passed and 12 warnings.
- Lint check: `uv run ruff check src/dbt_osmosis/core/transforms.py src/dbt_osmosis/core/diff.py tests/core/test_transforms.py tests/core/test_diff.py` passed with `All checks passed!`.

Records changed:

- Updated both ticket journals with red/green evidence and moved both tickets to `Status: review` because implementation and focused verification are complete, with review/audit as the next honest step before closure.

Unverified or unresolved risks:

- No full-suite pytest, pyright, or external audit pass was run in this packet.
- Existing test warnings remain: Click parser deprecation warnings and `datetime.utcnow()` deprecation warnings in diff tests.

## Parent Review

Outcome: accepted

Final verification from the main workspace:

- `uv run pytest` passed with 943 passed and 11 skipped.
- `uv run ruff check` passed.
- `uv run basedpyright` reported 0 errors with existing warning-only output.

Both referenced tickets were closed after parent review.
