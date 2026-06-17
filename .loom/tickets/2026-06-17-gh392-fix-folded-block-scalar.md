Status: done
Created: 2026-06-17
Updated: 2026-06-17
Parent: None
Depends-On: None

# Fix Folded Block Scalar Used for Multiline Strings (GitHub #392)

## Summary

dbt-osmosis writes multiline string descriptions using folded block scalar style (`>`) or its chomp-strip variant (`>-`/`|-`) instead of the literal block scalar style (`|`). The GitHub issue reports that users see `>-` where they expect `|`, meaning newlines in descriptions are lost or rendered as spaces when the YAML is later re-read by tools that do not preserve block chomping semantics.

Observed behavior: `uv run python -c "import io, yaml; from src.dbt_osmosis.core.schema.parser import create_yaml_instance; y = create_yaml_instance(); d = {'description': 'Line one.\nLine two.\nLine three.'}; o = io.StringIO(); y.dump(d, o); print(repr(o.getvalue()))"` returns `'description: |-\n  Line one.\n  Line two.\n  Line three.\n'` — the `|-` chomp-strip variant — rather than `description: |`.

The root cause is in `src/dbt_osmosis/core/schema/parser.py:172–174`. The `str_representer` function uses:
- `style=">"` for long single-line strings (correct — folded for word-wrapped descriptions)
- `style="|"` for strings with `newlines > 1`

The condition `newlines > 1` counts `splitlines()`, which returns a list of lines. A two-line string (`"foo\nbar"`) has `newlines == 2`, which is `> 1`, so it correctly uses `|`. But a string with exactly one newline that produces no trailing blank line may still get chomped down — this is a ruamel.yaml chomp normalization issue, not a style selection issue.

The concrete user-reported behavior is that **existing YAML files with `>-` block scalars are re-emitted as `>-`** (the chomp indicator is preserved by ruamel's round-trip mode), and when dbt-osmosis writes _new_ multiline descriptions it may strip the trailing newline, changing semantics.

The test at `tests/core/test_schema.py:632` (`test_yaml_read_write_preserves_managed_block_scalars_and_normalizes_quoted_keys_by_default`) already asserts `"description: >"` and `"description: |"` are preserved on round-trip. The test at line 721 (`test_yaml_read_write_formatter_contract_literalizes_quoted_multiline_managed_description`) asserts that quoted multi-line strings are converted to `|` on first write.

## Scope

In-scope:
- Confirm the exact reproduction case from issue #392 — whether the bug is in `str_representer` or in round-trip chomp normalization.
- If the representer uses `style=">"` for strings that have embedded newlines, fix the condition.
- If ruamel.yaml chomp behavior causes `>` to become `>-`, ensure the representer forces `|` explicitly.
- Add a regression test that directly exercises the reported scenario (long string with embedded newlines from issue #392).

Out-of-scope:
- Changing folded style behavior for long single-line strings (that is correct and tested).
- Modifying YAML indentation or width settings.
- Any changes to how specs, decisions, or other record types are written.

## Acceptance Criteria

- ACC-001: A multiline description containing `\n` characters is always written with literal block style (`|`) rather than folded style (`>` or `>-`).
  - Evidence: Add a unit test that creates a `create_yaml_instance()`, dumps a `description` value with embedded newlines, and asserts the output contains `description: |` and not `description: >`.
  - Audit: Confirm the fix does not regress `test_yaml_read_write_preserves_managed_block_scalars_and_normalizes_quoted_keys_by_default`.

- ACC-002: Existing files that already contain `>` folded scalars are preserved as-is on round-trip (ruamel.yaml preserves existing styles).
  - Evidence: The existing test at line 632 of test_schema.py continues to pass.
  - Audit: Run `uv run pytest tests/core/test_schema.py` and confirm all 29 tests pass.

- ACC-003: Long single-line strings (no embedded newlines) continue to use folded style (`>`).
  - Evidence: The existing test at line 202 (`test_yaml_string_representer_none_prefix_colon`) continues to pass.

## Implementation Notes

The relevant code is `src/dbt_osmosis/core/schema/parser.py:139–175`. The `str_representer` closure checks `newlines = len(data.splitlines())`. A string like `"foo\nbar"` has `newlines == 2` and correctly gets `|`. The issue may be that ruamel.yaml post-processes the chomping indicator. Look at whether `style="|"` being passed to `represent_scalar` is sufficient, or whether the dumper adds a `-` chomping modifier automatically.

Quickest investigation: add a print in a scratch test to see whether `style="|"` is what's being passed but the output shows `|-`.

## Progress and Notes

- 2026-06-17: Ticket opened from GitHub issue #392 scan. Baseline: `uv run pytest tests/core/test_schema.py` passes 29/29. The reported `|-` behavior was reproduced in a scratch run: `create_yaml_instance()` dumping `{'description': 'Line one.\nLine two.\nLine three.'}` yields `description: |-` not `description: |`. The condition `newlines > 1` should fire (3 lines → newlines=3), so the issue may be ruamel.yaml stripping the trailing newline and adding `-` chomp automatically.

## Blockers

None.

- 2026-06-17: Fix implemented. Root cause confirmed: ruamel.yaml adds `|-` (strip-chomp) when the string passed to `represent_scalar(..., style='|')` does not end with `\n`. Fixed in `src/dbt_osmosis/core/schema/parser.py` (lines 173-179) by normalizing the string to end with exactly one `\n` before handing it to the representer. Regression test `test_yaml_string_representer_uses_literal_not_folded_for_multiline` added to `tests/core/test_schema.py` covering 2-line, 3-line, and already-trailing-newline cases plus round-trip content verification. All 30 test_schema tests pass (was 29 before adding the new test). 60/61 restructuring tests pass (1 pre-existing skip unrelated to this fix). Ruff lint clean.
