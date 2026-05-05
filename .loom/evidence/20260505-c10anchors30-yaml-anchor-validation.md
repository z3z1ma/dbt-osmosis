---
id: evidence:c10anchors30-yaml-anchor-validation
kind: evidence
status: recorded
created_at: 2026-05-05T01:57:34Z
updated_at: 2026-05-05T04:05:03Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  tickets:
    - ticket:c10anchors30
  packets:
    - packet:ralph-ticket-c10anchors30-20260505T015001Z
    - packet:ralph-ticket-c10anchors30-20260505T020517Z
    - packet:ralph-ticket-c10anchors30-20260505T021703Z
    - packet:ralph-ticket-c10anchors30-20260505T022457Z
    - packet:ralph-ticket-c10anchors30-20260505T023255Z
    - packet:ralph-ticket-c10anchors30-20260505T024405Z
external_refs: {}
---

# Summary

Observed red/green validation for preserving cross-section YAML anchors, preserving unmanaged quote style, respecting default managed quote normalization, and parsing the anchor fixture with dbt.

# Procedure

Observed at: 2026-05-05T02:48:30Z

Source state: working tree on `loom/dbt-110-111-hardening` at baseline `2c3cf96e102360a2e09af6a4e4d9d4a892956276` plus the uncommitted `ticket:c10anchors30` implementation and Loom packet/ticket updates.

Procedure: Ralph child added failing regression tests across six iterations, implemented the fixes, and ran focused/full schema validation. Parent reran current-source verification after child return and ran a temporary dbt parse fixture for the anchor shape.

Expected result when applicable: Before implementation, focused anchor, unmanaged quote-style, managed quote-normalization, managed block-scalar, sequence item comment, boolean-like scalar, and formatter-contract tests fail against the relevant current behavior. After implementation, cross-section `x-common-tests: &common_tests` and managed `data_tests: *common_tests` remain linked and parseable, unmanaged quote style is preserved, default managed quote normalization remains intact for ordinary quoted scalars/keys and quoted sequence items, YAML boolean-like strings remain safely quoted, managed folded/literal block scalar styles remain intact, long and multiline managed quoted strings still use the normal formatter contract, sequence item comments remain intact, dbt parses the fixture, schema tests pass, formatting/lint/whitespace checks pass, and basedpyright reports zero errors.

Actual observed result: Expected red failures were observed before implementation. After implementation and parent verification, schema tests passed, Ruff checks passed, `git diff --check` passed, basedpyright returned zero errors, and the temporary dbt parse fixture exited 0.

Procedure verdict / exit code: mixed red/green evidence as expected; final current-source verification passed. basedpyright had `errorCount: 0` with warnings tolerated by current project policy.

# Artifacts

Child red evidence:

```text
uv run pytest tests/core/test_schema.py -q -k "cross_section_anchor"
1 failed, 20 deselected, 2 warnings in 0.22s
Expected failure: written YAML flattened the alias relationship; `x-common-tests: &common_tests` was missing and `data_tests` was expanded.

uv run pytest tests/core/test_schema.py -q -k "unmanaged_section_quote_style"
1 failed, 21 deselected, 2 warnings in 0.22s
Expected failure: unmanaged quoted scalars were emitted as plain scalars.

uv run pytest tests/core/test_schema.py -q -k "managed_quote"
Failed before iter-02 implementation as expected.
Expected failure: managed `description: "Test model"` remained quoted despite default `preserve_quotes=False`.

uv run pytest tests/core/test_schema.py -q -k "block_scalar or quoted_key"
1 failed, 23 deselected, 2 warnings before iter-03 implementation.
Expected failure: managed folded block scalar had been flattened to a quoted scalar string.

uv run pytest tests/core/test_schema.py -q -k "sequence_item_comment"
1 failed, 24 deselected before iter-04 implementation.
Expected failure: inline comments on managed quoted sequence items were missing after normalization.

uv run pytest tests/core/test_schema.py -q -k "boolean_like"
Failed before iter-05 implementation.
Expected failure: managed `flag: "on"` was emitted unquoted as `flag: on`.

uv run pytest tests/core/test_schema.py -q -k "formatter_contract"
2 failures before iter-06 implementation.
Expected failures: long quoted managed description did not emit folded style, and quoted newline managed description did not emit literal style.
```

Child green evidence:

```text
uv run pytest tests/core/test_schema.py -q -k "unmanaged_section_quote_style or cross_section_anchor or preserved_unknown_top_level or anchors"
4 passed, 18 deselected, 2 warnings in 0.10s

uv run pytest tests/core/test_schema.py -q
22 passed, 2 warnings in 0.46s

uv run ruff format src/dbt_osmosis/core/schema/reader.py src/dbt_osmosis/core/schema/writer.py tests/core/test_schema.py
3 files left unchanged

uv run ruff check src/dbt_osmosis/core/schema/reader.py src/dbt_osmosis/core/schema/writer.py tests/core/test_schema.py && git diff --check
All checks passed!

uv run basedpyright --outputjson src/dbt_osmosis/core/schema/reader.py src/dbt_osmosis/core/schema/writer.py
errorCount: 0, warningCount: 47, informationCount: 0

uv run pytest tests/core/test_schema.py -q -k "managed_quote or unmanaged_section_quote_style or cross_section_anchor"
3 passed, 20 deselected, 2 warnings

uv run pytest tests/core/test_schema.py -q
23 passed, 2 warnings

uv run basedpyright --outputjson src/dbt_osmosis/core/schema/reader.py src/dbt_osmosis/core/schema/writer.py
errorCount: 0, warningCount: 56, informationCount: 0

uv run pytest tests/core/test_schema.py -q -k "block_scalar or quoted_key or managed_quote or unmanaged_section_quote_style or cross_section_anchor"
4 passed, 20 deselected, 2 warnings

uv run pytest tests/core/test_schema.py -q
24 passed, 2 warnings

uv run basedpyright --outputjson src/dbt_osmosis/core/schema/reader.py src/dbt_osmosis/core/schema/writer.py
errorCount: 0, warningCount: 58, informationCount: 0

uv run pytest tests/core/test_schema.py -q -k "sequence_item_comment or block_scalar or managed_quote or unmanaged_section_quote_style or cross_section_anchor"
5 passed, 20 deselected, 2 warnings

uv run pytest tests/core/test_schema.py -q
25 passed, 2 warnings

uv run basedpyright --outputjson src/dbt_osmosis/core/schema/reader.py src/dbt_osmosis/core/schema/writer.py
errorCount: 0, warningCount: 57, informationCount: 0

uv run pytest tests/core/test_schema.py -q -k "boolean_like or sequence_item_comment or block_scalar or managed_quote or unmanaged_section_quote_style or cross_section_anchor"
6 passed, 20 deselected, 2 warnings

uv run pytest tests/core/test_schema.py -q
26 passed, 2 warnings

uv run pytest tests/core/test_schema.py -q -k "formatter_contract or boolean_like or sequence_item_comment or block_scalar or managed_quote or unmanaged_section_quote_style or cross_section_anchor"
8 passed, 20 deselected

uv run pytest tests/core/test_schema.py -q
28 passed
```

Parent current-source verification:

```text
uv run pytest tests/core/test_schema.py -q
28 passed, 2 warnings in 0.48s

uv run ruff format --check src/dbt_osmosis/core/schema/reader.py src/dbt_osmosis/core/schema/writer.py tests/core/test_schema.py
3 files already formatted

uv run ruff check src/dbt_osmosis/core/schema/reader.py src/dbt_osmosis/core/schema/writer.py tests/core/test_schema.py && git diff --check
All checks passed!

uv run basedpyright --outputjson src/dbt_osmosis/core/schema/reader.py src/dbt_osmosis/core/schema/writer.py
errorCount: 0, warningCount: 57, informationCount: 0

temporary dbt parse fixture with `x-common-tests: &common_tests` and managed `data_tests: *common_tests`
dbt=1.10.20, duckdb=1.10.0, exit 0
```

All `uv run` commands emitted the known local warning that `VIRTUAL_ENV=/Users/alexanderbutler/code_projects/personal/dbt-osmosis/.venv` does not match this worktree `.venv`; `uv` ignored it.

# Supports Claims

- `ticket:c10anchors30#ACC-001`: supports the implemented support policy for cross-section YAML anchor/alias preservation.
- `ticket:c10anchors30#ACC-002`: supports regression coverage for `x-common-tests: &common_tests` with managed `data_tests: *common_tests`.
- `ticket:c10anchors30#ACC-003`: supports that supported write/read round-trips keep aliases valid and parseable by ruamel, and that the same fixture is accepted by dbt parse.
- `ticket:c10anchors30#ACC-005`: supports that preserved unmanaged sections remain preserved, including explicit scalar quote style, while managed default quote normalization remains intact, YAML boolean-like scalar safety remains intact, managed folded/literal block scalars are not flattened, managed long/multiline string formatter behavior remains intact, and managed sequence item comments are retained.

# Challenges Claims

None - final current-source verification supports the scoped implementation claims. Red evidence intentionally challenged the pre-implementation behavior.

# Environment

Commit: `2c3cf96e102360a2e09af6a4e4d9d4a892956276` plus uncommitted `ticket:c10anchors30` changes

Branch: `loom/dbt-110-111-hardening`

Runtime: `uv run` project environment, Python 3.10 test environment from `.venv`

OS: macOS / Darwin

Relevant config: `tests/core/test_schema.py`, `src/dbt_osmosis/core/schema/reader.py`, `src/dbt_osmosis/core/schema/writer.py`

External service / harness / data source when applicable: none

# Validity

Valid for: the current uncommitted source state and the scoped schema reader/writer/test changes for `ticket:c10anchors30`, plus dbt parse compatibility for the temporary anchor fixture under dbt 1.10.20/duckdb 1.10.0.

Fresh enough for: parent reconciliation, recommended critique, and pre-commit/local acceptance review of `ticket:c10anchors30`.

Recheck when: schema reader/writer/parser code changes, tests are reformatted or rewritten, ruamel.yaml behavior changes, dbt schema YAML fixture behavior changes, or the branch is rebased/merged.

Invalidated by: failing schema tests, failing Ruff checks, basedpyright errors, changes that replace ruamel object identity for alias-bearing managed subtrees, or writer merge-order changes.

Supersedes / superseded by: none.

# Limitations

This evidence does not prove every possible YAML anchor topology. It specifically covers unmanaged top-level anchor definitions referenced from managed model column `data_tests`, preserved top-level unknown sections, explicit quote-style preservation during managed writes, default managed quote/key normalization for quoted scalars and sequence items, YAML boolean-like scalar safety, managed folded/literal block scalar preservation, managed long/multiline string formatter behavior, managed sequence item comment preservation, and dbt parse compatibility for this fixture. It does not prove behavior for callers that replace alias-bearing ruamel subtrees with newly constructed plain Python values.

# Result

The implementation produced the expected red failures before code changes and passed focused/full schema validation after code changes. Parent verification independently reproduced the final green state and observed dbt parse compatibility for the anchor fixture.

# Interpretation

The evidence supports the scoped support policy for cross-section anchors/aliases in normal `_read_yaml()`/`_write_yaml()` flows. It does not by itself close the ticket; the ticket still owns critique disposition and acceptance.

# Related Records

- ticket:c10anchors30
- packet:ralph-ticket-c10anchors30-20260505T015001Z
- packet:ralph-ticket-c10anchors30-20260505T020517Z
- packet:ralph-ticket-c10anchors30-20260505T021703Z
- packet:ralph-ticket-c10anchors30-20260505T022457Z
- packet:ralph-ticket-c10anchors30-20260505T023255Z
- packet:ralph-ticket-c10anchors30-20260505T024405Z
- initiative:dbt-110-111-hardening
