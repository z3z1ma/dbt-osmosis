---
id: evidence:c10diff31-schema-diff-validation
kind: evidence
status: recorded
created_at: 2026-05-05T00:46:02Z
updated_at: 2026-05-05T00:46:02Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  initiative:
    - initiative:dbt-110-111-hardening
  ticket:
    - ticket:c10diff31
  packets:
    - packet:ralph-ticket-c10diff31-20260505T003927Z
external_refs: {}
---

# Summary

Observed red/green and parent validation for `ticket:c10diff31` schema diff type-comparison and rename-detection behavior. The observations cover conservative type equality normalization, preservation of original type strings for real changes, deterministic one-to-one rename matching, and local static checks.

# Procedure

Observed at: 2026-05-05T00:46:02Z

Source state: branch `loom/dbt-110-111-hardening`, baseline and integration commit `4e21bdadff2c7ebb9288182cf86ba2de6ba53fe4`, with uncommitted `ticket:c10diff31` implementation/record diff.

Procedure: Ralph child added focused tests, observed red failures, implemented the bounded changes, and ran focused validation. Parent reviewed the diff and reran focused validation, formatting checks, Ruff, whitespace checks, and basedpyright on the touched core file.

Expected result when applicable: tests should fail before implementation for current case-only/whitespace-only type noise, duplicate rename matching, and ambiguous rename ordering; after implementation the focused diff tests, Ruff format/check, whitespace checks, and changed-source basedpyright should pass with zero errors.

Actual observed result: child reported the initial focused red command returned `4 failed, 4 passed, 12 deselected, 2 warnings` for expected type-noise and rename-detection failures. Parent post-child validation returned `20 passed, 2 warnings` for `tests/core/test_diff.py`, Ruff format/check passed, whitespace check passed, and basedpyright reported `errorCount: 0` with 12 existing warnings in `src/dbt_osmosis/core/diff.py`.

Procedure verdict / exit code: pass after implementation and parent validation.

# Artifacts

- Child red command from Ralph packet: `uv run pytest tests/core/test_diff.py -q -k "type_difference or rename"` -> expected red, `4 failed, 4 passed, 12 deselected, 2 warnings`.
- Expected red failures observed by child:
  - `test_schema_diff_rename_matches_added_columns_one_to_one` returned 2 renames for one added candidate.
  - `test_schema_diff_rename_output_is_deterministic_for_ambiguous_candidates` depended on added-list order.
  - `test_schema_diff_type_difference_ignores_case_only_changes` emitted `ColumnTypeChanged` for `VARCHAR` versus `varchar`.
  - `test_schema_diff_type_difference_ignores_whitespace_only_changes` emitted `ColumnTypeChanged` for `DECIMAL(10, 2)` versus `decimal(10,2)`.
- Parent focused test command: `uv run pytest tests/core/test_diff.py -q` -> `20 passed, 2 warnings in 17.22s`.
- Parent formatting check: `uv run ruff format --check src/dbt_osmosis/core/diff.py tests/core/test_diff.py` -> `2 files already formatted`.
- Parent lint/whitespace command: `uv run ruff check src/dbt_osmosis/core/diff.py tests/core/test_diff.py && git diff --check` -> `All checks passed!`.
- Parent type check: `uv run basedpyright --outputjson src/dbt_osmosis/core/diff.py` -> JSON summary `errorCount: 0`, `warningCount: 12`.
- Diff summary before evidence creation: `3 files changed, 170 insertions(+), 10 deletions(-)` across ticket/source/tests, plus the new Ralph packet and this evidence record.
- Code observation: `src/dbt_osmosis/core/diff.py` now skips `ColumnTypeChanged` when `_normalize_comparable_type()` matches and reserves each matched added rename candidate.
- Test observation: `tests/core/test_diff.py` now covers case-only type equality, whitespace-only type equality, original type-string preservation, one-to-one rename matching, and deterministic ambiguous rename output.

# Supports Claims

- `ticket:c10diff31#ACC-001` — red evidence showed case-only type differences emitted noise before the fix; green tests verify they no longer emit `ColumnTypeChanged`.
- `ticket:c10diff31#ACC-002` — red evidence showed whitespace-only type differences emitted noise before the fix; green tests verify they no longer emit `ColumnTypeChanged` for the covered whitespace-normalization case.
- `ticket:c10diff31#ACC-003` — red evidence showed two removed columns could map to one added column; green tests verify only one rename is emitted for a single added candidate.
- `ticket:c10diff31#ACC-004` — red evidence showed ambiguous rename output depended on caller list order; green tests verify stable output for the covered ambiguous candidate orderings.
- `ticket:c10diff31#ACC-005` — `tests/core/test_diff.py` now contains focused regression coverage for normalized type comparison and one-to-one deterministic rename matching.
- `initiative:dbt-110-111-hardening#OBJ-006` — focused tests and static checks support more trustworthy user-facing schema diff output.

# Challenges Claims

None - the observed post-implementation commands matched the expected result for the scoped claims.

# Environment

Commit: baseline `4e21bdadff2c7ebb9288182cf86ba2de6ba53fe4` with uncommitted `ticket:c10diff31` diff

Branch: `loom/dbt-110-111-hardening`

Runtime: OpenCode tool session; `uv` project environment

OS: darwin

Relevant config: `pyproject.toml`, `src/dbt_osmosis/core/diff.py`, `tests/core/test_diff.py`

External service / harness / data source when applicable: local filesystem and local test/type/lint commands only; no remote CI observed for this evidence

# Validity

Valid for: the uncommitted `ticket:c10diff31` implementation diff and local validation commands listed above.

Fresh enough for: ticket review, critique, and acceptance disposition for the named schema diff claims.

Recheck when: source, tests, dependency versions, fuzzy matching behavior, dbt metadata fixtures, or schema diff semantics change.

Invalidated by: a later implementation diff that changes type comparison, type-change emission, rename matching, or the focused tests without rerunning equivalent validation.

Supersedes / superseded by: none.

# Limitations

- The evidence is local-only and does not include GitHub Actions results for the eventual commit.
- Type normalization is intentionally conservative; this evidence does not support adapter alias equivalence such as `string` versus `varchar`.
- Rename matching remains greedy deterministic matching; this evidence does not prove globally optimal fuzzy assignment across all ambiguous candidate sets.
- Existing basedpyright warnings remain; this evidence only records that changed-source error count is zero.

# Result

The observed validation supports the implemented schema diff behavior for conservative type comparison and deterministic one-to-one rename matching under the local test/static-check surface.

# Interpretation

The evidence is sufficient to support review and acceptance evaluation for the scoped ticket claims if the ticket consumes the local-only and greedy-matching limitations. It does not by itself close the ticket, prove remote CI, or define broader adapter type alias semantics.

# Related Records

- ticket:c10diff31
- packet:ralph-ticket-c10diff31-20260505T003927Z
- initiative:dbt-110-111-hardening
