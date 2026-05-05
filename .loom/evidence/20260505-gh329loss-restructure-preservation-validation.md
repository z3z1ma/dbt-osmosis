---
id: evidence:gh329loss-restructure-preservation-validation
kind: evidence
status: recorded
created_at: 2026-05-05T06:51:54Z
updated_at: 2026-05-05T06:51:54Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  ticket:
    - ticket:gh329loss
  packets:
    - packet:ralph-ticket-gh329loss-20260505T062916Z
    - packet:ralph-ticket-gh329loss-20260505T064213Z
external_refs:
  github_issue: https://github.com/z3z1ma/dbt-osmosis/issues/329
  related_github_issue: https://github.com/z3z1ma/dbt-osmosis/issues/306
---

# Summary

Observed red/green and parent validation for preserving unmanaged top-level schema YAML content during restructure superseded-file cleanup.

# Procedure

Observed at: 2026-05-05T06:51:54Z
Source state: uncommitted diff on `loom/dbt-110-111-hardening` based on `db73d9e4a6fb2c28939d31aab611ca8d1f2f7021`.
Procedure: Ralph iteration 1 added a failing unmanaged superseded-file regression, implemented preservation, and ran focused tests. Oracle critique found missing-original-cache and dry-run/cache gaps. Ralph iteration 2 added failing regressions for those gaps, implemented fallback unfiltered loading, and reran focused validation. Parent reran focused pytest, Ruff, and whitespace checks.
Expected result when applicable: Superseded cleanup must preserve files with unmanaged root sections, delete truly empty superseded files, behave safely under missing original-cache and dry-run conditions, and avoid trailing whitespace.
Actual observed result: The new focused regressions failed before implementation per child report, then passed after implementation. Parent validation passed.
Procedure verdict / exit code: Pass; all parent commands exited 0.

# Artifacts

Child red observations:

- `uv run pytest tests/core/test_restructuring.py -q -k "unmanaged_top_level_content"` failed before the iteration 1 implementation because superseded YAML with `semantic_models` was deleted.
- `uv run pytest tests/core/test_restructuring.py -q -k "same_path_superseded"` failed before the iteration 1 implementation because same-path superseded cleanup removed the just-written managed section.
- `uv run pytest tests/core/test_restructuring.py -q -k "original_cache_missing or missing_original_cache"` failed before the iteration 2 implementation because missing original-cache unmanaged content could still be deleted or classified as fully superseded in dry-run.

Parent green commands:

```bash
uv run pytest tests/core/test_restructuring.py -q -k "superseded or original_cache_missing or missing_original_cache"
```

Observed result: `7 passed, 24 deselected, 2 warnings`.

```bash
uv run pytest tests/core/test_restructuring.py tests/core/test_schema.py -q -k "superseded or unknown_top_level or semantic_models"
```

Observed result: `10 passed, 49 deselected, 2 warnings`.

```bash
uv run ruff check src/dbt_osmosis/core/restructuring.py tests/core/test_restructuring.py
```

Observed result: `All checks passed!`.

```bash
git diff --check
```

Observed result: passed with no output.

# Supports Claims

- ticket:gh329loss#ACC-001: semantic-model superseded preservation regression passes.
- ticket:gh329loss#ACC-002: generic unmanaged-section logic and schema unknown top-level tests support arbitrary keys such as `databricks-tags`.
- ticket:gh329loss#ACC-003: existing empty superseded deletion tests remained green in the focused superseded run.
- ticket:gh329loss#ACC-004: same-path schema preservation tests remained green in the schema/restructure run.
- ticket:gh329loss#ACC-005: dry-run unmanaged superseded regression and missing-original-cache regression pass.

# Challenges Claims

None.

# Environment

Commit: uncommitted diff based on `db73d9e4a6fb2c28939d31aab611ca8d1f2f7021`.
Branch: `loom/dbt-110-111-hardening`.
Runtime: `uv run` project environment; warning observed that external `VIRTUAL_ENV` pointed at sibling checkout and was ignored.
OS: macOS Darwin.
Relevant config: focused pytest and Ruff checks only.
External service / harness / data source when applicable: GitHub issues #329 and #306 as external refs; no remote CI observed yet.

# Validity

Valid for: local source state and focused restructure/schema preservation behavior covered by the listed commands.
Fresh enough for: `ticket:gh329loss` parent acceptance review and mandatory critique.
Recheck when: restructure cleanup, schema reader/writer caches, parser top-level partitioning, or YAML dry-run behavior changes.
Invalidated by: code changes that affect `apply_restructure_plan()`, `_read_yaml()`, `_write_yaml()`, `_YAML_ORIGINAL_CACHE`, or unmanaged top-level section partitioning without rerunning the focused tests.
Supersedes / superseded by: N/A.

# Limitations

This evidence does not include full test suite or remote CI. It does not prove every possible unmanaged key shape; it proves the generic preserved-section code path plus concrete `semantic_models` and unknown top-level preservation coverage.

# Result

The observed implementation prevents the validated restructure data-loss path in focused local tests and keeps existing deletion/preservation behavior green.

# Interpretation

The evidence supports local acceptance of `ticket:gh329loss` after critique. It does not by itself establish remote CI health or release readiness.

# Related Records

- ticket:gh329loss
- critique:gh329loss-restructure-preservation-review
- packet:ralph-ticket-gh329loss-20260505T062916Z
- packet:ralph-ticket-gh329loss-20260505T064213Z
- initiative:issue-pr-zero
