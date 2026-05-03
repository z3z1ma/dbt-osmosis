---
id: evidence:c10col01-columninfo-config-red-green
kind: evidence
status: recorded
created_at: 2026-05-03T22:16:00Z
updated_at: 2026-05-03T22:16:00Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  ticket:
    - ticket:c10col01
  critique:
    - critique:c10col01-columninfo-config
  packet:
    - packet:ralph-ticket-c10col01-20260503T214308Z
    - packet:ralph-ticket-c10col01-20260503T215123Z
external_refs:
  github_issue: https://github.com/z3z1ma/dbt-osmosis/issues/352
  implementation_commit: https://github.com/z3z1ma/dbt-osmosis/commit/ba92d07d263090cd06eaa8939f1f727ed5420512
---

# Summary

Observed red/green and post-commit validation for ticket:c10col01. The evidence supports the local claims that injected missing columns keep dbt's `ColumnInfo.config` field, sync serialization succeeds, and empty classic/fusion-compatible `config` blocks do not leak into YAML-facing output.

# Procedure

Observed at: 2026-05-03T22:16:00Z

Source state: branch `loom/dbt-110-111-hardening`, commit `ba92d07d263090cd06eaa8939f1f727ed5420512`.

Procedure:

- Ralph iteration 1 ran `uv run pytest tests/core/test_transforms.py::test_inject_missing_columns_preserves_column_config_for_sync` before the production fix.
- Ralph iteration 1 removed the `ColumnInfo.config` deletion and reran the focused test and full transform test module.
- Ralph iteration 2 parameterized the regression test over `fusion_compat=False` and `fusion_compat=True`, then reran the focused test and full transform test module.
- Parent reran `uv run pytest tests/core/test_transforms.py` after commit `ba92d07d263090cd06eaa8939f1f727ed5420512`.

Expected result when applicable: the pre-fix focused regression should fail because injected columns have invalid missing `config`; the post-fix tests should pass and synced columns should omit empty `config` blocks.

Actual observed result:

- Pre-fix focused regression failed with `AttributeError: 'ColumnInfo' object has no attribute 'config'` at `sync_operations.py:81`.
- Post-fix focused regression passed.
- Post-coverage focused regression passed with `2 passed` after parameterizing over both output modes.
- Parent post-commit validation passed with `18 passed in 9.92s`.

Procedure verdict / exit code: mixed red/green as expected; final parent validation passed with exit code 0.

# Artifacts

- Command: `uv run pytest tests/core/test_transforms.py::test_inject_missing_columns_preserves_column_config_for_sync`
- Command: `uv run pytest tests/core/test_transforms.py`
- Final observed output excerpt:

```text
collected 18 items

tests/core/test_transforms.py ..................                         [100%]

============================== 18 passed in 9.92s ==============================
```

# Supports Claims

- ticket:c10col01#ACC-001: local implementation no longer deletes `ColumnInfo.config`.
- ticket:c10col01#ACC-002: injected missing columns serialize through `_sync_doc_section` without `AttributeError`.
- ticket:c10col01#ACC-003: empty `config` does not appear in YAML-facing synced output for classic or fusion-compatible mode.
- ticket:c10col01#ACC-004: focused regression coverage exists for the dbt `ColumnInfo` shape in the installed environment.

# Challenges Claims

None - the final observed tests passed. The red state challenged the pre-fix implementation and is captured as expected regression evidence.

# Environment

Commit: `ba92d07d263090cd06eaa8939f1f727ed5420512`

Branch: `loom/dbt-110-111-hardening`

Runtime: Python 3.13.9, pytest 8.3.5, local `uv` environment. Test output included a non-failing warning that `VIRTUAL_ENV=/Users/alexanderbutler/code_projects/personal/dbt-osmosis/.venv` did not match the sibling worktree `.venv` and was ignored.

OS: macOS / darwin.

Relevant config: `tests/core/test_transforms.py` unit-style mocked context; no external adapter-backed dbt 1.11 matrix was run.

External service / harness / data source when applicable: none.

# Validity

Valid for: local unit-level behavior in commit `ba92d07d263090cd06eaa8939f1f727ed5420512` for `inject_missing_columns()` and `_sync_doc_section` interaction.

Fresh enough for: critique and ticket support for ticket:c10col01#ACC-001 through ticket:c10col01#ACC-004.

Recheck when: `inject_missing_columns()`, `_sync_doc_section()`, dbt `ColumnInfo` serialization, or fusion-compatible YAML cleanup changes.

Invalidated by: a later change that reintroduces missing `config`, bypasses `_sync_doc_section`, or changes empty-config cleanup semantics.

Supersedes / superseded by: not superseded.

# Limitations

This evidence does not establish adapter-backed behavior under dbt 1.11.x, full YAML command behavior against `demo_duckdb`, or broader `config.meta` / `config.tags` compatibility. Those remain owned by ticket:c10ci06, ticket:c10cfg12, and ticket:c10meta02.

# Result

The observed red/green sequence reproduced the missing-config failure, the minimal fix removed that failure, and the final committed transform test module passed locally.

# Interpretation

The evidence supports local acceptance of ACC-001 through ACC-004 for ticket:c10col01. It does not by itself support closing ACC-005 or claiming full dbt 1.11 compatibility.

# Related Records

- ticket:c10col01
- critique:c10col01-columninfo-config
- packet:ralph-ticket-c10col01-20260503T214308Z
- packet:ralph-ticket-c10col01-20260503T215123Z
- initiative:dbt-110-111-hardening
