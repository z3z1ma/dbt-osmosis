---
id: evidence:c10ci06-main-ci-dbt18-test-fix
kind: evidence
status: recorded
created_at: 2026-05-03T23:00:07Z
updated_at: 2026-05-03T23:00:07Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  tickets:
    - ticket:c10ci06
  evidence:
    - evidence:c10ci06-ci-run-no-sync-fix
external_refs:
  main_tests_run: https://github.com/z3z1ma/dbt-osmosis/actions/runs/25293085888
  github_issue: https://github.com/z3z1ma/dbt-osmosis/issues/357
---

# Summary

Observed main CI run `25293085888` after landing the dbt 1.11 compatibility gate. The new dbt 1.10/1.11 gates passed, while dbt 1.8 rows failed on tests that assumed `ColumnInfo.config` exists. Recorded the version-aware test fix and local targeted validation under dbt 1.8 and dbt 1.11.

# Procedure

Observed at: 2026-05-03T23:00:07Z

Source state: `main` at `ff2b7da` for the CI observation, plus uncommitted test/ticket/evidence updates for the dbt 1.8 test compatibility fix.

Procedure:

- Watched GitHub Actions `Tests` run `25293085888` for `main`.
- Inspected failed logs with `gh run view 25293085888 --log-failed`.
- Updated `tests/core/test_inheritance_behavior.py` so config-meta/tag inheritance coverage skips when dbt's `ColumnInfo` has no `config` field.
- Updated `tests/core/test_transforms.py` so injected-column config preservation is asserted only when the dbt `ColumnInfo` type supports `config`, while YAML sync output remains asserted for all supported dbt versions.
- Installed `dbt-core~=1.8.0` and `dbt-duckdb~=1.8.0` into the sibling worktree venv and ran the failed test slice with `UV_NO_SYNC=1`.
- Installed `dbt-core~=1.11.0` and `dbt-duckdb~=1.10.1` and reran the same test slice with `UV_NO_SYNC=1`.

Expected result when applicable: dbt 1.8 should not fail tests for a `ColumnInfo.config` field it does not expose; dbt 1.10/1.11 should still exercise the config-field behavior.

Actual observed result: main CI passed all dbt 1.10 and dbt 1.11 rows and both latest-compat jobs, but dbt 1.8 rows failed on `test_config_tag_and_meta_inheritance` and `test_inject_missing_columns_preserves_column_config_for_sync`. After the test fix, the dbt 1.8 targeted run reported `2 passed, 1 skipped`, and the dbt 1.11 targeted run reported `3 passed`.

Procedure verdict / exit code: main CI failed before this fix; local dbt 1.8 and dbt 1.11 targeted checks passed after this fix.

# Artifacts

- `Tests` run `25293085888`: latest dbt 1.10 and 1.11 compatibility jobs passed.
- `Tests` run `25293085888`: main matrix dbt 1.10 and dbt 1.11 rows passed, including installed-version assertions, basedpyright, parse, pytest, and integration tests.
- Failed dbt 1.8 rows reported `AssertionError: assert 'config_pk' in []` and `AssertionError: assert False` for `hasattr(..., "config")`.
- Local dbt 1.8 targeted command output: `2 passed, 1 skipped`.
- Local dbt 1.11 targeted command output: `3 passed`.
- Ruff output for edited tests: `All checks passed!`.

# Supports Claims

- `ticket:c10ci06#ACC-003` - supports that the new dbt 1.10/1.11 gates execute successfully on main and that the remaining dbt 1.8 failures were test expectations for a field unavailable in dbt 1.8.
- `ticket:c10ci06#ACC-004` - supports the local matrix verification posture by reproducing the failed dbt 1.8 slice with `UV_NO_SYNC=1`.

# Challenges Claims

- `ticket:c10ci06#ACC-003` - main run `25293085888` still challenges full matrix acceptance until the dbt 1.8 test-compat fix is pushed and CI reruns.

# Environment

Commit: `ff2b7da` for observed main CI; uncommitted test/evidence updates for local checks.

Branch: delivery target `main`; local worktree branch `loom/dbt-110-111-hardening` used only as an isolated checkout.

Runtime: GitHub Actions Ubuntu for main CI; local macOS/darwin harness for targeted dbt 1.8 and 1.11 checks.

OS: GitHub Actions Ubuntu and local macOS/darwin.

Relevant config: `.github/workflows/tests.yml`, `tests/core/test_transforms.py`, `tests/core/test_inheritance_behavior.py`.

External service / harness / data source when applicable: GitHub Actions run `25293085888`.

# Validity

Valid for: classifying the first main CI result after the compatibility-gate landing and validating the dbt 1.8 test-compat fix locally.

Fresh enough for: committing the test compatibility fix and rerunning main CI.

Recheck when: dbt 1.8 support policy, `ColumnInfo` compatibility helpers, inheritance tests, transform tests, or CI matrix definitions change.

Invalidated by: a later test change that again assumes `ColumnInfo.config` exists across all supported dbt versions.

Supersedes / superseded by: should be superseded by the next main CI run after this test fix is pushed.

# Limitations

- This evidence does not prove the full matrix passes after the test fix; it records targeted local checks and the prior main CI failure.
- This evidence does not change the dbt 1.11 adapter boundary.
- This evidence does not close `ticket:c10ci06`; full main CI evidence and retrospective disposition are still needed.

# Result

The main CI run showed the new dbt 1.10/1.11 gates are functioning, and the remaining dbt 1.8 failures were caused by tests asserting config-field behavior unavailable in dbt 1.8. The targeted fix keeps config-specific coverage active for dbt 1.11 while avoiding false failures on dbt 1.8.

# Interpretation

The compatibility gate is now strong enough to expose version-specific test assumptions. The next acceptance step is to push this test fix to `main` and record the resulting CI evidence.

# Related Records

- `ticket:c10ci06`
- `evidence:c10ci06-ci-run-no-sync-fix`
- `critique:c10ci06-no-sync-follow-up`
