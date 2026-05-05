---
id: evidence:gh305wb-workbench-extra-preflight-validation
kind: evidence
status: recorded
created_at: 2026-05-05T08:25:59Z
updated_at: 2026-05-05T08:25:59Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  ticket:
    - ticket:gh305wb
  packets:
    - packet:ralph-ticket-gh305wb-20260505T081714Z
external_refs:
  github_issue: https://github.com/z3z1ma/dbt-osmosis/issues/305
---

# Summary

Observed red/green and parent validation for the workbench optional extra and dependency preflight fix.

# Procedure

Observed at: 2026-05-05T08:25:59Z
Source state: uncommitted diff after `72301c237cee12fff994747651d76e8d2563e74c`, with `gh305wb` package/preflight files dirty.
Procedure: Ralph added failing package/preflight tests, updated the workbench extra/lockfile and CLI preflight, then parent reran focused CLI/package tests, Ruff checks, format checks, lock validation, and whitespace checks.
Expected result when applicable: `dbt-osmosis[workbench]` should include the dependency needed for `ydata_profiling`/IPython startup; workbench preflight should catch transitive import failures before launching Streamlit; OpenAI should remain optional.
Actual observed result: Failing package/preflight tests turned green; parent validation and Oracle critique passed.
Procedure verdict / exit code: Pass; all parent validation commands exited 0.

# Artifacts

Child red observation:

- `uv run pytest tests/core/test_cli.py::test_workbench_transitive_import_failure_has_workbench_extra_hint tests/test_package_metadata.py::test_base_dependencies_and_optional_extras_are_intentional` failed before implementation with 2 failures: transitive `IPython` import failure was not caught before Streamlit launch, and `ipython` was absent from the workbench extra.

Parent green commands:

```bash
uv run pytest tests/core/test_cli.py tests/test_package_metadata.py -q
```

Observed result: `45 passed, 2 warnings`.

```bash
uv run ruff check src/dbt_osmosis/cli/main.py tests/core/test_cli.py tests/test_package_metadata.py
```

Observed result: `All checks passed!`.

```bash
uv run ruff format --check src/dbt_osmosis/cli/main.py tests/core/test_cli.py tests/test_package_metadata.py
```

Observed result: `3 files already formatted`.

```bash
uv lock --check
```

Observed result: `Resolved 179 packages in 5ms`.

```bash
git diff --check
```

Observed result: passed with no output.

# Supports Claims

- ticket:gh305wb#ACC-001: package metadata and lockfile include `ipython>=8,<9` in the `workbench` extra alongside `ydata-profiling`.
- ticket:gh305wb#ACC-002: CLI tests show direct and transitive workbench dependency import failures raise a workbench-extra hint before `subprocess.run` launches Streamlit.
- ticket:gh305wb#ACC-003: package metadata asserts OpenAI remains outside the workbench extra, and the preflight module list does not include OpenAI.
- ticket:gh305wb#ACC-004: tests cover package metadata and preflight behavior.
- ticket:gh305wb#ACC-005: existing workbench CLI smoke tests stayed green in the focused CLI run.

# Challenges Claims

None.

# Environment

Commit: uncommitted diff after `72301c237cee12fff994747651d76e8d2563e74c`.
Branch: `loom/dbt-110-111-hardening`.
Runtime: `uv run`; external `VIRTUAL_ENV` warning observed and ignored by `uv`.
OS: macOS Darwin.
Relevant config: focused CLI/package metadata tests and uv lock validation.
External service / harness / data source when applicable: GitHub issue #305.

# Validity

Valid for: local package metadata, lockfile, and CLI preflight behavior.
Fresh enough for: `ticket:gh305wb` parent acceptance review and critique.
Recheck when: workbench optional dependencies, `_WORKBENCH_APP_MODULES`, `workbench` command launch flow, or `uv.lock` changes.
Invalidated by: dependency metadata or workbench import/preflight changes without rerunning focused tests and `uv lock --check`.
Supersedes / superseded by: N/A.

# Limitations

This evidence is focused and local. It does not include an isolated fresh `dbt-osmosis[workbench]` install/start smoke, full-suite run, or remote CI.

# Result

The observed implementation adds the missing workbench extra dependency and catches transitive workbench import failures before launching Streamlit, while keeping OpenAI optional.

# Interpretation

The evidence supports local acceptance of `ticket:gh305wb` after critique. It does not establish remote CI health.

# Related Records

- ticket:gh305wb
- critique:gh305wb-workbench-extra-preflight-review
- packet:ralph-ticket-gh305wb-20260505T081714Z
- initiative:issue-pr-zero
