---
id: evidence:c10pyright32-ci-basedpyright-remediation
kind: evidence
status: recorded
created_at: 2026-05-04T17:34:36Z
updated_at: 2026-05-04T17:34:36Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  tickets:
    - ticket:c10pyright32
  critique:
    - critique:c10pyright32-ci-basedpyright-remediation-review
external_refs:
  github_actions:
    - https://github.com/z3z1ma/dbt-osmosis/actions/runs/25333133362
---

# Summary

Observed and remediated the CI-only basedpyright error that appeared after `ticket:c10pyright32` first added the pre-commit gate. GitHub Actions run `25333133362` failed with `basedpyright summary: 1 errors, 1869 warnings`; Linux container reproduction exposed the diagnostic in `src/dbt_osmosis/core/llm.py`, and commit `1d120731b5cdd36d78a394dd42be63a84c186501` reduced the reproduced CI error count to zero.

# Procedure

Observed at: 2026-05-04T17:34:36Z

Source state: `1d120731b5cdd36d78a394dd42be63a84c186501` on branch `loom/dbt-110-111-hardening`.

Procedure:

- Inspected GitHub Actions Tests run `25333133362` after push of `9b48d3057da465139d7a15b28489df7c67c4537b`.
- Reproduced the CI environment in a Linux container with Python 3.13, `uv --no-config`, editable install, `.[dev,openai,duckdb]`, and dbt 1.11 dependency pins.
- Applied commit `1d120731b5cdd36d78a394dd42be63a84c186501` to type the optional OpenAI rate-limit exception boundary as `type[Exception]` and avoid fallback-class redefinition.
- Ran `uv run ruff check src/dbt_osmosis/core/llm.py && uv run pre-commit run basedpyright --all-files --verbose`.
- Ran `uv run pytest tests/core/test_llm.py -q`.
- Re-ran Linux container basedpyright reproduction with dbt 1.8 and dbt 1.11 dependency sets.

Expected result when applicable: basedpyright reports zero errors locally and in the Linux CI reproduction; focused LLM tests continue to pass.

Actual observed result: pre-fix Linux reproduction reported one basedpyright error at `/workspace/src/dbt_osmosis/core/llm.py` line 18: `Type "type[RateLimitError]" is not assignable to declared type "type[_OpenAIRateLimitError]"`. After commit `1d120731b5cdd36d78a394dd42be63a84c186501`, local basedpyright hook reported `0 errors, 1869 warnings`; focused LLM tests reported `30 passed, 9 skipped`; Linux dbt 1.8 reproduction reported `errorCount: 0`; Linux dbt 1.11 reproduction reported `errorCount: 0`.

Procedure verdict / exit code: pass for post-fix local Ruff, basedpyright hook, focused tests, and Linux basedpyright reproductions.

# Artifacts

- Failed GitHub Actions run: `25333133362`.
- Failed CI summary: `basedpyright summary: 1 errors, 1869 warnings`.
- Reproduced diagnostic: `src/dbt_osmosis/core/llm.py` optional OpenAI SDK branch inferred `_OpenAIRateLimitError` as the fallback class type, then rejected assignment of `openai.RateLimitError` on Linux.
- Fix commit: `1d120731b5cdd36d78a394dd42be63a84c186501`.
- Final local hook summary: `basedpyright summary: 0 errors, 1869 warnings`.

# Supports Claims

- `ticket:c10pyright32#ACC-002`: the zero-error policy now catches and blocks the CI-only basedpyright error, and the remediated source returns to `errorCount: 0`.
- `ticket:c10pyright32#ACC-004`: current branch validation records `errorCount=0` locally and in Linux CI reproduction for dbt 1.8 and dbt 1.11 dependency sets.
- `initiative:dbt-110-111-hardening#OBJ-006`: the pushed hardening branch no longer has the reproduced basedpyright error that blocked CI before tests could run.

# Challenges Claims

- The earlier `evidence:c10pyright32-basedpyright-precommit-validation` limitation that full GitHub Actions validation was pending became material: CI found one Linux basedpyright error after the first push.

# Environment

Commit: `1d120731b5cdd36d78a394dd42be63a84c186501`

Branch: `loom/dbt-110-111-hardening`

Runtime: local `uv run` plus Docker `python:3.13-slim` Linux reproduction using `uv --no-config` and editable project install.

OS: macOS / Darwin for local checks; Linux container for CI reproduction.

Relevant config: `pyproject.toml` `tool.pyright`; `.github/workflows/tests.yml` basedpyright JSON summary gate; `.pre-commit-config.yaml` basedpyright hook.

External service / harness / data source when applicable: GitHub Actions run `25333133362`; Docker local Linux reproduction.

# Validity

Valid for: `ticket:c10pyright32` remediation at commit `1d120731b5cdd36d78a394dd42be63a84c186501` and the listed local/Linux reproduction environments.

Fresh enough for: re-push and CI remediation review.

Recheck when: `src/dbt_osmosis/core/llm.py`, optional OpenAI SDK typing, basedpyright version, dbt matrix dependencies, or CI basedpyright command changes.

Invalidated by: changes after commit `1d120731b5cdd36d78a394dd42be63a84c186501` that alter the optional OpenAI import boundary or type-check configuration.

Supersedes / superseded by: supersedes `evidence:c10pyright32-basedpyright-precommit-validation` for the post-push CI error remediation question; not superseded.

# Limitations

- Full GitHub Actions confirmation remains pending until after this remediation commit is pushed.
- Linux reproduction used Python 3.13 and dbt 1.8/1.11 dependency sets, not every Python/dbt matrix combination, but the original error was identical across CI matrix jobs.
- basedpyright warnings remain intentionally tolerated.

# Result

The CI-only basedpyright error was reproduced, patched, and driven to zero errors in local and Linux reproduction checks.

# Interpretation

The evidence supports pushing the remediation commit and expecting the basedpyright matrix failure to clear, while leaving final CI confirmation to the next GitHub Actions run.

# Related Records

- `ticket:c10pyright32`
- `critique:c10pyright32-ci-basedpyright-remediation-review`
- `evidence:c10pyright32-basedpyright-precommit-validation`
- `initiative:dbt-110-111-hardening`
