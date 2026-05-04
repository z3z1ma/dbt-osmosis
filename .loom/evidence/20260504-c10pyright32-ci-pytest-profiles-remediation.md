---
id: evidence:c10pyright32-ci-pytest-profiles-remediation
kind: evidence
status: recorded
created_at: 2026-05-04T18:01:26Z
updated_at: 2026-05-04T18:01:26Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  tickets:
    - ticket:c10pyright32
  critiques:
    - critique:c10pyright32-ci-pytest-profiles-remediation-review
external_refs:
  github_actions:
    - https://github.com/z3z1ma/dbt-osmosis/actions/runs/25333721046
---

# Summary

Observed and remediated the GitHub Actions `Tests` workflow failures that appeared after the CI basedpyright error was fixed. The failures were not basedpyright errors; all inspected pytest jobs failed because Click rejected the discovered default `--profiles-dir` path when `/home/runner/.dbt` did not exist.

# Procedure

Observed at: 2026-05-04T18:01:26Z

Source state: commit `37dcdc8a33882af415e22c180a819386d7d1b76f` for the failing GitHub Actions run; remediation commit `e151e760cce2bdeda8dcb9e4c269b1786be9a676`.

Procedure: inspected GitHub Actions `Tests` run `25333721046`, reproduced the missing default profiles directory condition locally with `DBT_PROFILES_DIR` and `DBT_PROJECT_DIR` unset and `HOME` pointing at a path without `.dbt`, applied the `src/dbt_osmosis/cli/main.py` remediation, then reran focused CLI validation and pre-commit.

Expected result when applicable: CLI tests that mock command execution should not be blocked by Click validation of a discovered default profiles directory that may legitimately be absent until dbt runtime loading.

Actual observed result: before the fix, representative local reproduction failed with two `tests/core/test_cli.py` failures and `SystemExit(2)`. After the fix, the affected tests passed under the same missing-home-profiles environment; the full CLI test file and pre-commit passed.

Procedure verdict / exit code: mixed red/green observation. Red reproduction failed as expected before the fix. Green validation passed after commit `e151e760cce2bdeda8dcb9e4c269b1786be9a676`.

# Artifacts

- GitHub Actions `Tests` run `25333721046` on `37dcdc8a33882af415e22c180a819386d7d1b76f` failed after basedpyright reported `0 errors, 1869 warnings` in inspected matrix jobs.
- Failed jobs included `Validate latest dbt compatibility (1.10.0)`, `Validate latest dbt compatibility (1.11.0)`, and all inspected pytest matrix jobs.
- The common pytest failure set was eight `tests/core/test_cli.py` failures caused by `Error: Invalid value for '--profiles-dir': Directory '/home/runner/.dbt' does not exist.`
- Red local reproduction: `env -u DBT_PROFILES_DIR -u DBT_PROJECT_DIR HOME="/var/folders/1b/6mg4g2fs2zx99h46b9j5r7mh0000gp/T/opencode/home-without-dbt" uv run pytest tests/core/test_cli.py::test_workbench_uses_streamlit_server_bind_flags_and_preserves_passthrough tests/core/test_cli.py::test_test_suggest_pattern_only_reports_ai_disabled -q` -> `2 failed`.
- Remediation: commit `e151e760cce2bdeda8dcb9e4c269b1786be9a676` changes `--profiles-dir` Click path types in the shared `dbt_opts` decorator and `workbench` command to stop requiring the path to exist before command logic runs.
- Green targeted reproduction after remediation: same missing-home-profiles environment over the affected workbench and test-suggest cases -> `8 passed in 0.20s`.
- Focused CLI validation: `uv run pytest tests/core/test_cli.py -q` -> `29 passed in 4.44s`.
- Repository hygiene validation: `uv run pre-commit run --all-files` -> all hooks passed, including `basedpyright`.

# Supports Claims

- Supports `ticket:c10pyright32#ACC-004` for the current remediated branch state through `uv run pre-commit run --all-files`, including the basedpyright hook passing.
- Supports the ticket journal claim that GitHub Actions run `25333721046` moved beyond basedpyright and exposed a separate CLI/default-profile validation failure.
- Supports the ticket journal claim that commit `e151e760cce2bdeda8dcb9e4c269b1786be9a676` remediates the reproduced local failure mode.

# Challenges Claims

- Challenges the earlier residual-risk assumption that the post-`1d120731b5cdd36d78a394dd42be63a84c186501` GitHub Actions `Tests` workflow might pass after basedpyright remediation alone; run `25333721046` failed in pytest jobs.

# Environment

Commit: `e151e760cce2bdeda8dcb9e4c269b1786be9a676`

Branch: `loom/dbt-110-111-hardening`

Runtime: local `uv` environment; GitHub Actions Linux matrix logs from run `25333721046`

OS: local macOS for reproduction and validation; GitHub Actions Ubuntu for failed CI observations

Relevant config: `DBT_PROFILES_DIR` and `DBT_PROJECT_DIR` unset for red/green local reproduction; `HOME` pointed to a path without `.dbt`

External service / harness / data source when applicable: GitHub Actions run `25333721046`

# Validity

Valid for: the `--profiles-dir` Click validation behavior in `src/dbt_osmosis/cli/main.py` and the observed `tests/core/test_cli.py` failure pattern.

Fresh enough for: deciding whether the local remediation is ready to push and re-run GitHub Actions.

Recheck when: CLI option declarations change, profile discovery order changes, Click version changes, or GitHub Actions reruns against a newer commit.

Invalidated by: a later failing GitHub Actions run that reports the same `--profiles-dir` validation symptom after commit `e151e760cce2bdeda8dcb9e4c269b1786be9a676`.

Supersedes / superseded by: supersedes the failing observation portion of `evidence:c10pyright32-ci-basedpyright-remediation` for the post-basedpyright pytest blocker; should be superseded by a green GitHub Actions run evidence record if remote CI passes.

# Limitations

This evidence does not establish that the next GitHub Actions `Tests` workflow is green. It validates the reproduced failure mode locally and preserves the failed remote run. Full remote confirmation remains pending until commit `e151e760cce2bdeda8dcb9e4c269b1786be9a676` or a later commit is pushed and the workflow completes.

# Result

The observed failure moved from basedpyright to pytest: Click rejected a missing discovered default profiles directory before command logic or test mocks could run. Removing the premature existence requirement for `--profiles-dir` resolved the reproduced failure locally, while pre-commit continued to pass.

# Interpretation

The narrow remediation is justified because `discover_profiles_dir()` intentionally falls back to `~/.dbt` without proving that directory exists, and commands that truly require dbt profile loading still validate through dbt runtime setup. This evidence alone does not close the ticket; it supports pushing and waiting for remote CI confirmation.

# Related Records

- ticket:c10pyright32
- evidence:c10pyright32-ci-basedpyright-remediation
- critique:c10pyright32-ci-pytest-profiles-remediation-review
