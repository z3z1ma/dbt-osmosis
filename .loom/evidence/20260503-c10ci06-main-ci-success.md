---
id: evidence:c10ci06-main-ci-success
kind: evidence
status: recorded
created_at: 2026-05-03T23:36:45Z
updated_at: 2026-05-03T23:36:45Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  tickets:
    - ticket:c10ci06
  evidence:
    - evidence:c10ci06-ci-gate-local-verification
    - evidence:c10ci06-ci-run-no-sync-fix
    - evidence:c10ci06-main-ci-dbt18-test-fix
  critique:
    - critique:c10ci06-dbt-111-ci-gate
external_refs:
  tests_run: https://github.com/z3z1ma/dbt-osmosis/actions/runs/25293813881
  lint_run: https://github.com/z3z1ma/dbt-osmosis/actions/runs/25293813907
  github_issue: https://github.com/z3z1ma/dbt-osmosis/issues/357
---

# Summary

Observed final `main` GitHub Actions for commit `b3470bff42566dfb475a8a21ec19e45cc7faaf0d` after the dbt 1.11 CI gate, `UV_NO_SYNC=1` fix, dbt 1.8 test compatibility fix, and formatter cleanup landed. The `Tests` workflow and `lint` workflow both completed successfully.

# Procedure

Observed at: 2026-05-03T23:36:45Z

Source state: `main` at `b3470bff42566dfb475a8a21ec19e45cc7faaf0d` (`b3470bf style: format column metadata helper`).

Procedure:

- Queried recent `main` runs with `gh run list --branch main --limit 10`.
- Inspected `Tests` run `25293813881` with `gh run view 25293813881 --json status,conclusion,updatedAt,headSha,name,url,jobs`.
- Inspected `lint` run `25293813907` with `gh run view 25293813907 --json status,conclusion,updatedAt,headSha,name,url,jobs`.
- Listed `Tests` job conclusions with `gh run view 25293813881 --json jobs --jq '.jobs[] | [.name, .conclusion] | @tsv'`.
- Ran local `pre-commit run --all-files` in the sibling worktree.
- Ran local focused tests: `uv run pytest tests/core/test_transforms.py tests/core/test_inheritance_behavior.py tests/core/test_settings_resolver.py tests/core/test_property_accessor.py`.

Expected result when applicable: the post-fix `main` workflows should pass, including dbt 1.11 matrix rows for each supported Python version, latest dbt 1.10/1.11 compatibility jobs, installed-version assertions, parse, pytest, integration tests, and lint/pre-commit.

Actual observed result: `Tests` run `25293813881` completed with conclusion `success`; `lint` run `25293813907` completed with conclusion `success`. Local pre-commit passed, and the focused local core suite reported `68 passed, 3 skipped`.

Procedure verdict / exit code: pass; all observed commands and GitHub workflow runs completed successfully.

# Artifacts

- `Tests` run `25293813881`: success, updated at `2026-05-03T23:31:53Z`, head SHA `b3470bff42566dfb475a8a21ec19e45cc7faaf0d`.
- `lint` run `25293813907`: success, updated at `2026-05-03T23:24:17Z`, head SHA `b3470bff42566dfb475a8a21ec19e45cc7faaf0d`.
- Successful dbt 1.11 matrix jobs: `Run pytest (3.10, 1.11.0)`, `Run pytest (3.11, 1.11.0)`, `Run pytest (3.12, 1.11.0)`, and `Run pytest (3.13, 1.11.0)`.
- Successful dbt 1.10 matrix jobs: `Run pytest (3.10, 1.10.0)`, `Run pytest (3.11, 1.10.0)`, `Run pytest (3.12, 1.10.0)`, and `Run pytest (3.13, 1.10.0)`.
- Successful latest compatibility jobs: `Validate latest dbt compatibility (1.10.0)` and `Validate latest dbt compatibility (1.11.0)`.
- Successful older supported rows observed in the same workflow: dbt 1.8 on Python 3.10-3.12 and dbt 1.9 on Python 3.10-3.12.
- Local `pre-commit run --all-files`: passed all hooks, including `ruff-format`, `ruff`, gitleaks, and actionlint.
- Local focused tests: `68 passed, 3 skipped in 26.71s`.

# Supports Claims

- `ticket:c10ci06#ACC-001` - dbt 1.11 matrix rows passed for Python 3.10, 3.11, 3.12, and 3.13 with the published `dbt-duckdb~=1.10.1` adapter boundary.
- `ticket:c10ci06#ACC-002` - each matrix job includes and executed the `Assert installed versions` step before type checks, parse, pytest, and integration tests.
- `ticket:c10ci06#ACC-003` - dbt 1.10 and dbt 1.11 matrix rows ran basedpyright, manifest parse, full pytest, and integration tests successfully; latest compatibility jobs also ran parse/import/CLI smoke plus pytest successfully.
- `ticket:c10ci06#ACC-004` - the local `Taskfile.yml` compatibility story was implemented before this run and the final CI evidence confirms the same matrix semantics on `main`.
- `ticket:c10ci06#ACC-005` - the scheduled latest-core compatibility job is present in `.github/workflows/tests.yml` and its dbt 1.10/1.11 job logic passed on the `main` push observation.

# Challenges Claims

None - the observed final `main` workflows support the ticket acceptance claims. The first future cron-triggered execution of the scheduled canary remains a normal recheck event, not a current challenge to the workflow configuration or job logic.

# Environment

Commit: `b3470bff42566dfb475a8a21ec19e45cc7faaf0d`

Branch: `main` for GitHub Actions; local sibling worktree branch `loom/dbt-110-111-hardening` for local verification.

Runtime: GitHub Actions Ubuntu for workflows; local macOS/darwin for pre-commit and focused pytest.

OS: GitHub Actions Ubuntu and local macOS/darwin.

Relevant config: `.github/workflows/tests.yml`, `.github/workflows/lint.yml`, `Taskfile.yml`, `tests/core/test_transforms.py`, `tests/core/test_inheritance_behavior.py`, `src/dbt_osmosis/core/logger.py`.

External service / harness / data source when applicable: GitHub Actions runs `25293813881` and `25293813907`.

# Validity

Valid for: final acceptance evidence for `ticket:c10ci06` at source state `b3470bff42566dfb475a8a21ec19e45cc7faaf0d`.

Fresh enough for: closing `ticket:c10ci06` after retrospective disposition is recorded.

Recheck when: the CI matrix, `UV_NO_SYNC` behavior, dbt version support policy, Python version support policy, adapter mapping, or relevant compatibility tests change.

Invalidated by: a later workflow failure on the same acceptance path, removal of installed-version assertions, removal of dbt 1.11 matrix rows, or changing the adapter boundary without fresh evidence.

Supersedes / superseded by: supersedes `evidence:c10ci06-main-ci-dbt18-test-fix` for final matrix acceptance.

# Limitations

- This evidence observes the scheduled canary job logic on a `push` event; it does not observe the first future cron event itself.
- This evidence does not prove support for a newer `dbt-duckdb` adapter boundary than `~=1.10.1` under dbt 1.11.
- This evidence does not address release workflow ordering, dependency lockfile policy, or future dbt 1.12 compatibility.

# Result

The final `main` CI state is green for the dbt 1.11 compatibility gate and lint workflow. The expanded matrix, latest compatibility jobs, installed-version assertions, parse, pytest, integration tests, and pre-commit checks all passed for the observed source state.

# Interpretation

The dbt 1.11 compatibility gate is now strong enough for the current acceptance scope: it runs across the supported Python versions, verifies the installed dbt runtime, preserves the overlay runtime with `UV_NO_SYNC=1`, and exercises meaningful dbt workflows. The remaining adapter-boundary and future cron-trigger observations are normal maintenance risks rather than blockers for this ticket.

# Related Records

- `ticket:c10ci06`
- `evidence:c10ci06-ci-gate-local-verification`
- `evidence:c10ci06-ci-run-no-sync-fix`
- `evidence:c10ci06-main-ci-dbt18-test-fix`
- `critique:c10ci06-dbt-111-ci-gate`
- `critique:c10ci06-no-sync-follow-up`
- `critique:c10ci06-dbt18-test-fix`
