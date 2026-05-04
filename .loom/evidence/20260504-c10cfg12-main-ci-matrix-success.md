---
id: evidence:c10cfg12-main-ci-matrix-success
kind: evidence
status: recorded
created_at: 2026-05-04T07:04:39Z
updated_at: 2026-05-04T07:04:39Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  tickets:
    - ticket:c10cfg12
    - ticket:c10col01
    - ticket:c10meta02
  evidence:
    - evidence:c10cfg12-real-config-shape-fixtures
  critique:
    - critique:c10cfg12-real-config-shape-fixtures
external_refs:
  github_actions:
    - https://github.com/z3z1ma/dbt-osmosis/actions/runs/25305202544
    - https://github.com/z3z1ma/dbt-osmosis/actions/runs/25305202551
---

# Summary

Observed successful post-commit GitHub Actions validation for `ticket:c10cfg12` implementation commit `77bb9dd364f43ed11fdb493661f78e6b8218910d`. This evidence supplements local real parsed config-shape fixture verification with main-branch `Tests` matrix and `lint` validation; it does not decide ticket closure by itself.

# Procedure

Observed at: 2026-05-04T07:04:39Z
Source state: `origin/main` at `77bb9dd364f43ed11fdb493661f78e6b8218910d`.
Procedure: Queried GitHub Actions with `gh run view` for the post-push `Tests` and `lint` runs, including the `Tests` job matrix outcomes.
Expected result when applicable: `Tests` and `lint` should complete successfully for implementation commit `77bb9dd364f43ed11fdb493661f78e6b8218910d`, including dbt 1.10 and dbt 1.11 matrix rows that exercise the new real parsed config-shape tests.
Actual observed result: `Tests` and `lint` completed with conclusion `success`. The `Tests` run included successful dbt 1.11 pytest rows for Python 3.10, 3.11, 3.12, and 3.13, plus successful latest dbt compatibility validation for 1.11.0.
Procedure verdict / exit code: pass; `gh run view` reported `status: completed` and `conclusion: success` for both cited runs and success for all queried `Tests` jobs.

# Artifacts

- `Tests` run `25305202544`: event `push`, head SHA `77bb9dd364f43ed11fdb493661f78e6b8218910d`, created `2026-05-04T06:50:51Z`, updated `2026-05-04T07:00:31Z`, conclusion `success`, URL `https://github.com/z3z1ma/dbt-osmosis/actions/runs/25305202544`.
- `lint` run `25305202551`: event `push`, head SHA `77bb9dd364f43ed11fdb493661f78e6b8218910d`, created `2026-05-04T06:50:51Z`, updated `2026-05-04T06:52:03Z`, conclusion `success`, URL `https://github.com/z3z1ma/dbt-osmosis/actions/runs/25305202551`.
- `Tests` matrix dbt 1.11 rows completed successfully: `Run pytest (3.10, 1.11.0)`, `Run pytest (3.11, 1.11.0)`, `Run pytest (3.12, 1.11.0)`, and `Run pytest (3.13, 1.11.0)`.
- `Tests` matrix dbt 1.10 rows completed successfully: `Run pytest (3.10, 1.10.0)`, `Run pytest (3.11, 1.10.0)`, `Run pytest (3.12, 1.10.0)`, and `Run pytest (3.13, 1.10.0)`.
- Latest compatibility jobs completed successfully for `Validate latest dbt compatibility (1.10.0)` and `Validate latest dbt compatibility (1.11.0)`.
- Docs build, pip install smoke, and lockfile check jobs in the same `Tests` run also completed successfully.

# Supports Claims

- ticket:c10cfg12#ACC-001: post-commit `Tests` passed the real parsed fixture coverage for dbt-osmosis options under node `config.meta` and column `config.meta`.
- ticket:c10cfg12#ACC-002: post-commit dbt 1.10 and 1.11 matrix rows passed tests that assert actual `node.config.meta`, `node.config.extra`, and `node.unrendered_config` fields.
- ticket:c10cfg12#ACC-003: post-commit matrix rows passed tests that assert column `config.tags` and `config.meta` without relying only on mocks.
- ticket:c10cfg12#ACC-004: post-commit matrix rows passed while keeping the real config-shape assertions isolated to dbt 1.10+ and compatible with older configured rows through honest skips.
- ticket:c10cfg12#ACC-005: post-commit `Tests` passed after the local test-first red/green implementation captured in `evidence:c10cfg12-real-config-shape-fixtures`.
- ticket:c10cfg12#ACC-006: post-commit dbt 1.11 matrix rows passed the converted fixture-gap coverage from `ticket:c10col01#ACC-005` and `ticket:c10meta02#ACC-005`.
- ticket:c10col01#ACC-005: post-commit dbt 1.11 matrix rows passed the adapter-backed missing-column injection/sync fixture coverage added by `ticket:c10cfg12`.
- ticket:c10meta02#ACC-005: post-commit dbt 1.11 matrix rows passed the real dbt-parsed column `config.meta` and `config.tags` fixture coverage consumed through `SettingsResolver` and `PropertyAccessor`.
- initiative:dbt-110-111-hardening#OBJ-001, #OBJ-002, and #OBJ-004: main CI now exercises dbt 1.10/1.11 config-shape fixture coverage without source-tree generated fixture artifacts.

# Challenges Claims

None - the observed CI results matched the expected successful post-commit validation for the cited claims.

# Environment

Commit: `77bb9dd364f43ed11fdb493661f78e6b8218910d`
Branch: `main`
Runtime: GitHub-hosted Actions runners for `Tests` and `lint` workflows.
OS: GitHub Actions Ubuntu runners for the queried workflow jobs.
Relevant config: `.github/workflows/tests.yml`, `.github/workflows/lint.yml`, `tests/core/test_real_config_shapes.py`, `tests/support.py`, `tests/core/test_config_resolution.py`, `tests/core/test_settings_resolver.py`, `tests/core/test_property_accessor.py`, and `tests/core/test_transforms.py` at the cited commit.
External service / harness / data source when applicable: GitHub Actions via `gh` CLI.

# Validity

Valid for: post-commit `Tests` and `lint` validation of `ticket:c10cfg12` implementation commit `77bb9dd364f43ed11fdb493661f78e6b8218910d` on `main`, including dbt 1.10 and dbt 1.11 matrix rows.
Fresh enough for: final ticket acceptance review and closure consideration for `ticket:c10cfg12`, including the dbt 1.11 portions of converted `ticket:c10col01#ACC-005` and `ticket:c10meta02#ACC-005` fixture gaps.
Recheck when: source changes after `77bb9dd364f43ed11fdb493661f78e6b8218910d`, workflow configuration changes, dbt compatibility matrix changes, dbt-duckdb adapter mappings change, or GitHub reruns replace these observations.
Invalidated by: a later failed required run for the same commit, source changes that alter real config-shape tests or fixture helpers, or evidence that the cited runs did not execute the relevant matrix/jobs.
Supersedes / superseded by: Supplements `evidence:c10cfg12-real-config-shape-fixtures` with post-commit dbt 1.10/1.11 matrix and lint validation.

# Limitations

- GitHub Actions success covers the repository's configured dbt/Python matrix at the cited commit, not every future dbt adapter/version combination.
- The downstream `Release` workflow was still in progress when this evidence was recorded; release packaging is not a `ticket:c10cfg12` acceptance criterion.
- This evidence does not prove unrelated resolver behavior outside the real config-shape fixture surfaces asserted by the tests.

# Result

The observed main-branch `Tests` and `lint` runs succeeded for implementation commit `77bb9dd364f43ed11fdb493661f78e6b8218910d`, and all queried dbt 1.10 and dbt 1.11 matrix rows completed successfully.

# Interpretation

The evidence supports resolving `critique:c10cfg12-real-config-shape-fixtures#FIND-001` in the ticket-owned acceptance dossier and moving `ticket:c10cfg12` from post-commit CI review to final acceptance. It does not itself close the ticket or replace retrospective / promotion disposition.

# Related Records

- ticket:c10cfg12
- ticket:c10col01
- ticket:c10meta02
- evidence:c10cfg12-real-config-shape-fixtures
- critique:c10cfg12-real-config-shape-fixtures
- commit `77bb9dd364f43ed11fdb493661f78e6b8218910d`
