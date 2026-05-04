---
id: evidence:c10race13-main-ci-success
kind: evidence
status: recorded
created_at: 2026-05-04T08:19:55Z
updated_at: 2026-05-04T08:19:55Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  tickets:
    - ticket:c10race13
  evidence:
    - evidence:c10race13-yaml-sync-serialization-verification
  critique:
    - critique:c10race13-yaml-sync-serialization-review
external_refs:
  github_actions:
    - https://github.com/z3z1ma/dbt-osmosis/actions/runs/25307685047
    - https://github.com/z3z1ma/dbt-osmosis/actions/runs/25307685079
---

# Summary

Observed successful post-commit GitHub Actions `Tests` and `lint` validation for `ticket:c10race13` implementation commit `6899aabe9f86cbea896ccf5b9240e6967371dd65`. This evidence supplements local red/green verification; it does not decide ticket closure by itself.

# Procedure

Observed at: 2026-05-04T08:19:55Z
Source state: `origin/main` at `6899aabe9f86cbea896ccf5b9240e6967371dd65`.
Procedure: Queried GitHub Actions with `gh run view` for the post-push `Tests` and `lint` runs. No additional waiting was performed for downstream Release validation because it is not a `ticket:c10race13` acceptance criterion.
Expected result when applicable: `Tests` and `lint` should complete successfully for implementation commit `6899aabe9f86cbea896ccf5b9240e6967371dd65`.
Actual observed result: `Tests` and `lint` completed with conclusion `success`.
Procedure verdict / exit code: pass; `gh run view` reported `status: completed` and `conclusion: success` for both cited runs.

# Artifacts

- `Tests` run `25307685047`: event `push`, head SHA `6899aabe9f86cbea896ccf5b9240e6967371dd65`, created `2026-05-04T07:57:29Z`, updated `2026-05-04T08:08:18Z`, conclusion `success`, URL `https://github.com/z3z1ma/dbt-osmosis/actions/runs/25307685047`.
- `lint` run `25307685079`: event `push`, head SHA `6899aabe9f86cbea896ccf5b9240e6967371dd65`, created `2026-05-04T07:57:29Z`, updated `2026-05-04T07:58:27Z`, conclusion `success`, URL `https://github.com/z3z1ma/dbt-osmosis/actions/runs/25307685079`.
- `gh run view 25307685047 --json jobs` returned no non-success jobs after completion.

# Supports Claims

- ticket:c10race13#ACC-001 through ticket:c10race13#ACC-006: post-commit `Tests` passed after the YAML sync serialization implementation landed on `main`.
- initiative:dbt-110-111-hardening#OBJ-003: main CI accepted the source state containing path-grouped YAML sync scheduling and unique temp writes.

# Challenges Claims

None - the observed CI results matched the expected successful post-commit validation for the cited claims.

# Environment

Commit: `6899aabe9f86cbea896ccf5b9240e6967371dd65`
Branch: `main`
Runtime: GitHub-hosted Actions runners for `Tests` and `lint` workflows.
OS: GitHub Actions Ubuntu runners.
Relevant config: `.github/workflows/tests.yml`, `.github/workflows/lint.yml`, `src/dbt_osmosis/core/sync_operations.py`, `src/dbt_osmosis/core/schema/writer.py`, and `tests/core/test_sync_operations.py` at the cited commit.
External service / harness / data source when applicable: GitHub Actions via `gh` CLI.

# Validity

Valid for: post-commit `Tests` and `lint` validation of `ticket:c10race13` implementation commit `6899aabe9f86cbea896ccf5b9240e6967371dd65` on `main`.
Fresh enough for: final ticket acceptance review and closure consideration for `ticket:c10race13`.
Recheck when: source changes after `6899aabe9f86cbea896ccf5b9240e6967371dd65`, workflow configuration changes, YAML sync scheduling changes, or GitHub reruns replace these observations.
Invalidated by: a later failed required run for the same commit or source changes that alter the cited implementation behavior.
Supersedes / superseded by: Supplements `evidence:c10race13-yaml-sync-serialization-verification` with post-commit main `Tests` and `lint` validation.

# Limitations

- Downstream Release validation was in progress when this evidence was recorded and is intentionally not part of this ticket's closure basis.
- GitHub Actions success covers the repository's configured matrix at the cited commit, not future dbt adapter/version changes.

# Result

The observed main-branch `Tests` and `lint` runs succeeded for implementation commit `6899aabe9f86cbea896ccf5b9240e6967371dd65`.

# Interpretation

The evidence supports moving `ticket:c10race13` from post-commit CI review to final acceptance. It does not itself replace retrospective / promotion disposition.

# Related Records

- ticket:c10race13
- evidence:c10race13-yaml-sync-serialization-verification
- critique:c10race13-yaml-sync-serialization-review
- commit `6899aabe9f86cbea896ccf5b9240e6967371dd65`
