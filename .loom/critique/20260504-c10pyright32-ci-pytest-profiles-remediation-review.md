---
id: critique:c10pyright32-ci-pytest-profiles-remediation-review
kind: critique
status: final
created_at: 2026-05-04T18:01:26Z
updated_at: 2026-05-04T18:01:26Z
scope:
  kind: repository
  repositories:
    - repo:root
review_target: "commit e151e760cce2bdeda8dcb9e4c269b1786be9a676 CLI profiles-dir validation remediation"
links:
  tickets:
    - ticket:c10pyright32
  evidence:
    - evidence:c10pyright32-ci-pytest-profiles-remediation
external_refs: {}
---

# Summary

Reviewed the local remediation for the GitHub Actions pytest matrix failures that occurred after basedpyright passed. The review focused on whether removing Click's eager existence check for `--profiles-dir` causes a behavioral regression or leaves the CI failure insufficiently covered.

# Review Target

Target: commit `e151e760cce2bdeda8dcb9e4c269b1786be9a676`, which changes the shared dbt CLI option surface and workbench CLI option surface in `src/dbt_osmosis/cli/main.py`.

The change was reviewed because it affects validation behavior for every command using `dbt_opts`, plus `workbench`, and because the previous CI run failed across the pytest matrix.

# Verdict

`pass`

The remediation is narrow and consistent with the existing discovery contract: `discover_profiles_dir()` may fall back to `~/.dbt` without proving the directory exists, so Click should not reject that discovered default before command-specific logic or dbt runtime loading can run.

# Findings

None - no findings.

# Evidence Reviewed

- Diff for `src/dbt_osmosis/cli/main.py` in commit `e151e760cce2bdeda8dcb9e4c269b1786be9a676`.
- GitHub Actions `Tests` run `25333721046` failure excerpts showing pytest failures caused by `Error: Invalid value for '--profiles-dir': Directory '/home/runner/.dbt' does not exist.`
- Red local reproduction under missing `HOME/.dbt`: two representative `tests/core/test_cli.py` failures before the fix.
- Green local validation after the fix: affected CLI tests under the missing-home-profiles environment -> `8 passed in 0.20s`.
- Focused CLI validation: `uv run pytest tests/core/test_cli.py -q` -> `29 passed in 4.44s`.
- Hygiene validation: `uv run pre-commit run --all-files` -> all hooks passed, including `basedpyright`.
- Oracle review `ses_20bde45aaffe98T7tkmG8PRQOQ`, which returned no blocking findings and noted that explicit missing profile-dir typos now reach dbt loading instead of Click.

# Residual Risks

- An explicitly mistyped `--profiles-dir /missing/path` now reaches downstream dbt loading instead of failing immediately in Click, so the eventual error may be less direct.
- Remote GitHub Actions confirmation for the remediation is still pending until the commit is pushed and the `Tests` workflow completes.

# Required Follow-up

No code follow-up is required before pushing the remediation. The ticket should remain pending acceptance until a new GitHub Actions `Tests` run confirms the remote matrix no longer fails with this `--profiles-dir` Click validation symptom.

# Acceptance Recommendation

`ticket-acceptance-review-needed` - no critique blockers remain for the local remediation, but ticket-owned acceptance should wait for remote GitHub Actions confirmation.
