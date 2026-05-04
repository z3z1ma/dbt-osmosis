---
id: critique:c10lock07-adapter-constraint-follow-up
kind: critique
status: final
created_at: 2026-05-04T00:47:15Z
updated_at: 2026-05-04T00:47:15Z
scope:
  kind: repository
  repositories:
    - repo:root
review_target: "ticket:c10lock07 adapter-constraint follow-up diff"
links:
  tickets:
    - ticket:c10lock07
  evidence:
    - evidence:c10lock07-adapter-bound-verification
  critique:
    - critique:c10lock07-dependency-resolution
    - critique:c10lock07-integration-path-follow-up
external_refs:
  github_issue: https://github.com/z3z1ma/dbt-osmosis/issues/358
---

# Summary

Follow-up critique for the c10lock07 adapter-bound patch that addresses dbt 1.8 main CI failures under CI-pinned `uv==0.5.13`.

# Review Target

Reviewed the current uncommitted diff after the parent added `DBT_ADAPTERS_CONSTRAINT="dbt-adapters>=1.16.3,<2.0"` to workflow/Taskfile uv-resolved matrix and latest compatibility installs. The review focused on `.github/workflows/tests.yml` and `Taskfile.yml`, with c10lock07 ticket acceptance, prior critiques, and local evidence as context.

# Verdict

`pass`.

The first review pass found the workflow constraint was scoped only to the `lockfile` job and therefore would not reach the primary pytest matrix job. The parent moved the constraint to workflow-level `env`, reran local checks, and the final review found no remaining critique blocker.

# Findings

None - no open findings remain in the final follow-up pass.

Review note: the pre-final env-scope issue was valid when observed, but it was fixed before this critique was finalized. The ticket should preserve that reconciliation in its journal and evidence references rather than treating it as an unresolved critique finding.

# Evidence Reviewed

- Current uncommitted diff for `.github/workflows/tests.yml` and `Taskfile.yml`.
- `.github/workflows/tests.yml` workflow-level `DBT_ADAPTERS_CONSTRAINT` and matrix/latest uv install/version assertion steps.
- `Taskfile.yml` matrix/latest uv install/version assertion steps.
- `ticket:c10lock07` acceptance criteria and claim matrix.
- `evidence:c10lock07-local-dependency-resolution-verification`.
- `evidence:c10lock07-adapter-bound-verification`.
- `critique:c10lock07-dependency-resolution` and `critique:c10lock07-integration-path-follow-up`.
- Parent-reported local checks: pinned-uv dbt 1.8 targeted test, `uv lock --check`, pre-commit hooks, and `task pip-install-smoke`.

# Residual Risks

- Full GitHub Actions evidence is still pending.
- The adapter constraint is a CI resolver stabilizer, not package metadata cleanup; `ticket:c10pkg10` still owns broader package metadata and extras decisions.
- The plain pip smoke logs `dbt-adapters` but intentionally does not assert the floor, so records should not claim all install paths enforce `dbt-adapters>=1.16.3`.
- `demo_duckdb/integration_tests.sh` still contains `uv run` and remains unsafe to reintroduce into matrix CI unchanged.

# Required Follow-up

- Commit and push the follow-up patch.
- Record final GitHub Actions evidence before accepting or closing `ticket:c10lock07`.
- Keep broader package metadata cleanup routed to `ticket:c10pkg10`.

# Acceptance Recommendation

`ticket-acceptance-review-needed`.

No critique blocker remains, but the ticket still needs final `main` CI evidence before acceptance.
