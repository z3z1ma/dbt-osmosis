---
id: critique:c10lock07-dependency-resolution
kind: critique
status: final
created_at: 2026-05-04T00:19:53Z
updated_at: 2026-05-04T00:19:53Z
scope:
  kind: repository
  repositories:
    - repo:root
review_target: "ticket:c10lock07 implementation diff before integration-step follow-up"
links:
  tickets:
    - ticket:c10lock07
  packets:
    - packet:ralph-ticket-c10lock07-20260503T234103Z
  evidence:
    - evidence:c10lock07-local-dependency-resolution-verification
external_refs:
  github_issue: https://github.com/z3z1ma/dbt-osmosis/issues/358
---

# Summary

Mandatory high-risk critique of the `ticket:c10lock07` dependency-resolution implementation using release-packaging and test-coverage lenses.

# Review Target

Reviewed the uncommitted diff after the Ralph implementation and before the follow-up integration-step fix. The target changed `.github/workflows/tests.yml` and `Taskfile.yml`, with supporting records from the consumed Ralph packet, local evidence, owning ticket, and `wiki:ci-compatibility-matrix`.

# Verdict

`changes_required`.

The implementation mostly satisfied the ticket shape, but one high-severity finding showed the CI integration step could still invoke `uv run` through `demo_duckdb/integration_tests.sh`, reintroducing the lockfile-runtime failure mode this ticket is meant to prevent.

# Findings

## FIND-001: Matrix Integration Step Reintroduced `uv run`

Severity: high

Confidence: high

State: open

Observation:

The matrix job removed the workflow-level `UV_NO_SYNC` environment and created a clean matrix virtual environment, but the integration step still called `bash -x integration_tests.sh`. That script invokes `uv run dbt-osmosis` for YAML integration commands.

References:

- `demo_duckdb/integration_tests.sh:15-18`
- `wiki:ci-compatibility-matrix` failure mode for missing `UV_NO_SYNC=1` and matrix runtime resync.
- `evidence:c10lock07-local-dependency-resolution-verification` local evidence did not run the full matrix integration step.

Why it matters:

Prior accepted evidence showed `uv run` can resync a matrix job back to lockfile/project dependency state. If CI integration coverage uses `uv run`, the matrix can either fail for the wrong reason or silently exercise a different dbt runtime than the matrix row claims.

Follow-up:

Change the integration step to use the already-installed matrix environment directly, or otherwise prove that the integration `uv run` path cannot sync or switch away from the matrix runtime. Record follow-up critique before acceptance.

Challenges:

- `ticket:c10lock07#ACC-002`
- `ticket:c10lock07#ACC-003`
- `ticket:c10lock07#ACC-004`

# Evidence Reviewed

- Current uncommitted diff for `.github/workflows/tests.yml` and `Taskfile.yml` before the integration-step follow-up.
- `demo_duckdb/integration_tests.sh:15-18` showing `uv run dbt-osmosis` calls.
- `packet:ralph-ticket-c10lock07-20260503T234103Z` child output and parent merge notes.
- `ticket:c10lock07` claim matrix and critique disposition.
- `evidence:c10lock07-local-dependency-resolution-verification` local verification record.
- `wiki:ci-compatibility-matrix` accepted explanation of the `uv run` resync failure mode.

# Residual Risks

- Full GitHub Actions matrix evidence is still pending.
- Local evidence did not run the full pytest/integration matrix.
- The existing uv-only protobuf override remains outside this ticket and is owned by packaging follow-up.

# Required Follow-up

- Fix the integration-step `uv run` environment/sync hole before acceptance.
- Run a follow-up critique pass after the fix.
- Commit/push and record final GitHub Actions evidence before closure.

# Acceptance Recommendation

`follow-up-needed-before-acceptance`.

The ticket must disposition `critique:c10lock07-dependency-resolution#FIND-001` before acceptance.
