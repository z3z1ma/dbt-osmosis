---
id: critique:c10lock07-integration-path-follow-up
kind: critique
status: final
created_at: 2026-05-04T00:19:53Z
updated_at: 2026-05-04T00:19:53Z
scope:
  kind: repository
  repositories:
    - repo:root
review_target: "ticket:c10lock07 integration-step follow-up diff"
links:
  tickets:
    - ticket:c10lock07
  critique:
    - critique:c10lock07-dependency-resolution
  evidence:
    - evidence:c10lock07-local-dependency-resolution-verification
external_refs:
  github_issue: https://github.com/z3z1ma/dbt-osmosis/issues/358
---

# Summary

Follow-up critique for the c10lock07 integration-step fix that removed `uv run` from CI matrix integration coverage.

# Review Target

Reviewed the current uncommitted diff after the follow-up patch. The key target is `.github/workflows/tests.yml:139-154`, where CI now invokes `dbt-osmosis` directly from the matrix environment instead of calling `demo_duckdb/integration_tests.sh`.

# Verdict

`pass`.

The high-severity matrix integration finding is resolved in the implementation shape. Matrix installation and dependency checks use explicit clean environments, `uv --no-config pip install/check`, and direct commands from the matrix environment for parse, pytest, and YAML integration coverage.

# Findings

None - no open critique findings remain in this follow-up pass.

# Evidence Reviewed

- Current `git status --short` and uncommitted diff for `.github/workflows/tests.yml` and `Taskfile.yml`.
- `.github/workflows/tests.yml:35-37` lock freshness job.
- `.github/workflows/tests.yml:85-124` clean matrix environment, `uv --no-config pip install`, version assertions, and dependency consistency check.
- `.github/workflows/tests.yml:139-154` fixed integration step using direct `dbt-osmosis` commands from the matrix environment.
- `.github/workflows/tests.yml:190-220` latest-compat clean smoke and `pip check`.
- `.github/workflows/tests.yml:229-278` pip resolver smoke.
- `Taskfile.yml:64-184`, `Taskfile.yml:187-278`, and `Taskfile.yml:280-311` for local lock, clean matrix, latest-compat, and pip smoke parity.
- `ticket:c10lock07`, `evidence:c10lock07-local-dependency-resolution-verification`, and consumed Ralph packet.
- `demo_duckdb/integration_tests.sh:15-18` still contains `uv run`, but CI no longer calls it.

# Residual Risks

- Full GitHub Actions matrix evidence is still pending.
- `demo_duckdb/integration_tests.sh` remains unsafe for matrix use if reintroduced unchanged.
- Existing uv-only protobuf override remains out of scope and owned by packaging follow-up.

# Required Follow-up

- Commit/push, run GitHub Actions, and record final CI evidence.
- Do not close `ticket:c10lock07` until CI evidence and critique disposition are reconciled.
- Avoid reusing `demo_duckdb/integration_tests.sh` in matrix CI unless its `uv run` calls are removed or made sync-safe.

# Acceptance Recommendation

`ticket-acceptance-review-needed`.

No further critique blockers remain, but the ticket still needs final CI evidence before acceptance.
