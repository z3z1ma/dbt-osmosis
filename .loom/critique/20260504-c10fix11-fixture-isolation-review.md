---
id: critique:c10fix11-fixture-isolation-review
kind: critique
status: final
created_at: 2026-05-04T05:28:41Z
updated_at: 2026-05-04T05:28:41Z
scope:
  kind: repository
  repositories:
    - repo:root
review_target: "ticket:c10fix11 fixture-isolation diff from b7bdc45 working tree"
links:
  tickets:
    - ticket:c10fix11
  evidence:
    - evidence:c10fix11-fixture-isolation-verification
  packets:
    - packet:ralph-ticket-c10fix11-20260504T050248Z
external_refs: {}
---

# Summary

Reviewed the `ticket:c10fix11` fixture-isolation diff after Ralph implementation and critique-driven follow-up fixes. The review focused on fixture-copy exclusions, DuckDB profile path isolation, source-tree manifest removal, integration smoke safety, CI guard coverage, and evidence strength for `ticket:c10fix11#ACC-001` through `ticket:c10fix11#ACC-006`.

# Review Target

Target: `ticket:c10fix11` uncommitted working-tree diff on branch `loom/dbt-110-111-hardening`, based on commit `b7bdc45db5b97e41ffdbf2cd8d7174585a9769bf`.

Reviewed changed surfaces:

- `tests/support.py`
- `tests/conftest.py`
- `tests/core/conftest.py`
- `tests/core/test_legacy.py`
- `tests/core/test_demo_fixture_support.py`
- `tests/test_fixture_isolation.py`
- `demo_duckdb/integration_tests.sh`
- `.github/workflows/tests.yml`
- `Taskfile.yml`
- `packet:ralph-ticket-c10fix11-20260504T050248Z`
- `evidence:c10fix11-fixture-isolation-verification`

# Verdict

`pass_with_findings`.

The initial review found two medium-severity fixture/CI coverage gaps. The current diff addresses them by expanding default fixture-copy exclusions for representative ignored/local artifacts and adding CI source-artifact guards before and after matrix validation. No critique blocker remains before the ticket acceptance gate consumes the evidence and finding dispositions.

# Findings

## FIND-001: Fixture artifact exclusion was too narrow

Severity: medium
Confidence: high
State: open

Observation:

The initial helper excluded known dbt build directories and database files, but common ignored/local artifacts such as `.cache`, `.env`, `.DS_Store`, `*.log`, sqlite files, and bytecode could still be copied into temp fixtures.

Why it matters:

`ticket:c10fix11#ACC-005` requires fixture copies to exclude generated and ignored artifacts unless intentionally included. A narrow denylist could let local or sensitive-ish operator files leak into copied test projects and make compatibility checks depend on ambient checkout state.

Follow-up:

Resolved in the reviewed diff by expanding the default fixture-copy denylist and adding regression cases for representative ignored/local artifacts. The ticket should consume this finding as `resolved` with evidence from focused pytest.

Challenges:

- ticket:c10fix11#ACC-005

## FIND-002: CI artifact guards did not cover pytest-created source artifacts

Severity: medium
Confidence: high
State: open

Observation:

The integration script guarded against source artifacts only relative to its own start state. In CI, pytest runs before the integration script, so a pytest-created repo-root `test.db` or source `demo_duckdb/target` would be treated as pre-existing by the script rather than failed as a source-tree artifact leak.

Why it matters:

`ticket:c10fix11#ACC-002` and `ticket:c10fix11#ACC-006` rely on tests not creating or requiring source-tree fixture artifacts. The CI workflow needs a job-level guard around parse, pytest, and integration behavior, not only an integration-script-local guard.

Follow-up:

Resolved in the reviewed diff by adding CI source-artifact guard checks before and after matrix validation and latest-core validation. The ticket should consume this finding as `resolved` with evidence from the workflow diff, focused static test coverage, and targeted hooks.

Challenges:

- ticket:c10fix11#ACC-002
- ticket:c10fix11#ACC-006

# Evidence Reviewed

Reviewed local diff, packet child output, focused red/green test notes, current focused pytest output, integration smoke output, `task parse-demo` output, post-run artifact guard output, targeted pre-commit output, `git diff --check`, and follow-up critique output.

Key evidence record:

- evidence:c10fix11-fixture-isolation-verification

# Residual Risks

- Fixture exclusion remains denylist-based rather than fully `.gitignore`-aware; future ignored artifact classes must be added deliberately.
- Local evidence does not execute the full GitHub Actions matrix; final acceptance should still cite post-commit CI evidence.
- `task parse-demo` temp copies are left to operating-system temp cleanup, but the source-tree isolation claims are covered.
- Existing operator/wiki guidance that describes the old unsafe integration script should be reconciled during acceptance/retrospective follow-through.

# Required Follow-up

No critique-required implementation follow-up remains. Before closure, the ticket should record both open findings above with ticket-owned disposition `resolved`, preserve post-commit CI evidence if required by the workflow, and complete or explicitly disposition retrospective / promotion follow-through.

# Acceptance Recommendation

`no-critique-blockers`.
