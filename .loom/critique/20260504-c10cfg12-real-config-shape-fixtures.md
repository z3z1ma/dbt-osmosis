---
id: critique:c10cfg12-real-config-shape-fixtures
kind: critique
status: final
created_at: 2026-05-04T06:49:00Z
updated_at: 2026-05-04T06:49:00Z
scope:
  kind: repository
  repositories:
    - repo:root
review_target: "ticket:c10cfg12 real parsed config-shape fixture diff from a92a225 working tree"
links:
  tickets:
    - ticket:c10cfg12
    - ticket:c10col01
    - ticket:c10meta02
  evidence:
    - evidence:c10cfg12-real-config-shape-fixtures
  packets:
    - packet:ralph-ticket-c10cfg12-20260504T063203Z
external_refs: {}
---

# Summary

Reviewed the `ticket:c10cfg12` real parsed config-shape fixture diff after Ralph implementation and the critique-driven missing-column precondition fix. The review focused on test coverage, dbt compatibility, fixture isolation, and support for `ticket:c10cfg12#ACC-001` through `#ACC-006` plus converted follow-up gaps from `ticket:c10col01#ACC-005` and `ticket:c10meta02#ACC-005`.

# Review Target

Target: `ticket:c10cfg12` uncommitted working-tree diff on branch `loom/dbt-110-111-hardening`, based on commit `a92a225f0d0f616bc6b7d41788a5542c22e7bc9d`.

Reviewed changed surfaces:

- `tests/core/test_real_config_shapes.py`
- `packet:ralph-ticket-c10cfg12-20260504T063203Z`
- `evidence:c10cfg12-real-config-shape-fixtures`
- `ticket:c10cfg12`

# Verdict

`pass_with_findings`.

The implementation is acceptable to commit: no blocking or medium implementation finding remains after adding the missing-column precondition assertion. The ticket must not close or fully accept the dbt 1.11 portions of the converted claims until post-commit matrix CI or equivalent dbt 1.11 evidence passes.

# Findings

## FIND-001: dbt 1.11 converted claims still need post-commit matrix evidence

Severity: high
Confidence: high
State: open

Observation:

Local evidence observed dbt-core `1.10.20` and dbt-duckdb `1.10.0`. The new tests are designed to run in the CI matrix for dbt 1.10/1.11, but local evidence alone does not prove the exact dbt 1.11 converted fixture gaps are closed.

Why it matters:

The ticket specifically owns dbt 1.10/1.11 parsed fixture coverage and converted exact dbt 1.11 gaps from `ticket:c10col01#ACC-005` and `ticket:c10meta02#ACC-005`. Closing those claims from local 1.10 evidence would overstate the support proof.

Follow-up:

Commit the implementation and record post-commit `Tests` matrix evidence for dbt 1.11 before ticket-owned acceptance/closure. The ticket should consume this finding as `resolved` only after dbt 1.11 matrix evidence passes.

Challenges:

- ticket:c10cfg12#ACC-002
- ticket:c10cfg12#ACC-006
- ticket:c10col01#ACC-005
- ticket:c10meta02#ACC-005

## FIND-002: Missing-column injection test lacked an absence precondition

Severity: medium
Confidence: high
State: open

Observation:

The initial missing-column injection test asserted the final injected/sync-serialized state but did not assert `warehouse_only_col` was absent before `inject_missing_columns()` ran.

Why it matters:

Without the precondition, the test weakly proved adapter-backed missing-column injection because the target column could have already existed in the parsed manifest.

Follow-up:

Resolved in the reviewed diff by adding `assert "warehouse_only_col" not in fixture.node.columns` before injection and rerunning focused/broader tests. The ticket should consume this finding as `resolved` with evidence from `evidence:c10cfg12-real-config-shape-fixtures`.

Challenges:

- ticket:c10col01#ACC-005

# Evidence Reviewed

Reviewed the current diff, Ralph child output, local dbt version output, focused test output, broader test output, artifact guard output, targeted pre-commit output, `git diff --check`, and follow-up critique output.

Key evidence record:

- evidence:c10cfg12-real-config-shape-fixtures

# Residual Risks

- dbt-core `<1.10` skips the new shape assertions because the fixture targets dbt 1.10+ column config namespace.
- Local evidence is dbt 1.10 only; dbt 1.11 acceptance depends on post-commit matrix CI or equivalent evidence.
- The implementation is test-only and does not validate unrelated resolver behavior outside the asserted config-shape surfaces.

# Required Follow-up

No critique-required implementation follow-up remains before committing. Before closure, the ticket must record dbt 1.11 post-commit matrix CI evidence and consume `FIND-001` with a ticket-owned disposition. The ticket should also reconcile the claim matrix, evidence links, critique disposition, and retrospective / promotion disposition.

# Acceptance Recommendation

`ticket-acceptance-review-needed`.
