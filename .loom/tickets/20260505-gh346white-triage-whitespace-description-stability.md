---
id: ticket:gh346white
kind: ticket
status: proposed
change_class: code-behavior
risk_class: medium
created_at: 2026-05-05T08:33:14Z
updated_at: 2026-05-05T08:33:14Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  initiative:
    - initiative:issue-pr-zero
external_refs:
  github_pr: https://github.com/z3z1ma/dbt-osmosis/pull/346
  related_github_issue: https://github.com/z3z1ma/dbt-osmosis/issues/311
depends_on: []
---

# Summary

Triage and, if still reproducible, fix the residual whitespace-only description and folded-scalar idempotency behaviors surfaced in external PR #346.

# Context

PR #346 is conflicting, failing CI, and too broad to merge. It mixes at least three concerns: whitespace-only descriptions in inheritance, folded-scalar idempotency, and a large rename-description enrichment feature with a direct git dependency on a forked lineage package. The narrow nested-description trailing-space behavior from issue #311 was already fixed in `ticket:gh311wrap` and pushed in commit `72301c2`.

# Why Now

The PR should be closed rather than merged, but one remaining claim may be valid: whitespace-only descriptions may still be treated as meaningful text in inheritance, and folded scalar values may still need an idempotency check outside the already-fixed nested wrap threshold.

# Scope

- Reproduce whether whitespace-only local descriptions block upstream inheritance on current `main`.
- Reproduce whether whitespace-only upstream descriptions propagate downstream on current `main`.
- Reproduce whether existing folded scalar descriptions oscillate across consecutive YAML refactor/sync runs on current `main`.
- If confirmed, implement the smallest fix using existing schema/inheritance paths and add focused regressions.

# Out Of Scope

- Merging PR #346 as-is.
- Adding `dbt-col-lineage` or any direct git dependency.
- Implementing rename-description enrichment.
- Broad YAML formatter redesign.

# Acceptance Criteria

- ACC-001: Current-main behavior for whitespace-only local descriptions is reproduced and either fixed or recorded as already acceptable.
- ACC-002: Current-main behavior for whitespace-only upstream descriptions is reproduced and either fixed or recorded as already acceptable.
- ACC-003: Folded-scalar idempotency across consecutive refactor/sync runs is reproduced and either fixed or recorded as already acceptable.
- ACC-004: Any fix includes focused tests and preserves existing description inheritance and YAML scalar behavior.
- ACC-005: The external PR is closed with a comment explaining the accepted, rejected, and follow-up portions.

# Coverage

Covers:

- initiative:issue-pr-zero#OBJ-003
- initiative:issue-pr-zero#OBJ-005

# Claim Matrix

| Claim | Evidence | Critique | Status |
| --- | --- | --- | --- |
| ticket:gh346white#ACC-001 | None yet | None yet | open |
| ticket:gh346white#ACC-002 | None yet | None yet | open |
| ticket:gh346white#ACC-003 | None yet | None yet | open |
| ticket:gh346white#ACC-004 | None yet | None yet | open |
| ticket:gh346white#ACC-005 | None yet | None yet | open |

# Execution Notes

Parent triage found current `main` still checks `node_column.description` truthiness in `transforms.py` and `graph_edge.get("description") == ""` in `inheritance.py`, so whitespace-only strings may still be behaviorally relevant. The fold-point claim needs fresh reproduction because `ticket:gh311wrap` changed nested description wrapping but did not intentionally normalize existing `FoldedScalarString` values.

# Blockers

None.

# Evidence

Expected evidence: reproduce-first output for the whitespace-only and folded-scalar claims, focused tests if fixed, and critique if behavior changes.

# Critique Disposition

Risk class: medium

Critique policy: recommended

Policy rationale: Description inheritance and YAML formatting behavior are user-visible and can create silent documentation or diff churn.

Required critique profiles:

- inheritance-behavior
- yaml-formatting

Findings:

None - no critique yet.

Disposition status: pending

Deferral / not-required rationale: N/A.

# Retrospective / Promotion Disposition

Disposition status: pending

Promoted:

None yet.

Deferred / not-required rationale: N/A.

# Wiki Disposition

N/A - no wiki promotion selected yet.

# Acceptance Decision

Accepted by: Pending reproduction and evidence.
Accepted at: N/A.
Basis: N/A.
Residual risks: N/A.

# Dependencies

None.

# Journal

- 2026-05-05T08:33:14Z: Created while closing external PR #346. The PR is not mergeable and contains broad rejected scope, but whitespace-only description and folded-scalar idempotency claims are worth a bounded reproduce-first follow-up.
