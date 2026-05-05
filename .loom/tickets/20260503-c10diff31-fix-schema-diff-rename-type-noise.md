---
id: ticket:c10diff31
kind: ticket
status: complete_pending_acceptance
change_class: code-behavior
risk_class: medium
created_at: 2026-05-03T21:10:43Z
updated_at: 2026-05-05T00:52:53Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  initiative:
    - initiative:dbt-110-111-hardening
  evidence:
    - evidence:oracle-backlog-scan
    - evidence:c10diff31-schema-diff-validation
  critique:
    - critique:c10diff31-schema-diff-review
depends_on: []
---

# Summary

Make schema diff type comparisons and rename detection deterministic and semantically accurate.

# Context

`src/dbt_osmosis/core/diff.py:343-355` compares raw type strings before classifying type changes, so case-only or formatting-only differences can emit noisy changes. `diff.py:305-328` and `407-437` detect renames but do not reserve matched added columns, so multiple removed columns can map to the same added column.

# Why Now

`diff schema` is a user-facing command and may become part of compatibility/release validation. Noisy or ambiguous diffs reduce trust in the tool.

# Scope

- Normalize comparable data types before emitting changes.
- Preserve original type strings in output when a real change exists.
- Make rename matching one-to-one between removed and added columns.
- Add tests for case-only type differences and competing rename matches.

# Out Of Scope

- Full semantic type compatibility across every adapter.
- Changing severity taxonomy unless tests show current taxonomy is wrong.

# Acceptance Criteria

- ACC-001: Case-only type differences such as `VARCHAR` versus `varchar` do not emit changes.
- ACC-002: Whitespace/format-only type differences are normalized when safe.
- ACC-003: Rename detection cannot map two removed columns to the same added column.
- ACC-004: Ambiguous rename candidates produce deterministic output.
- ACC-005: Tests cover normalized type comparisons and one-to-one rename matching.

# Coverage

Covers:

- ticket:c10diff31#ACC-001
- ticket:c10diff31#ACC-002
- ticket:c10diff31#ACC-003
- ticket:c10diff31#ACC-004
- ticket:c10diff31#ACC-005
- initiative:dbt-110-111-hardening#OBJ-006

# Claim Matrix

| Claim | Evidence | Critique | Status |
| --- | --- | --- | --- |
| ticket:c10diff31#ACC-001 | evidence:c10diff31-schema-diff-validation | critique:c10diff31-schema-diff-review | supported_pending_remote_ci |
| ticket:c10diff31#ACC-002 | evidence:c10diff31-schema-diff-validation | critique:c10diff31-schema-diff-review#FIND-001 accepted_risk | supported_pending_remote_ci |
| ticket:c10diff31#ACC-003 | evidence:c10diff31-schema-diff-validation | critique:c10diff31-schema-diff-review | supported_pending_remote_ci |
| ticket:c10diff31#ACC-004 | evidence:c10diff31-schema-diff-validation | critique:c10diff31-schema-diff-review#FIND-002 accepted_risk | supported_pending_remote_ci |
| ticket:c10diff31#ACC-005 | evidence:c10diff31-schema-diff-validation | critique:c10diff31-schema-diff-review#FIND-002 accepted_risk | supported_pending_remote_ci |

# Execution Notes

Keep normalization conservative. Adapter-specific aliases such as `string` versus `varchar` may need separate policy if not already supported.

# Blockers

None.

# Evidence

Existing evidence: evidence:oracle-backlog-scan; evidence:c10diff31-schema-diff-validation.

Missing evidence: remote CI after commit/push.

# Critique Disposition

Risk class: medium

Critique policy: recommended

Policy rationale: User-facing diff output can affect migration decisions.

Required critique profiles: code-change, test-coverage, operator-clarity

Findings:

- critique:c10diff31-schema-diff-review#FIND-001 — accepted_risk. Rationale: conservative whitespace normalization is intended for normal adapter type strings; exotic quoted/custom type strings with semantic whitespace are outside the ticket's explicit support target.
- critique:c10diff31-schema-diff-review#FIND-002 — accepted_risk. Rationale: helper-level tests cover the changed rename matching logic, existing `compare_node()` rename coverage still exercises the integration path, and critique marked full competing-rename `compare_node()` coverage optional rather than acceptance-blocking.

Disposition status: completed

Deferral / not-required rationale: Recommended critique completed as critique:c10diff31-schema-diff-review; no required code follow-up remains before acceptance.

# Retrospective / Promotion Disposition

Disposition status: not_required

Promoted: None.

Deferred / not-required rationale: Localized schema diff behavior fix is covered by tests and evidence; no durable explanation promotion is needed unless diff becomes a broader release gate.

# Wiki Disposition

N/A - no wiki promotion selected yet.

# Acceptance Decision

Accepted by: Not accepted yet.
Accepted at: N/A.
Basis: Local red/green evidence and recommended critique exist; remote CI is pending.
Residual risks: Adapter type aliases can be broader than simple normalization; rename matching remains greedy deterministic matching rather than globally optimal assignment.

# Dependencies

Coordinate with ticket:c10lint24 for CLI diff selector behavior if implemented nearby.

# Journal

- 2026-05-03T21:10:43Z: Created from core architecture oracle finding.
- 2026-05-05T00:39:27Z: Promoted `proposed -> ready -> active` after readiness review. No blockers remain, acceptance is concrete, and `packet:ralph-ticket-c10diff31-20260505T003927Z` was compiled for a bounded test-first implementation iteration.
- 2026-05-05T00:46:30Z: Ralph iteration returned `stop`; parent diff review and validation passed (`tests/core/test_diff.py`, Ruff format/check, `git diff --check`, basedpyright zero errors). Recorded evidence:c10diff31-schema-diff-validation and moved ticket to `review_required` for recommended medium-risk critique.
- 2026-05-05T00:52:53Z: Recommended critique completed as critique:c10diff31-schema-diff-review with `pass_with_findings`. Accepted both low findings as scoped risks and moved ticket to `complete_pending_acceptance`; remaining gate is commit/push and remote CI evidence.
