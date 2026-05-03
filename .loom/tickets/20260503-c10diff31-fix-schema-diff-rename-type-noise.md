---
id: ticket:c10diff31
kind: ticket
status: proposed
change_class: code-behavior
risk_class: medium
created_at: 2026-05-03T21:10:43Z
updated_at: 2026-05-03T21:10:43Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  initiative:
    - initiative:dbt-110-111-hardening
  evidence:
    - evidence:oracle-backlog-scan
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
| ticket:c10diff31#ACC-001 | evidence:oracle-backlog-scan | None | open |
| ticket:c10diff31#ACC-003 | evidence:oracle-backlog-scan | None | open |

# Execution Notes

Keep normalization conservative. Adapter-specific aliases such as `string` versus `varchar` may need separate policy if not already supported.

# Blockers

None.

# Evidence

Existing evidence: evidence:oracle-backlog-scan. Missing evidence: focused diff tests.

# Critique Disposition

Risk class: medium

Critique policy: recommended

Policy rationale: User-facing diff output can affect migration decisions.

Required critique profiles: code-change, test-coverage, operator-clarity

Findings: None - no critique yet.

Disposition status: pending

Deferral / not-required rationale: None.

# Retrospective / Promotion Disposition

Disposition status: pending

Promoted: None - implementation not complete.

Deferred / not-required rationale: Not likely wiki-worthy unless diff becomes a release gate.

# Wiki Disposition

N/A - no wiki promotion selected yet.

# Acceptance Decision

Accepted by: Not accepted yet.
Accepted at: N/A.
Basis: Pending tests.
Residual risks: Adapter type aliases can be broader than simple normalization.

# Dependencies

Coordinate with ticket:c10lint24 for CLI diff selector behavior if implemented nearby.

# Journal

- 2026-05-03T21:10:43Z: Created from core architecture oracle finding.
