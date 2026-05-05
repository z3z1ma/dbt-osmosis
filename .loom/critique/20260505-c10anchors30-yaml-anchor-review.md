---
id: critique:c10anchors30-yaml-anchor-review
kind: critique
status: final
created_at: 2026-05-05T04:11:56Z
updated_at: 2026-05-05T04:11:56Z
scope:
  kind: repository
  repositories:
    - repo:root
review_target: "ticket:c10anchors30 dirty diff against 2c3cf96e102360a2e09af6a4e4d9d4a892956276"
links:
  initiative:
    - initiative:dbt-110-111-hardening
  ticket:
    - ticket:c10anchors30
  evidence:
    - evidence:c10anchors30-yaml-anchor-validation
  packets:
    - packet:ralph-ticket-c10anchors30-20260505T015001Z
    - packet:ralph-ticket-c10anchors30-20260505T020517Z
    - packet:ralph-ticket-c10anchors30-20260505T021703Z
    - packet:ralph-ticket-c10anchors30-20260505T022457Z
    - packet:ralph-ticket-c10anchors30-20260505T023255Z
    - packet:ralph-ticket-c10anchors30-20260505T024405Z
external_refs: {}
---

# Summary

Reviewed the `ticket:c10anchors30` implementation that preserves cross-section YAML anchors and aliases through normal schema read/write flows while preserving managed/unmanaged formatting behavior.

# Review Target

Review target was the dirty worktree diff against `2c3cf96e102360a2e09af6a4e4d9d4a892956276` for:

- `src/dbt_osmosis/core/schema/reader.py`
- `src/dbt_osmosis/core/schema/writer.py`
- `tests/core/test_schema.py`
- `ticket:c10anchors30`
- six Ralph packets for `ticket:c10anchors30`
- `evidence:c10anchors30-yaml-anchor-validation`

Profiles applied: code-change, data-preservation, test-coverage.

# Verdict

`pass_with_findings`

ACC-001, ACC-002, ACC-003, and ACC-005 are satisfied within the ticket's support policy. ACC-004 is not applicable because the supported path was chosen. No blocking code findings remain. The findings below are low-severity record/coverage/residual-risk items that do not require code changes before acceptance.

# Findings

## FIND-001: Ticket record had stale closure fields during review

Severity: low

Confidence: high

State: resolved_by_parent

Observation: During review, `.loom/tickets/20260503-c10anchors30-validate-yaml-anchors-aliases-preservation.md` still had pending implementation, critique, retrospective, and acceptance fields even though implementation and evidence had landed.

Why it matters: Stale owner fields can make future continuation ambiguous even when code and evidence are sound.

Follow-up: Parent reconciled the ticket after this critique by updating critique disposition, retrospective/promotion disposition, acceptance basis, residual risks, and claim statuses.

Challenges:

- None - record hygiene issue only.

## FIND-002: Quoted top-level managed keys are not normalized

Severity: low

Confidence: medium

State: accepted_risk

Observation: `src/dbt_osmosis/core/schema/reader.py` normalizes nested managed mapping keys, but `src/dbt_osmosis/core/schema/writer.py` preserves original top-level order and key objects. A quoted top-level `"models":` key can remain quoted under default `preserve_quotes=False`.

Why it matters: This is an edge-case formatting inconsistency, not data loss. dbt schema top-level keys are rarely quoted, and preserving original top-level key objects is part of keeping unmanaged anchors before managed aliases.

Follow-up: Accepted risk. Do not add top-level key rewriting for this ticket.

Challenges:

- None - does not challenge the ticket's anchor/alias preservation acceptance criteria.

## FIND-003: dbt parse compatibility evidence is not automated in pytest

Severity: low

Confidence: high

State: accepted_risk

Observation: `tests/core/test_schema.py` covers ruamel parse/object identity for the cross-section anchor fixture. dbt parse compatibility is recorded in `evidence:c10anchors30-yaml-anchor-validation` from a temporary dbt 1.10.20 / duckdb 1.10.0 fixture rather than a durable automated pytest.

Why it matters: Future dbt parser drift for this exact fixture would not be caught by the current test module alone.

Follow-up: Accepted risk for this ticket. CI will still run the schema test module; broader dbt parser matrix coverage can be added later if this edge case recurs.

Challenges:

- None - the ticket has direct evidence for the required dbt parse compatibility claim.

# Evidence Reviewed

- `ticket:c10anchors30`
- `packet:ralph-ticket-c10anchors30-20260505T015001Z`
- `packet:ralph-ticket-c10anchors30-20260505T020517Z`
- `packet:ralph-ticket-c10anchors30-20260505T021703Z`
- `packet:ralph-ticket-c10anchors30-20260505T022457Z`
- `packet:ralph-ticket-c10anchors30-20260505T023255Z`
- `packet:ralph-ticket-c10anchors30-20260505T024405Z`
- `evidence:c10anchors30-yaml-anchor-validation`
- Dirty diff for the schema reader/writer/test changes
- Reviewer fresh local validation: `uv run pytest tests/core/test_schema.py -q` -> `28 passed, 2 warnings`

# Residual Risks

- Accepted risk: mutating an alias-bearing managed subtree in place can mutate the shared unmanaged anchor object because the implementation preserves YAML shared-node semantics.
- Accepted risk: dbt parse compatibility is evidenced for dbt 1.10.20 / duckdb 1.10.0 and not automated across a broader dbt matrix.
- Accepted risk: quoted top-level managed keys may remain quoted.
- Support remains scoped to normal `_read_yaml()` / `_write_yaml()` flows that retain ruamel object identity.

# Required Follow-up

None before acceptance.

# Acceptance Recommendation

`accept_with_recorded_risks`

The ticket should accept the implementation after local packaging checks and remote CI complete. The accepted residual risks are documented and do not require additional code changes for this ticket.
