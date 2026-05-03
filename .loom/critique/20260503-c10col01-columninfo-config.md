---
id: critique:c10col01-columninfo-config
kind: critique
status: final
created_at: 2026-05-03T22:04:00Z
updated_at: 2026-05-03T22:04:00Z
scope:
  kind: repository
  repositories:
    - repo:root
review_target: "ticket:c10col01 implementation diff on branch loom/dbt-110-111-hardening"
links:
  ticket:
    - ticket:c10col01
  packet:
    - packet:ralph-ticket-c10col01-20260503T214308Z
    - packet:ralph-ticket-c10col01-20260503T215123Z
external_refs:
  github_issue: https://github.com/z3z1ma/dbt-osmosis/issues/352
---

# Summary

Reviewed the `ticket:c10col01` implementation that stops `inject_missing_columns()` from deleting `ColumnInfo.config` and adds regression coverage for sync serialization under classic and fusion-compatible output modes.

# Review Target

Target: current uncommitted diff on branch `loom/dbt-110-111-hardening` for:

- `src/dbt_osmosis/core/transforms.py`
- `tests/core/test_transforms.py`
- Ralph packets for ticket:c10col01 iterations 1 and 2

The review focused on code-change, test-coverage, and dbt-compatibility concerns because the ticket is high risk and has mandatory critique policy.

# Verdict

`pass_with_findings` - the minimal code fix is correct and the local regression coverage supports ACC-001 through ACC-004, but dbt 1.11 adapter-backed evidence for ACC-005 is still missing and must be dispositioned by the ticket before closure.

# Findings

## FIND-001: Adapter-backed dbt 1.11 evidence remains absent

Severity: high
Confidence: high
State: open

Observation:

The implementation removes the invalid `ColumnInfo.config` deletion and local tests pass under the installed environment, but no dbt 1.11 adapter-backed fixture or matrix run has been recorded for the missing-column scenario required by `ticket:c10col01#ACC-005`.

Why it matters:

The initiative explicitly targets dbt 1.10.x and 1.11.x compatibility. Source-level evidence supports the shape of `ColumnInfo.config`, but the ticket acceptance criterion asks for runtime adapter-backed coverage under both version families.

Follow-up:

Before closing ticket:c10col01, either gather the dbt 1.11 adapter-backed evidence directly or record a ticket-owned disposition that converts `ticket:c10col01#ACC-005` to follow-up work under the matrix/config-shape tickets that own broader dbt 1.11 execution.

Challenges:

- ticket:c10col01#ACC-005
- initiative:dbt-110-111-hardening#OBJ-001

## FIND-002: Fusion-compatible empty config coverage gap was resolved

Severity: medium
Confidence: high
State: withdrawn

Observation:

Initial critique observed that the first regression test forced `context.fusion_compat = False`, leaving the dbt 1.10+ fusion-compatible output path unexercised.

Why it matters:

`ticket:c10col01#ACC-003` requires empty generated config blocks not to leak into YAML output; dbt 1.10+ compatibility work needs coverage for the fusion-compatible path as well as classic output.

Follow-up:

No further follow-up required for this finding. Ralph iteration 2 parameterized the regression test over `fusion_compat = False` and `True`, then asserted the YAML-facing synced column does not contain `config` in either mode.

Withdrawal rationale:

The current diff includes `tests/core/test_transforms.py::test_inject_missing_columns_preserves_column_config_for_sync` parameterized over both output modes. Parent reran `uv run pytest tests/core/test_transforms.py` and observed `18 passed in 9.76s`.

Challenges:

- ticket:c10col01#ACC-003

# Evidence Reviewed

- `src/dbt_osmosis/core/transforms.py:356-369` after the change.
- `tests/core/test_transforms.py:227-282` after the second Ralph iteration.
- `src/dbt_osmosis/core/sync_operations.py:81-88` and `src/dbt_osmosis/core/sync_operations.py:175-253` for serialization and empty-config cleanup behavior.
- `packet:ralph-ticket-c10col01-20260503T214308Z` child red/green output.
- `packet:ralph-ticket-c10col01-20260503T215123Z` child coverage output.
- Parent verification: `uv run pytest tests/core/test_transforms.py` passed with `18 passed in 9.76s` on Python 3.13.9.

# Residual Risks

- dbt 1.11 adapter-backed runtime evidence is still absent for this exact missing-column scenario.
- Broader `config.meta` and `config.tags` compatibility remains owned by ticket:c10meta02.

# Required Follow-up

- Ticket-owned disposition is required for critique:c10col01-columninfo-config#FIND-001 before ticket:c10col01 can close.
- The most natural follow-up route is to satisfy or convert ACC-005 through ticket:c10ci06 and/or ticket:c10cfg12, which own dbt 1.11 CI/config-shape fixture coverage.

# Acceptance Recommendation

`risk-disposition-needed` - accept the minimal code change after ticket-owned disposition of FIND-001, but do not close the ticket as fully accepted until ACC-005 has runtime evidence or a truthful conversion to follow-up.
