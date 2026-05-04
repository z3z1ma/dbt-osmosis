---
id: ticket:c10val27
kind: ticket
status: closed
change_class: code-behavior
risk_class: medium
created_at: 2026-05-03T21:10:43Z
updated_at: 2026-05-04T16:26:17Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  initiative:
    - initiative:dbt-110-111-hardening
  evidence:
    - evidence:oracle-backlog-scan
    - evidence:c10val27-validation-timeout-verification
  critique:
    - critique:c10val27-validation-timeout-review
  wiki:
    - wiki:model-validation-timeouts
  packets:
    - packet:ralph-ticket-c10val27-20260504T152817Z
depends_on: []
---

# Summary

Make model validation `timeout_seconds` real, or rename/remove it so the validation API does not imply protection from long-running queries.

# Context

`src/dbt_osmosis/core/validation.py:123-128` exposes `timeout_seconds`, but `_validate_single_model()` at `validation.py:264-302` executes normally and only classifies timeout by checking whether an exception string contains `timeout`. Long-running queries can hang indefinitely.

# Why Now

Validation commands may be used as compatibility or release checks. A misleading timeout parameter can hang CI or user workflows.

# Scope

- Enforce timeout through adapter/query settings, worker futures, or another safe execution boundary.
- If enforcing timeout is not feasible per adapter, rename/remove the parameter and document limitations.
- Add tests with mocked long-running execution to prove behavior returns `TIMEOUT` without waiting indefinitely.
- Preserve compile-error and execution-error classification semantics.

# Out Of Scope

- Building adapter-specific cancellation for every warehouse unless needed for a supported behavior claim.
- Changing SQL compile behavior beyond coordinating with ticket:c10sql21 if necessary.

# Acceptance Criteria

- ACC-001: A validation query that exceeds `timeout_seconds` returns `ModelValidationStatus.TIMEOUT` without waiting for full query completion in tests.
- ACC-002: Timeout handling does not leave leaked worker threads/connections where the chosen mechanism can control them.
- ACC-003: If true timeout cannot be guaranteed, the API and docs no longer imply it can.
- ACC-004: Compile errors and execution errors remain classified correctly.
- ACC-005: Tests cover success, compile error, execution error, and timeout cases.

# Coverage

Covers:

- ticket:c10val27#ACC-001
- ticket:c10val27#ACC-002
- ticket:c10val27#ACC-003
- ticket:c10val27#ACC-004
- ticket:c10val27#ACC-005
- initiative:dbt-110-111-hardening#OBJ-006

# Claim Matrix

| Claim | Evidence | Critique | Status |
| --- | --- | --- | --- |
| ticket:c10val27#ACC-001 | evidence:c10val27-validation-timeout-verification | critique:c10val27-validation-timeout-review | accepted |
| ticket:c10val27#ACC-002 | evidence:c10val27-validation-timeout-verification | critique:c10val27-validation-timeout-review | accepted |
| ticket:c10val27#ACC-003 | evidence:c10val27-validation-timeout-verification | critique:c10val27-validation-timeout-review | accepted |
| ticket:c10val27#ACC-004 | evidence:c10val27-validation-timeout-verification | critique:c10val27-validation-timeout-review | accepted |
| ticket:c10val27#ACC-005 | evidence:c10val27-validation-timeout-verification | critique:c10val27-validation-timeout-review | accepted |

# Execution Notes

Be honest about adapter limitations. A test-level future timeout may classify timeout but not cancel the underlying database query; document that if it is the chosen behavior.

# Blockers

None. Adapter-level cancellation support varies, so accepted behavior is explicitly scoped to best-effort local timeout reporting rather than guaranteed warehouse-side cancellation.

# Evidence

Existing evidence: `evidence:oracle-backlog-scan`.

Validation evidence: `evidence:c10val27-validation-timeout-verification`.

Implementation commit: `1df6090494e8b19c5523267cbd47772fad4df8ec` (`fix: enforce model validation timeouts`).

Observed validation:

- `uv run pytest tests/core/test_model_validation.py tests/core/test_cli.py tests/core/test_llm.py -q` passed `73 passed, 9 skipped`.
- `uv run ruff check ... && git diff --check` passed.
- Local basedpyright summary reported `errorCount: 0` after also fixing the main-branch CI type errors observed in GitHub Actions Tests run `25327641090`.
- Targeted `uv run pre-commit run --files ...` passed.

# Critique Disposition

Risk class: medium

Critique policy: recommended

Policy rationale: Validation runtime behavior can hang CI/user workflows.

Required critique profiles: code-change, test-coverage, operator-clarity

Critique record: `critique:c10val27-validation-timeout-review`.

Verdict: `pass`.

Findings: None - no open findings. Initial critique concerns about daemon-thread leakage, cancellation overclaiming, and missing positive-timeout success/error tests were resolved before final acceptance.

Disposition status: completed

Deferral / not-required rationale: None.

# Retrospective / Promotion Disposition

Disposition status: completed

Promoted: `wiki:model-validation-timeouts`.

Deferred / not-required rationale: Retrospective found one durable lesson: validation timeouts are best-effort local wait boundaries, not guaranteed warehouse cancellation. That explanation was promoted to wiki.

# Wiki Disposition

Completed: `wiki:model-validation-timeouts` preserves the accepted timeout semantics, limitations, and future adapter-specific follow-up boundary.

# Acceptance Decision

Accepted by: OpenCode.
Accepted at: 2026-05-04T16:26:17Z.
Basis: `evidence:c10val27-validation-timeout-verification`, `critique:c10val27-validation-timeout-review`, and retrospective promotion `wiki:model-validation-timeouts` cover all scoped acceptance criteria with no critique blockers.
Residual risks: Signal-based timeout is POSIX/main-thread/process-global and best-effort for real adapter/C-extension calls; warehouse-side query cancellation remains adapter-specific and not guaranteed; full GitHub Actions validation after push remains pending outside this ticket's local acceptance.

# Dependencies

Coordinate with ticket:c10sql21 if execution helpers change.

# Journal

- 2026-05-03T21:10:43Z: Created from core architecture oracle finding.
- 2026-05-04T15:28:17Z: Activated ticket and compiled Ralph packet `packet:ralph-ticket-c10val27-20260504T152817Z` for test-first timeout semantics work in `core/validation.py` and `tests/core/test_model_validation.py`.
- 2026-05-04T16:26:17Z: Accepted and closed after implementation commit `1df6090494e8b19c5523267cbd47772fad4df8ec`, evidence `evidence:c10val27-validation-timeout-verification`, critique `critique:c10val27-validation-timeout-review`, and retrospective promotion `wiki:model-validation-timeouts`.
