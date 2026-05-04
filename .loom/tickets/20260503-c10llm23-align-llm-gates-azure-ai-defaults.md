---
id: ticket:c10llm23
kind: ticket
status: closed
change_class: code-behavior
risk_class: medium
created_at: 2026-05-03T21:10:43Z
updated_at: 2026-05-04T15:16:22Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  initiative:
    - initiative:dbt-110-111-hardening
  evidence:
    - evidence:oracle-backlog-scan
    - evidence:c10llm23-llm-gate-validation
  critique:
    - critique:c10llm23-llm-gate-review
  packets:
    - packet:ralph-ticket-c10llm23-20260504T144951Z
depends_on: []
---

# Summary

Align `test-llm`, default AI behavior, Azure OpenAI AD wiring, error handling, and optional extras so LLM paths fail clearly and behave consistently.

# Context

`src/dbt_osmosis/cli/main.py:64-97` calls `get_llm_client()` before `test_llm_connection()` and then `test_llm_connection()` ignores its parameter and requires `LLM_PROVIDER`, even though `get_llm_client()` defaults to OpenAI. `src/dbt_osmosis/core/llm.py:228-273` acquires an Azure AD token but constructs a generic `OpenAI` client instead of an Azure-specific client path, and its missing dependency message references a non-existent `dbt-osmosis[azure]` extra. `src/dbt_osmosis/cli/main.py:1729-1802` defaults `--use-ai` to true and can silently fall back.

# Why Now

Optional AI paths are part of the CLI surface and the constitution says optional paths must fail clearly. Hidden fallbacks and invalid install messages make debugging harder during compatibility work.

# Scope

- Make `test-llm` use one client resolution path and report friendly `ClickException` errors.
- Decide whether `LLM_PROVIDER` is required or defaults to OpenAI, and align code/docs/tests.
- Fix Azure OpenAI AD client construction and API-version handling, or clearly mark it unsupported.
- Add or correct an `azure` optional extra if code/docs recommend it.
- Make `test suggest` AI default/fallback behavior explicit, with `--no-use-ai` or opt-in semantics if needed.
- Add tests for missing dependencies/env and successful default OpenAI resolution with mocked clients.

# Out Of Scope

- Improving prompt quality or generated content.
- Adding real network calls in tests.

# Acceptance Criteria

- ACC-001: `test-llm` succeeds with the same default provider logic used by generation paths.
- ACC-002: Missing optional packages or env vars produce clear nonzero CLI errors.
- ACC-003: Azure OpenAI AD uses a correct SDK client/auth shape or is removed/disabled with clear messaging.
- ACC-004: Install instructions reference only real extras or direct dependencies.
- ACC-005: AI test suggestion default behavior is explicit and user-visible on fallback/failure.
- ACC-006: Tests cover missing OpenAI package, missing provider env, default OpenAI, Azure identity absence, and pattern-only behavior.

# Coverage

Covers:

- ticket:c10llm23#ACC-001
- ticket:c10llm23#ACC-002
- ticket:c10llm23#ACC-003
- ticket:c10llm23#ACC-004
- ticket:c10llm23#ACC-005
- ticket:c10llm23#ACC-006
- initiative:dbt-110-111-hardening#OBJ-006
- initiative:dbt-110-111-hardening#OBJ-007

# Claim Matrix

| Claim | Evidence | Critique | Status |
| --- | --- | --- | --- |
| ticket:c10llm23#ACC-001 | evidence:c10llm23-llm-gate-validation | critique:c10llm23-llm-gate-review | accepted |
| ticket:c10llm23#ACC-002 | evidence:c10llm23-llm-gate-validation | critique:c10llm23-llm-gate-review | accepted |
| ticket:c10llm23#ACC-003 | evidence:c10llm23-llm-gate-validation | critique:c10llm23-llm-gate-review | accepted |
| ticket:c10llm23#ACC-004 | evidence:c10llm23-llm-gate-validation | critique:c10llm23-llm-gate-review | accepted |
| ticket:c10llm23#ACC-005 | evidence:c10llm23-llm-gate-validation | critique:c10llm23-llm-gate-review | accepted |
| ticket:c10llm23#ACC-006 | evidence:c10llm23-llm-gate-validation | critique:c10llm23-llm-gate-review | accepted |

# Execution Notes

Mock OpenAI/Azure clients rather than making network calls. Keep sensitive env values out of logs and Loom records.

# Blockers

None. `test suggest` stayed AI-on-by-default and now makes default/fallback behavior explicit, so no human decision was needed for an opt-in behavior change.

# Evidence

Existing evidence: `evidence:oracle-backlog-scan`.

Validation evidence: `evidence:c10llm23-llm-gate-validation`.

Implementation commit: `3fbb1e16acab450c7855e806a144e04613670985` (`fix: clarify LLM optional path failures`).

Observed validation:

- `uv run pytest tests/core/test_llm.py tests/core/test_cli.py tests/core/test_test_suggestions.py tests/test_package_metadata.py -q` passed `106 passed, 9 skipped`.
- `uv run ruff check ... && git diff --check` passed.
- Targeted `uv run pre-commit run --files ...` passed.
- `uv run python -c "import importlib.util; print(importlib.util.find_spec('openai') is not None)"` printed `False`, confirming the base worktree `uv` environment validated without OpenAI installed.

# Critique Disposition

Risk class: medium

Critique policy: recommended

Policy rationale: Optional dependency and auth paths are user-facing and can leak confusing errors, but tests can isolate behavior.

Required critique profiles: code-change, operator-clarity, security

Critique record: `critique:c10llm23-llm-gate-review`.

Verdict: `pass`.

Findings: None - no findings.

Disposition status: completed

Deferral / not-required rationale: None.

# Retrospective / Promotion Disposition

Disposition status: not_required

Promoted: None.

Deferred / not-required rationale: The implementation updated the user-facing Docusaurus CLI and installation docs directly. No durable Loom wiki or research promotion was needed for this ticket-local optional LLM error-surface change.

# Wiki Disposition

Disposition status: not_required

Rationale: The accepted explanation lives in the CLI and installation docs touched by this ticket; no cross-cutting Loom wiki page was needed.

# Acceptance Decision

Accepted by: OpenCode.
Accepted at: 2026-05-04T15:16:22Z.
Basis: `evidence:c10llm23-llm-gate-validation` and `critique:c10llm23-llm-gate-review` cover all scoped acceptance criteria with no critique blockers.
Residual risks: Real OpenAI/Azure service connectivity and long-lived Azure AD token refresh behavior were not tested by scope; final initiative-level CI/GitHub Actions validation remains pending outside this ticket.

# Dependencies

Coordinate with ticket:c10pkg10 for extras.

# Journal

- 2026-05-03T21:10:43Z: Created from CLI/SQL/workbench oracle findings.
- 2026-05-04T14:49:51Z: Activated ticket and compiled Ralph packet `packet:ralph-ticket-c10llm23-20260504T144951Z` for test-first LLM client/default-provider, Azure AD, optional dependency, and AI fallback visibility work with recommended critique before acceptance.
- 2026-05-04T15:16:22Z: Accepted and closed after implementation commit `3fbb1e16acab450c7855e806a144e04613670985`, parent validation evidence `evidence:c10llm23-llm-gate-validation`, and final critique `critique:c10llm23-llm-gate-review` with no findings.
