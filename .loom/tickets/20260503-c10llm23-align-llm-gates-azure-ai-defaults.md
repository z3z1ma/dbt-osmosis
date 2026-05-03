---
id: ticket:c10llm23
kind: ticket
status: ready
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
| ticket:c10llm23#ACC-004 | evidence:oracle-backlog-scan | None | open |
| ticket:c10llm23#ACC-006 | None - optional dependency tests not written yet | None | open |

# Execution Notes

Mock OpenAI/Azure clients rather than making network calls. Keep sensitive env values out of logs and Loom records.

# Blockers

Human decision may be needed if changing `test suggest` from AI-by-default to opt-in changes public behavior.

# Evidence

Existing evidence: evidence:oracle-backlog-scan. Missing evidence: CLI tests and docs diff.

# Critique Disposition

Risk class: medium

Critique policy: recommended

Policy rationale: Optional dependency and auth paths are user-facing and can leak confusing errors, but tests can isolate behavior.

Required critique profiles: code-change, operator-clarity, security

Findings: None - no critique yet.

Disposition status: pending

Deferral / not-required rationale: None.

# Retrospective / Promotion Disposition

Disposition status: pending

Promoted: None - implementation not complete.

Deferred / not-required rationale: Docs update likely needed.

# Wiki Disposition

N/A - no wiki promotion selected yet.

# Acceptance Decision

Accepted by: Not accepted yet.
Accepted at: N/A.
Basis: Pending tests and docs.
Residual risks: Azure SDK behavior may vary by OpenAI package version.

# Dependencies

Coordinate with ticket:c10pkg10 for extras.

# Journal

- 2026-05-03T21:10:43Z: Created from CLI/SQL/workbench oracle findings.
