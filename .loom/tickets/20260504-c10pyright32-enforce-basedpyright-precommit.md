---
id: ticket:c10pyright32
kind: ticket
status: active
change_class: developer-tooling
risk_class: medium
created_at: 2026-05-04T17:16:38Z
updated_at: 2026-05-04T17:16:38Z
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

Add a pre-commit basedpyright gate so type errors are caught before commits and CI can require zero basedpyright errors on every pushed commit.

# Context

The operator reported basedpyright errors surfacing in CI and requested that basedpyright error count be zero on each commit. CI already runs basedpyright and fails when `summary.errorCount` is nonzero, but `.pre-commit-config.yaml` does not currently run basedpyright locally.

# Why Now

The dbt 1.10/1.11 hardening branch is pushing completed ticket slices directly to `origin/main`. A local pre-commit gate reduces the chance that code commits pass local tests but fail CI type checks.

# Scope

- Add a repository pre-commit hook that runs basedpyright for the configured `tool.pyright` scope.
- Keep the hook filename-independent so it checks the full configured type surface on each commit.
- Match the CI zero-error policy by failing on basedpyright errors while allowing existing warnings.
- Include optional OpenAI/Azure SDKs in the hook environment so optional imports do not hide CI-only type errors.
- Validate the hook locally.

# Out Of Scope

- Reducing the existing basedpyright warning count.
- Changing `tool.pyright` include/exclude scope.
- Replacing the CI basedpyright jobs.
- Folding lint/diff CLI behavior from `ticket:c10lint24` into this tooling gate.

# Acceptance Criteria

- ACC-001: `pre-commit run basedpyright --all-files` executes basedpyright rather than file-scoped partial checks.
- ACC-002: The hook fails when basedpyright reports one or more errors and passes when `errorCount` is zero.
- ACC-003: The hook includes the optional SDK surface used to reproduce CI type issues.
- ACC-004: Current branch validation records `errorCount=0` for the hook.
- ACC-005: The hook is documented in `.pre-commit-config.yaml` without changing unrelated hooks.

# Coverage

Covers:

- ticket:c10pyright32#ACC-001
- ticket:c10pyright32#ACC-002
- ticket:c10pyright32#ACC-003
- ticket:c10pyright32#ACC-004
- ticket:c10pyright32#ACC-005
- initiative:dbt-110-111-hardening#OBJ-006

# Claim Matrix

| Claim | Evidence | Critique | Status |
| --- | --- | --- | --- |
| ticket:c10pyright32#ACC-001 | None - implementation pending | None | open |
| ticket:c10pyright32#ACC-004 | None - validation pending | None | open |

# Execution Notes

Prefer a local pre-commit hook that uses `uv run` to reuse the project environment rather than duplicating dependency setup in a separate pre-commit virtualenv.

# Blockers

None.

# Evidence

Existing evidence: local basedpyright checks on recent commits reported `errorCount=0`. Missing evidence: hook-level validation after configuration change.

# Critique Disposition

Risk class: medium

Critique policy: recommended

Policy rationale: This changes developer/CI gating behavior, but the implementation should be a narrow hook addition.

Required critique profiles: developer-tooling, operator-clarity

Findings: None - no critique yet.

Disposition status: pending

Deferral / not-required rationale: None.

# Retrospective / Promotion Disposition

Disposition status: pending

Promoted: None - implementation not complete.
Deferred / not-required rationale: Likely not wiki-worthy; pre-commit config is the accepted surface.

# Wiki Disposition

N/A - no wiki promotion selected yet.

# Acceptance Decision

Accepted by: Not accepted yet.
Accepted at: N/A.
Basis: Pending hook implementation and validation.
Residual risks: Hook runtime may add commit latency because basedpyright checks the full configured project surface.

# Dependencies

None.

# Journal

- 2026-05-04T17:16:38Z: Created from operator request to keep basedpyright errors at zero on each commit and add a pre-commit check.
