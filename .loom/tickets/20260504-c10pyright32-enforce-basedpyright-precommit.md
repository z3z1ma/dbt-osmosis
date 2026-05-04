---
id: ticket:c10pyright32
kind: ticket
status: closed
change_class: developer-tooling
risk_class: medium
created_at: 2026-05-04T17:16:38Z
updated_at: 2026-05-04T17:19:12Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  initiative:
    - initiative:dbt-110-111-hardening
  evidence:
    - evidence:oracle-backlog-scan
    - evidence:c10pyright32-basedpyright-precommit-validation
  critique:
    - critique:c10pyright32-basedpyright-precommit-review
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
| ticket:c10pyright32#ACC-001 | evidence:c10pyright32-basedpyright-precommit-validation | critique:c10pyright32-basedpyright-precommit-review | supported |
| ticket:c10pyright32#ACC-002 | evidence:c10pyright32-basedpyright-precommit-validation | critique:c10pyright32-basedpyright-precommit-review | supported |
| ticket:c10pyright32#ACC-003 | evidence:c10pyright32-basedpyright-precommit-validation | critique:c10pyright32-basedpyright-precommit-review | supported |
| ticket:c10pyright32#ACC-004 | evidence:c10pyright32-basedpyright-precommit-validation | critique:c10pyright32-basedpyright-precommit-review | supported |
| ticket:c10pyright32#ACC-005 | evidence:c10pyright32-basedpyright-precommit-validation | critique:c10pyright32-basedpyright-precommit-review | supported |

# Execution Notes

Prefer a local pre-commit hook that uses `uv run` to reuse the project environment rather than duplicating dependency setup in a separate pre-commit virtualenv.

# Blockers

None.

# Evidence

Evidence `evidence:c10pyright32-basedpyright-precommit-validation` records hook-level validation after commit `7716997dfbbf0d0ec9a465aba48a7ff981369fc3`. `uv run pre-commit run basedpyright --all-files --verbose` passed and printed `basedpyright summary: 0 errors, 1869 warnings`.

Evidence disposition: sufficient for the scoped developer-tooling gate.

# Critique Disposition

Risk class: medium

Critique policy: recommended

Policy rationale: This changes developer/CI gating behavior, but the implementation should be a narrow hook addition.

Required critique profiles: developer-tooling, operator-clarity

Findings: `critique:c10pyright32-basedpyright-precommit-review` records no open findings. Initial recordkeeping finding `C10PYRIGHT32-EVID-001` was resolved by creating and linking the evidence/critique records and updating this acceptance dossier.

Disposition status: completed

Deferral / not-required rationale: None.

# Retrospective / Promotion Disposition

Disposition status: not_required

Promoted: None.
Deferred / not-required rationale: No durable explanation beyond `.pre-commit-config.yaml` is needed. The accepted project practice is already to run `pre-commit run --all-files`; this ticket adds the missing hook.

# Wiki Disposition

Disposition status: not_required

Rationale: The pre-commit configuration is the accepted operator surface for this local gate.

# Acceptance Decision

Accepted by: OpenCode parent acceptance gate.
Accepted at: 2026-05-04T17:19:12Z.
Basis: `evidence:c10pyright32-basedpyright-precommit-validation` and `critique:c10pyright32-basedpyright-precommit-review` support all ticket-local acceptance criteria. The committed hook reported `0 errors` through pre-commit.
Residual risks: Hook runtime adds full-project basedpyright latency and depends on `bash`, `uv`, and `python` on PATH. The nonzero-error path was verified by command inspection rather than an induced failing run.

# Dependencies

None.

# Journal

- 2026-05-04T17:16:38Z: Created from operator request to keep basedpyright errors at zero on each commit and add a pre-commit check.
- 2026-05-04T17:19:12Z: Added basedpyright pre-commit gate in commit `7716997dfbbf0d0ec9a465aba48a7ff981369fc3`, validated hook output with `0 errors`, recorded evidence/critique, and closed ticket.
