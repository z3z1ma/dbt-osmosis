---
id: ticket:c10pyright32
kind: ticket
status: complete_pending_acceptance
change_class: developer-tooling
risk_class: medium
created_at: 2026-05-04T17:16:38Z
updated_at: 2026-05-04T18:01:26Z
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
    - evidence:c10pyright32-ci-basedpyright-remediation
    - evidence:c10pyright32-ci-pytest-profiles-remediation
  critique:
    - critique:c10pyright32-basedpyright-precommit-review
    - critique:c10pyright32-ci-basedpyright-remediation-review
    - critique:c10pyright32-ci-pytest-profiles-remediation-review
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
| ticket:c10pyright32#ACC-002 | evidence:c10pyright32-ci-basedpyright-remediation | critique:c10pyright32-ci-basedpyright-remediation-review | supported |
| ticket:c10pyright32#ACC-003 | evidence:c10pyright32-basedpyright-precommit-validation | critique:c10pyright32-basedpyright-precommit-review | supported |
| ticket:c10pyright32#ACC-004 | evidence:c10pyright32-ci-basedpyright-remediation | critique:c10pyright32-ci-basedpyright-remediation-review | supported |
| ticket:c10pyright32#ACC-005 | evidence:c10pyright32-basedpyright-precommit-validation | critique:c10pyright32-basedpyright-precommit-review | supported |

# Execution Notes

Prefer a local pre-commit hook that uses `uv run` to reuse the project environment rather than duplicating dependency setup in a separate pre-commit virtualenv.

# Blockers

None.

# Evidence

Evidence `evidence:c10pyright32-basedpyright-precommit-validation` records hook-level validation after commit `7716997dfbbf0d0ec9a465aba48a7ff981369fc3`. `uv run pre-commit run basedpyright --all-files --verbose` passed and printed `basedpyright summary: 0 errors, 1869 warnings`.

Evidence `evidence:c10pyright32-ci-basedpyright-remediation` records the post-push GitHub Actions failure, Linux reproduction of the CI-only optional OpenAI typing error, and the remediation commit `1d120731b5cdd36d78a394dd42be63a84c186501` that returns the Linux basedpyright reproduction to `errorCount: 0` for dbt 1.8 and 1.11 dependency sets.

Evidence `evidence:c10pyright32-ci-pytest-profiles-remediation` records the later GitHub Actions `Tests` run `25333721046` failure after basedpyright passed, the reproduced missing-home-profiles Click validation failure, and the remediation commit `e151e760cce2bdeda8dcb9e4c269b1786be9a676` that returns the affected CLI tests to green locally.

Evidence disposition: sufficient for the scoped developer-tooling gate and local CI pytest remediation before re-push. Full GitHub Actions confirmation for `e151e760cce2bdeda8dcb9e4c269b1786be9a676` remains pending.

# Critique Disposition

Risk class: medium

Critique policy: recommended

Policy rationale: This changes developer/CI gating behavior, but the implementation should be a narrow hook addition.

Required critique profiles: developer-tooling, operator-clarity

Findings: `critique:c10pyright32-basedpyright-precommit-review`, `critique:c10pyright32-ci-basedpyright-remediation-review`, and `critique:c10pyright32-ci-pytest-profiles-remediation-review` record no open findings. Initial recordkeeping finding `C10PYRIGHT32-EVID-001` was resolved by creating and linking the evidence/critique records and updating this acceptance dossier.

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

Current gate status: complete pending GitHub Actions acceptance.
Prior accepted by: OpenCode parent acceptance gate.
Prior accepted at: 2026-05-04T17:19:12Z.
Basis: `evidence:c10pyright32-basedpyright-precommit-validation`, `evidence:c10pyright32-ci-basedpyright-remediation`, `evidence:c10pyright32-ci-pytest-profiles-remediation`, and linked critiques support the local remediation state. The committed hook reported `0 errors` through pre-commit, the Linux CI reproduction reported `errorCount: 0` for dbt 1.8 and 1.11 dependency sets, and the missing-home-profiles CLI failure reproduced locally before passing after commit `e151e760cce2bdeda8dcb9e4c269b1786be9a676`.
Residual risks: Hook runtime adds full-project basedpyright latency and depends on `bash`, `uv`, and `python` on PATH. Full GitHub Actions confirmation for commit `e151e760cce2bdeda8dcb9e4c269b1786be9a676` remains pending until re-push and workflow completion.

# Dependencies

None.

# Journal

- 2026-05-04T17:16:38Z: Created from operator request to keep basedpyright errors at zero on each commit and add a pre-commit check.
- 2026-05-04T17:19:12Z: Added basedpyright pre-commit gate in commit `7716997dfbbf0d0ec9a465aba48a7ff981369fc3`, validated hook output with `0 errors`, recorded evidence/critique, and closed ticket.
- 2026-05-04T17:34:36Z: GitHub Actions Tests run `25333133362` exposed one Linux basedpyright error after closure. Reproduced the diagnostic, fixed the optional OpenAI rate-limit error type boundary in commit `1d120731b5cdd36d78a394dd42be63a84c186501`, recorded remediation evidence/critique, and kept ticket closed with updated acceptance basis.
- 2026-05-04T18:01:26Z: GitHub Actions Tests run `25333721046` passed basedpyright but exposed a pytest matrix failure where Click rejected the discovered default `--profiles-dir` `/home/runner/.dbt` before command logic ran. Reproduced the missing-home-profiles failure locally, fixed the premature Click existence check in commit `e151e760cce2bdeda8dcb9e4c269b1786be9a676`, recorded evidence/critique, and reopened the ticket to `complete_pending_acceptance` pending a green remote workflow.
