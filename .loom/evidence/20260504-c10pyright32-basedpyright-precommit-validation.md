---
id: evidence:c10pyright32-basedpyright-precommit-validation
kind: evidence
status: recorded
created_at: 2026-05-04T17:19:12Z
updated_at: 2026-05-04T17:19:12Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  tickets:
    - ticket:c10pyright32
  critique:
    - critique:c10pyright32-basedpyright-precommit-review
external_refs: {}
---

# Summary

Observed validation for `ticket:c10pyright32` after commit `7716997dfbbf0d0ec9a465aba48a7ff981369fc3` added a local basedpyright pre-commit hook. The hook runs full configured basedpyright checks through `uv`, includes optional OpenAI/Azure SDK dependencies, parses JSON output, and passed locally with zero errors.

# Procedure

Observed at: 2026-05-04T17:19:12Z

Source state: `7716997dfbbf0d0ec9a465aba48a7ff981369fc3` on branch `loom/dbt-110-111-hardening`.

Procedure:

- `uv run pre-commit run basedpyright --all-files --verbose`
- `uv run pre-commit run check-yaml --files .pre-commit-config.yaml .loom/tickets/20260504-c10pyright32-enforce-basedpyright-precommit.md && git diff --check`

Expected result when applicable: the basedpyright pre-commit hook runs the full configured type-check surface, reports zero basedpyright errors, and YAML/whitespace checks pass.

Actual observed result: `pre-commit run basedpyright --all-files --verbose` passed and printed `basedpyright summary: 0 errors, 1869 warnings`; `check-yaml` passed; `git diff --check` produced no output.

Procedure verdict / exit code: pass / exit code 0 for basedpyright hook validation, YAML validation, and whitespace validation.

# Artifacts

- Implementation commit: `7716997dfbbf0d0ec9a465aba48a7ff981369fc3`.
- Changed config: `.pre-commit-config.yaml`.
- Hook id: `basedpyright`.
- Hook command shape: `uv run --with 'openai~=1.58.1' --with 'azure-identity>=1.19,<2' basedpyright --outputjson`, followed by Python JSON summary parsing that exits with status 1 when `summary["errorCount"]` is nonzero.
- Hook settings: `language: system`, `pass_filenames: false`, `always_run: true`.

# Supports Claims

- `ticket:c10pyright32#ACC-001`: the hook uses `pass_filenames: false` and `always_run: true`, so `pre-commit run basedpyright --all-files` runs the configured project scope rather than a file-scoped partial check.
- `ticket:c10pyright32#ACC-002`: the hook parses `summary["errorCount"]` and exits with status 1 when it is nonzero; the observed zero-error run passed.
- `ticket:c10pyright32#ACC-003`: the hook includes `openai~=1.58.1` and `azure-identity>=1.19,<2` through `uv run --with`.
- `ticket:c10pyright32#ACC-004`: current branch validation reported `basedpyright summary: 0 errors, 1869 warnings`.
- `ticket:c10pyright32#ACC-005`: `.pre-commit-config.yaml` contains the local hook without changing unrelated hook definitions.
- `initiative:dbt-110-111-hardening#OBJ-006`: local pre-commit now mirrors the CI zero-error basedpyright gate earlier in the developer loop.

# Challenges Claims

None - no observed validation result challenged the scoped claims.

# Environment

Commit: `7716997dfbbf0d0ec9a465aba48a7ff981369fc3`

Branch: `loom/dbt-110-111-hardening`

Runtime: `uv run` project environment; warning noted that an unrelated active `VIRTUAL_ENV` was ignored.

OS: macOS / Darwin.

Relevant config: `.pre-commit-config.yaml`; `pyproject.toml` `tool.pyright` includes `src/dbt_osmosis/core` and `src/dbt_osmosis/cli`.

External service / harness / data source when applicable: no external service or GitHub Actions execution was used for this validation.

# Validity

Valid for: `ticket:c10pyright32` hook configuration at commit `7716997dfbbf0d0ec9a465aba48a7ff981369fc3` and the listed local environment.

Fresh enough for: ticket acceptance review and critique disposition.

Recheck when: `.pre-commit-config.yaml`, `pyproject.toml` `tool.pyright`, basedpyright version, optional SDK dependency versions, or CI basedpyright policy changes.

Invalidated by: changes after commit `7716997dfbbf0d0ec9a465aba48a7ff981369fc3` that alter the basedpyright hook command, remove optional SDK coverage, or change the pre-commit runtime environment.

Supersedes / superseded by: not superseded.

# Limitations

- The nonzero-error failure path was verified by command inspection, not by intentionally introducing a type error.
- The hook depends on `bash`, `uv`, and `python` being available on the developer PATH.
- The hook uses a fixed `/tmp/dbt-osmosis-basedpyright-precommit.json` path; this is acceptable for normal single-hook local pre-commit use but is not concurrency-hardened.
- The hook adds full-project basedpyright latency to pre-commit runs.
- Full GitHub Actions validation remains pending until after guarded push.

# Result

The committed basedpyright pre-commit hook passed locally and reported zero basedpyright errors.

# Interpretation

The evidence supports accepting the scoped tooling gate: local pre-commit now checks basedpyright errors before commits, while preserving the repository's existing warning-tolerant policy.

# Related Records

- `ticket:c10pyright32`
- `critique:c10pyright32-basedpyright-precommit-review`
- `initiative:dbt-110-111-hardening`
