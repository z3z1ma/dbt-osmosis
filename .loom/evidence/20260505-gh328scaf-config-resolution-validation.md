---
id: evidence:gh328scaf-config-resolution-validation
kind: evidence
status: recorded
created_at: 2026-05-05T06:59:39Z
updated_at: 2026-05-05T06:59:39Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  ticket:
    - ticket:gh328scaf
  packets:
    - packet:ralph-ticket-gh328scaf-20260505T065313Z
external_refs:
  github_issue: https://github.com/z3z1ma/dbt-osmosis/issues/328
---

# Summary

Observed red/green and parent validation for resolving `scaffold-empty-configs` through existing config-resolution paths during YAML sync.

# Procedure

Observed at: 2026-05-05T06:59:39Z
Source state: uncommitted diff on `loom/dbt-110-111-hardening` based on `db73d9e4a6fb2c28939d31aab611ca8d1f2f7021`, with unrelated `gh329loss` diff present outside this ticket scope.
Procedure: Ralph added failing tests for project vars and node config `scaffold-empty-configs`, implemented `resolve_setting()` usage in sync, and parent reran focused tests plus Ruff and whitespace checks.
Expected result when applicable: Project/node config and CLI fallback should keep empty scaffold fields, while absent/default false should continue dropping them.
Actual observed result: Failing tests turned green; parent validation passed.
Procedure verdict / exit code: Pass; all parent commands exited 0.

# Artifacts

Child red observation:

- `uv run pytest tests/core/test_sync_operations.py -k 'scaffold_empty_configs'` after tests-only change: `2 failed, 1 passed`; failures showed missing `description` for project/node config cases.

Parent green commands:

```bash
uv run pytest tests/core/test_sync_operations.py -q -k "scaffold_empty_configs"
```

Observed result: `3 passed, 31 deselected, 2 warnings`.

```bash
uv run pytest tests/core/test_sync_operations.py -q
```

Observed result: `34 passed, 2 warnings`.

```bash
uv run ruff check src/dbt_osmosis/core/sync_operations.py tests/core/test_sync_operations.py
```

Observed result: `All checks passed!`.

```bash
git diff --check
```

Observed result: passed with no output.

# Supports Claims

- ticket:gh328scaf#ACC-001: project vars config path keeps placeholder model description and empty column description.
- ticket:gh328scaf#ACC-002: CLI/global fallback remains covered.
- ticket:gh328scaf#ACC-003: absent/default false behavior still drops empty column descriptions.
- ticket:gh328scaf#ACC-004: tests cover project vars, node config, and column cleanup paths.
- ticket:gh328scaf#ACC-005: implementation uses `resolve_setting()`.

# Challenges Claims

None.

# Environment

Commit: uncommitted diff based on `db73d9e4a6fb2c28939d31aab611ca8d1f2f7021`.
Branch: `loom/dbt-110-111-hardening`.
Runtime: `uv run`; external `VIRTUAL_ENV` warning observed and ignored by `uv`.
OS: macOS Darwin.
Relevant config: focused sync-operation unit tests.
External service / harness / data source when applicable: GitHub issue #328.

# Validity

Valid for: local source state and focused sync-operation config behavior.
Fresh enough for: `ticket:gh328scaf` parent acceptance review and critique.
Recheck when: `sync_operations.py`, `SettingsResolver`, `resolve_setting()`, or YAML scaffold behavior changes.
Invalidated by: changes to config precedence or empty-field cleanup without rerunning the focused tests.
Supersedes / superseded by: N/A.

# Limitations

This evidence is unit-level. It does not include a full CLI run against a dbt project file, and it does not broaden existing scaffold semantics beyond model placeholder descriptions and empty column cleanup behavior.

# Result

The observed implementation makes `scaffold-empty-configs` project/node config affect sync decisions through `resolve_setting()` while preserving CLI/default fallback behavior.

# Interpretation

The evidence supports local acceptance of `ticket:gh328scaf` after critique. It does not establish remote CI health.

# Related Records

- ticket:gh328scaf
- critique:gh328scaf-config-resolution-review
- packet:ralph-ticket-gh328scaf-20260505T065313Z
- initiative:issue-pr-zero
