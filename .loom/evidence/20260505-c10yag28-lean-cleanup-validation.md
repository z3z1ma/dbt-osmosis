---
id: evidence:c10yag28-lean-cleanup-validation
kind: evidence
status: recorded
created_at: 2026-05-05T05:03:18Z
updated_at: 2026-05-05T05:03:18Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  ticket:
    - ticket:c10yag28
  packets:
    - packet:ralph-ticket-c10yag28-20260505T045616Z
external_refs: {}
---

# Summary

Observed test-first validation for the `ticket:c10yag28` cleanup of private executor mutation, unsupported database property-source behavior, resolver/source YAGNI status, and stale Black/isort formatter config.

# Procedure

Observed at: 2026-05-05T05:03:18Z

Source state: branch `loom/dbt-110-111-hardening` at baseline `d812dd42c519fc732005b5615beb460bc7456b24`, plus uncommitted `ticket:c10yag28` source, test, config, ticket, packet, and evidence changes.

Procedure: Ralph child ran focused red tests before implementation, implemented the bounded cleanup, then ran focused green validation. Parent inspected the diff and reran focused pytest, Ruff format check, Ruff lint plus whitespace check, and basedpyright on touched source files.

Expected result when applicable: red tests should fail before implementation because `YamlRefactorContext` mutates a default executor after construction, caller-supplied pools are resized, and `PropertySource.DATABASE` / `source="database"` silently falls back to manifest data. After implementation, default pools should be constructed with dbt threads or the old default formula, caller-supplied pools should remain unchanged, database property source should raise a clear unsupported-source error, stale Black/isort config should be removed only if unused by active tooling, focused tests should pass, Ruff should pass, whitespace checks should pass, and basedpyright should report zero errors.

Actual observed result: red tests failed for the expected reasons before implementation. After implementation, focused and parent validation passed. basedpyright reported zero errors with existing warnings tolerated by current project policy.

Procedure verdict / exit code: pass after implementation and parent validation.

# Artifacts

Child red evidence:

```text
uv run pytest tests/core/test_settings.py::TestYamlRefactorContext::test_context_constructs_default_pool_with_dbt_threads tests/core/test_settings.py::TestYamlRefactorContext::test_context_preserves_custom_pool_worker_count tests/core/test_introspection.py::TestPropertyAccessorHasUnrenderedJinja::test_database_source_is_explicitly_unsupported -q
4 failed

Expected failure reasons:
- default pool was constructed with the old default worker count and then mutated to dbt threads;
- caller-supplied pool was mutated from 2 to 4 workers;
- PropertySource.DATABASE / source="database" did not raise.
```

Child green evidence:

```text
uv run pytest tests/core/test_settings.py::TestYamlRefactorContext::test_context_constructs_default_pool_with_dbt_threads tests/core/test_settings.py::TestYamlRefactorContext::test_context_preserves_custom_pool_worker_count tests/core/test_introspection.py::TestPropertyAccessorHasUnrenderedJinja::test_database_source_is_explicitly_unsupported -q
4 passed

uv run pytest tests/core/test_settings.py tests/core/test_config_resolution.py tests/core/test_introspection.py -q
134 passed, 2 warnings

uv run ruff format --check src/dbt_osmosis/core/settings.py src/dbt_osmosis/core/introspection.py tests/core/test_settings.py tests/core/test_config_resolution.py tests/core/test_introspection.py pyproject.toml
5 files already formatted

uv run ruff check src/dbt_osmosis/core/settings.py src/dbt_osmosis/core/introspection.py tests/core/test_settings.py tests/core/test_config_resolution.py tests/core/test_introspection.py && git diff --check
All checks passed!

uv run basedpyright --outputjson src/dbt_osmosis/core/settings.py src/dbt_osmosis/core/introspection.py
errorCount: 0, warningCount: 309, informationCount: 0
```

Parent current-source verification:

```text
uv run pytest tests/core/test_settings.py tests/core/test_config_resolution.py tests/core/test_introspection.py -q
134 passed, 2 warnings in 5.25s

uv run ruff format --check src/dbt_osmosis/core/settings.py src/dbt_osmosis/core/introspection.py tests/core/test_settings.py tests/core/test_config_resolution.py tests/core/test_introspection.py pyproject.toml
5 files already formatted

uv run ruff check src/dbt_osmosis/core/settings.py src/dbt_osmosis/core/introspection.py tests/core/test_settings.py tests/core/test_config_resolution.py tests/core/test_introspection.py && git diff --check
All checks passed!

uv run basedpyright --outputjson src/dbt_osmosis/core/settings.py src/dbt_osmosis/core/introspection.py
errorCount: 0, warningCount: 309, informationCount: 0
```

Observed code/config facts:

- `src/dbt_osmosis/core/settings.py` no longer contains `_max_workers` references in production code.
- `src/dbt_osmosis/core/introspection.py` now raises `NotImplementedError` for `PropertySource.DATABASE` and `source="database"` instead of falling back to manifest data.
- `SettingsResolver` still uses `ConfigMetaSource`, `UnrenderedConfigSource`, `SupplementaryFileSource`, and `ProjectVarsSource`; no active resolver/source helper removal was appropriate.
- `_get_setting_for_node()` remains a compatibility wrapper and was not removed.
- `pyproject.toml` no longer contains `[tool.black]` or `[tool.isort]`; active formatting/linting remains Ruff-based.

All `uv run` commands emitted the known local warning that `VIRTUAL_ENV=/Users/alexanderbutler/code_projects/personal/dbt-osmosis/.venv` does not match this worktree `.venv`; `uv` ignored it.

# Supports Claims

- `ticket:c10yag28#ACC-001`: supports that production code no longer mutates private `ThreadPoolExecutor._max_workers`.
- `ticket:c10yag28#ACC-002`: supports that unsupported database property access is explicit and cannot be mistaken for real database introspection.
- `ticket:c10yag28#ACC-003`: supports that resolver helper/source classes are active in `SettingsResolver`, while `_get_setting_for_node()` remains compatibility-only.
- `ticket:c10yag28#ACC-004`: supports that stale Black/isort config was removed after active tooling checks showed Ruff is canonical.
- `ticket:c10yag28#ACC-005`: supports that the cleanup diff stayed small and focused, with targeted tests and static checks.

# Challenges Claims

None observed after implementation. Red evidence intentionally challenged pre-implementation behavior.

# Environment

Commit: baseline `d812dd42c519fc732005b5615beb460bc7456b24` with uncommitted `ticket:c10yag28` diff

Branch: `loom/dbt-110-111-hardening`

Runtime: OpenCode tool session and `uv` project environment

OS: darwin

Relevant config: `pyproject.toml`, `src/dbt_osmosis/core/settings.py`, `src/dbt_osmosis/core/introspection.py`, `tests/core/test_settings.py`, `tests/core/test_introspection.py`, `tests/core/test_config_resolution.py`

External service / harness / data source when applicable: local filesystem and local test/type/lint commands only; no remote CI observed for this evidence

# Validity

Valid for: the uncommitted `ticket:c10yag28` implementation diff and local validation commands listed above.

Fresh enough for: ticket review, implementation commit packaging, and pre-remote-CI acceptance evaluation for the named cleanup claims.

Recheck when: source, tests, dependency versions, ThreadPoolExecutor lifecycle, PropertyAccessor source policy, pyproject tooling configuration, or resolver source usage changes.

Invalidated by: failing focused tests, failing Ruff checks, basedpyright errors, reintroducing production `_max_workers` references, restoring database-source manifest fallback, or adding active Black/isort tooling without reinstating clear config.

Supersedes / superseded by: none.

# Limitations

This evidence is local-only and does not include GitHub Actions results for the eventual implementation commit. It does not remove compatibility exports or prove all external callers have stopped importing `_get_setting_for_node()`. basedpyright warning debt remains, but the changed source files report zero errors.

# Result

The observed validation supports the scoped cleanup behavior for `ticket:c10yag28` under the local test/static-check surface.

# Interpretation

This evidence supports moving `ticket:c10yag28` to `complete_pending_acceptance` pending implementation commit packaging and remote CI. It does not itself close the ticket; the ticket owns final acceptance after commit and remote workflow evidence.

# Related Records

- ticket:c10yag28
- packet:ralph-ticket-c10yag28-20260505T045616Z
- ticket:c10res14
- initiative:dbt-110-111-hardening
