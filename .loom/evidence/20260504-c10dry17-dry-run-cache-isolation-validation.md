---
id: evidence:c10dry17-dry-run-cache-isolation-validation
kind: evidence
status: recorded
created_at: 2026-05-04T12:24:01Z
updated_at: 2026-05-04T12:24:01Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  tickets:
    - ticket:c10dry17
  packets:
    - packet:ralph-ticket-c10dry17-20260504T115035Z
  critique:
    - critique:c10dry17-dry-run-cache-isolation-review
  wiki:
    - wiki:yaml-sync-safety
external_refs: {}
---

# Summary

Observed test-first local verification for `ticket:c10dry17` dry-run YAML cache isolation and test cache-fixture fidelity. The evidence records red/green behavior, parent follow-up validation, hooks, post-commit checks, and critique input; it does not decide ticket closure by itself.

# Procedure

Observed at: 2026-05-04T12:24:01Z
Source state: branch `loom/dbt-110-111-hardening`, implementation commit `d9eee85a212485c5d6f2944e52eafc2fad3c345e`, based on prior main commit `6dc2c61e0092ee296c40f8b69900724dbcd5d73c`.
Procedure: Reviewed Ralph child output, resolved parent-side cache fixture fallout, added no-op dry-run cache coverage, ran focused dry-run cache checks, ran all changed/related pytest modules, ran Ruff format/check, ran `git diff --check`, ran targeted pre-commit hooks, committed the implementation, reran the changed/related pytest modules and Ruff/diff checks against the committed source state, and ran a read-only adversarial critique.
Expected result when applicable: dry-run and `--check` paths should report mutations without leaving dirty YAML buffer or original-cache entries that affect later same-process reads; test cache fixtures should clear production cache instances without replacing cache globals; preserved original sections should not leak from stale `_YAML_ORIGINAL_CACHE` state.
Actual observed result: Initial red checks failed before implementation because dry-run `_write_yaml`, dry-run `commit_yamls`, dry-run `_finalize_synced_document(commit=False)`, and stale original-cache state leaked cached YAML into later reads/writes, and central `fresh_caches` replaced `_YAML_BUFFER_CACHE` with a plain dict. Final focused, broader, hook, lint, and post-commit acceptance checks passed locally. Final critique reported no medium/high findings.
Procedure verdict / exit code: mixed red/green sequence. Final green checks observed exit 0 for focused pytest, changed/related pytest, Ruff, targeted pre-commit, and `git diff --check`.

# Artifacts

Red observations recorded in `packet:ralph-ticket-c10dry17-20260504T115035Z`:

- `uv run pytest tests/core/test_schema.py::test_fresh_caches_preserves_production_cache_instances tests/core/test_schema.py::test_write_yaml_dry_run_discards_dirty_buffer_before_fresh_read tests/core/test_schema.py::test_commit_yamls_dry_run_discards_buffer_after_tracking_mutation tests/core/test_schema.py::test_dry_run_write_does_not_leak_original_preserved_sections tests/core/test_sync_operations.py::test_finalize_synced_document_dry_run_commit_false_discards_cache` returned 5 failures before implementation.
- Failures showed `fresh_caches` replaced production cache objects, dry-run writer paths left mutated buffers visible to later reads, stale `_YAML_ORIGINAL_CACHE` restored removed preserved sections, and dry-run sync `commit=False` left mutated YAML cached.

Parent follow-up observations:

- Removed remaining local test fixtures that monkeypatched `_COLUMN_LIST_CACHE`, `_YAML_BUFFER_CACHE`, or `_YAML_ORIGINAL_CACHE` and moved those tests to the central `fresh_caches` fixture.
- Added a no-op dry-run `_write_yaml` regression proving a processed dry-run path is discarded even when no mutation is counted.
- Tests that intentionally inspect non-committed in-memory YAML after `sync_node_to_yaml(..., commit=False)` now set `yaml_context.settings.dry_run = False` to distinguish the non-dry-run helper contract from dry-run isolation.
- A grep check for direct test monkeypatches of the production cache globals returned no files.

Final green observations:

- Focused dry-run cache checks returned `4 passed in 0.13s`.
- Changed/related modules returned `186 passed, 3 skipped in 96.81s` before commit.
- Ruff format/check over changed source/test files passed.
- `git diff --check` passed.
- Targeted `pre-commit run --files ...` over changed source/test files passed all applicable hooks.
- Implementation commit: `d9eee85a212485c5d6f2944e52eafc2fad3c345e`.
- Post-implementation-commit acceptance validation: `uv run pytest tests/core/test_schema.py tests/core/test_sync_operations.py tests/core/test_inheritance_behavior.py tests/core/test_property_accessor.py tests/core/test_transforms.py tests/core/test_introspection.py tests/core/test_diff.py tests/core/test_migration.py tests/core/test_pipeline_integration.py tests/test_yaml_inheritance.py -q` returned `186 passed, 3 skipped in 96.52s`.
- Post-implementation-commit `uv run ruff check ... && git diff --check` passed.
- Final critique `critique:c10dry17-dry-run-cache-isolation-review` reported no medium/high findings.

# Supports Claims

- ticket:c10dry17#ACC-001: dry-run `_write_yaml`, dry-run `commit_yamls`, and dry-run `_finalize_synced_document(commit=False)` regressions prove later same-process reads return disk-backed YAML instead of dry-run mutations.
- ticket:c10dry17#ACC-002: mutation tracker assertions prove dry-run `_write_yaml` and dry-run `commit_yamls` still count mutations before cache cleanup, preserving `--check` helper behavior.
- ticket:c10dry17#ACC-003: central `fresh_caches` now clears `_COLUMN_LIST_CACHE`, `_YAML_BUFFER_CACHE`, and `_YAML_ORIGINAL_CACHE` in place before and after each test.
- ticket:c10dry17#ACC-004: `test_fresh_caches_preserves_production_cache_instances` proves production cache objects remain their production classes under the fixture, and the grep check found no remaining direct test monkeypatches of the production cache globals.
- ticket:c10dry17#ACC-005: stale preserved-section regression proves old `_YAML_ORIGINAL_CACHE` state cannot restore removed preserved top-level sections after dry-run cleanup.
- initiative:dbt-110-111-hardening#OBJ-003: local validation supports schema YAML data-preservation hardening.
- initiative:dbt-110-111-hardening#OBJ-004: fixture changes increase test fidelity for cache behavior.

# Challenges Claims

None - final observed checks matched the expected post-fix results for the cited claims.

# Environment

Commit: implementation commit `d9eee85a212485c5d6f2944e52eafc2fad3c345e`.
Branch: `loom/dbt-110-111-hardening`
Runtime: local `uv run` environment.
OS: macOS Darwin.
Relevant config: `src/dbt_osmosis/core/schema/writer.py`, `src/dbt_osmosis/core/sync_operations.py`, `tests/conftest.py`, and related YAML/schema/core test modules from the reviewed source state.
External service / harness / data source when applicable: no production service exercised; tests used local temp files, mocks, and existing demo fixture where applicable.

# Validity

Valid for: local dry-run YAML writer/sync cache isolation, mutation tracking, preserved-section cache cleanup, and test cache fixture behavior in the observed implementation commit.
Fresh enough for: critique and local ticket acceptance under the current directive to defer per-ticket GitHub Actions waiting to final initiative validation.
Recheck when: YAML cache internals change, writer preservation strategy changes, sync finalization changes, cache fixture behavior changes, or final initiative-level CI fails for the same claims.
Invalidated by: source changes after this evidence that alter dry-run writer cleanup, `commit_yamls`, sync finalization, schema reader caches, or central test cache fixture behavior.
Supersedes / superseded by: Supplements `evidence:oracle-backlog-scan` with local implementation verification; final initiative-level CI should supplement this evidence later.

# Limitations

- Full repository test suite and GitHub Actions matrix were not run for this ticket; per-ticket CI waiting is intentionally deferred to final initiative validation.
- `--check` coverage is helper-level through mutation trackers, not a Click CLI integration test.
- Dry-run sync still mutates a shared cached document before finalization discards it, so concurrent same-process readers could observe transient dry-run state.
- A long-lived caller that dry-runs with filtered YAML data and then applies that same data without a fresh read could lose preserved top-level sections; the safe pattern is to reread before the real write after a dry-run preview.

# Result

The observed checks showed dry-run writer and sync paths now discard processed YAML cache entries after mutation reporting, test fixtures preserve production cache instances, and final critique has no medium/high findings.

# Interpretation

The evidence supports ticket acceptance with the noted low residual risks and final initiative-level CI still pending outside this ticket. It does not replace broader initiative validation.

# Related Records

- ticket:c10dry17
- packet:ralph-ticket-c10dry17-20260504T115035Z
- critique:c10dry17-dry-run-cache-isolation-review
- wiki:yaml-sync-safety
- evidence:oracle-backlog-scan
