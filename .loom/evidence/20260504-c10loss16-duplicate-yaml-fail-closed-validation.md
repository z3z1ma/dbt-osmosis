---
id: evidence:c10loss16-duplicate-yaml-fail-closed-validation
kind: evidence
status: recorded
created_at: 2026-05-04T11:48:07Z
updated_at: 2026-05-04T11:48:07Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  tickets:
    - ticket:c10loss16
  packets:
    - packet:ralph-ticket-c10loss16-20260504T112322Z
  critique:
    - critique:c10loss16-duplicate-yaml-fail-closed-review
  wiki:
    - wiki:yaml-sync-safety
external_refs: {}
---

# Summary

Observed test-first and critique-driven local verification for `ticket:c10loss16` fail-closed duplicate YAML handling. The evidence records red/green behavior, parent preflight refinement, local validation, hooks, and final review; it does not decide ticket closure by itself.

# Procedure

Observed at: 2026-05-04T11:48:07Z
Source state: branch `loom/dbt-110-111-hardening`, implementation commit `1bf5d4b7f45b749da36fb098133bbf3086c7d0fc`, based on prior main commit `132f5eae04c016313f5ea9a909e7a592a157563e`.
Procedure: Reviewed Ralph child output, inspected and refined the implementation diff, addressed mandatory critique findings, added targeted regression coverage, reran focused and broader local pytest suites, ran Ruff format/check, ran `git diff --check`, ran targeted pre-commit hooks, committed the implementation, reran acceptance validation against the committed source state, and reran final critique until no medium/high findings remained.
Expected result when applicable: duplicate `models[]` entries and duplicate model `versions[]` entries should not be silently deleted during sync; duplicate handling should fail before writing with actionable errors; validation should catch duplicate top-level model names; tests should prove duplicate user-authored content remains intact when the failure is raised.
Actual observed result: Initial red checks failed before implementation because duplicate model/version sync did not raise and validation accepted duplicate model entries. Final focused, broader, and post-commit acceptance checks passed locally; targeted hooks passed; and final critique reported no medium/high findings.
Procedure verdict / exit code: mixed red/green sequence. Final green checks observed exit 0 for focused pytest, broader pytest, post-commit acceptance pytest, Ruff, targeted pre-commit, and `git diff --check`.

# Artifacts

Red observations recorded in `packet:ralph-ticket-c10loss16-20260504T112322Z`:

- `uv run pytest tests/core/test_sync_operations.py::test_get_or_create_model_rejects_duplicate_entries_without_deleting_user_content tests/core/test_sync_operations.py::test_get_or_create_version_rejects_duplicate_entries_without_deleting_user_content tests/core/test_validation.py::TestModelValidator::test_validate_duplicate_model_names` returned 3 failures before implementation.
- Duplicate model sync did not raise `YamlValidationError` and only warned, preserving the existing path that could delete duplicate entries.
- Duplicate version sync did not raise `YamlValidationError`.
- `ModelValidator` returned valid for duplicate top-level model names.

Parent critique-driven observations:

- `_deduplicate_model_entries()` now fails closed with `YamlValidationError` instead of removing same-name duplicate entries.
- `_deduplicate_versions()` now indexes versions without rewriting the `versions` list and fails closed on duplicate version identity.
- `ModelValidator` now emits `DUPLICATE_MODEL_NAME` for duplicate top-level model entries.
- All-node sync now preflights grouped target YAML documents before dispatching sync workers, so a duplicate in any target aborts before any `_finalize_synced_document(..., commit=True)` call.
- Error messages explain that dbt-osmosis refuses to sync because choosing one duplicate would delete user-authored YAML content.

Final green observations:

- Focused duplicate/preflight/version-selector regressions returned `6 passed`.
- Broader related suites returned `90 passed`.
- Ruff format/check over changed source/test files passed.
- `git diff --check` passed.
- Targeted `pre-commit run --files ...` over changed source/test/packet/ticket files passed all applicable hooks.
- Post-implementation-commit acceptance validation: `uv run pytest tests/core/test_sync_operations.py tests/core/test_validation.py tests/core/test_schema.py -q` returned `90 passed in 17.36s`.
- Post-implementation-commit `uv run ruff check src/dbt_osmosis/core/sync_operations.py src/dbt_osmosis/core/schema/validation.py tests/core/test_sync_operations.py tests/core/test_validation.py` passed.
- Post-implementation-commit `git diff --check` produced no output.
- Final critique `critique:c10loss16-duplicate-yaml-fail-closed-review` reported no medium/high findings.

# Supports Claims

- ticket:c10loss16#ACC-001: duplicate model sync regression proves duplicate `models[]` entries fail closed and are not deleted.
- ticket:c10loss16#ACC-002: duplicate version sync regression proves duplicate `models[].versions[]` entries fail closed and are not deleted.
- ticket:c10loss16#ACC-003: all-node preflight regression proves command-level sync aborts before finalizing unrelated documents when duplicate YAML is detected.
- ticket:c10loss16#ACC-004: the selected fail-closed policy preserves all conflicting user-authored content by refusing to choose one entry.
- ticket:c10loss16#ACC-005: regression tests cover duplicate model entries and duplicate version entries with distinct descriptions; validation covers duplicate model names.
- ticket:c10loss16#ACC-006: `YamlValidationError` messages name the duplicate resource, indexes, and consolidation action.
- initiative:dbt-110-111-hardening#OBJ-003: local validation supports dbt 1.10/1.11 schema YAML data preservation hardening.

# Challenges Claims

None - final observed checks matched the expected post-fix results for the cited claims.

# Environment

Commit: implementation commit `1bf5d4b7f45b749da36fb098133bbf3086c7d0fc`.
Branch: `loom/dbt-110-111-hardening`
Runtime: local `uv run` environment.
OS: macOS Darwin.
Relevant config: `src/dbt_osmosis/core/sync_operations.py`, `src/dbt_osmosis/core/schema/validation.py`, and related `tests/core/` files from the reviewed source state.
External service / harness / data source when applicable: no production service exercised; tests used local mocks, monkeypatching, and the existing demo fixture where applicable.

# Validity

Valid for: local duplicate model/version YAML sync, validation, and all-node preflight behavior in the observed implementation commit.
Fresh enough for: mandatory critique and local ticket acceptance under the current directive to avoid per-ticket GitHub Actions waiting.
Recheck when: sync grouping/finalization changes, duplicate version identity changes, schema validation changes, or generate/NL writer migration changes.
Invalidated by: source changes after this evidence that alter duplicate sync handling, validation, all-node preflight, or failed final initiative-level CI for the same claims.
Supersedes / superseded by: Supplements `evidence:oracle-backlog-scan` with local implementation verification; final initiative-level CI should supplement this evidence later.

# Limitations

- Full repository test suite and GitHub Actions matrix were not run for this ticket; per-ticket CI waiting is intentionally deferred to final initiative validation.
- Generated/NL writer migration remains out of scope for `ticket:c10gen20`.
- Single-node seed duplicate errors still use the shared model-entry wording; this is a low user-experience limitation outside the central model/version data-loss claim.
- Malformed non-list `versions` shapes remain primarily validation-owned and may have less polished sync errors if validation is bypassed.

# Result

The observed checks showed that duplicate model/version YAML entries now fail closed instead of being silently deleted, all-node sync preflights duplicate YAML before writing any target document, and mandatory critique has no medium/high findings.

# Interpretation

The evidence supports ticket acceptance and closure with final initiative-level CI still pending outside this ticket. It does not replace broader initiative validation.

# Related Records

- ticket:c10loss16
- packet:ralph-ticket-c10loss16-20260504T112322Z
- critique:c10loss16-duplicate-yaml-fail-closed-review
- wiki:yaml-sync-safety
- evidence:oracle-backlog-scan
