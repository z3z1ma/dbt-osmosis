---
id: evidence:c10ver15-versioned-yaml-access-validation
kind: evidence
status: recorded
created_at: 2026-05-04T11:20:10Z
updated_at: 2026-05-04T11:20:10Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  tickets:
    - ticket:c10ver15
  packets:
    - packet:ralph-ticket-c10ver15-20260504T095044Z
  critique:
    - critique:c10ver15-versioned-yaml-access-review
  wiki:
    - wiki:versioned-model-yaml
external_refs: {}
---

# Summary

Observed test-first and critique-driven local verification for `ticket:c10ver15` versioned model YAML access, validation, and selector-preserving sync behavior. The evidence records red/green behavior, parent fixes, local validation, hooks, and final review; it does not decide ticket closure by itself.

# Procedure

Observed at: 2026-05-04T11:20:10Z
Source state: branch `loom/dbt-110-111-hardening`, implementation commit `ef1d5409bcceee40c3403c06d75bf5cbe4cc4bb1`, based on prior main commit `df8dace880951768aaa0d3aa4ae98ec4cf89c862`.
Procedure: Reviewed Ralph child output, inspected and refined the implementation diff, addressed mandatory critique findings, added targeted regression coverage, reran focused and broader local pytest suites, ran Ruff format/check, ran `git diff --check`, ran targeted pre-commit hooks, committed the implementation, reran acceptance validation against the committed source state, and reran final critique until no medium/high findings remained.
Expected result when applicable: versioned `ModelNode` YAML access should select the matching `models[].versions[]` block; version-level columns should be visible to YAML property access and inheritance; top-level model fallback behavior should match dbt where scoped; versioned schema validation should catch invalid version entries, tests, column selectors, duplicates, and `latest_version`; sync/refactor should preserve dbt include/exclude selector rows and avoid duplicate version blocks from int/string version skew.
Actual observed result: Initial red checks failed before implementation because version-level YAML access and validation were absent. Final focused, broader, and post-commit acceptance checks passed locally; targeted hooks passed; and final critique reported no medium/high findings.
Procedure verdict / exit code: mixed red/green sequence. Final green checks observed exit 0 for focused pytest, broader pytest, post-commit acceptance pytest, Ruff, targeted pre-commit, and `git diff --check`.

# Artifacts

Red observations recorded in `packet:ralph-ticket-c10ver15-20260504T095044Z`:

- `uv run pytest tests/core/test_property_accessor.py::TestPropertyAccessor::test_yaml_source_reads_version_level_column_properties tests/core/test_inheritance_behavior.py::test_get_node_yaml_returns_versioned_block_with_top_level_fallback tests/core/test_inheritance_behavior.py::test_versioned_ancestor_unrendered_description_reads_version_columns tests/core/test_validation.py::TestModelValidator::test_validate_versioned_model_versions_and_columns -q` returned `4 failed in 5.99s` before production changes.
- `PropertyAccessor(source="yaml")` returned the manifest/top-level value instead of version-level `versions[].columns` YAML for `stg_customers.v2`.
- `_get_node_yaml()` returned the top-level `models[]` entry rather than the selected `versions[].v: 2` block.
- Unrendered inheritance inherited a rendered ancestor description instead of the raw version-level `{{ doc(...) }}` description.
- `ModelValidator` accepted malformed `versions[]` content, proving version-level validation was skipped.

Parent critique-driven observations:

- Added exact-raw-first version matching with restricted numeric fallback so dbt-distinct string versions such as `1.1`, `1.10`, and `01` are not accidentally collapsed.
- Added selected-version YAML views for versioned `ModelNode`s, with version-level columns owning column YAML and top-level description/meta/tags fallback for node-level access.
- Added blank version description fallback to top-level model description, matching dbt patch behavior.
- Preserved version-level raw `{{ doc(...) }}` descriptions for inheritance when `use_unrendered_descriptions` is enabled.
- Added model version validation for `versions`, `v`, duplicate version identity, `latest_version`, version-level tests, version-level columns, and include/exclude selector controls.
- Preserved dbt version column selector rows during sync/refactor and made sync version lookup update YAML `v: "2"` for manifest version `2` instead of appending a duplicate block.

Final green observations:

- Focused selector/version regressions before formatting returned `9 passed`.
- Broader related suites before formatting returned `114 passed, 3 skipped`.
- Key identity regressions after formatting returned `4 passed`.
- `uv run ruff format` over changed source/test files completed; final formatting run left files unchanged before commit.
- `uv run ruff check` over changed source/test files passed.
- `git diff --check` passed.
- Targeted `pre-commit run --files ...` over changed source/test/packet/ticket files passed all applicable hooks.
- Post-implementation-commit acceptance validation: `uv run pytest tests/core/test_property_accessor.py tests/core/test_inheritance_behavior.py tests/core/test_validation.py tests/core/test_sync_operations.py -q` returned `114 passed, 3 skipped in 30.56s`.
- Post-implementation-commit `uv run ruff check src/dbt_osmosis/core/inheritance.py src/dbt_osmosis/core/schema/validation.py src/dbt_osmosis/core/sync_operations.py tests/core/test_property_accessor.py tests/core/test_inheritance_behavior.py tests/core/test_validation.py tests/core/test_sync_operations.py` passed.
- Post-implementation-commit `git diff --check` produced no output.
- Final critique `critique:c10ver15-versioned-yaml-access-review` reported no medium/high findings.

# Supports Claims

- ticket:c10ver15#ACC-001: `_get_node_yaml()` regression coverage shows versioned `ModelNode`s read the selected `versions[].v` block, with exact string identity safeguards.
- ticket:c10ver15#ACC-002: `PropertyAccessor(source="yaml")`, inheritance, and sync tests show version-level column descriptions/meta/tags/tests and selector rows are handled without top-level column merging.
- ticket:c10ver15#ACC-003: top-level fallback behavior is explicit and tested, including blank version description fallback.
- ticket:c10ver15#ACC-004: validation tests cover invalid version entries, missing/invalid `v`, duplicate versions, invalid `latest_version`, invalid version columns/tests, and invalid include/exclude selectors.
- ticket:c10ver15#ACC-005: inheritance tests cover a version-level `{{ doc(...) }}` description under a version block.
- initiative:dbt-110-111-hardening#OBJ-003: local validation supports dbt 1.10/1.11 versioned model YAML compatibility.

# Challenges Claims

None - final observed checks matched the expected post-fix results for the cited claims.

# Environment

Commit: implementation commit `ef1d5409bcceee40c3403c06d75bf5cbe4cc4bb1`.
Branch: `loom/dbt-110-111-hardening`
Runtime: local `uv run` environment.
OS: macOS Darwin.
Relevant config: `src/dbt_osmosis/core/inheritance.py`, `src/dbt_osmosis/core/schema/validation.py`, `src/dbt_osmosis/core/sync_operations.py`, and related `tests/core/` files from the reviewed source state.
External service / harness / data source when applicable: no production service exercised; tests used local mocks, temp/cached YAML, and the existing demo fixture where applicable.

# Validity

Valid for: local versioned model YAML access, validation, inheritance, and sync/refactor behavior in the observed implementation commit.
Fresh enough for: mandatory critique and local ticket acceptance under the current directive to avoid per-ticket GitHub Actions waiting.
Recheck when: dbt model version parsing changes, YAML version matching changes, versioned sync/deduplication changes, schema validation expands, or `stg_customers` fixture shape changes.
Invalidated by: source changes after this evidence that alter version matching, YAML access, versioned validation, sync/refactor, or failed final initiative-level CI for the same claims.
Supersedes / superseded by: Supplements `evidence:oracle-backlog-scan` with local implementation verification; final initiative-level CI should supplement this evidence later.

# Limitations

- Full repository test suite and GitHub Actions matrix were not run for this ticket; per-ticket CI waiting is intentionally deferred to final initiative validation.
- The schema validator remains a focused dbt-osmosis validator, not a complete dbt schema replacement for every version-level field.
- `_get_node_yaml()` returns a shallow read-only mapping; nested lists/dicts remain mutable through existing cache objects.
- Non-finite numeric version values are not explicitly modeled beyond current validation and dbt parse expectations.

# Result

The observed checks showed that versioned model YAML access, inheritance, validation, and sync/refactor now handle dbt `models[].versions[]` blocks, version-level columns, include/exclude selectors, top-level fallback, and common version identity skew without unresolved medium/high critique findings.

# Interpretation

The evidence supports ticket acceptance and closure with final initiative-level CI still pending outside this ticket. It does not replace broader initiative validation.

# Related Records

- ticket:c10ver15
- packet:ralph-ticket-c10ver15-20260504T095044Z
- critique:c10ver15-versioned-yaml-access-review
- wiki:versioned-model-yaml
- evidence:oracle-backlog-scan
