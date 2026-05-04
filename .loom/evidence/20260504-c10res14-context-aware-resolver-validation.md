---
id: evidence:c10res14-context-aware-resolver-validation
kind: evidence
status: recorded
created_at: 2026-05-04T09:45:38Z
updated_at: 2026-05-04T09:45:38Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  tickets:
    - ticket:c10res14
  packets:
    - packet:ralph-ticket-c10res14-20260504T082226Z
  critique:
    - critique:c10res14-context-aware-settings-resolver-review
external_refs: {}
---

# Summary

Observed test-first and critique-driven local verification for `ticket:c10res14` context-aware settings resolution. The evidence records red/green behavior, parent fixes, local validation, hooks, and final review; it does not decide ticket closure by itself.

# Procedure

Observed at: 2026-05-04T09:45:38Z
Source state: branch `loom/dbt-110-111-hardening`, implementation commit `e4047ad46529dcecc40a6a68b27e8fcd5716b314`, based on prior main commit `5e7f7b69a5c100d0778e1bb37803cf622d0ebdca`.
Procedure: Reviewed Ralph child output, inspected and refined the uncommitted implementation diff, addressed mandatory critique findings, added targeted regression coverage, reran focused and broader local pytest suites, ran Ruff format/check, ran `git diff --check`, ran targeted pre-commit hooks, and reran final critique until no medium/high findings remained.
Expected result when applicable: documented settings precedence should be resolved through a single context-aware path; supplementary `dbt-osmosis.yml` and project vars should affect real transform/sync/introspection/inheritance behavior; explicit falsey values should not fall through; production call sites with context should not call `_get_setting_for_node()`; mandatory critique should have no unresolved medium/high findings.
Actual observed result: Initial red checks failed before implementation because transform and sync paths ignored context-backed sources. Final focused and broader checks passed locally, targeted hooks passed, and final critique reported no medium/high findings.
Procedure verdict / exit code: mixed red/green sequence. Final green checks observed exit 0 for focused pytest, broader pytest, Ruff, targeted pre-commit, and `git diff --check`.

# Artifacts

Red observations recorded in `packet:ralph-ticket-c10res14-20260504T082226Z`:

- `uv run pytest tests/core/test_transforms.py::test_inject_missing_columns_honors_supplementary_file_skip tests/core/test_sync_operations.py::test_sync_doc_section_honors_project_vars_output_to_lower -q` returned 2 failures before production changes.
- The transform failure showed `inject_missing_columns()` called `get_columns()` despite `dbt-osmosis.yml` containing `skip-add-columns: true`.
- The sync failure showed `_sync_doc_section()` wrote `MIXED_CASE` instead of project-vars-driven `mixed_case`.

Parent critique-driven observations:

- Replaced context-bearing production `_get_setting_for_node()` use with context-aware `resolve_setting()` in transforms, inheritance, sync, plugins, path routing, and precise dtype introspection.
- Added sentinel-based source resolution so `False`, `0`, and `""` are preserved rather than treated as missing.
- Added nested snake-case options coverage for node/config/supplementary sources and expanded project-vars key-shape support.
- Fixed Click tuple versus dataclass list default handling for `add-inheritance-for-specified-keys` so normal CLI defaults do not shadow project config.
- Made `SettingsResolver.has()` and `get_precedence_chain()` context-aware and exposed `CONTEXT_SETTINGS` in debug chains.
- Added a shared mtime/size-keyed cache for parsed supplementary file content.
- Moved all-node `inject_missing_columns()` fan-out before project-level skip resolution so node-level false overrides can still win.
- Passed full `YamlRefactorContext` into inheritance plugin hooks so project-level `prefix` reaches `FuzzyPrefixMatching`.

Final green observations:

- Focused inheritance/resolver/transform check: `uv run pytest tests/core/test_inheritance_behavior.py tests/core/test_plugins.py tests/core/test_transforms.py tests/core/test_config_resolution.py tests/core/test_introspection.py -q` returned `110 passed in 24.69s`.
- Broader c10res14 check: `uv run pytest tests/core/test_settings_resolver.py tests/core/test_config_resolution.py tests/core/test_transforms.py tests/core/test_real_config_shapes.py tests/core/test_inheritance_behavior.py tests/core/test_sync_operations.py tests/core/test_path_management.py tests/core/test_introspection.py tests/core/test_plugins.py -q` returned `151 passed, 1 skipped in 43.46s`.
- The single skip was the existing `tests/core/test_path_management.py:59` demo-source absence skip.
- `uv run ruff format src/dbt_osmosis/core/introspection.py src/dbt_osmosis/core/transforms.py src/dbt_osmosis/core/inheritance.py tests/core/test_config_resolution.py tests/core/test_introspection.py tests/core/test_transforms.py tests/core/test_inheritance_behavior.py` left 7 files unchanged.
- `uv run ruff check` over the same files passed.
- `git diff --check` produced no output.
- Targeted `pre-commit run --files ...` over changed source/test files passed all applicable hooks.
- Final critique `critique:c10res14-context-aware-settings-resolver-review` reported no medium/high findings.

# Supports Claims

- ticket:c10res14#ACC-001: resolver tests and migrated call-site tests show one context-aware `SettingsResolver` path handles node, context, supplementary, vars, context settings, and fallback sources.
- ticket:c10res14#ACC-002: `test_inject_missing_columns_honors_supplementary_file_skip` proves supplementary `dbt-osmosis.yml` affects the real inject transform.
- ticket:c10res14#ACC-003: `test_sync_doc_section_honors_project_vars_output_to_lower` and `test_get_columns_honors_context_project_vars_for_precise_dtype` prove project vars affect real sync and introspection behavior.
- ticket:c10res14#ACC-004: source grep and final review found no production `_get_setting_for_node()` calls outside compatibility wrapper/re-export/test surfaces.
- ticket:c10res14#ACC-005: resolver tests cover `False`, `0`, and empty-string values across context-backed resolution.
- ticket:c10res14#ACC-006: `_get_setting_for_node()` remains isolated as a compatibility wrapper and public facade re-export; new production use goes through `resolve_setting()`.
- initiative:dbt-110-111-hardening#OBJ-002 and #OBJ-008: local validation supports dbt 1.10+ config-shape hardening and central resolver behavior.

# Challenges Claims

None - final observed checks matched the expected post-fix results for the cited claims.

# Environment

Commit: implementation commit `e4047ad46529dcecc40a6a68b27e8fcd5716b314`.
Branch: `loom/dbt-110-111-hardening`
Runtime: local `uv run` environment.
OS: macOS Darwin.
Relevant config: `src/dbt_osmosis/core/introspection.py`, `src/dbt_osmosis/core/transforms.py`, `src/dbt_osmosis/core/inheritance.py`, `src/dbt_osmosis/core/sync_operations.py`, `src/dbt_osmosis/core/plugins.py`, `src/dbt_osmosis/core/path_management.py`, and related `tests/core/` files from the reviewed source state.
External service / harness / data source when applicable: no production service exercised; tests used local mocks, temp files, and the existing demo fixture where applicable.

# Validity

Valid for: local context-aware settings resolution behavior and migrated call sites in the observed implementation commit.
Fresh enough for: mandatory critique and local ticket acceptance under the current directive to avoid per-ticket GitHub Actions waiting.
Recheck when: settings precedence, CLI fallback construction, supplementary file parsing, project vars shape, inheritance plugins, transform fan-out, sync serialization, or dbt fixture behavior changes.
Invalidated by: source changes after this evidence that alter resolver precedence or migrated call sites, failed final initiative-level CI for the same claims, or evidence that project config no longer reaches production behavior.
Supersedes / superseded by: Supplements `evidence:oracle-backlog-scan` with local implementation verification; final initiative-level CI should supplement this evidence later.

# Limitations

- Full repository test suite and GitHub Actions matrix were not run for this ticket; per-ticket CI waiting is intentionally deferred to final initiative validation.
- Explicit `None` remains indistinguishable from missing in class-backed sources; the accepted scope only required preservation of `False`, `0`, and empty strings.
- `CONTEXT_SETTINGS` values win only when production call sites pass the current context setting as fallback; this mirrors migrated call-site behavior but remains an API footgun for future direct callers.
- Third-party plugins that expected `context.project` exactly may need to tolerate full `YamlRefactorContext` after inheritance now passes the complete context to hook implementations.

# Result

The observed checks showed that context-backed project settings now participate in real transform, sync, introspection, and inheritance behavior; falsey values remain meaningful; production call sites use the context-aware resolver helper; and mandatory critique has no medium/high findings.

# Interpretation

The evidence supports ticket acceptance and closure with final initiative-level CI still pending outside this ticket. It does not replace broader initiative validation.

# Related Records

- ticket:c10res14
- packet:ralph-ticket-c10res14-20260504T082226Z
- critique:c10res14-context-aware-settings-resolver-review
- wiki:config-resolution
- evidence:oracle-backlog-scan
