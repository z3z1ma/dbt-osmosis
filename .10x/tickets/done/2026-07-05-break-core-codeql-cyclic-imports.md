Status: done
Created: 2026-07-05
Updated: 2026-07-05
Parent: None
Depends-On: None

# Break core CodeQL cyclic imports

## Scope

Address the remaining CodeQL `py/cyclic-import` findings by moving helper ownership out of modules that currently import each other at runtime.

In scope:

- Move catalog load/generate helpers out of `src/dbt_osmosis/core/introspection.py` into a dedicated core helper module.
- Update `YamlRefactorContext.read_catalog()` to depend on the dedicated catalog helper module instead of importing back from `introspection.py`.
- Move node YAML lookup ownership out of `src/dbt_osmosis/core/inheritance.py` into a dedicated core helper module with no dependency on `introspection.py`.
- Keep existing private import compatibility where practical for `_get_node_yaml`, `_load_catalog`, and `_generate_catalog`.
- Update tests that patch the catalog helper owner.
- Verify focused tests, Ruff, basedpyright error-level for touched source files, and CodeQL cycle delta.

Out of scope:

- Changing catalog generation behavior, catalog persistence, or YAML lookup semantics.
- Refactoring settings resolution or plugin candidate behavior.
- Renaming public CLI/API surfaces.
- Addressing non-CodeQL architecture tooling or broader layering policy.

## Acceptance Criteria

- ACC-001: CodeQL reports zero `py/cyclic-import` findings.
- ACC-002: Catalog load/generate behavior remains routed through `YamlRefactorContext.read_catalog()`.
- ACC-003: `dbt_osmosis.core.inheritance._get_node_yaml` remains importable for existing private callers.
- ACC-004: Focused settings, inheritance, introspection/property accessor, sync, and plugin tests pass.
- ACC-005: Ruff passes and basedpyright reports no errors for touched source files.

## Evidence Expectations

- Capture before CodeQL findings from `/tmp/dbt-osmosis-ai-quality/continuation/codeql-results-stub-bodies-after.sarif`.
- Capture focused pytest output.
- Capture Ruff and basedpyright output.
- Capture CodeQL after-check proving cyclic-import findings cleared.

## Progress and Notes

- 2026-07-05: CodeQL after stub cleanup reported 7 total findings, all `py/cyclic-import`.
- 2026-07-05: Source inspection found two practical cycle roots: `introspection.py` imports `_get_node_yaml` from `inheritance.py`, while `inheritance.py` and `plugins.py` import settings helpers from `introspection.py`; `settings.py` imports catalog helpers from `introspection.py`, while `introspection.py` imports `YamlRefactorSettings` for default comparison.
- 2026-07-05: Added `catalog_operations.py`, `model_versions.py`, and `node_yaml.py` as neutral helper owners.
- 2026-07-05: Kept `inheritance._get_node_yaml`, `inheritance._raw_model_version_value`, `inheritance._version_values_match`, `inheritance._versioned_model_yaml_view`, `introspection._load_catalog`, and `introspection._generate_catalog` importable as compatibility aliases.
- 2026-07-05: Updated settings/property accessor tests to patch the new helper owners.
- 2026-07-05: Ruff check and Ruff format check passed for touched files.
- 2026-07-05: basedpyright error-level passed for touched source files.
- 2026-07-05: Focused tests passed: 226 passed, 3 skipped, 2 warnings.
- 2026-07-05: Full pytest passed: 959 passed, 15 skipped, 2 warnings.
- 2026-07-05: CodeQL after-check reported zero Python security-and-quality findings.

## Blockers

None known.

## References

- `/tmp/dbt-osmosis-ai-quality/continuation/codeql-results-stub-bodies-after.sarif`
- `/tmp/dbt-osmosis-ai-quality/continuation/codeql-results-cycles-after.sarif`
- `.10x/evidence/2026-07-05-core-cyclic-import-break.md`
- `.10x/reviews/2026-07-05-core-cyclic-import-break-review.md`
- `src/dbt_osmosis/core/catalog_operations.py`
- `src/dbt_osmosis/core/introspection.py`
- `src/dbt_osmosis/core/inheritance.py`
- `src/dbt_osmosis/core/model_versions.py`
- `src/dbt_osmosis/core/node_yaml.py`
- `src/dbt_osmosis/core/settings.py`
- `src/dbt_osmosis/core/plugins.py`
- `tests/core/test_property_accessor.py`
- `tests/core/test_settings.py`
