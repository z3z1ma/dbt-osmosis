Status: recorded
Created: 2026-07-05
Updated: 2026-07-05
Relates-To: .10x/tickets/done/2026-07-05-break-core-codeql-cyclic-imports.md

# Core cyclic import break

## What Was Observed

The CodeQL scan after protocol/contract stub cleanup reported 7 total Python security-and-quality results, all `py/cyclic-import`:

```text
src/dbt_osmosis/core/inheritance.py:349
src/dbt_osmosis/core/inheritance.py:396
src/dbt_osmosis/core/inheritance.py:875
src/dbt_osmosis/core/introspection.py:1695
src/dbt_osmosis/core/introspection.py:279
src/dbt_osmosis/core/plugins.py:51
src/dbt_osmosis/core/settings.py:360
```

After extracting neutral helper ownership for catalog operations, node YAML lookup, and model-version matching, the full CodeQL security-and-quality scan reported zero results.

## Procedure

- Added `src/dbt_osmosis/core/catalog_operations.py` to own `_load_catalog()` and `_generate_catalog()`.
- Added `src/dbt_osmosis/core/model_versions.py` to own model-version normalization and versioned YAML view helpers.
- Added `src/dbt_osmosis/core/node_yaml.py` to own `_get_node_yaml()` without importing `introspection.py`.
- Updated `YamlRefactorContext.read_catalog()` to import catalog helpers from `catalog_operations.py`.
- Updated `PropertyAccessor._get_from_yaml()` to import `_get_node_yaml()` from `node_yaml.py`.
- Kept compatibility aliases through existing private modules where practical.
- Updated tests that patch helper owners.
- Ran Ruff:

```bash
uvx ruff==0.15.17 check \
  src/dbt_osmosis/core/catalog_operations.py \
  src/dbt_osmosis/core/model_versions.py \
  src/dbt_osmosis/core/node_yaml.py \
  src/dbt_osmosis/core/inheritance.py \
  src/dbt_osmosis/core/introspection.py \
  src/dbt_osmosis/core/settings.py \
  tests/core/test_property_accessor.py \
  tests/core/test_settings.py
```

Result: all checks passed.

- Ran Ruff format check:

```bash
uvx ruff==0.15.17 format --check \
  src/dbt_osmosis/core/catalog_operations.py \
  src/dbt_osmosis/core/model_versions.py \
  src/dbt_osmosis/core/node_yaml.py \
  src/dbt_osmosis/core/inheritance.py \
  src/dbt_osmosis/core/introspection.py \
  src/dbt_osmosis/core/settings.py \
  tests/core/test_property_accessor.py \
  tests/core/test_settings.py
```

Result: files already formatted.

- Ran basedpyright at error level:

```bash
uv run basedpyright --level error \
  src/dbt_osmosis/core/catalog_operations.py \
  src/dbt_osmosis/core/model_versions.py \
  src/dbt_osmosis/core/node_yaml.py \
  src/dbt_osmosis/core/inheritance.py \
  src/dbt_osmosis/core/introspection.py \
  src/dbt_osmosis/core/settings.py
```

Result: 0 errors, 0 warnings, 0 notes.

- Ran focused tests:

```bash
PYTHONPYCACHEPREFIX=/tmp/dbt-osmosis-ai-quality/continuation/pycache-cycle-tests \
  uv run pytest \
  tests/core/test_settings.py \
  tests/core/test_inheritance_behavior.py \
  tests/test_yaml_inheritance.py \
  tests/core/test_sync_operations.py \
  tests/core/test_introspection.py \
  tests/core/test_property_accessor.py \
  tests/core/test_plugins.py \
  tests/core/test_validation.py \
  -o cache_dir=/tmp/dbt-osmosis-ai-quality/continuation/pytest-cache-cycle-break
```

Result: 226 passed, 3 skipped, 2 warnings.

- Ran full pytest:

```bash
PYTHONPYCACHEPREFIX=/tmp/dbt-osmosis-ai-quality/continuation/pycache-cycle-full \
  uv run pytest \
  -o cache_dir=/tmp/dbt-osmosis-ai-quality/continuation/pytest-cache-cycle-full
```

Result: 959 passed, 15 skipped, 2 warnings.

- Rebuilt and analyzed a CodeQL database:

```bash
rm -rf /tmp/dbt-osmosis-ai-quality/continuation/codeql-db-cycles-after \
  /tmp/dbt-osmosis-ai-quality/continuation/codeql-results-cycles-after.sarif

codeql database create /tmp/dbt-osmosis-ai-quality/continuation/codeql-db-cycles-after \
  --language=python \
  --source-root . \
  --overwrite

codeql database analyze \
  /tmp/dbt-osmosis-ai-quality/continuation/codeql-db-cycles-after \
  codeql/python-queries:codeql-suites/python-security-and-quality.qls \
  --format=sarif-latest \
  --output=/tmp/dbt-osmosis-ai-quality/continuation/codeql-results-cycles-after.sarif
```

Result: exit 0; SARIF result count 0.

## What This Supports Or Challenges

This supports closing `.10x/tickets/done/2026-07-05-break-core-codeql-cyclic-imports.md`:

- `ACC-001`: CodeQL reports zero `py/cyclic-import` findings and zero total findings.
- `ACC-002`: `YamlRefactorContext.read_catalog()` still calls `_load_catalog()` and `_generate_catalog()`, now from `catalog_operations.py`.
- `ACC-003`: `dbt_osmosis.core.inheritance._get_node_yaml` remains importable by compatibility import from `node_yaml.py`.
- `ACC-004`: Focused tests passed.
- `ACC-005`: Ruff passed and basedpyright reported no errors.

## Limits

CodeQL validates the import-cycle result but not every possible import-time path. Full pytest passed after the extraction, including inheritance, property accessor, settings, sync, and validation coverage.
