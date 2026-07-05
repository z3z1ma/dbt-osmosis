Status: recorded
Created: 2026-07-05
Updated: 2026-07-05
Relates-To: .10x/tickets/done/2026-07-05-clarify-schema-package-exports.md

# Schema package export clarity

## What Was Observed

The continuation CodeQL scan before this ticket reported 76 total Python security-and-quality results. Three results used rule `py/undefined-export` and pointed at `src/dbt_osmosis/core/schema/__init__.py` for `_YAML_BUFFER_CACHE`, `_read_yaml`, and `_write_yaml`.

After replacing wildcard imports in `src/dbt_osmosis/core/schema/__init__.py` with explicit package bindings, the full CodeQL security-and-quality scan reported 73 total results and zero `py/undefined-export` results.

## Procedure

- Inspected the prior SARIF at `/tmp/dbt-osmosis-ai-quality/continuation/codeql-results.sarif`.
- Replaced schema package wildcard imports with explicit imports from `parser`, `reader`, `validation`, and `writer`.
- Preserved the existing `__all__` list. Direct compatibility bindings that are not in `__all__` remain available on the module.
- Ran a runtime import check:

```bash
PYTHONPYCACHEPREFIX=/tmp/dbt-osmosis-ai-quality/continuation/pycache-schema-import \
  uv run python - <<'PY'
from dbt_osmosis.core.schema import _YAML_BUFFER_CACHE, _read_yaml, _write_yaml
import dbt_osmosis.core.schema as schema

for name in [
    "_YAML_BUFFER_CACHE",
    "_YAML_ORIGINAL_CACHE",
    "_read_yaml",
    "_write_yaml",
    "_merge_preserved_sections",
    "OsmosisYAML",
    "ValidationResult",
    "commit_yamls",
]:
    print(name, hasattr(schema, name))

print(_YAML_BUFFER_CACHE, _read_yaml, _write_yaml)
PY
```

All checked attributes were present and the direct import succeeded.

- Ran focused schema tests:

```bash
PYTHONPYCACHEPREFIX=/tmp/dbt-osmosis-ai-quality/continuation/pycache-schema-tests \
  uv run pytest tests/core/test_validation.py tests/core/test_schema.py \
  -o cache_dir=/tmp/dbt-osmosis-ai-quality/continuation/pytest-cache-schema
```

Result: 80 passed, 2 warnings.

- Ran Ruff on the touched Python file:

```bash
uvx ruff==0.15.17 check src/dbt_osmosis/core/schema/__init__.py
```

Result: all checks passed.

- Rebuilt and analyzed a CodeQL database:

```bash
rm -rf /tmp/dbt-osmosis-ai-quality/continuation/codeql-db-after \
  /tmp/dbt-osmosis-ai-quality/continuation/codeql-results-after.sarif

codeql database create /tmp/dbt-osmosis-ai-quality/continuation/codeql-db-after \
  --language=python \
  --source-root . \
  --overwrite

codeql database analyze \
  /tmp/dbt-osmosis-ai-quality/continuation/codeql-db-after \
  codeql/python-queries:codeql-suites/python-security-and-quality.qls \
  --format=sarif-latest \
  --output=/tmp/dbt-osmosis-ai-quality/continuation/codeql-results-after.sarif
```

Parsed result summary:

```text
total: 73
py/undefined-export: 0
```

## What This Supports Or Challenges

This supports closing `.10x/tickets/done/2026-07-05-clarify-schema-package-exports.md`:

- `ACC-001`: Direct imports of `_YAML_BUFFER_CACHE`, `_read_yaml`, and `_write_yaml` succeeded.
- `ACC-002`: Focused schema tests passed.
- `ACC-003`: Ruff passed on the touched Python file.
- `ACC-004`: CodeQL no longer reports schema package undefined exports.
- `ACC-005`: The implementation change is limited to schema package exports.

## Limits

The CodeQL after-scan still reports other rules, including cyclic imports, ineffectual statements, unnecessary lambdas, import style, mixed returns, and uninitialized locals. Those residual findings are outside this ticket.
