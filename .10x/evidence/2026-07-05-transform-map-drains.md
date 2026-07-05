Status: recorded
Created: 2026-07-05
Updated: 2026-07-05
Relates-To: .10x/tickets/done/2026-07-05-clarify-transform-map-drains.md

# Transform map drains

## What Was Observed

The CodeQL scan after the rshift protocol cleanup reported 53 total Python security-and-quality results:

```text
py/cyclic-import: 7
py/ineffectual-statement: 46
```

Eleven `py/ineffectual-statement` results were intentional map-drain loop bodies:

```text
src/dbt_osmosis/core/sync_operations.py:1004
src/dbt_osmosis/core/transforms.py:233
src/dbt_osmosis/core/transforms.py:338
src/dbt_osmosis/core/transforms.py:423
src/dbt_osmosis/core/transforms.py:475
src/dbt_osmosis/core/transforms.py:528
src/dbt_osmosis/core/transforms.py:574
src/dbt_osmosis/core/transforms.py:604
src/dbt_osmosis/core/transforms.py:824
src/dbt_osmosis/core/transforms.py:883
src/dbt_osmosis/core/transforms.py:1048
```

After replacing those loop bodies with `pass`, the full CodeQL security-and-quality scan reported 42 total results:

```text
py/cyclic-import: 7
py/ineffectual-statement: 35
```

The after-scan reported no `py/ineffectual-statement` findings in `src/dbt_osmosis/core/transforms.py` or `src/dbt_osmosis/core/sync_operations.py`.

## Procedure

- Replaced the scoped ellipsis-only map-drain loop bodies with `pass`.
- Ran focused transform and sync tests:

```bash
PYTHONPYCACHEPREFIX=/tmp/dbt-osmosis-ai-quality/continuation/pycache-map-drain-tests \
  uv run pytest \
  tests/core/test_pipeline_integration.py \
  tests/core/test_transforms.py \
  tests/core/test_sync_operations.py \
  -o cache_dir=/tmp/dbt-osmosis-ai-quality/continuation/pytest-cache-map-drain
```

Result: 76 passed, 2 warnings.

- Ran Ruff on touched files:

```bash
uvx ruff==0.15.17 check \
  src/dbt_osmosis/core/transforms.py \
  src/dbt_osmosis/core/sync_operations.py
```

Result: all checks passed.

- Ran basedpyright at error level on touched files:

```bash
uv run basedpyright --level error \
  src/dbt_osmosis/core/transforms.py \
  src/dbt_osmosis/core/sync_operations.py
```

Result: 0 errors, 0 warnings, 0 notes.

- Rebuilt and analyzed a CodeQL database:

```bash
rm -rf /tmp/dbt-osmosis-ai-quality/continuation/codeql-db-map-drain-after \
  /tmp/dbt-osmosis-ai-quality/continuation/codeql-results-map-drain-after.sarif

codeql database create /tmp/dbt-osmosis-ai-quality/continuation/codeql-db-map-drain-after \
  --language=python \
  --source-root . \
  --overwrite

codeql database analyze \
  /tmp/dbt-osmosis-ai-quality/continuation/codeql-db-map-drain-after \
  codeql/python-queries:codeql-suites/python-security-and-quality.qls \
  --format=sarif-latest \
  --output=/tmp/dbt-osmosis-ai-quality/continuation/codeql-results-map-drain-after.sarif
```

Result: exit 0.

## What This Supports Or Challenges

This supports closing `.10x/tickets/done/2026-07-05-clarify-transform-map-drains.md`:

- `ACC-001`: The loops still iterate over `context.pool.map(...)`; only the no-op loop body changed from ellipsis to `pass`.
- `ACC-002`: CodeQL no longer reports the scoped map-drain bodies as ineffectual statements.
- `ACC-003`: Focused transform/sync tests passed.
- `ACC-004`: Ruff passed and basedpyright reported no errors for touched source files.

## Limits

The CodeQL after-scan still reports 35 `py/ineffectual-statement` findings outside this ticket and 7 `py/cyclic-import` findings. These residual findings are not in the touched map-drain locations.
