Status: recorded
Created: 2026-07-05
Updated: 2026-07-05
Relates-To: .10x/tickets/done/2026-07-05-align-transform-pipeline-rshift-protocol.md

# Transform pipeline rshift protocol

## What Was Observed

The CodeQL scan after explicit return cleanup reported 54 total Python security-and-quality results. One result used rule `py/unexpected-raise-in-special-method` in `src/dbt_osmosis/core/transforms.py`, where `TransformPipeline.__rshift__` raised `ValueError` for unsupported operands.

After returning `NotImplemented` for unsupported operands, the full CodeQL security-and-quality scan reported 53 total results and zero `py/unexpected-raise-in-special-method` results.

## Procedure

- Imported `NotImplementedType` for the `__rshift__` return annotation.
- Changed `TransformPipeline.__rshift__` to return `NotImplemented` for unsupported right-hand operands.
- Added `test_transform_pipeline_rejects_unsupported_rshift_operand`, which asserts Python raises the standard unsupported-operand `TypeError`.
- Ran focused pipeline tests:

```bash
PYTHONPYCACHEPREFIX=/tmp/dbt-osmosis-ai-quality/continuation/pycache-rshift-tests \
  uv run pytest tests/core/test_pipeline_integration.py \
  -o cache_dir=/tmp/dbt-osmosis-ai-quality/continuation/pytest-cache-rshift
```

Result: 18 passed, 2 warnings.

- Ran Ruff on touched files:

```bash
uvx ruff==0.15.17 check \
  src/dbt_osmosis/core/transforms.py \
  tests/core/test_pipeline_integration.py
```

Result: all checks passed.

- Ran basedpyright at error level for the touched core file:

```bash
uv run basedpyright --level error src/dbt_osmosis/core/transforms.py
```

Result: 0 errors, 0 warnings, 0 notes.

- Also ran warning-level basedpyright on the same file:

```bash
uv run basedpyright src/dbt_osmosis/core/transforms.py
```

Result: 0 errors, 100 warnings, 0 notes; exit code 1 because existing warnings are treated as failing diagnostics.

- Rebuilt and analyzed a CodeQL database:

```bash
rm -rf /tmp/dbt-osmosis-ai-quality/continuation/codeql-db-rshift-after \
  /tmp/dbt-osmosis-ai-quality/continuation/codeql-results-rshift-after.sarif

codeql database create /tmp/dbt-osmosis-ai-quality/continuation/codeql-db-rshift-after \
  --language=python \
  --source-root . \
  --overwrite

codeql database analyze \
  /tmp/dbt-osmosis-ai-quality/continuation/codeql-db-rshift-after \
  codeql/python-queries:codeql-suites/python-security-and-quality.qls \
  --format=sarif-latest \
  --output=/tmp/dbt-osmosis-ai-quality/continuation/codeql-results-rshift-after.sarif
```

Parsed result summary:

```text
total: 53
py/unexpected-raise-in-special-method: 0
```

## What This Supports Or Challenges

This supports closing `.10x/tickets/done/2026-07-05-align-transform-pipeline-rshift-protocol.md`:

- `ACC-001`: Existing valid pipeline tests passed.
- `ACC-002`: Unsupported operands now use Python operator protocol and are covered by a focused test.
- `ACC-003`: Focused tests passed.
- `ACC-004`: Ruff passed and basedpyright reported no errors.
- `ACC-005`: Targeted CodeQL finding cleared.

## Limits

CodeQL still reports pre-existing `py/ineffectual-statement` findings in `src/dbt_osmosis/core/transforms.py`, and basedpyright warning-level still reports pre-existing warnings in that file. Those are outside this ticket.
