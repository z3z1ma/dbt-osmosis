Status: recorded
Created: 2026-07-05
Updated: 2026-07-05
Relates-To: .10x/tickets/done/2026-07-05-tighten-final-type-tool-feedback.md

# Final type tool feedback

## What Was Observed

After the CodeQL-zero cycle cleanup, whole-repo `ty check` and mypy were run as non-mutating final static checks. Both are not currently clean project-wide.

The whole-repo `ty check` output includes broad diagnostics across optional extras, tests, CLI table APIs, and protocol strictness. Two actionable diagnostics mapped directly to recently touched optimizer surfaces:

- `src/dbt_osmosis/core/introspection.py`: `ty` preferred `pass` for `@overload` bodies.
- `src/dbt_osmosis/core/transforms.py`: valid `TransformPipeline >> operation` chains were typed as possibly `NotImplementedType`.

The targeted `ty` check on `src/dbt_osmosis/core/introspection.py`, `src/dbt_osmosis/core/transforms.py`, and `src/dbt_osmosis/cli/main.py` dropped from 7 diagnostics to 4 after this patch. The remaining 4 diagnostics are pre-existing CLI diagnostics in `src/dbt_osmosis/cli/main.py`:

- `kwargs["profiles_dir"]` typed as an invalid assignment.
- Three `rich.table.Table.print_table` attribute diagnostics.

## Procedure

- Changed `_find_first()` overload bodies from `raise NotImplementedError` to `pass`.
- Added overloads to `TransformPipeline.__rshift__()` so valid operands return `TransformPipeline`, while unsupported operands keep the Python operator fallback.
- Changed `suggest_improved_documentation()` all-node dispatch to map `operation.func` when forwarding `threshold` and `learning_mode`.
- Ran Ruff:

```bash
uvx ruff==0.15.17 check \
  src/dbt_osmosis/core/introspection.py \
  src/dbt_osmosis/core/transforms.py \
  tests/core/test_pipeline_integration.py
```

Result: all checks passed.

- Ran basedpyright at error level:

```bash
uv run basedpyright --level error \
  src/dbt_osmosis/core/introspection.py \
  src/dbt_osmosis/core/transforms.py
```

Result: 0 errors, 0 warnings, 0 notes.

- Ran targeted `ty`:

```bash
uv run --no-sync --with ty ty check \
  src/dbt_osmosis/core/introspection.py \
  src/dbt_osmosis/core/transforms.py \
  src/dbt_osmosis/cli/main.py
```

Result: exit 1 with 4 diagnostics, all in `src/dbt_osmosis/cli/main.py` and outside this ticket.

- Ran focused tests:

```bash
PYTHONPYCACHEPREFIX=/tmp/dbt-osmosis-ai-quality/continuation/pycache-final-type-tests \
  uv run pytest \
  tests/core/test_pipeline_integration.py \
  tests/core/test_introspection.py \
  -o cache_dir=/tmp/dbt-osmosis-ai-quality/continuation/pytest-cache-final-type
```

Result: 42 passed, 2 warnings.

- Rebuilt and analyzed a CodeQL database:

```bash
rm -rf /tmp/dbt-osmosis-ai-quality/continuation/codeql-db-final-type-after \
  /tmp/dbt-osmosis-ai-quality/continuation/codeql-results-final-type-after.sarif

codeql database create /tmp/dbt-osmosis-ai-quality/continuation/codeql-db-final-type-after \
  --language=python \
  --source-root . \
  --overwrite

codeql database analyze \
  /tmp/dbt-osmosis-ai-quality/continuation/codeql-db-final-type-after \
  codeql/python-queries:codeql-suites/python-security-and-quality.qls \
  --format=sarif-latest \
  --output=/tmp/dbt-osmosis-ai-quality/continuation/codeql-results-final-type-after.sarif
```

Result: exit 0; SARIF result count 0.

## What This Supports Or Challenges

This supports closing `.10x/tickets/done/2026-07-05-tighten-final-type-tool-feedback.md`:

- `ACC-001`: The scoped overload-body and valid transform-chain `ty` diagnostics are gone.
- `ACC-002`: Focused pipeline tests still cover unsupported `pipeline >> object()` raising Python's standard `TypeError`.
- `ACC-003`: Ruff, basedpyright, focused tests, and CodeQL passed.

## Limits

Whole-repo `ty` and mypy are not clean and remain broader adoption work. This ticket only addressed the diagnostics tied to recently touched optimizer surfaces.
