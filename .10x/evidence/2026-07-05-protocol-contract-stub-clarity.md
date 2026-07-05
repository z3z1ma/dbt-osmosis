Status: recorded
Created: 2026-07-05
Updated: 2026-07-05
Relates-To: .10x/tickets/done/2026-07-05-clarify-protocol-contract-stubs.md

# Protocol contract stub clarity

## What Was Observed

The CodeQL scan after map-drain cleanup reported 42 total Python security-and-quality results:

```text
py/cyclic-import: 7
py/ineffectual-statement: 35
```

The 35 `py/ineffectual-statement` findings were all in Protocol, overload, or specification contract stub bodies:

- `src/dbt_osmosis/core/dbt_protocols.py`: 21 findings
- `src/dbt_osmosis/core/introspection.py`: 5 findings
- `src/dbt_osmosis/core/logger.py`: 1 finding
- `src/dbt_osmosis/workbench/components/dashboard.py`: 1 finding
- `specs/001-unified-config-resolution/contracts/config-resolver.py`: 7 findings

After replacing scoped stub bodies with explicit non-implementation bodies, the full CodeQL security-and-quality scan reported 7 total results:

```text
py/cyclic-import: 7
```

CodeQL reported zero `py/ineffectual-statement` results and did not introduce a replacement special-method finding.

## Procedure

- Replaced ordinary Protocol, overload, and contract stub bodies with `raise NotImplementedError`.
- Replaced the special Protocol methods `LogMethod.__call__` and `ImplementsBool.__bool__` with `@abstractmethod` plus `pass` bodies to avoid special-method raise semantics.
- Ran Ruff:

```bash
uvx ruff==0.15.17 check \
  src/dbt_osmosis/core/dbt_protocols.py \
  src/dbt_osmosis/core/introspection.py \
  src/dbt_osmosis/core/logger.py \
  src/dbt_osmosis/workbench/components/dashboard.py \
  specs/001-unified-config-resolution/contracts/config-resolver.py
```

Result: all checks passed.

- Ran Ruff format check:

```bash
uvx ruff==0.15.17 format --check \
  src/dbt_osmosis/core/dbt_protocols.py \
  src/dbt_osmosis/core/introspection.py \
  src/dbt_osmosis/core/logger.py \
  src/dbt_osmosis/workbench/components/dashboard.py \
  specs/001-unified-config-resolution/contracts/config-resolver.py
```

Result: 5 files already formatted.

- Ran basedpyright at error level:

```bash
uv run basedpyright --level error \
  src/dbt_osmosis/core/dbt_protocols.py \
  src/dbt_osmosis/core/introspection.py \
  src/dbt_osmosis/core/logger.py \
  src/dbt_osmosis/workbench/components/dashboard.py
```

Result: 0 errors, 0 warnings, 0 notes.

- Ran focused tests:

```bash
PYTHONPYCACHEPREFIX=/tmp/dbt-osmosis-ai-quality/continuation/pycache-stub-tests \
  uv run pytest \
  tests/core/test_logger.py \
  tests/core/test_introspection.py \
  tests/core/test_config_resolution.py \
  tests/core/test_property_accessor.py \
  tests/core/test_real_config_shapes.py \
  tests/core/test_workbench_app.py \
  tests/workbench/test_ai_assistant.py \
  -o cache_dir=/tmp/dbt-osmosis-ai-quality/continuation/pytest-cache-stub-bodies
```

Result: 154 passed, 3 skipped, 2 warnings.

- Rebuilt and analyzed a CodeQL database:

```bash
rm -rf /tmp/dbt-osmosis-ai-quality/continuation/codeql-db-stub-bodies-after \
  /tmp/dbt-osmosis-ai-quality/continuation/codeql-results-stub-bodies-after.sarif

codeql database create /tmp/dbt-osmosis-ai-quality/continuation/codeql-db-stub-bodies-after \
  --language=python \
  --source-root . \
  --overwrite

codeql database analyze \
  /tmp/dbt-osmosis-ai-quality/continuation/codeql-db-stub-bodies-after \
  codeql/python-queries:codeql-suites/python-security-and-quality.qls \
  --format=sarif-latest \
  --output=/tmp/dbt-osmosis-ai-quality/continuation/codeql-results-stub-bodies-after.sarif
```

Result: exit 0.

## What This Supports Or Challenges

This supports closing `.10x/tickets/done/2026-07-05-clarify-protocol-contract-stubs.md`:

- `ACC-001`: CodeQL no longer reports `py/ineffectual-statement` for the scoped stub-body locations.
- `ACC-002`: Touched function and property signatures remain unchanged; only bodies and special-method abstract decorators changed.
- `ACC-003`: Ruff check and format check passed.
- `ACC-004`: basedpyright error-level passed for touched source files.
- `ACC-005`: Focused tests passed.

## Limits

The CodeQL after-scan still reports 7 `py/cyclic-import` findings. Those architecture findings are outside this ticket.
