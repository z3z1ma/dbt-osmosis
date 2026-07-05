Status: recorded
Created: 2026-07-05
Updated: 2026-07-05
Relates-To: .10x/tickets/done/2026-07-05-reduce-vulture-boilerplate-noise.md

# Vulture boilerplate cleanup evidence

## What was observed

The high-confidence source-only Vulture pass was reduced from six findings to zero.

Before cleanup, `uv run --no-sync --with vulture vulture src/dbt_osmosis --min-confidence 100` reported:

- `src/dbt_osmosis/core/catalog_operations.py`: `generated_at`, `compile_results`
- `src/dbt_osmosis/core/dbt_protocols.py`: `omit_none`
- `src/dbt_osmosis/core/logger.py`: `msg`, `kwds`
- `src/dbt_osmosis/sql/proxy.py`: `attrs`

After cleanup, the same Vulture command exited 0 with no output.

## Procedure

- `uv run --no-sync --with vulture vulture src/dbt_osmosis --min-confidence 100` → exit 0 after cleanup.
- `uvx ruff==0.15.17 check src/dbt_osmosis/core/catalog_operations.py src/dbt_osmosis/core/config.py src/dbt_osmosis/core/settings.py src/dbt_osmosis/core/dbt_protocols.py src/dbt_osmosis/core/logger.py src/dbt_osmosis/sql/proxy.py tests/core/test_sql_proxy.py` → exit 0.
- `uvx ruff==0.15.17 format --check src/dbt_osmosis/core/catalog_operations.py src/dbt_osmosis/core/config.py src/dbt_osmosis/core/settings.py src/dbt_osmosis/core/dbt_protocols.py src/dbt_osmosis/core/logger.py src/dbt_osmosis/sql/proxy.py tests/core/test_sql_proxy.py` → exit 0.
- `uv run --no-sync --with pydoclint pydoclint src/dbt_osmosis/core/catalog_operations.py src/dbt_osmosis/core/config.py src/dbt_osmosis/core/settings.py src/dbt_osmosis/core/dbt_protocols.py src/dbt_osmosis/core/logger.py` → exit 0.
- `uv run pytest tests/core/test_sql_proxy.py tests/core/test_config.py tests/core/test_settings.py tests/core/test_config_resolution.py tests/core/test_property_accessor.py` → exit 0; 157 passed, 3 skipped, 2 warnings.

An earlier focused pytest attempt included nonexistent filenames `tests/core/test_config_context.py` and `tests/core/test_yaml_refactor_context.py`; it collected no tests and exited 4. It was corrected with the command above and is not acceptance evidence.

## What this supports or challenges

This supports ACC-001 and ACC-002 for `.10x/tickets/done/2026-07-05-reduce-vulture-boilerplate-noise.md`: the safe Vulture boilerplate findings were removed without changing callback signatures or public compatibility imports.

## Limits

Vulture at lower confidence still reports many CLI command callbacks, protocol members, properties, and public compatibility surfaces. Those were not treated as dead code because they are Click entrypoints, protocol shape, or public/adapter-facing API. This record does not prove those lower-confidence findings are removable.
