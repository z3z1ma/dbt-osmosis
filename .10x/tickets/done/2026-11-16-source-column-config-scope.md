Status: done
Created: 2026-11-16
Updated: 2026-11-16

# Source column configuration scope (#421)

## Scope and acceptance

Reproduce #421 on current main, verify global vars and source metadata with real
dbt parsing and YAML writeback, clarify scope, and fix any demonstrated shared
resolution defect. Preserve model/source separation and existing precedence.
No release.

## Evidence

- Baseline: `d55a5f7959531b2d4464b2154595fc2ecea4d7f4`.
- Python 3.12.8 / dbt Core 1.11.8 / dbt-duckdb 1.10.1: seven end-to-end
  cases passed on unchanged runtime code. A `models:` option does not configure
  sources; vars, project `sources.+meta`, source-table meta and CLI work.
- Python 3.12.8 / dbt Core 1.8.10 / dbt-duckdb 1.8.4: baseline was six passed,
  one failed. Project `sources.+meta` is retained in `SourceConfig.extra.meta`
  instead of a native `meta` attribute. The resolver ignored that representation.
- Fix: ConfigMetaSource falls back to extra.meta only when the native meta
  attribute is absent. Resolution and precedence introspection use that same
  source. Native metadata and higher-priority node settings retain precedence.
- After fix, dbt 1.8: source-column end-to-end and config-resolution suites,
  **66 passed**, with dependency deprecation warnings.
- After fix, dbt 1.11.8: source-column end-to-end, settings resolver, config
  resolution, transforms and sync operations, **138 passed**.
- Ruff check and format pass for all touched Python files; diff whitespace check
  passes. Basedpyright: **0 errors**, 1983 non-blocking warnings.
- End-to-end cases verify model columns still populate, explicit false overrides
  global vars, and source columns are actually persisted or omitted in YAML.

## Review

Separate diff review checked the fallback against parsed dbt 1.8 manifest evidence,
native-meta precedence, missing config behavior, explicit false, and matching
precedence-chain reporting. No blocking findings. The fix belongs in the shared
resolver; the source transform already calls it correctly.

Docs now distinguish models from sources, provide global and source-scoped
examples, and remove an invalid `--skip-add-source-columns=false` invocation.
The reporter's exact dbt-osmosis 1.3.0 package was not retested; conclusions about
current behavior apply to the baseline commit above and the validated patch.

Remote CI, exact head review, merge, and main verification are recorded in the
linked PR and maintainer handoff rather than presumed by these local results.

## Reference

https://github.com/z3z1ma/dbt-osmosis/issues/421
