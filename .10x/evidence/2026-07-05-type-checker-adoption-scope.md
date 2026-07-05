Status: recorded
Created: 2026-07-05
Updated: 2026-07-05
Relates-To: .10x/tickets/2026-07-05-ratify-type-checker-adoption-scope.md

# Type checker adoption scope evidence

## What Was Observed

`ty` and mypy are not clean as whole-repo gates on the current branch. The diagnostics are broad and mostly fall into policy/configuration categories rather than one bounded production fix.

Raw output was captured under `/tmp/dbt-osmosis-ai-quality/type-adoption/`:

- `ty-concise.txt`
- `mypy.txt`
- `ty-core-cli.txt`
- `mypy-core-cli.txt`

Whole-repo `ty check --output-format concise` exited 1 with 343 parsed diagnostics:

```text
299 tests
 21 optional workbench extra
 14 core/source
  5 optional sql proxy extra
  4 optional llm extra
```

Top `ty` rule categories:

```text
235 invalid-argument-type
 39 unresolved-attribute
 31 unresolved-import
 12 unsupported-operator
  8 not-subscriptable
  6 possibly-missing-submodule
  5 redundant-cast
```

Whole-repo `mypy .` exited 1 with 263 parsed errors, plus notes:

```text
222 tests
 20 optional workbench extra
 11 core/source
  5 optional llm extra
  5 optional sql proxy extra
```

Top mypy error categories:

```text
177 arg-type
 26 import-not-found
 11 import-untyped
 11 operator
  8 index
  8 var-annotated
  6 union-attr
```

The currently basedpyright-covered source surface is much closer but still not clean:

- `ty check src/dbt_osmosis/core src/dbt_osmosis/cli`: 18 diagnostics.
- `mypy src/dbt_osmosis/core src/dbt_osmosis/cli`: non-zero, with 19 output lines including notes.

Representative core/CLI `ty` diagnostics:

- `src/dbt_osmosis/cli/main.py`: `kwargs["profiles_dir"]` assignment and three `rich.table.Table.print_table` attribute diagnostics.
- `src/dbt_osmosis/core/llm.py`: unresolved optional `openai` and `azure.identity` imports.
- `src/dbt_osmosis/core/schema/validation.py`: two `_validate_resource_tests` argument type diagnostics.
- `src/dbt_osmosis/core/diff.py` and `src/dbt_osmosis/core/migration.py`: deprecated `datetime.utcnow()` warnings.

Representative core/CLI mypy diagnostics:

- missing or untyped third-party stubs for `dbt_core_interface`, `agate.table`, `yaml`, `openai`, and `azure.identity`;
- optional LLM assignment diagnostics in `src/dbt_osmosis/core/llm.py`;
- one `src/dbt_osmosis/core/path_management.py` lambda inference diagnostic;
- one `src/dbt_osmosis/core/inheritance.py` optional assignment diagnostic;
- one `src/dbt_osmosis/core/transforms.py` `TransformOperation` name argument diagnostic.

## Procedure

Commands run:

```text
uv run --no-sync --with ty ty check --output-format concise --no-progress
uv run --no-sync --with mypy mypy . --cache-dir /tmp/dbt-osmosis-ai-quality/type-adoption/mypy-cache --show-error-codes --no-error-summary --no-color-output
uv run --no-sync --with ty ty check --output-format concise --no-progress src/dbt_osmosis/core src/dbt_osmosis/cli
uv run --no-sync --with mypy mypy src/dbt_osmosis/core src/dbt_osmosis/cli --cache-dir /tmp/dbt-osmosis-ai-quality/type-adoption/mypy-core-cli-cache --show-error-codes --no-error-summary --no-color-output
```

The mypy cache was redirected to `/tmp` to avoid repository cache churn.

## What This Supports Or Challenges

This supports `ACC-001` for `.10x/tickets/2026-07-05-ratify-type-checker-adoption-scope.md`: current `ty` and mypy diagnostics are categorized with representative examples.

This challenges adopting either tool as a whole-repo gate immediately. Whole-repo adoption would require decisions on optional extras, missing stubs, test fixture typing, whether tests are in scope, and whether broad suppressions or baselines are acceptable.

The smallest practical recommendation is to keep basedpyright as the enforced type gate for now. If another gate is desired, open child tickets for a narrower source-only adoption path after ratifying:

- which tool, `ty`, mypy, or both;
- whether optional extras must be installed for the gate or excluded;
- whether tests are excluded initially;
- whether missing third-party stubs may be ignored, installed, or locally stubbed;
- whether warnings count as failures.

## Limits

The raw artifacts live in `/tmp` and are not durable repository files. This evidence records the summaries needed for durable decision-making.

No code or configuration changes were made by this investigation.
