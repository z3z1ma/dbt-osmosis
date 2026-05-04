---
id: evidence:c10opt03-nested-options-validation
kind: evidence
status: recorded
created_at: 2026-05-04T17:06:20Z
updated_at: 2026-05-04T17:10:34Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  tickets:
    - ticket:c10opt03
  packets:
    - packet:ralph-ticket-c10opt03-20260504T165420Z
  critique:
    - critique:c10opt03-nested-options-review
external_refs: {}
---

# Summary

Observed validation for `ticket:c10opt03` after final implementation/coverage commit `3c644601c5812fd1333e3fb1627e252baddfd40a`. The checks show nested dbt-osmosis option containers resolve kebab-case and snake_case inner keys across the targeted config sources under focused tests, and local lint/type gates report no blocking errors.

# Procedure

Observed at: 2026-05-04T17:10:34Z

Source state: `3c644601c5812fd1333e3fb1627e252baddfd40a` on branch `loom/dbt-110-111-hardening`.

Procedure:

- `uv run pytest tests/core/test_config_resolution.py -q`
- `uv run ruff check src/dbt_osmosis/core/introspection.py tests/core/test_config_resolution.py && git diff --check`
- `uv run --with 'openai~=1.58.1' --with 'azure-identity>=1.19,<2' basedpyright --outputjson`, with the JSON summary reduced to the error and warning counts

Expected result when applicable: focused config-resolution tests pass, Ruff and whitespace checks pass, and optional-SDK basedpyright reports zero errors.

Actual observed result: targeted pytest reported `56 passed`; Ruff reported `All checks passed!`; `git diff --check` produced no output; optional-SDK basedpyright summary reported `errorCount=0 warningCount=1869`.

Procedure verdict / exit code: pass / exit code 0 for pytest, Ruff, whitespace, and basedpyright error gate. basedpyright warnings remain pre-existing/non-blocking under repository policy.

# Artifacts

- Ralph packet observation: current `src/dbt_osmosis/core/introspection.py` already used `_get_options_value()` to check kebab inner keys before snake inner keys, preserve explicit falsey values through `_MISSING`, and read both `dbt-osmosis-options` and `dbt_osmosis_options` containers in `ConfigMetaSource`, `UnrenderedConfigSource`, and `SupplementaryFileSource`.
- Implementation/coverage commits: `0e3c3565dc9d662f53b3f547562d2d92f629f657` and `3c644601c5812fd1333e3fb1627e252baddfd40a`.
- Critique follow-up observation: commit `3c644601c5812fd1333e3fb1627e252baddfd40a` added the literal `UnrenderedConfigSource.get("output-to-lower")` assertion for `dbt_osmosis_options.output_to_lower`.
- Changed test file: `tests/core/test_config_resolution.py`.
- Final focused pytest: `56 passed in 4.84s`.
- Final optional-SDK basedpyright summary: `errorCount=0 warningCount=1869`.

# Supports Claims

- `ticket:c10opt03#ACC-001`: `ConfigMetaSource.get("output-to-lower")` resolves from `dbt_osmosis_options.output_to_lower`.
- `ticket:c10opt03#ACC-002`: `UnrenderedConfigSource.get("output-to-lower")` resolves from snake_case nested option keys in `dbt_osmosis_options`.
- `ticket:c10opt03#ACC-003`: `SupplementaryFileSource` resolves snake_case keys inside both `dbt-osmosis-options` and `dbt_osmosis_options` containers when invoked.
- `ticket:c10opt03#ACC-004`: existing kebab-case nested behavior continues to pass for the targeted source classes.
- `ticket:c10opt03#ACC-005`: tests document the mixed inner-key precedence contract; when both variants appear in one nested options object, kebab-case wins over snake_case.
- `initiative:dbt-110-111-hardening#OBJ-002`: dbt 1.10+ config migration paths remain covered for nested option spelling variants.

# Challenges Claims

None - no observed validation result challenged the scoped claims.

# Environment

Commit: `3c644601c5812fd1333e3fb1627e252baddfd40a`

Branch: `loom/dbt-110-111-hardening`

Runtime: `uv run` project environment; warning noted that an unrelated active `VIRTUAL_ENV` was ignored.

OS: macOS / Darwin.

Relevant config: optional-SDK type check included `openai~=1.58.1` and `azure-identity>=1.19,<2` through `uv run --with`; base local environment behavior for tests was otherwise unchanged.

External service / harness / data source when applicable: no network service, real dbt parse, external warehouse, OpenAI request, Azure request, or GitHub Actions execution was used for this validation.

# Validity

Valid for: `ticket:c10opt03` implementation/coverage state at commit `3c644601c5812fd1333e3fb1627e252baddfd40a` and the listed local environment.

Fresh enough for: ticket acceptance review and critique disposition for the scoped nested option source classes.

Recheck when: `src/dbt_osmosis/core/introspection.py`, `tests/core/test_config_resolution.py`, resolver source ordering, option namespace names, or nested option precedence expectations change.

Invalidated by: changes after commit `3c644601c5812fd1333e3fb1627e252baddfd40a` that alter nested dbt-osmosis option lookup behavior or remove the focused tests.

Supersedes / superseded by: supersedes `evidence:oracle-backlog-scan` for `ticket:c10opt03` scoped validation; not superseded.

# Limitations

- This evidence is based on focused unit tests and direct local validation; it does not exercise a full dbt parse against every supported dbt version.
- The implementation source did not change in this ticket because the current resolver already satisfied the scoped behavior; the ticket resolved a coverage gap.
- Full GitHub Actions validation for pushed commits remains pending until after guarded push.
- The evidence does not cover broader resolver precedence integration owned by adjacent config-resolution tickets.

# Result

The committed `c10opt03` coverage passed focused tests, lint/whitespace checks, and optional-SDK basedpyright with zero errors.

# Interpretation

The evidence supports the scoped claim that nested dbt-osmosis option source classes handle kebab-case and snake_case inner keys, preserve explicit falsey values in those paths, and document the mixed-key precedence contract through tests.

# Related Records

- `ticket:c10opt03`
- `packet:ralph-ticket-c10opt03-20260504T165420Z`
- `critique:c10opt03-nested-options-review`
- `initiative:dbt-110-111-hardening`
