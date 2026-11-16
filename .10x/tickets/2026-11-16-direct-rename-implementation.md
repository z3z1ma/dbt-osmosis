Status: active
Created: 2026-11-16
Updated: 2026-11-16

# Implement bounded direct-rename inheritance

## Scope
Implement opt-in DuckDB/Trino direct-column inheritance for issue #440 under `.10x/specs/direct-rename-implementation.md`.
## Non-goals
No dependency edits, broad dialect/derived-expression support, release, research-only PR, or retry of denied communications/cleanup.
## Acceptance Criteria
Every contract scenario has meaningful tests; default unchanged; existing inheritance suite, Ruff/type checks and exact-head CI pass; independent review; document actual supported semantics and limits.
## References
`.10x/specs/direct-rename-implementation.md`; `.10x/knowledge/maintenance-roadmap.md`; research branch0011fad and its experiment.
## Assumptions
User authorized concrete roadmap implementation with maintainer judgment. Existing source owns precedence; conservative opt-in slice and structural limits chosen for reversible implementation. Unsupported dialects remain legacy behavior, not newly claimed support.
## Journal
- Created isolated branch from main8b86daef after reading source and prior research. Chose no persistent cache, DuckDB-only scope, distinct semantic-rejection vs unavailable fallback. No denied action retried.
## Blockers
None for local implementation; external actions remain subject to review.
## Evidence
Pending executor journal.
## Review
Pending independent reviewer.
## Retrospective
Pending.

- Parent highlighted Trino reporter environment; offline SQLGlot probe supports quoted catalog/schema/column normalization. Official Trino identifier docs confirm case-insensitivity. Scope expanded to Trino with explicit fixture/collision tests, no warehouse claim.

- Real local DuckDB probe established create_dbt_project_context discards compiled_code. Parent directed targeted opt-in model compilation through established dbt runner instead of an artifactinput surface. Superseded self-imposed no-query assumption: macros may query; no model SQL execution; dry-run YAML contract preserved. Tests must prove ordinary CLI alias inheritance with fresh context.


### Executor observations (local, 2026-11-16)
- Initial resolver/inheritance suite: 22 focused assertions passed (5.55s). Expanded existing inheritance run: 90 passed (42.88s), after correcting test-only YAML snapshot to ignore generated directories named `.yml` and disabling SQLGlot projection trimming to detect duplicate aliases.
- `/tmp/lineage-loader-probe.py` compiled a real temporary DuckDB ref model, then loaded `create_dbt_project_context`: compiled SQL existed in the on-disk artifact before loading, but was None after parsing. This invalidated a compiled-input-only CLI claim. Parent approved targeted actual-ModelNode compilation through SqlCompileRunner, with explicit macro-query caveat, no execution of model SQL, warning/fallback for DbtRuntimeError, and no retry until manifest identity changes.
- 29 focused compile/lineage tests passed (6.73s), including actual model ref/this compilation and once-per-context failure diagnostics. Current fresh CLI regression and manifest-generation retry coverage in progress.
- Basedpyright with explicit `/tmp/osmosis-supported/bin/python` interpreter: zero errors in five changed core modules; 136 warnings include existing permissive Any contracts. Initial invocation without explicit interpreter could not resolve installed dependencies, so that was not product evidence.
- Tests require sandbox escalation because preexisting osmosis logger writes `~/.dbt-osmosis/logs`; all fixtures use temporary local DuckDB projects. No external warehouse used. No dependency files changed or denied action retried.

- Fresh subprocess CLI and core lineage suite: `PYTHONPATH=src /tmp/osmosis-supported/bin/python -m pytest tests/test_lineage_cli.py tests/core/test_column_lineage.py -q`: **32 passed in 17.93s**. The CLI starts from fresh parse, compiles the model, writes renamed donor docs references, repeats without byte changes, and preserves YAML bytes/mtime under dry-run. The new generation test resets failed compilation attempts when the loaded manifest object changes; no persistent donor cache.
- Exact installed validation environment (queried with importlib.metadata): **dbt-core 1.11.17, dbt-duckdb 1.10.1, SQLGlot 30.31.0**, Python 3.13.14. Earlier research used SQLGlot 30.8.0; that is not this implementation test environment.
- Changed Python files: Ruff check passed, Ruff format check passed (7 files), `git diff --check` passed. Basedpyright with explicit environment interpreter analyzed five changed core modules: **0 errors, 136 warnings** (permissive existing Any contracts remain).
- Final combined regression command running: `PYTHONPATH=src /tmp/osmosis-supported/bin/python -m pytest tests/test_lineage_cli.py tests/core/test_column_lineage.py tests/core/test_inheritance_behavior.py tests/core/test_inheritance_complex.py tests/core/test_inheritance_shared_ancestors.py tests/test_yaml_inheritance.py tests/core/test_sql_operations.py -q`; log `/tmp/lineage-final-tests.log`.

### Independent review (review agent)
- Scope: inspected resolver, inheritance integration, actual-model compiler helper, context/protocol state, public documentation, spec and new test assertions. No runtime code authored by reviewer.
- Findings: **significant, resolved** — an in-memory-only suite did not establish ordinary fresh-CLI usefulness. Added subprocess CLI cases now prove actual alias docs-reference write, repeat byte idempotence, and dry-run YAML bytes/mtime preservation. **minor, resolved** — once-per-invocation wording overstated retry suppression; documentation/spec now correctly say once per loaded manifest generation, with manifest-replacement test.
- Adversarial checks: examined semantic rejection versus unavailable fallback, intermediate CTE projections, duplicate outputs, same-name fallback suppression, source/model donor identity, quoting/collisions, explicit descriptions and progenitor overrides, manifest RLock nesting, real ModelNode compilation (no temporary SQL-operation context), compile failure logging, generation reset, and opt-out guard. No unresolved significant or critical finding in the reviewed diff.
- Additional compatibility evidence: installed SQLGlot 28.0.0 into isolated `/tmp/lineage-sqlglot28` without modifying repository dependencies. Confirmed imported version/path `28.0.0 /tmp/lineage-sqlglot28/sqlglot/__init__.py`. Command `PYTHONPATH=/tmp/lineage-sqlglot28:src /tmp/osmosis-supported/bin/python -m pytest tests/core/test_column_lineage.py -q -k 'direct_and_cte or changed_or_ambiguous or unknown_star or escaped_identifiers or unavailable_inputs or stale_candidate or duplicate_output'`: **22 passed, 8 deselected in 0.56s**. This tests candidate resolver semantics at declared SQLGlot minimum; it is not a full dbt-version audit.
- A separate targeted Unicode probe compared DuckDB rejection of Ä/ä and ß/ss identifier substitutions with resolver behavior: both returned rejected, not a donor. No general Unicode support certification is inferred.
- Verdict: **pass for reviewed implementation**, contingent on completion of the executor's combined suite and required exact-head CI before merge. This is not merge authorization or a claim that CI has run.
- Residual risk: Trino is covered offline rather than by live adapter/warehouse execution. Structural budgets are not a wall-clock limit, and opt-in compilation macros can query a warehouse. The documentation states these limits. Full cross-version CI and re-review of any subsequent runtime changes remain required.

- Final combined regression completed: **102 passed in 51.64s** (CLI, lineage, all existing inheritance modules, SQL operations).
- Final narrow hardening clears partially mutated `compiled_code` after a dbt compilation error; the failure test now injects partial output and asserts it is cleared, no second attempt occurs, and explicit legacy fallback is retained. Independent reviewer notified of this post-review change; focused rerun pending.

- Final hardening focused rerun: **30 passed in 8.57s** (`tests/core/test_column_lineage.py`); Ruff check/format and diff check passed.

### Executor retrospective
The key rejected premise was that a prior dbt compile artifact survives normal osmosis context loading: a real probe showed reparse removes compiled SQL. Targeted actual-model compilation makes this usable through the CLI, while preserving model context and documenting macro-side effects. SQLGlot projection trimming also hid duplicate output aliases until an adversarial test exposed it; preserving full selects allows safe rejection. Use typed donor identities and keep raw donor lookup separate from target settings. Structural limits do not imply a wall-clock timeout. No persistent candidate cache avoids schema/description staleness; compilation attempts reset with manifest identity.

- Final review amendment: inspected the last `DbtRuntimeError` handler change clearing partially populated `node.compiled_code`, and the strengthened regression injecting partial SQL then asserting clear/once-only attempt/legacy fallback across two invocations with force inheritance. This prevents a failed compilation's partial SQL from becoming a subsequent donor-resolution input. **PASS includes this final runtime change.** Executor recorded combined 102 passed before this narrow hardening and focused 30 passed in 8.57s afterward; no duplicate suite was run by reviewer. Exact-head CI remains a merge gate.
