Status: active
Created: 2026-11-21
Updated: 2026-11-21
Parent: .10x/knowledge/maintenance-roadmap.md

# Prefer exact catalog relation identity

## Scope
Fix reproduced wrong-column selection for case-distinct catalog relation names using a bounded exact-identity precedence pass. Worktree osmosis-catalog-correctness, branch fix/catalog-correctness-20261121, baseline b4c03c70864e1db49fe609e7a32812187ae8ef6c.
## Non-goals
No adapter-wide semantic changes, strict ambiguity rejection, dependency/lock edits, #468 actions, release, network warehouse or cloud operations.
## Acceptance Criteria
1. Before/after regression reproduces a quoted Orders request incorrectly selecting orders metadata; after fix selects exact Orders metadata regardless of entry order, including nodes/sources crossing.
2. Preserve approximate-only compatibility fallback, normal exact matches, optional absent/null database behavior, missing relation/no catalog behavior; no input mutation.
3. Exercise actual public get_columns path with real dbt relation/catalog objects where possible, proving returned column types/comments as well as names. Use offline adapter objects; no live warehouse.
4. Appropriate focused tests, lint/type checks, independent adversarial review, exact-head required CI and post-merge required checks pass before calling complete.
5. Roadmap records actual evidence and bounded remaining quoting limitations.
## References
.10x/specs/catalog-exact-relation-precedence.md; .10x/specs/catalog-input-contract.md; .10x/knowledge/maintenance-roadmap.md; root/core AGENTS.md; /Users/alexanderbut/code_projects/personal/10x/SKILL.md.
## Assumptions
Record-backed: current lookup lowercases CatalogTable.key() and catches ApproximateMatchError as True, scanning nodes before sources. User-ratified: delegated ownership authorizes one evidence-backed compatible correctness increment with offline reproduction and independent review. Exact raw match precedence preserves existing broad fallback instead of silently changing global adapter matching semantics.
## Journal
- 2026-11-21: Fresh fetch and isolated worktree at b4c03c7. Parent verified no issues, only excluded draft #468; current main Tests44761010974/lint44761011055 successful. Read roadmap, catalog boundary and #474 warning evidence.
- Current primary catalog and quoting documentation checked. Existing /tmp/osmosis-supported is incomplete (missing venv config/importable dbt); created /tmp/osmosis-correctness-env from cached Python/dependencies. Full dev offline install failed missing virtualenv wheel; runtime + dbt-duckdb + pytest offline succeeded without manifest/lock changes. Initial resolver output suggested Core1.12.7; direct installed-version inspection below corrects that observation. Import opens existing home log and required approved local execution outside sandbox.
- Repro: BaseRelation.create(database='db',schema='public',identifier='Orders'); real CatalogTable.from_dict entries orders/wrong_column before Orders/right_column. Current _catalog_columns_for_relation returned ['wrong_column']. No warehouse/network query. Same-case identity is lost by CatalogTable.key before matching.
## Blockers
None for bounded offline implementation. External integration remains subject to available permissions and CI.
## Evidence
1. Before-fix matrix: 12 failing wrong-column selections; after-fix matrix included in 166 passing focused tests. Cases cover each raw identity component, reversed ordering, nodes/sources individually and cross-section precedence.
2. Six optional-database/approximate-only controls and two missing catalog/relation recording-adapter controls pass. Catalog snapshots remain identical after reads and catalog hits make no adapter call.
3. Tests call public get_columns with actual BaseRelation and CatalogArtifact; require right_column, INTEGER and its specific documentation. Catalog compatibility suite also exercises generated Core artifacts and non-dry-run CLI input preservation.
4. Ruff check/format and git diff --check passed. Basedpyright reports zero errors. Independent review and exact-head/post-merge CI remain orchestrator-owned gates.
5. Roadmap updated with reproduced gap, bounded implementation and residual fallback limits; no broad engine support claim.
## Review
- 2026-11-21 independent adversarial review: **Verdict: pass**, subject to exact-head and post-merge CI integration gates. **Findings: none** (no critical, significant, minor or nit findings).
- Reviewed source/test diff against b4c03c70864e1db49fe609e7a32812187ae8ef6c, governing exact-identity and catalog-input specs, caller conversion to BaseRelation, repository instructions, roadmap and executor evidence. No executor verification was repeated.
- Exact raw database/schema/name equality is checked across both maps before the unchanged lowercased adapter matcher. The fallback traversal and ApproximateMatchError behavior are unchanged when no exact tuple exists; absent/null database continues through that fallback where necessary. No new mutation, cache, exception, public API or dependency behavior is introduced.
- Regression assertions check selected column name, type and specific comment via public get_columns using real dbt relation/catalog types. The matrix covers case collisions in each identity component, both insertion orders, same-section and crossed node/source placement. Controls prove approximate-only and omitted/null database acceptance, catalog/no-match warehouse fallback through a recording adapter, no adapter calls on hits and unchanged catalog snapshots. Existing assertions were not weakened.
- Recorded baseline 12 failures/20 passes and final 166 focused passes establish the bounded before/after change. Read final test and type log summaries; Ruff/format and diff checks are journaled. Actual installed Core 1.11.8 correction is consistent across spec, ticket and roadmap.
- **Residual risk:** local runtime evidence is Python 3.13.14/Core 1.11.8/adapters 1.24.5/DuckDB 1.11.1; supported Core 1.8–1.12 matrix remains required. Approximate-only matching deliberately remains permissive, so quoted missing-exact relations are not newly certified safe. Duplicate exact identities retain first-match behavior. This is neither actual Fusion producer nor live warehouse certification.
## Retrospective
The catalog key helper intentionally lowercases relation identity; using it as the only lookup key loses information before adapter matching. Exact raw precedence fixes a demonstrated collision without redefining every adapter's quoting policy. Crossing nodes and sources in tests matters because traversal order favors all nodes even when dictionary insertion order is reversed. Installed-version evidence must come from importlib.metadata, not a resolver's intermediate choices. Existing runtime-only cached setup avoided unavailable optional dev packages; local validation is bounded to that actual combination.

### Executor journal
- Added public `get_columns` regression matrix with real BaseRelation/CatalogArtifact objects: database/schema/name case collisions, both entry orders, same-section and cross-section node/source placement. Added optional database and approximate-only fallback cases and a recording adapter for missing-relation/no-catalog fallback.
- Baseline `PYTHONPATH=src /tmp/osmosis-correctness-env/bin/python -m pytest tests/core/test_introspection.py -k 'exact_catalog_identity or catalog_fallback_and_optional or missing_catalog_relation' -q`: **12 failed, 20 passed, 24 deselected**. Log `/tmp/osmosis-exact-catalog-before.log`. All failures select wrong_column instead of right_column when an approximate candidate precedes the exact one in traversal order; fallback controls passed.
- Added raw identity first pass across both sections, retaining the unchanged adapter/lowercased fallback only if no exact tuple exists. No artifact mutation, new dependencies or public API change.

- `PYTHONPATH=src /tmp/osmosis-correctness-env/bin/python -m pytest tests/core/test_introspection.py tests/test_catalog_contract.py tests/core/test_settings.py -q`: **166 passed in 27.96s**; `/tmp/osmosis-exact-catalog-after.log`. Offline local DuckDB fixture seed/run/docs only; no external warehouse. This includes real generated Core catalog and CLI validation/preservation regressions.
- `ruff check --no-cache src/dbt_osmosis/core/introspection.py tests/core/test_introspection.py` and matching `ruff format --check --no-cache` passed after applying canonical formatting. `git diff --check` passed.
- Installed existing declared basedpyright tool from local cache, without manifest/lock edits. `/tmp/osmosis-correctness-env/bin/basedpyright --pythonpath /tmp/osmosis-correctness-env/bin/python src/dbt_osmosis/core/introspection.py`: **0 errors, 258 warnings, 0 notes**; `/tmp/osmosis-exact-catalog-types.log`. Warnings are the module's broad Any/unknown and existing style diagnostics; no type suppressions added.
- Direct `importlib.metadata.version` inspection: Python **3.13.14**, dbt-core **1.11.8**, dbt-adapters **1.24.5**, dbt-duckdb **1.11.1**, pytest **9.2.0**. Initial parent resolver output mentioning Core1.12.7 was an intermediate choice, not the installed version; corrected governing spec and this ticket. Supported matrix remains remote CI gate.
- Source contract additionally checked by orchestrator at https://raw.githubusercontent.com/dbt-labs/dbt-adapters/main/dbt-adapters/src/dbt/adapters/base/relation.py. Exact raw database/schema/identifier identity is available separately from approximate match behavior.
- Residual limitation: no-exact approximate-only matching is intentionally unchanged. This does not certify safety for every case-sensitive quoted missing relation, actual Fusion producers or live adapter behavior; stricter fallback policy needs separate adapter-specific compatibility evidence.
- Source/tests ready for independent review; no commit, publication, merge, release, #468 modification, dependency manifests or lockfiles changed by executor.
