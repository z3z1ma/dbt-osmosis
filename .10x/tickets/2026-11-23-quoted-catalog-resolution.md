Status: done
Created: 2026-11-23
Updated: 2026-11-23

# Adapter-specific no-exact catalog resolution

## Scope
Investigate whether actual installed supported adapter Relation objects select metadata from an unrelated case-distinct catalog resource when the exact quoted identity is absent. Use offline constructed adversarial catalog inputs, clearly distinguished from actual producer fixtures, and current primary adapter/database contracts. Reconcile #482 completion records. Patch only a reproduced wrong result after a bounded contract is established and reviewed.

## Non-goals
No warehouse, credentials, cloud, spend, releases/tags/package publication, support policy or dependency queue changes, yanked Core 1.13 RC, or excluded #468 action. No global quote-policy-equals-case-sensitive assumption. Snowflake session/account overrides, DuckDB/Trino insensitive identifiers and adapter folding must be respected.

## Acceptance Criteria
1. Verify live main/inventory and #482 integration/release-validation facts; reconcile stale records.
2. Reproduce public get_columns result with real installed adapter relation classes and constructed catalog metadata, recording exact versions and primary semantic sources. Prefer stable Core 1.12.7; do not infer support from a spot test.
3. Establish valid case-fold compatibility controls and missing-relation behavior; propose scoped correction only for proven wrong metadata. If no defect is demonstrated, retain runtime behavior and record no-action findings.
4. Any patch requires focused regression before/after evidence, independent review, complete exact-head normal CI and post-merge verification; draft-first publication.

## References
.10x/specs/postgres-quoted-catalog-resolution.md; .10x/specs/catalog-input-contract.md; .10x/specs/catalog-exact-relation-precedence.md; .10x/knowledge/maintenance-roadmap.md; .10x/tickets/2026-11-21-catalog-exact-relation-precedence.md; root/core AGENTS.md; mandatory 10x skill.

## Assumptions
User-ratified investigation/fix scope and constraints; existing no-exact behavior is a compatibility boundary pending adapter-specific evidence. Offline constructed catalog is not genuine producer certification. Actual adapter APIs and documented database semantics are the authority.

## Journal
- Fresh main `ec92d9b02ec7e790cea3020ce57900a709b219c4`; isolated worktree osmosis-quoted-catalog. Only draft #468 open; zero issues and no dependency PRs. Original worktrees preserved.
- Re-read #482 main Tests45063489067:25success/1skip, lint45063489117 success. Release45065501839 success with both publication steps skipped; latestreleasev1.6.0. Reconciled roadmap and prior ticket.

## Blockers
None. Merged #483 at `bebaeb9ea78ae8f32c250bf4c4a4a9b1ef39894e`; independent review and exact-head/post-merge CI passed.

## Evidence
Actual installed adapter reproduction, 25 failing regressions before correction, 207 focused tests after correction and adapter compatibility controls are journaled below. Independent review and exact-head/post-merge CI passed for merged #483.

## Review
- 2026-11-23 independent adversarial review: **pass; no findings in the bounded change**. Reviewer inspected the contract, complete runtime/test/documentation diff, actual installed PostgresRelation and BaseRelation source, and executor's journaled before/after evidence. No verification suites were repeated.
- The guard uses the existing configured credentials type, not guessed inheritance or a global quote-policy rule. It checks raw catalog database/schema/name before key folding and before either success or ApproximateMatchError can accept a wrong donor. The database-null wildcard, exact pass, unquoted matching and non-PostgreSQL branches remain unchanged. A rejected candidate permits later candidates and otherwise reaches the existing introspection policy; public documentation states that enabled introspection may run.
- The 38 new cases exercise both maps, all three components, inverse case, mixed quote policy, optional database, later valid source, adapter compatibility and enabled/disabled introspection with a recording stub. They assert wrong-donor rejection, expected metadata, stable catalog data and call behavior; existing fresh-cache isolation remains active. BaseRelation-based CI cases are explicitly identified as representative inherited API coverage, backed by actual installed PostgresRelation reproduction rather than an unsupported producer claim or optional skipped tests.
- Accepted journaled validation: 25 failing new regressions before correction; 207 focused tests after; actual PostgreSQL/DuckDB/Trino object controls after; pinned lint/format and zero type errors. Normal exact-head matrix and post-merge verification remain required integration gates.
- Residual risk: no live PostgreSQL/warehouse certification; other adapter and session-specific semantics remain unchanged. Reviewer identified the separate unquoted raw exact-first precedence risk; executor reproduced it with the actual PostgreSQL object and recorded `.10x/tickets/2026-11-23-postgres-unquoted-catalog-precedence.md`. It is pre-existing, explicitly outside this fix, and not claimed resolved.

## Retrospective
CatalogTable.key erases case before either exactish matching or an approximate-match exception, so changing the exception handler alone misses the inverse-case bug. Adapter quoting semantics are not portable: narrow the guard using configured PostgreSQL type and raw metadata. Existing unquoted raw exact-first precedence is a separate reproduced issue with its own follow-up. Actual adapter objects supply reproduction evidence while identically inherited BaseRelation APIs keep CI regressions always active without optional dependency additions.

### Offline adapter reproduction (executor, 2026-11-23)
- Created isolated `/tmp/osmosis-quoted-core12-env` using Python 3.12.8. Installed stable Core 1.12.7, dbt-adapters 1.24.5, dbt-postgres 1.11.0, dbt-duckdb 1.11.1, dbt-trino 1.12.0. No repository dependency/lock changes or 1.13 prerelease. Core's transitive experimental parser package was installed but no parser mode was invoked; this is a relation-object experiment, not producer evidence.
- Read actual installed relation classes. `PostgresRelation.matches is BaseRelation.matches` and `_is_exactish_match` identity both true; PostgreSQL relation adds length/materialized-view behavior, not matching. DuckDB also inherits matching; Trino overrides `_is_exactish_match` to lowercase, matching its documented insensitive identifiers.
- Primary sources read 2026-11-23: PostgreSQL19 lexical structure https://www.postgresql.org/docs/19/sql-syntax-lexical.html states quoted identifiers sensitive/unquoted lowercase; Trino484 https://trino.io/docs/current/language/reserved.html states identifiers not case sensitive. DuckDB documented quoted-insensitive behavior is a preserved control, not a reason to generalize Postgres semantics.
- Temporary reproducible script `/tmp/osmosis-quoted-repro.py` constructs explicitly synthetic catalog metadata (not real producer output), instantiates actual PostgresRelation/DuckDBRelation/TrinoRelation, then calls public `get_columns` with introspection disabled and a mock adapter. Run `PYTHONPATH=src /tmp/osmosis-quoted-core12-env/bin/python /tmp/osmosis-quoted-repro.py`; output `/tmp/osmosis-quoted-repro.log`. Import requires osmosis's normal local warning log; sandbox error was resolved via approved bounded exec escalation, no HOME changes.
- Confirmed wrong PostgreSQL result: quoted `db.public.Orders` with only catalog `db.public.orders` returns `catalog_column`; inverse case also returns it. Separate quoted database/schema case differences produce the same incorrect donor. No exact entry exists and no adapter introspection occurs. Documented PostgreSQL semantics distinguish these identifiers.
- Controls: unquoted PostgreSQL `Orders` with physical lowercase `orders` returns catalog metadata; exact quoted identity returns catalog metadata. Both quoted case directions and unquoted folding continue to match on actual DuckDB/Trino objects, consistent with their documented behavior. No warehouse connection created.
- Proposal pending parent contract: PostgreSQL credentials type only; before existing no-exact matcher, reject candidate raw components that differ from a requested quoted component (retain optional null/absent catalog database wildcard). Compare raw metadata rather than CatalogTable.key(), which erases case before matching. Leave all other adapter semantics and unquoted fallback unchanged; use existing missing-relation path after rejection. Existing base matching class can provide always-executed CI regression without adding optional Postgres dependency, backed by this actual installed-object identity observation. No production edits yet.

### Authorized implementation scope
Actual installed PostgreSQL objects reproduced wrong metadata in both identifier-case directions and quoted schema/database mismatches; actual DuckDB/Trino and PostgreSQL unquoted controls preserved. The bounded contract is `.10x/specs/postgres-quoted-catalog-resolution.md`: reject raw mismatching quoted components only for configured postgres in no-exact fallback, preserve unknown database wildcard, all unquoted and other-adapter behavior, and existing missing behavior. No broader runtime normalization is authorized.

### Scoped correction and validation (executor)
- Parent established `.10x/specs/postgres-quoted-catalog-resolution.md` and authorized the bounded PostgreSQL-only no-exact guard. Added raw quoted component check before folded matcher only when configured credentials type is `postgres`. Missing/null catalog database remains wildcard, exact-first pass unchanged, unquoted components unchanged; every other adapter still follows its existing matcher. Missing relation uses existing introspection enabled/disabled policy.
- Added 38 always-executed offline regressions/controls: 24 quoted mismatch cases across model/source maps, database/schema/name, inverse case and introspection enabled/disabled; six mixed-quote/optional database cases; four unquoted adapter controls; three other-adapter quoted controls; and a later valid source after a rejected node. No optional-adapter skip or dependency changes. Before production edit, focused new cases: **25 failed, 13 passed, 56 deselected** (`/tmp/osmosis-quoted-before.log`). Mock adapter calls prove fallback policy; no warehouse was used.
- After correction, `PYTHONPATH=src /tmp/osmosis-quoted-core12-env/bin/python -m pytest tests/test_catalog_contract.py tests/core/test_introspection.py tests/core/test_settings.py -q`: **207 passed in 32.17s**, `/tmp/osmosis-quoted-after.log`. Includes genuine Core/Fusion producer regression, prior exact precedence and input preservation tests. Stable local Core1.12.7; no support-policy inference.
- Reran actual installed PostgresRelation/DuckDBRelation/TrinoRelation reproduction: quoted PostgreSQL case-distinct candidates now return empty with introspection disabled (both directions, all components); exact quoted and unquoted lowercase-fold controls remain successful. DuckDB/Trino controls unchanged. `/tmp/osmosis-quoted-repro-after.log`; no adapter calls. Actual Postgres matching method identity with BaseRelation remains verified.
- Pinned Ruff0.16.2 lint/format passed on source/tests. Basedpyright with explicit temporary interpreter: **0 errors / 2013 nonblocking warnings**,39 files (`/tmp/osmosis-quoted-types.json`). No warnings policy changed.
- Independently reproduced reviewer's distinct unquoted exact-first case using actual PostgresRelation: unquoted Orders with catalog raw Orders+orders selects mixed-case donor. Left exact-first behavior unchanged as scoped, recorded durable follow-up `.10x/tickets/2026-11-23-postgres-unquoted-catalog-precedence.md`. This unresolved case is not claimed fixed.

### Requested-null compatibility check (2026-11-23)
- Parent/reviewer requested checking requested relation database=None or schema=None against nonnull candidate metadata with quoted flags true. `/tmp/osmosis-quoted-none-request.py` uses actual PostgresRelation and constructed catalog; bypassing only the new guard exercises the unchanged historical exact/fallback path, then compares the current public get_columns result. Both requested-null components returned empty before and after with introspection disabled; zero adapter calls. Output `/tmp/osmosis-quoted-none-request.log`. Existing BaseRelation.matches already rejects the nonnull candidate component against a null requested component, so the guard introduces no compatibility change here. No production correction or broad test rerun was needed.

### Completed integration reconciled 2026-11-23
- #483 merged as `bebaeb9ea78ae8f32c250bf4c4a4a9b1ef39894e`. Exact-head PR Tests45209220318 and push Tests45209168058 each passed 25 executed jobs, with scheduled canary skipped; lint passed. Post-merge Tests45216264367 passed 25 jobs, canary skipped; lint45216264330 passed. Merged tree matched reviewed tree. No manual release or #468 action.
