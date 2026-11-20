Status: active
Created: 2026-11-16
Updated: 2026-11-20

# dbt-osmosis maintenance roadmap

## Baseline and ownership
Observed main: 7279fec9 (2026-11-16). GitHub remains canonical for issues and PRs; counts are a snapshot, not a promise of current state. `gh pr list --limit 100` returned 11 open PRs, all dependency updates: #430, #432, #433, #434, #435, #436, #437, #446, #447, #448, #449. The separate dependency maintainer owns that queue and its manifests/lockfiles. Re-read remote main and exact-head checks before integrating either workstream. No release is authorized by this roadmap.

Three open issues: [#452](https://github.com/z3z1ma/dbt-osmosis/issues/452), [#443](https://github.com/z3z1ma/dbt-osmosis/issues/443), [#440](https://github.com/z3z1ma/dbt-osmosis/issues/440). Prior maintenance through #458 is on main; open status does not imply every reported claim remains reproducible or has been fixed.

## Architecture constraints
The CLI constructs a Python dbt context; config.py loads dbt project/runtime objects. SettingsResolver and PropertyAccessor own configuration precedence. Bulk YAML selection, path management, transform/inheritance and schema reader/writer form distinct boundaries. Preserve round-trip YAML and atomic writes. Existing `specs/001-unified-config-resolution` governs configuration work; do not invent another precedence system. `FUSION_COMPAT.md` records historical project experiments, not a current portable compatibility certification.

## Evidence and implications
Primary sources checked 2026-11-16:

- [Core 1.12 GA announcement](https://www.getdbt.com/blog/dbt-core-v1-12-is-ga) describes the August 2026 release, vars.yml, semantic specifications and ad-hoc SQL improvements. Our blocking audit ending at 1.11 misses a current stable Python runtime: promote only after the existing full gates pass.
- [dbt versions](https://docs.getdbt.com/docs/dbt-versions) distinguishes Python 1.x from Rust 2.x. [dbt v2 GA](https://docs.getdbt.com/blog/dbt-v2-is-ga) describes the September release and Parquet information schema. Neither is evidence that Python adapter or compilation APIs are interchangeable. A Rust rewrite is not justified by current defects.
- [Manifest contract](https://docs.getdbt.com/reference/artifacts/manifest-json) says compiled SQL is available only for executed nodes. Its version table does not establish the 1.12 schema: inspect generated metadata rather than infer it. Artifact readers must validate actual schema and producer metadata.
- [Catalog contract](https://docs.getdbt.com/reference/artifacts/catalog-json) retains catalog v1 for external tools while v2 docs uses the information schema. Historical empty BigQuery catalog observations must not be generalized to all current Fusion installations.
- [Metadata configuration](https://docs.getdbt.com/reference/resource-configs/meta) distinguishes config-level metadata and backward-compatible top-level forms. Preserve unknown non-null semantic fields and old-version source behavior; do not normalize away meaningful data.

- [SQLGlot lineage API](https://sqlglot.com/sqlglot/lineage.html) accepts schema, sources and dialect inputs; verify the installed API before using current documentation examples. [dbt MCP tools](https://docs.getdbt.com/docs/dbt-ai/mcp-available-tools) offer local LSP-backed and platform lineage. These are evaluation inputs for #440, not a reason to introduce a new service dependency or duplicate MCP execution surfaces.

These sources motivate compatibility tests and cautious artifact boundaries, not dependency upgrades or blanket engine support claims. Current upstream tooling increasingly supplies SQL execution and lineage; prioritize osmosis's YAML preservation and documentation inheritance value instead of duplicating editor/runtime capabilities.

## Ordered work

| Priority / owner | Outcome and rationale | Acceptance / safety gate |
| --- | --- | --- |
| P0 / Core audit ticket | Add Core 1.12 to blocking compatibility coverage. Existing 1.12 local full-suite evidence makes this the smallest useful first iteration. | Existing full pytest, integration, type, parse and pip checks pass on every added Python row; retain old rows and future canary. README range changes only with passing exact-head CI. No manifests or lockfile edits. |
| P1 / #443 | Close the default cross-package write defect with evidence, retaining unresolved allegations as separate scope. | Reference real-parser regression and merged fix. Default writes preserve package/root YAML and reparsing succeeds. Do not claim the un-reproduced dry-run writes fixed. Explicit external/direct-node paths retain behavior and require a separate bounded design before changing. |
| P1 / #452 | Reconcile false-cycle report with merged traversal fix. | Reference diamond/shared-ancestor and genuine-cycle coverage; verify warning correction separately from the reporter's inheritance-loss claim. Only close the proven warning defect; inheritance loss needs a reproducer. |
| P2 / #440 | Prototype opt-in direct-column rename lineage using existing SQLGlot, before broad expression inheritance. | Real compiled SQL fixtures with CTEs and qualified aliases; explicit descriptions win; docs blocks survive; ambiguous, unavailable or unsupported lineage does not invent a donor. Existing behavior stays default. Specify resource/time bounds and cache invalidation before implementation; preserve license notices if adapting external code. No automatic acceptance of contributor-proposed transformed-column semantics. |
| P2 / artifact boundary follow-up | Establish a supported artifact-consumption contract before any Fusion claim. | Fixtures from identified Core/Fusion producers; check schema/version, missing compiled SQL/catalog, semantic metadata and unchanged YAML. Separate read-only artifact operations from Python compilation and warehouse introspection. No manifest-only CLI until it can fail clearly on unsupported operations. |
| P3 / support policy follow-up | Review legacy Python/Core support and prerelease detection after 1.12 audit. | Measure adapter availability and user impact; explicit policy decision before removing old rows or changing package bounds. Current latest-stable canary is not prerelease coverage; add a bounded 1.13 beta audit separately if warranted. |

## Completed work and active iteration
Core 1.12 coverage merged in #459; issue reconciliation merged in #464 and real CLI residual regressions in #465. Direct-rename lineage merged in #466 at `f1d8947f`. Each completed exact-head and post-merge CI. The original #443/#452 defects were resolved with bounded reproductions; #440 was resolved as direct-rename support plus an explicit decision against automatic derived/ambiguous documentation copying. The five superseded docs dependency PRs were closed after #463 incorporated their changes. GitHub inventory was zero open PRs/issues on 2026-11-17; this is a snapshot, not a completion criterion for the roadmap.

The existing catalog input boundary merged in #467 at `a7632b5b`: invalid supplied artifacts fail before YAML mutation; valid generated Core catalogs and legacy schema-less metadata objects remain accepted. Follow-up #469 at `271b269` completed optional relation database identifier validation, preserving absent/null/string values while rejecting malformed values during preflight. Exact-head and post-merge Tests/lint passed.

Release validation's Core 1.12.6 raw unrendered-config representation assertion was corrected by test-only #470 at `0cdf4a7`. The rendered value and resolver still require boolean false; only the raw version-dependent representation is tolerated. Main Release validation succeeded with 1,144 tests passed and 11 skipped; publication steps were skipped. This is validation evidence, not a new package release.

The artifact support documentation is now on main, describing the implemented boundary, missing-path and missing-relation fallback, and engine support limits. The next bounded iteration is `.10x/tickets/2026-11-20-catalog-error-observability.md`: surface generation errors in supplied compatible catalogs without rejecting usable metadata or printing raw warehouse errors. Official [catalog documentation](https://docs.getdbt.com/reference/artifacts/catalog-json), rechecked 2026-11-20, identifies `errors` as metadata-query failures. Generated catalogs already warn; supplied catalogs lacked that visibility. Focused validation and integration evidence belongs in the ticket. Actual current Fusion producer fixtures remain a separate evidence gap. Draft #468 owns prerelease auditing and is blocked pending explicit approval to resume the previously denied publication/ready/merge actions; this slice does not change it.

Direct lineage remains opt-in for DuckDB/Trino with raw donor docs references, conservative rejection, structural budgets and no persistent candidate cache. Compilation macros may query the warehouse; YAML dry-run remains non-writing. Automatic derived/aggregate/multi-donor copying remains deliberately excluded because origin does not prove unchanged meaning. Trino evidence is offline, not warehouse certification. Actual Fusion producer fixtures, other dialects, artifact-only operations and future Core prerelease auditing remain evidence-gated follow-ups. No broad delivery dates are committed.

## Integration gates
Use isolated branches; publish draft first; commission independent review; require all executed blocking checks on the exact proposed head. Fetch main again before merge, integrate concurrent dependency changes and rerun affected/full remote gates as required. Never treat scheduled-canary skip as an executed pass. Keep release operations separate. Preserve original checkouts and unrelated work.

Observed 2026-11-20: main `4843795` includes merged [#473](https://github.com/z3z1ma/dbt-osmosis/pull/473), a certifi-only lock refresh; stable [v1.6.0](https://github.com/z3z1ma/dbt-osmosis/releases/tag/v1.6.0) was published 2026-11-18. This iteration performs no release or dependency changes. Next evidence-gated artifact priority remains actual versioned producer fixtures before broader engine claims.
