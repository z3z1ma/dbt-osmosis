Status: active
Created: 2026-11-16
Updated: 2026-11-16

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

## First iteration and next action
Executable owner: `.10x/tickets/2026-11-16-core-112-audit.md`. The compatibility promotion is in progress until exact-head CI and independent review pass. The next iteration is issue reconciliation for #443/#452 using the already recorded regression evidence, followed by a bounded #440 direct-rename specification and experiment. No broad feature delivery dates are committed.

## Integration gates
Use isolated branches; publish draft first; commission independent review; require all executed blocking checks on the exact proposed head. Fetch main again before merge, integrate concurrent dependency changes and rerun affected/full remote gates as required. Never treat scheduled-canary skip as an executed pass. Keep release operations separate. Preserve original checkouts and unrelated work.
