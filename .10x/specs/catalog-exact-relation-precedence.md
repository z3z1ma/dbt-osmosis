Status: active
Created: 2026-11-21
Updated: 2026-11-23

# Exact catalog relation identity precedence

When multiple catalog resources differ only by identifier case, the existing catalog lookup MUST prefer an entry whose raw database, schema and name exactly equal the requested relation identity before invoking the existing adapter/folded fallback. This exact pass MUST consider both nodes and sources, independent of dictionary ordering between an approximate and exact candidate. It MUST NOT normalize identifiers or mutate artifacts. If no exact raw identity exists, the existing matching and missing-relation fallback MUST remain unchanged except for the evidence-backed PostgreSQL quoted-component restriction in `postgres-quoted-catalog-resolution.md`; this bounded fix does not redefine adapter quoting or reject approximate-only catalogs.

Given a quoted db.public.Orders relation and a catalog containing orders with wrong_column followed by Orders with right_column, lookup returns right_column. Reversing the entries gives the same result. A source exact match must win over an earlier approximate node, and vice versa. Existing same-case, case-insensitive fallback, absent/null database behavior and no-match behavior remain compatible. No uniqueness rule for duplicate exact entries is introduced.

Sources checked 2026-11-21: https://docs.getdbt.com/reference/artifacts/catalog-json identifies catalog resource database/schema/name and unique IDs; https://docs.getdbt.com/reference/resource-properties/quoting states quoting depends on adapters and quoted Snowflake identifiers become case sensitive. Installed Core 1.11.8 CatalogTable.key lowercases the three identifiers; installed BaseRelation.matches raises ApproximateMatchError for approximate but not exact matches; osmosis currently treats it as success. These contracts justify preserving exact catalog identity before existing compatibility fallback, not a global quoting rewrite or Fusion engine claim.

No new dependencies, CLI flags, warehouse access, releases, freshness policies, prerelease work, #468 changes, or generic schema hardening.

Compatibility limit: when no exact raw identity exists, the historical approximate-only fallback remains; this fix does not prove that fallback is safe for every quoted case-sensitive relation. Tightening it needs a separate adapter-specific compatibility investigation, not an implicit global behavior change.

The PostgreSQL quoted-component investigation on 2026-11-23 established the bounded exception above. Other no-exact adapter behavior remains unchanged; see `.10x/tickets/2026-11-23-quoted-catalog-resolution.md`.

2026-11-23 authorized refinement: `postgres-unquoted-catalog-precedence.md` now governs PostgreSQL ASCII SQL-included unquoted schema/identifier precedence and fallback. Its proven canonical identity exception supersedes raw exact-first/unquoted-preservation requirements only for that scope; the remaining contract stays unchanged.
