Status: active
Created: 2026-11-23
Updated: 2026-11-23

# PostgreSQL quoted catalog resolution

## Contract
For a project whose configured adapter type is exactly `postgres`, the no-exact catalog fallback MUST reject a candidate when a quoted requested relation component differs from the corresponding raw catalog metadata component. Compare raw names before `CatalogTable.key()` lowercases them. The components are database, schema, and identifier (catalog name). Missing/null database metadata MUST retain the existing wildcard behavior, rather than be treated as a case conflict. This is a narrow exception to the historical no-exact fallback preservation in `catalog-exact-relation-precedence.md`.

The exact-identity pass MUST remain unchanged. Unquoted components MUST retain existing matching/folding behavior. Non-PostgreSQL adapters, including DuckDB and Trino (case-insensitive quoted identifiers) and Snowflake (session/account-dependent quoted behavior), MUST remain unchanged. Rejected candidates continue through remaining catalog entries; if none qualifies, the existing missing-relation behavior applies, including empty output with introspection disabled or existing adapter fallback when enabled. This change MUST NOT add warehouse access or mutate catalog input.

## Scenarios and acceptance
- Quoted PostgreSQL `Orders`, catalog only `orders`: do not consume unrelated columns. The inverse (`orders` requested, raw `Orders` catalog) also rejects, even though the folded key appears exact.
- Equivalent quoted schema/database mismatches reject; a later valid candidate is still selected. Cover nodes and sources.
- Quoted same-case identity succeeds; unquoted `Orders` and catalog `orders` continue to match. A correctly equal quoted component must survive case differences in another unquoted component.
- Null/missing catalog database retains historical matching. DuckDB/Trino quoted case-fold controls and generic existing fallback remain valid.
- Public `get_columns` tests assert column identity/type/comment, missing behavior, no unintended adapter calls, and stable input; genuine producer tests stay intact.
- Reproduce before/after with installed PostgresRelation and stable Core 1.12.7. Regular CI tests may use its identically inherited BaseRelation matching API plus a PostgreSQL context, with equivalence evidence recorded; tests must not be skipped for absent optional adapters.

## Evidence and boundaries
Offline investigation in the owning ticket reproduced wrong columns using installed PostgresRelation, Core 1.12.7, dbt-postgres 1.11.0. PostgresRelation.matches and _is_exactish_match are inherited unchanged from BaseRelation. Constructed adversarial catalog inputs are not producer artifacts or warehouse evidence.
Primary sources checked 2026-11-23: PostgreSQL19 lexical structure https://www.postgresql.org/docs/19/sql-syntax-lexical.html (quoted identifiers case-sensitive, unquoted lowercased); DuckDB https://duckdb.org/docs/2.0/sql/dialect/keywords_and_identifiers; Trino https://trino.io/docs/current/language/reserved.html (case-insensitive identifiers); Snowflake https://docs.snowflake.com/en/sql-reference/identifiers-syntax (QUOTED_IDENTIFIERS_IGNORE_CASE changes semantics). No general quote-policy-to-sensitivity inference is valid.

User authorized adapter-specific investigation and a narrowly reproduced correctness fix while preserving working combinations. This contract does not redefine unquoted exact-first precedence, other adapters, arbitrary session settings, full Fusion parity or supported dependency bounds. Broader concerns need independently reproduced scope.
