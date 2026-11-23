Status: active
Created: 2026-11-23
Updated: 2026-11-23

# PostgreSQL unquoted catalog identity

## Contract
For configured adapter type exactly `postgres`, catalog lookup MUST resolve an ASCII schema or identifier included in the rendered SQL and marked unquoted to its lowercase SQL identity. It MUST compare that normalized requested component with raw catalog metadata, not lowercase the candidate. Quoted components remain literal under the existing quoted contract. Both the preferred-identity pass and subsequent candidate filtering MUST enforce the applicable unquoted identity, so a missing canonical candidate cannot fall back to a distinct literal mixed-case object. Normalized candidates across nodes and sources MUST beat literal mixed-case donors regardless of insertion order.

This is a bounded refinement of `catalog-exact-relation-precedence.md` and `postgres-quoted-catalog-resolution.md`. PostgreSQL database identity and its existing quoted/null-database rules MUST remain unchanged: a connection database is not assumed to be a SQL-folded token. SQL-excluded schema/identifier components, missing components, and non-ASCII unquoted components MUST retain existing handling; this work does not establish encoding-, locale-, search-path-, or identifier-truncation semantics. Do not mutate relation objects or artifact data.

Non-PostgreSQL adapters MUST retain existing raw exact precedence and fallback. PostgreSQL valid lowercase unquoted matches, mixed quoted/unquoted matches, #483 quoted rejection, optional catalog database wildcard and existing missing-relation behavior MUST remain intact. Rejection can reach existing adapter introspection only when already enabled; no new connection policy or warehouse execution is introduced.

## Acceptance scenarios
- Actual installed PostgresRelation requests unquoted `Orders` and catalogs contain raw `Orders` (wrong donor) plus `orders` (correct donor): public get_columns returns the lowercase physical object. Cover either section, both cross-map directions and insertion order.
- Equivalent ASCII schema differences and mixed per-component quoting follow SQL inclusion and quoting flags. Quoted `Orders` remains literal.
- Only raw mixed-case object present, including already lowercase unquoted requests: return no catalog match rather than consume unrelated metadata. Preserve existing disabled/enabled introspection paths with offline fakes.
- Null/absent catalog database and SQL-excluded database retain prior behavior. Non-ASCII and excluded schema/identifier controls demonstrate deliberate unchanged limits.
- Actual DuckDB/Trino controls and all existing catalog tests remain valid. Constructed catalogs MUST be identified as offline adversarial inputs, not producer evidence.
- Normal CI regressions use the inherited BaseRelation API plus PostgreSQL context with actual PostgresRelation equivalence recorded; no optional-adapter skips or dependency changes.

## Sources and authorization
User explicitly authorized this follow-up after #483. Actual Core 1.12.7 / dbt-postgres 1.11.0 public-consumer matrix is recorded in `.10x/tickets/2026-11-23-postgres-unquoted-catalog-precedence.md`.
Primary sources checked 2026-11-23: https://www.postgresql.org/docs/current/sql-syntax-lexical.html ; https://www.postgresql.org/docs/current/ddl-schemas.html ; https://raw.githubusercontent.com/postgres/postgres/REL_19_STABLE/src/backend/parser/scansup.c . PostgreSQL's parser always lowercases ASCII A-Z, while high-bit behavior depends on encoding/locale; generic Unicode lower/casefold is not justified. Actual PostgresRelation inherits BaseRelation creation, matching and inclusion behavior; create_from preserves configured paths. Database connection identity remains outside this normalization scope.
