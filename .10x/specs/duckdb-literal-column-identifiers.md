Status: active
Created: 2026-11-26
Updated: 2026-11-26

# DuckDB literal column identity

## Contract
For adapter type exactly `duckdb`, physical catalog/adapter column names and parsed YAML column names are literal identifier values, not SQL expressions. Normalization MUST preserve their characters, including leading/trailing brackets, backticks and literal double quotes. SQL identifier quoting is a separate representation; the YAML `quote` property controls quoting and MUST remain preserved. Other adapters MUST retain their existing normalization behavior. Explicit output-to-upper/lower configuration remains a separate existing transformation and MUST NOT be silently changed.

The shared normalization boundary MUST preserve identity consistently through column lookup/conversion, add/remove/sort/update transforms, YAML merge, unrendered documentation inheritance and schema comparison. A catalog-only key change that still collapses names later does not satisfy this contract. Existing indexes, types, comments, raw ignore-regex matching, ordinary case behavior, database/alphabetical ordering and artifact bytes MUST remain preserved. Repeated document runs MUST be idempotent. No automatic new quote flag, general SQL unquoting parser, dependency/support-policy change or behavior change for other adapters is included.

## Scenarios and acceptance
- An original local DuckDB table with distinct physical `id`, `[id]`, backtick-bearing `id`, nested brackets and double-quote-bearing `id` produces a real catalog. Actual yaml document with disabled introspection MUST keep every model/source column and correct type, with consumer database files absent and input catalog unchanged.
- Given existing YAML entries with distinct descriptions, types, quote flags and custom metadata, synchronization MUST NOT merge them or transfer one column's metadata to another. Source-only catalog input MUST independently exercise source handling.
- Given name `id` with quote true, normal SQL-quoted usage remains `id`; given a literal name containing double-quote characters, those characters survive. Actual parser/producer evidence must establish this distinction before activation.
- Raw ignore patterns remain raw; explicit order/case settings remain operative. Dry-run MUST preserve YAML bytes/mtime; a second ordinary document run MUST preserve the first output bytes.
- Shared helper callers receive focused controls for YAML merge, inheritance and diff, alongside normal broad CI. Other-adapter normalization controls, older/newer Core conversion and existing producer fixtures MUST retain their original bounded expectations except the explicitly corrected DuckDB literal-name semantics.

## Authority and boundaries
Current DuckDB identifier documentation permits quoted names containing special characters and distinguishes escaped SQL delimiters from stored values: https://duckdb.org/docs/2.0/sql/dialect/keywords_and_identifiers . dbt column properties define name and quote separately: https://docs.getdbt.com/reference/resource-properties/columns . Installed dbt-duckdb1.11.1 catalog macros select duckdb_columns().column_name directly, without SQL wrapping. Real pre-fix Core1.12.7 producer and Core1.8.10/1.12.7 consumer observations are owned by `.10x/tickets/2026-11-26-importer-boundary.md`.

This deliberately refines the historical DuckDB stripping behavior, including the preservation-only scope of `.10x/specs/catalog-column-ordinals.md`; the ordinal contract itself is unchanged. It does not certify other adapters, arbitrary SQL expressions as YAML names, or full v2/OSS engine parity. Activated after actual five-column producer catalog and parsed model/source names confirmed the literal-doublequote/conventional-quote distinction; parent approved the DuckDB-only shared-boundary correction2026-11-26. The measured dbt-duckdb1.11.1 generic not_null SQL escaping failure for literal embedded quotes remains an upstream limitation, not covered by this importer correction.
