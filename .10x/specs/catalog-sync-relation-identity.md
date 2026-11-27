Status: active
Created: 2026-11-27
Updated: 2026-11-27

# Catalog relation identity through final YAML synchronization

## Contract
Final catalog-backed YAML column type synchronization MUST use the same adapter relation construction and catalog relation selection as column collection. Logical model names and source names MUST NOT independently select catalog donors: model alias, source identifier, database, schema and adapter quoting semantics determine physical identity under the existing canonical resolver. A correct earlier resolution MUST NOT be overwritten with a different donor during YAML merge.

This correction MUST remain catalog-only during final sync: absent catalog or absent matching relation MUST retain existing manifest/current-YAML fallback without triggering warehouse introspection. Existing skip-add-data-types behavior MUST remain operative. Catalog columns MUST use literal metadata names consistently with the existing column normalization contract; artifact mapping keys MUST NOT silently override metadata identity. Existing catalog exact/fallback ordering, PostgreSQL quoted and unquoted guards, optional database compatibility, other-adapter matching, literal DuckDB names and catalog ordinals MUST remain unchanged. No new general matching policy is authorized.

## Scenarios and acceptance
- Given original physical tables physical_accounts.id INTEGER and accounts.id VARCHAR, a logical model accounts aliased to physical_accounts MUST write INTEGER to YAML, including stale initial type, regardless of catalog member order.
- Given source accounts with identifier physical_accounts, actual CLI MUST import the physical table's type and refresh stale initial YAML. Source-only derived catalogs must exercise sources independently.
- Given database-distinct or mixed-quoting donor candidates, final sync MUST agree with canonical selection; adversarial inputs must be labeled as constructed, not producer evidence.
- Given a missing catalog/matching entry, no new introspection occurs. Given skip-add-data-types, current type remains preserved.
- Actual CLI dry-run preserves YAML bytes/mtime; normal repeated runs are idempotent; descriptions, quote flags and semantic metadata survive; input catalog remains unchanged and disabled-introspection consumers create no database.
- Core1.8 and stable1.12 consumer controls plus existing #483/#484/#486/#487 regressions preserve their scoped contracts. Normal exact-head CI and independent review precede integration.

## Authority and scope
Official dbt alias documentation defines the database name separately from logical model name: https://docs.getdbt.com/reference/resource-configs/alias . Source identifier is the stored table name: https://docs.getdbt.com/reference/resource-properties/identifier . Catalog metadata describes physical objects: https://docs.getdbt.com/reference/artifacts/catalog-json . The current canonical resolver and prior PostgreSQL/literal/index specs govern matching and preservation. Actual original local producer and wrong final CLI output are journaled in `.10x/tickets/2026-11-27-owned-importer-boundary.md`.

User-authorized narrow offline correctness work and measured wrong-donor evidence justify reusing the existing authority rather than inventing another matcher. No dependency/support policy changes, upstream DuckDB generic-test repair/contact, #468 action, warehouse, credentials, cloud, release or tag.
