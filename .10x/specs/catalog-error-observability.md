Status: active
Created: 2026-11-20
Updated: 2026-11-20

# Supplied catalog generation errors

A successfully deserialized catalog with nonempty `errors` MUST emit one warning per load identifying the supplied path, error count and potentially incomplete metadata. It MUST direct users to the artifact's errors field without printing raw producer errors. Missing, null or empty errors MUST remain quiet. Existing valid partial catalog ingestion, relation fallback, malformed-input failure, input preservation and context caching MUST remain unchanged. This is observability, not new schema validation or Fusion certification.

Given a compatible catalog with one or more generation errors, loading returns its nodes and sources unchanged and warns. Given the same cached context is read again, no duplicate load warning occurs. Given malformed input, existing contextual failure remains authoritative.

Primary sources checked 2026-11-20: https://docs.getdbt.com/reference/artifacts/catalog-json defines errors as metadata-query errors; https://schemas.getdbt.com/dbt/catalog/v1.json defines nullable string-array errors. Existing `_generate_catalog` warns about generation errors, whereas `_load_catalog` returns partial catalogs silently. User authorizes an independent small compatibility/observability roadmap iteration; no release or #468 changes.
