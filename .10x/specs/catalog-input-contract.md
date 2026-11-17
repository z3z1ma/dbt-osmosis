Status: active
Created: 2026-11-17
Updated: 2026-11-17

# Existing catalog input boundary

The existing `--catalog-path` accepts a dbt catalog, not a manifest. Malformed or explicitly incompatible existing artifacts MUST fail with a path-specific actionable diagnostic before YAML mutation; they MUST NOT silently fall back to warehouse queries or catalog regeneration. Absent path configuration and nonexistent paths retain current fallback behavior. Legacy catalog inputs with a schema-less metadata object remain accepted when structurally compatible. An explicitly declared schema MUST identify catalog v1; unknown additive producer metadata must not establish runtime compatibility or be rejected solely because the producer is newer. Validate the structural fields actually consumed and retain dbt's typed deserialization for supported data. Do not rewrite the input artifact or unrelated YAML metadata.

Given a valid generated Core catalog, loading preserves catalog model/source column identity, types and comments. Given manifest JSON, invalid JSON, non-object roots, wrong section shapes or declared unsupported schema, the user receives the catalog path and cause before YAML changes. Given absent catalog path, behavior is unchanged. Given a future producer with the supported catalog schema and compatible structure, acceptance is artifact compatibility only, not Python/Fusion engine certification.

No manifest-only runtime, new CLI flag, dependencies, warehouse execution change, global exception handling refactor, or Fusion producer certification in this slice. Real Core fixture and synthetic future-producer/negative fixtures MUST be distinguished.

Sources verified 2026-11-17: https://docs.getdbt.com/reference/artifacts/catalog-json and https://schemas.getdbt.com/dbt/catalog/v1.json. Published schema is authority for shapes (columns object, despite prose's array label). Existing `_load_catalog` and context fallback behavior supply compatibility authority. User explicitly delegates pragmatic roadmap implementation and compatibility preservation.
