---
sidebar_position: 4
---

# Catalog artifacts and engine support

`--catalog-path` supplies column metadata to dbt-osmosis. It does **not** replace the Python dbt project runtime with an artifact-only or Fusion runtime. dbt-osmosis still uses Python dbt Core and its adapter to load the project and perform compilation where needed.

## Accepted catalog inputs

Supply `catalog.json`, not `manifest.json`. The [dbt catalog contract](https://docs.getdbt.com/reference/artifacts/catalog-json) currently identifies catalog schema v1. dbt v2 documentation uses the Information Schema instead, but the catalog artifact remains available for external tools; that does not establish engine compatibility.

The existing catalog loader accepts:

- A JSON object with `metadata`, `nodes`, and `sources` objects.
- A declared `metadata.dbt_schema_version` equal to `https://schemas.getdbt.com/dbt/catalog/v1.json`, or a legacy metadata object without that field. Entirely missing metadata is not accepted.
- Structurally compatible relation and column entries. Relation `type`, `schema`, and `name` must be strings; `database` may be absent, null, or a string. Columns are an object of entries with string `name` and `type`, an integer `index`, and an optional string or null `comment`.

These checks cover fields consumed by dbt-osmosis, followed by dbt's typed deserialization. They are not a complete JSON Schema validator, a freshness check, or a guarantee that the catalog belongs to your current project. Generate it for the intended project, target, and warehouse state. New or additional producer metadata alone neither rejects an otherwise compatible artifact nor certifies that producer's runtime.

## Failure and fallback behavior

An existing invalid artifact produces an actionable error naming its path. Validation happens before YAML mutation; invalid input does not silently trigger warehouse fallback or catalog regeneration. The input artifact is not rewritten. Project loading may already have occurred before validation, so this is not a promise of zero dbt activity.

A compatible catalog with recorded generation errors remains usable, but dbt-osmosis warns with its path and error count because metadata may be incomplete. Inspect the catalog's `errors` field for details; the warning does not print the producer's raw error messages. Repeated reads within the same context reuse the loaded catalog without repeating the warning.

The YAML and migration CLI commands require an explicitly supplied `--catalog-path` to exist: a nonexistent path is rejected during argument validation, not silently treated as a missing catalog. Omitting the option retains normal lookup behavior. Internally, the catalog loader returns no catalog when its path setting is absent or no longer exists; that internal fallback also covers a file disappearing after CLI argument validation.

Catalog generation can be requested with `--create-catalog-if-not-exists`; it requires introspection to be enabled. For a relation not found in a loaded catalog, column lookup may fall back to the warehouse unless `--disable-introspection` is enabled.

For catalog-backed column lookup without live column introspection:

```shell
dbt-osmosis yaml refactor --catalog-path target/catalog.json --disable-introspection --dry-run
```

This disables dbt-osmosis's live column introspection and catalog generation. It does not make every project operation offline: project macros or compilation paths can still query the warehouse. `--dry-run` previews YAML changes without applying them; it is not a network isolation mechanism. See the [settings reference](./settings.md).

## Core and Fusion boundaries

[dbt versions](https://docs.getdbt.com/docs/dbt-versions) distinguishes the Python 1.x engine from dbt v2. Matching artifact schema numbers do not make their parsing, compilation, or adapter APIs interchangeable.

Catalog validation tests use real Python Core/DuckDB-generated artifacts, a recorded catalog from the official full dbt 2.0.12 distribution with local DuckDB, and synthetic malformed or future-producer cases. The [versioned fixture and provenance](https://github.com/z3z1ma/dbt-osmosis/tree/main/tests/fixtures/fusion_2_0_12) cover model and source column names, types, comments, and ordering through the catalog loader and column lookup without warehouse fallback. The loader preserves producer ordinals; column lookup output indexes follow the existing Core-version-dependent representation. This is evidence for that producer/project/catalog combination, not complete Fusion runtime compatibility. Synthetic metadata remains separate from actual producer evidence. Historical project-specific Fusion experiments are not portable certification of current Fusion versions or adapters. In particular, an empty BigQuery catalog observed in one older installation does not establish that all current Fusion catalogs are empty or unsupported.

There is no manifest-only CLI in this contract. `--fusion-compat` controls YAML compatibility behavior; it does not remove the Python Core dependency or certify the complete workflow against Fusion. Broader support needs versioned producer fixtures and validation of the requested operations, not just successful JSON loading.
