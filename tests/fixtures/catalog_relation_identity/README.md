# Original local catalog relation identity fixture

Generated 2026-11-27 with Python 3.12.8, dbt-core 1.12.7,
dbt-duckdb 1.11.1, dbt-adapters 1.24.5 and DuckDB 2.0.0.
The SQL/YAML inputs are original and contain no credentials or private data.
The profile is changed only to use the relative local paths `fixture.duckdb` and `other.duckdb`.

Reproduce in an isolated environment with those versions:

```sh
cd project
python -c "import duckdb; duckdb.connect('other.duckdb').close()"
dbt run --project-dir . --profiles-dir .
dbt docs generate --project-dir . --profiles-dir .
```

Both commands succeeded. Read-only `DESCRIBE physical_accounts` returns
`id INTEGER`, while `DESCRIBE accounts` returns `id VARCHAR`. A third model
builds `other.main.physical_accounts.id BIGINT` in a local attached database;
its catalog metadata and source declaration exercise database identity. The logical
model `accounts` aliases the first table; `decoy` aliases the second. The source
logical name `accounts` also uses `identifier: physical_accounts`.

`catalog.json` is the unedited producer artifact (catalog v1), SHA256:
`3dc887e540a5aad748aae6cc385900e0c9a0b985a3b9ef4acca9ce35f5c2cb9f`.
No database, binary or bundled manifest/macros are redistributed.

The regression executes the actual osmosis CLI with introspection disabled,
including stale types and source-only/reordered derived catalog copies. These
copies are consumer test inputs, not independent producer artifacts. This
fixture proves the recorded Core/DuckDB boundary only; it is not full v2/OSS
or warehouse compatibility evidence.
