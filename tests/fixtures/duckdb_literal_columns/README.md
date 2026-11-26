# Literal DuckDB column identifiers

`catalog.json` is the unchanged output of Python dbt Core 1.12.7 with
`dbt-duckdb` 1.11.1, `dbt-adapters` 1.24.5 and DuckDB 2.0.0, generated locally
on macOS ARM64 on 2026-11-26. SHA256:
`23cb26b9f77a95473f7d1a5bc40533e1d4e5daced6fc8aa9bca695f80788bfd8`.
The original SQL and YAML inputs are included; only the profile's local absolute
database path is replaced by `fixture.duckdb`. No credentials, real data, binary,
database or bundled macro-bearing manifest is included. The project is original
regression material, and the catalog contains only its names/types and standard
dbt metadata. This is a Core producer fixture, not Fusion/OSS output.

In an isolated environment with the versions above, from `project/`:

```sh
DBT_SEND_ANONYMOUS_USAGE_STATS=false dbt run --profiles-dir .
DBT_SEND_ANONYMOUS_USAGE_STATS=false dbt docs generate --profiles-dir .
```

The five physical column names are `id`, `[id]`, backtick-bearing `id`,
`[[edge]]`, and double-quote-bearing `id`. Their respective types are INTEGER,
BIGINT, SMALLINT, INTEGER and VARCHAR; ordinals are 1 through 5. A read-only
DuckDB cursor confirmed the names/types and the row `(1, 2, 3, 4, '5')`.
Both catalog maps refer to this same physical table. A source-only copy in tests
is explicitly derived, not a second producer output.

The original experiment used `dbt build`, which created the table and passed
8 generic tests but failed the 2 tests for the literal double-quote name:
this installed adapter emits invalid `select ""id""` rather than escaping the
identifier's literal quotes. Subsequent `dbt docs generate` succeeded and produced
this catalog. The manifest preserved literal names and separate `quote: true`
flags for models and sources; ordinary `id` with `quote: true` passed its tests.
Preserving importer identity does **not** fix that upstream generic-test quoting
limitation or promise that arbitrary SQL expressions work as YAML names.

Before correction, `yaml document --catalog-path catalog.json
--disable-introspection` collapsed these distinct names into `id` and `edge`
when adding columns, and transferred the VARCHAR type to multiple preexisting
columns. The regression checks actual YAML output, model/source identity,
metadata, dry-run and repeat-run preservation; it does not require a warehouse
or a producer download in CI.

Authority: [DuckDB identifier rules](https://duckdb.org/docs/2.0/sql/dialect/keywords_and_identifiers),
[dbt column name and quote properties](https://docs.getdbt.com/reference/resource-properties/columns).
The installed adapter's catalog macro selects `duckdb_columns().column_name`
directly, without surrounding SQL delimiters. Reproduction produces fresh
invocation IDs/timestamps, so the hash identifies this retained artifact only.
