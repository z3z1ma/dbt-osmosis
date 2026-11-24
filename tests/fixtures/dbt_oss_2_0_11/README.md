# Apache-only dbt OSS 2.0.11 / local DuckDB catalog

`catalog.json` is the **unedited, byte-for-byte** output of the official Apache-only
`dbt-oss==2.0.11` distribution on macOS ARM64, generated on 2026-11-24. It is not
the full proprietary distribution, a Python Core v2-parser experiment, or synthetic
JSON. CI consumes the recorded artifact with Python Core; it does not install v2.

## Provenance and retained content

The [official OSS installation guide](https://docs.getdbt.com/docs/local/install-dbt-v2)
identifies `dbt-oss` and requires an isolated environment because it shares the
`dbt` entry point. Official [PyPI metadata](https://pypi.org/pypi/dbt-oss/2.0.11/json)
identified the Apache-2.0 package and checksum-pinned bootstrap. At generation time,
the latest available OSS release was 2.0.11; 2.0.12 metadata returned HTTP 404.
The [full-product comparator](../fusion_2_0_12/README.md) is a different release,
2.0.12. These observations do not establish equal engine features or release cadence.

Verified downloads and installed native payload:

- PyPI `dbt_oss-2.0.11.tar.gz` SHA256:
  `ed5cccb16863d25b1c358ee2f2d2ef6288043b7e4fa9e81acc5763a4317a871e`.
- Its inspected `assets.json` selects the official release wheel
  `https://github.com/dbt-labs/dbt/releases/download/v2.0.11/dbt_oss-2.0.11-cp310-abi3-macosx_11_0_arm64.whl`,
  SHA256 `e4a0cbc60f02bee1d12117d952bb9f11a74fc33959d5b7e8ab83ff761f222bef`.
- Installed `dbt/_core.abi3.so` SHA256:
  `a01c4fd7834400f22f65127e11c5bf9a230ce4bf025f0260b8b2101d58257dfc`.

Both download hashes matched before installation in a separate Python 3.13.14
environment. `dbt --version` printed `dbt-oss 2.0.11`; help identified the OSS
system command and stated that lint requires the full distribution. The
[pinned upstream license](https://github.com/dbt-labs/dbt/blob/v2.0.11/LICENSE)
is Apache 2.0. No producer binary, wheel, database, or upstream macro payload is
redistributed. The SQL/YAML inputs are original osmosis test inputs reused from
the full-product experiment, with only the local database profile path changed.
The historical project name `osmosis_fusion_fixture` remains unchanged to control
the comparison; it does not identify the producing distribution.

The catalog contains original relation/column metadata, standard artifact stats,
empty `env`, a timestamp and a random invocation UUID. Inspection found no user
ID, username, credential, user-specific path, SQL or macro payload. The profile's
sole absolute path is the deliberately chosen generic local path below. No
sanitization or producer metadata rewriting was performed.

## Reproduction

Use a fresh `/tmp/osmosis-oss-2011-project` containing the included `project/`
files, and an isolated official `dbt-oss==2.0.11` installation. Do not install it
alongside full dbt or Python Core in the same environment.

```sh
dbt --version
dbt parse --project-dir /tmp/osmosis-oss-2011-project \
  --profiles-dir /tmp/osmosis-oss-2011-project \
  --no-version-check --no-manage-state --no-send-anonymous-usage-stats
# Preserve target/manifest.json here if comparing parse-only output.
dbt build --write-catalog --project-dir /tmp/osmosis-oss-2011-project \
  --profiles-dir /tmp/osmosis-oss-2011-project \
  --no-version-check --no-manage-state --no-send-anonymous-usage-stats
```

Observed execution used an absolute pinned executable in a clean environment with
`DBT_SEND_ANONYMOUS_USAGE_STATS=false`, no credentials, packages, extensions or cloud
services. The bundled DuckDB adapter ran in-process against the local file without
a separate adapter installation. dbt required its normal temporary `~/.dbt/leases`
state; no home/profile configuration was changed. `parse` succeeded; `build`
succeeded for both physical table models and wrote the catalog. Read-only local
DuckDB queries confirmed `accounts` held `(1, 'active', true)` and `account_labels`
held `(1, 'active')`, with the recorded INTEGER/VARCHAR/BOOLEAN types.

| Observed artifact | Schema | SHA256 |
| --- | --- | --- |
| Parse manifest | v12 | `8e99f0d08c37d99dcdaf31b24ecac5d81c714f9fc25a96ded18571b2f511a535` |
| Build manifest | v12 | `14b8b79ab6ca246c39273b1e700e8301d48ae978cd750b22c13950b50ebf7f61` |
| Catalog (2,338 bytes) | v1 | `27ea7acac2bf49d23fd04442c4e2d17ba26cfe4db39bfa26cbcff8c7ae66a314` |
| Run results | v6 | `c23794e125f452e74c6734cca92d3dd62dd4b8363942cfb75cf074c3bc8f190e` |

Regeneration changes timestamps and invocation IDs; hashes identify this observed
run, not reproducible bytes across invocations. Both manifests have two models and
one source; only built model nodes have compiled SQL. They contain 525 bundled
macros unnecessary for the catalog regression and are not retained. Core 1.12.7
deserialized both actual manifests with `Manifest.from_dict`; that experiment does
not establish a manifest-only osmosis runtime.

## Consumer contract and limits

The regression checks the exact raw and typed-loader names, types, comments and
one-based ordinals for both models and the source. Public `get_columns` preserves
order/names/types/comments without warehouse fallback or input mutation. Public
output indexes remain Core-version-dependent, as documented for the full fixture.
The source map is isolated in memory so the same physical model cannot mask source
consumption. Each case starts with fresh caches; both fixture and working-copy bytes
remain unchanged.

For these same original inputs, all non-metadata catalog content matches the
separately recorded full dbt 2.0.12 fixture. The only differing top-level metadata
fields are `dbt_version`, `generated_at` and `invocation_id`; `dbt_schema_version`
and the empty `env` are equal. This is a measured artifact comparison
across distinct producer versions, not OSS/full-engine parity. It does not certify
SQL comprehension, Python APIs, other adapters, live warehouses, arbitrary OSS
projects, or a complete v2-based osmosis runtime. Python Core remains required;
this fixture changes no runtime behavior or dependency bounds.
