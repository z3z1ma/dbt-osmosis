# Full dbt 2.0.12 / local DuckDB catalog

`catalog.json` is the **unedited, byte-for-byte** output of the official full dbt
2.0.12 distribution (formerly Fusion), generated on macOS ARM64 on 2026-11-22.
It is not synthetic JSON, dbt OSS output, or Python Core with a v2 parser.
The regression consumes this recorded catalog with the supported Python Core
runtime; CI does not download or execute the proprietary producer.

## Provenance

The [official installation instructions](https://docs.getdbt.com/docs/local/install-dbt)
identify the `dbt` Python distribution. Its pinned
[PyPI metadata](https://pypi.org/pypi/dbt/2.0.12/json) identified this bootstrap:

- `dbt-2.0.12.tar.gz`, SHA256
  `b12c71e1fb10734f310b002892a2a44a98db5c09e9d8671b60b5eb34c235e239`.
- The bootstrap's `assets.json` identified
  `https://public.cdn.getdbt.com/fs/wheels/dbt-2.0.12-cp310-abi3-macosx_11_0_arm64.whl`,
  SHA256 `6af97acc6f8ce22cb0a1a65a905f659c8942580014c59a52dbbe1b5dc24d27a0`.
- Both downloaded hashes matched. The wheel was installed in a separate Python
  3.13.14 environment. `dbt --version` printed `dbt 2.0.12`; help identified the
  proprietary distribution. Its entrypoint invokes the native `dbt._core.run_cli`.
  Installed `_core.abi3.so` SHA256:
  `e741763a414a00dc104081bf526fd7e6d869ac5f4bfcda0ba6436a843c0f0360`.

No producer binary, wheel, bundled macro code, database file, or credentials are
redistributed here. The project SQL/YAML and descriptions are original test inputs.
The catalog contains their local relation/column metadata, empty `env`, standard
artifact statistics, a generation timestamp, and a random invocation UUID. It has
no user ID, username, credential, user-specific path, or model/macro source payload.
All retained project files were inspected; the sole absolute path is the deliberately
chosen generic `/tmp/osmosis-fusion-2012-project/fixture.duckdb` in the original profile.
No sanitization or producer metadata rewriting was performed.

The [Product Licensing Agreement](https://www.getdbt.com/dbt-product-license-agreement)
governs use and redistribution of the proprietary product. The
[official licensing FAQ](https://www.getdbt.com/licenses-faq) expressly permits
metadata artifact integrations. Review found no restriction on retaining this
original generated metadata fixture. This fixture does not redistribute the product
or expose it as a service. Reproduction requires independently installing and using
the official product under its applicable terms.

## Reproduction

Copy the included `project/` files to `/tmp/osmosis-fusion-2012-project` (a fresh
path), then use an isolated installation of the pinned full `dbt==2.0.12`:

```sh
dbt --version
dbt parse --project-dir /tmp/osmosis-fusion-2012-project \
  --profiles-dir /tmp/osmosis-fusion-2012-project \
  --no-version-check --no-manage-state --no-send-anonymous-usage-stats
# Preserve target/manifest.json here if comparing parse-only output.
dbt build --write-catalog --project-dir /tmp/osmosis-fusion-2012-project \
  --profiles-dir /tmp/osmosis-fusion-2012-project \
  --no-version-check --no-manage-state --no-send-anonymous-usage-stats
```

The observed commands used a clean environment with
`DBT_SEND_ANONYMOUS_USAGE_STATS=false`, explicit absolute executable paths, no
credentials, packages, extensions, or cloud services. DuckDB ran in-process against
the local file. dbt also required its normal temporary `~/.dbt/leases` files; explicit
profile/project paths do not relocate those. No home/profile settings were changed.

`parse` succeeded. `build --write-catalog` succeeded for both table models. Read-only
DuckDB queries independently confirmed `accounts` held `(1, 'active', true)` and
`account_labels` held `(1, 'active')`, with the types recorded in the catalog.

| Observed output | Schema | Content / SHA256 |
| --- | --- | --- |
| Parse manifest | v12 | 2 models, 1 source; no model `compiled_code`; `1c97c66729d7f829135df21ab5e6b21a0d522ca34acb3ca5d48f43b7b8c6627e` |
| Build manifest | v12 | 2 models with compiled SQL, 1 source; `39ff9dac3d6dd2c22f87d82c96b9fdcc383075a072a39b829f0d9a270fe91c89` |
| Catalog | v1 | 2 models, 1 source, persisted comments, no errors; `7e6faa5cedf317d7321350b6461518d622072bf12c306d4c950e6e4e1a1171aa` |
| Run results | v6 | 2 successful models; `1d7913ee73586d25a02706b08bf016f66d76740da17a079b71844545d4da75a0` |

Regeneration changes timestamps/invocation IDs, so these hashes describe the observed
run, not reproducible byte identity across runs. The raw catalog is retained unchanged.
The manifests are not retained: they include 525 bundled upstream macros unnecessary
for the catalog regression. Python Core 1.11.8 successfully deserialized both actual
manifests with `Manifest.from_dict`; that is an experiment result, not a manifest-only
runtime contract.

## What the regression establishes

The catalog loader and public `get_columns` preserve this producer's model and source
column names, INTEGER/VARCHAR/BOOLEAN types, comments, and indexes without warehouse
fallback or rewriting input bytes. The source case isolates the source map in memory,
so it cannot pass by selecting the model for the same physical table. The on-disk
fixture remains unchanged.

This does not certify all Fusion projects, SQL compilation, Python APIs, other
adapters, live warehouses, or a complete Fusion-based osmosis runtime. Python Core
remains required. No runtime behavior or dependency bounds were changed for this test.
