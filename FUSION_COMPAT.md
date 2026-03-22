# dbt-fusion Compatibility Findings

Tracked from real-world testing on [oem-dbt-bigquery](https://github.com/RicardoAGL/oem-dbt-bigquery) (BigQuery, Kimball star schema, 15+ models).

dbt-fusion version: **2.0.0-preview.154**

---

## Finding 1: `+dbt-osmosis` config keys rejected by fusion parser

**Severity**: Blocking (parse fails)
**dbt-core behavior**: Ignores unknown `+` prefixed keys in `dbt_project.yml`
**dbt-fusion behavior**: Strict schema validation rejects them as invalid config

```
error: dbt1013: Invalid model definition `oem_data_platform.staging.+dbt-osmosis`:
  invalid type: string "_stg_{parent}__models.yml", expected struct ProjectModelConfig
  --> dbt_project.yml:25:21
```

**Affected config** (from `dbt_project.yml`):
```yaml
models:
  oem_data_platform:
    staging:
      +dbt-osmosis: "_stg_{parent}__models.yml"
    intermediate:
      +dbt-osmosis: "_int_{parent}__models.yml"
    marts:
      +dbt-osmosis: "_marts_{parent}__models.yml"
seeds:
  +dbt-osmosis: "_seeds__models.yml"
```

**Impact on osmosis**: dbt-osmosis uses these keys to route YAML schema files. If fusion rejects them, projects that use both osmosis and fusion cannot parse. This is the most critical compat issue for the osmosis + fusion combination.

**Solution implemented**: Move routing config to `vars.dbt-osmosis.models` in `dbt_project.yml`. The `vars:` section is standard dbt and accepted by both core and fusion.

**Before** (breaks fusion):
```yaml
models:
  oem_data_platform:
    staging:
      +dbt-osmosis: "_stg_{parent}__models.yml"
```

**After** (works with both core and fusion):
```yaml
vars:
  dbt-osmosis:
    models:
      staging: "_stg_{parent}__models.yml"
      intermediate: "_int_{parent}__models.yml"
      marts: "_marts_{parent}__models.yml"
    seeds: "_seeds__models.yml"
```

Routing matches against the node's FQN folder path. For nested folders, use dot notation (e.g., `staging.oem_raw`). The most specific match wins. Existing `+dbt-osmosis` config keys still take priority for backward compatibility.

**Status**: Implemented in `feat/fusion-vars-routing` branch. 18 tests passing (16 unit + 2 integration). 678 total tests green.

---

## Finding 2: Source freshness syntax must use `config:` block

**Severity**: Blocking (parse fails)
**dbt-core behavior**: Warns (deprecated), still works
**dbt-fusion behavior**: Rejects as unexpected key

```
error: dbt1060: Ignored unexpected key `"freshness"`. YAML path: `freshness`.
  --> models/staging/oem_raw/_src_oem_raw.yml:10:5
error: dbt1060: Ignored unexpected key `"loaded_at_field"`. YAML path: `loaded_at_field`.
  --> models/staging/oem_raw/_src_oem_raw.yml:13:5
```

**Fix**: Move `freshness` and `loaded_at_field` into `config:` block per dbt 1.9+ syntax. This is a known deprecation that core still accepts but fusion enforces.

**Impact on osmosis**: If osmosis generates or reads freshness config at the old location, it needs to handle the new `config:` location too. Check if `yaml refactor` handles this migration.

**Status**: Fix is straightforward in the project; osmosis needs testing

---

## Finding 3: Deprecated test argument format

**Severity**: Blocking (parse fails)
**dbt-core behavior**: Warns (deprecated), still works
**dbt-fusion behavior**: Rejects

```
error: dbt0102: Deprecated test arguments: ["field", "to"] at top-level detected.
  Please migrate to the new format under the 'arguments' field.
  --> models/marts/_marts_marts__models.yml:201:22
```

**Old format** (core accepts, fusion rejects):
```yaml
tests:
  - relationships:
      to: ref('dim_date')
      field: date_key
```

**New format** (both accept):
```yaml
tests:
  - relationships:
      arguments:
        to: ref('dim_date')
        field: date_key
```

**Impact on osmosis**: Check if `yaml refactor` generates old or new format. If old, osmosis needs a flag or default to emit fusion-compatible syntax. The `--fusion-compat` flag may already handle this -- needs verification.

**Status**: Fix is straightforward in the project; osmosis `--fusion-compat` needs testing

---

## Performance (tested 2026-03-22, oem-dbt-bigquery, 16 models + 34 tests + 5 seeds)

| Operation | dbt-core 1.11.7 | dbt-fusion 2.0.0-preview.154 |
| --------- | ---------------- | ---------------------------- |
| Parse     | ~2.0s            | 854ms (2.3x faster)          |
| Build     | ~30s             | 62s (includes catalog gen)    |
| Build (no catalog) | ~30s    | ~50s (estimated)              |

Note: fusion build includes `--write-catalog` overhead. The 55/55 pass rate is identical to core.

---

## Architectural Problem: osmosis depends on dbt-core internals

This is the biggest challenge. dbt-osmosis uses dbt-core's Python engine internally (manifest parsing, adapter calls, YAML generation). It does not just read/write YAML files -- it calls into dbt-core to resolve refs, compile SQL, and understand the project graph.

This means:
- **osmosis requires dbt-core as a Python dependency** -- you cannot uninstall core and run osmosis alone
- **In a fusion project, you have two engines**: fusion (Rust, fast, strict) and core (Python, slower, lenient). They may disagree on what is valid.
- **osmosis-generated YAML may be valid for core but invalid for fusion** (Finding 1 and 3 above are examples)

### The hybrid approach (documented in this fork)

The intended workflow is: **install both core and fusion**. Osmosis uses dbt-core internally to resolve the project graph and generate YAML schemas. The output targets the fusion project. This works for most things -- osmosis reads models, resolves columns, writes YAML that fusion can consume.

**The catch**: `dbt_project.yml` is shared by both engines. Osmosis stores its routing config there as `+dbt-osmosis` keys. Core ignores unknown `+` prefixed keys. Fusion's strict parser rejects them. This is the one file where the hybrid breaks.

### Solution: vars-based routing (implemented)

The `feat/fusion-vars-routing` branch adds a new resolution step in `_get_yaml_path_template` (path_management.py). Instead of requiring `+dbt-osmosis` config keys, users can place routing under `vars.dbt-osmosis.models`:

```yaml
vars:
  dbt-osmosis:
    models:
      staging: "_stg_{parent}__models.yml"
      intermediate: "_int_{parent}__models.yml"
      marts: "_marts_{parent}__models.yml"
    seeds: "_seeds__models.yml"
    sources:
      oem_raw:
        path: "staging/oem_raw/_src_oem_raw.yml"
```

The `vars:` section is standard dbt -- both core and fusion accept it. Routing matches against the node's FQN folder path, with most-specific match winning (e.g., `staging.oem_raw` beats `staging`). Existing `+dbt-osmosis` config keys still take priority for backward compatibility.

### Fusion artifact compatibility (tested 2026-03-22)

| Artifact | Fusion produces? | Format | Compatible with osmosis? |
| -------- | ---------------- | ------ | ----------------------- |
| manifest.json | Yes | v12 (same as core) | Yes -- osmosis can read it |
| catalog.json | Yes (`--write-catalog`) | v1 (same as core) | No -- empty due to TABLE_STORAGE permission bug |
| run_results.json | Yes | v6 (same as core) | N/A |
| semantic_manifest.json | Yes | Standard | N/A |

**Key finding**: Fusion's `--write-catalog` queries `INFORMATION_SCHEMA.TABLE_STORAGE` which requires additional BigQuery permissions. Core's `dbt docs generate` uses a different query path that works with standard `dataEditor` + `jobUser` roles. This is likely a fusion bug or missing macro override for BigQuery.

**Workaround**: Use `dbt docs generate` (core) to produce the catalog, then use fusion for everything else.

### dbt-core dependency analysis

Osmosis depends on dbt-core at 4 levels:

| Level | What | Can replace with fusion? |
| ----- | ---- | ----------------------- |
| Engine | Parse project, load manifest | Yes -- read fusion's manifest.json instead |
| Catalog | Query warehouse for column metadata | Not yet -- fusion catalog empty (permission bug) |
| SQL compilation | SqlCompileRunner, process_node | No -- fusion has no Python API |
| Data types | ResultNode, NodeType, ColumnInfo | No -- used in every function signature |

**Bottom line**: Osmosis cannot run without dbt-core today. The hybrid approach (both engines installed) is the practical path.

### Other options (future)

1. **Read fusion manifest instead of re-parsing** (medium effort): Since fusion produces v12 manifests, osmosis could skip its own `dbt parse` and read from disk. Still needs core for catalog.

2. **Fusion fixes TABLE_STORAGE bug** (fusion-side): Once fusion's catalog works on BigQuery, osmosis could use it instead of `dbt docs generate`.

3. **Manifest-only mode** (medium effort): Osmosis reads manifest + catalog from disk without invoking core's parser. Core only needed for SQL compilation.

4. **Rust port** (high effort): Rewrite as fusion plugin. Maximum performance, zero Python dependency.

### Recommendation for OEM project

- **Use `dbtf build` as primary engine** -- all 55 nodes pass
- **Use `dbt docs generate` (core) for catalog** -- needed until fusion catalog bug is fixed
- **Use vars-based routing** -- `+dbt-osmosis` keys replaced with `vars.dbt-osmosis.models`
- **Validate with `dbtf parse`** as source of truth for YAML correctness
- osmosis can be used for YAML management in the hybrid setup

### Questions answered

- **Does fusion produce manifest.json?** Yes, v12 schema, fully compatible with core and osmosis.
- **Does fusion produce catalog.json?** Yes with `--write-catalog`, but empty on BigQuery due to TABLE_STORAGE permission query. Needs bug report or permission fix.
- **Can osmosis work without core?** No. Needs core for catalog generation and SQL compilation. Manifest can be read from fusion.

### Remaining questions

- Does `dbt-osmosis yaml refactor --fusion-compat` produce fusion-valid output? (Test with `dbtf parse` after running it)
- dbt MCP server -- does it work with fusion artifacts?
- ModuLens -- can it read fusion-generated manifest/catalog?

---

## Test Plan Results (2026-03-22)

| Test | Result |
| ---- | ------ |
| `dbtf parse` | PASS -- 0 errors after fixing freshness, test args, and +dbt-osmosis keys |
| `dbtf build` | PASS -- 55/55 (16 models, 34 tests, 5 seeds) |
| `dbtf build --write-catalog` | PARTIAL -- catalog.json generated but empty (BQ permission bug) |
| `dbt docs generate` (core) | PASS -- 30.9K catalog with full column metadata |
| osmosis vars routing | PASS -- 18 tests (feat/fusion-vars-routing branch) |
