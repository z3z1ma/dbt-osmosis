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

## Performance

| Operation | dbt-core 1.11.7 | dbt-fusion 2.0.0-preview.154 |
| --------- | ---------------- | ---------------------------- |
| Parse     | ~2.0s            | 637ms (3.1x faster)          |
| Build     | ~30s             | Not tested (parse must pass)  |

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

### Other options (not yet needed)

1. **Fusion allows unknown `+` prefixed keys** (fusion-side): Contribute a fix to fusion. May be on their roadmap.

2. **osmosis reads fusion artifacts instead of core** (medium effort): If fusion produces a compatible `manifest.json`, osmosis could read that instead of calling dbt-core.

3. **Rust port of osmosis** (high effort, high payoff): Rewrite in Rust, using fusion's parser. Most aligned with where dbt is heading.

### Recommendation for OEM project (immediate)

For the OEM project specifically:
- **Remove osmosis from pre-commit and CI** until the `+dbt-osmosis` key issue is resolved
- Keep osmosis installed locally for manual YAML management if needed
- All new models validate with `dbtf parse` as the source of truth
- Track fusion compat issues in this file
- First fix to try: option 1 (move routing config to `.dbt-osmosis.yml` or `meta:`)

### Questions to answer

- Does fusion produce a `manifest.json`? If so, is the schema compatible with what osmosis reads?
- Does `dbt-osmosis yaml refactor --fusion-compat` actually produce fusion-valid output? (Test with `dbtf parse` after running it)
- Can osmosis work with fusion's catalog instead of calling `dbt docs generate` on core?

---

## Test Plan

Once parse errors are fixed:
1. `dbtf build` -- does it compile and run against BigQuery?
2. `dbtf test` -- do all 55 tests pass?
3. `dbt-osmosis yaml refactor --fusion-compat` then `dbtf parse` -- does osmosis output pass fusion?
4. dbt MCP server -- does it work with fusion artifacts?
5. ModuLens -- can it read fusion-generated manifest/catalog?
