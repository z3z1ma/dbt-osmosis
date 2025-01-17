---
sidebar_position: 4
---
# Migration Guide: Upgrading to dbt-osmosis 1.x.x

## 1. Changes to `vars.dbt-osmosis` Structure

**Old** (pre–1.x.x):

```yaml
vars:
  dbt-osmosis:
    # everything was directly under dbt-osmosis
    <source_name>: ...
    _blacklist:
      - ...
```

**New** (1.x.x+):

```yaml
vars:
  dbt-osmosis:
    sources:
      <source_name>: ...
    column_ignore_patterns:
      - ...
    yaml_settings:
      <kwargs for ruamel.yaml>
```

### Why It’s Breaking

- Previously, users placed all source definitions **directly** under `vars.dbt-osmosis`.
- Now, all source definitions **must** be nested under `vars.dbt-osmosis.sources`.
- Other keys like `column_ignore_patterns` and `yaml_settings` now have **their own** top-level keys under `dbt-osmosis`, instead of living in the same dict.

**Migration**: If your `dbt_project.yml` currently has a line like:

```yaml
vars:
  dbt-osmosis:
    salesforce: "staging/salesforce/source.yml"
    # ...
```

You must nest that under `sources:`:

```yaml
vars:
  dbt-osmosis:
    sources:
      salesforce: "staging/salesforce/source.yml"
    # ...
```

---

## 2. Renamed CLI Flags

The following CLI flags have been renamed for clarity:

1. `--char-length` → `--string-length`
2. `--numeric-precision` → `--numeric-precision-and-scale`
3. `--catalog-file` → `--catalog-path`

If you scripted these old flags, update them to the new names:

```bash
# Old (pre-1.x.x)
dbt-osmosis yaml refactor --char-length --catalog-file=target/catalog.json

# New (1.x.x+)
dbt-osmosis yaml refactor --string-length --catalog-path=target/catalog.json
```

---

## 3. `--auto-apply` Prompt for File Moves

In `1.x.x`, both `organize` and `refactor` commands may prompt you to confirm file moves if dbt-osmosis detects a restructure operation. By default, it asks:

```
Apply the restructure plan? [y/N]
```

### `--auto-apply`

- **Pass** `--auto-apply` to automatically confirm and avoid prompts (helpful in CI/CD).
- If you do **not** pass `--auto-apply`, you will be prompted to confirm any file shuffle.

This is a **behavioral** change: previously, `organize`/`refactor` would just move files without an interactive confirmation step.

---

## 4. Seeds Must Have `+dbt-osmosis: <path>`

To manage seeds with dbt-osmosis in `1.x.x`, you now **must** include a `+dbt-osmosis` directive in your `dbt_project.yml` seeds config. If it’s missing, dbt-osmosis raises an exception.

**Before** (pre-1.x.x), you might not have needed anything for seeds.
**Now**:

```yaml
seeds:
  my_project:
    +dbt-osmosis: "_schema.yml"
```

Without this, seeds are not properly recognized for YAML syncing, and an error occurs.

---

## 5. More Flexible Configuration Resolution

dbt-osmosis `1.x.x` allows you to set options at multiple levels:

- **Global defaults / fallbacks** (via CLI flags)
- **Folder-level** (the preferred, canonical approach via `+dbt-osmosis-options`)
- **Node-level** (via `config(dbt_osmosis_options=...)` in the `.sql` file)
- **Column-level** (via column `meta:` in the schema file)

**Why It Matters**: Now you can override or skip merges, re-lowercase columns, or specify prefixes to handle fuzzy matching—**all** at different granularities. This includes new keys like `prefix`, or existing ones like `output-to-lower`, or `numeric-precision-and-scale` that you can apply per-node or per-column.

Example folder-level override:

```yaml
models:
  my_project:
    staging:
      +dbt-osmosis: "{parent}.yml"
      +dbt-osmosis-options:
        numeric-precision-and-scale: false
        output-to-lower: true
```

Example node-level override in `.sql`:

```sql
{{ config(materialized="view", dbt_osmosis_options={"prefix": "account_"}) }}

SELECT
  id AS account_id,
  ...
FROM ...
```

---

## 6. Inheritance Defaults: No Overwriting Child Docs Without `--force-inherit-descriptions`

In `1.x.x`, **by default**, if your child model has **any** existing column description, dbt-osmosis **won’t** override it with upstream docs. This is a shift from older versions where child descriptions might have been overwritten if upstream had a doc.

- **New**: Must pass `--force-inherit-descriptions` to forcibly overwrite child docs with ancestor docs.
- The old `osmosis_keep_description` approach is effectively **deprecated** (now a no-op). The new approach is simpler: child nodes keep their doc unless you specifically **force** override.

Also, **meta** merges are more **additive**. Child meta keys are **merged** with upstream rather than overwriting them wholesale.

---

## 7. New Plugin System for Fuzzy Matching

dbt-osmosis `1.x.x` adds a **plugin system** (via [pluggy](https://pluggy.readthedocs.io)) so you can supply custom logic for how to match/alias columns across the lineage. Built-in “fuzzy” logic includes:

- Case transformations (upper, lower, camelCase, PascalCase).
- Prefix stripping (if you systematically rename columns like `stg_contact_id → contact_id`).

If you have advanced naming patterns, you can **author** your own plugin that provides additional candidate column matches. This is a new feature and not strictly a breaking change, but important if you rely on custom inheritance logic.

---

## 8. Potential PyPI Release Changes

We’re planning to unify the stable release on `1.1.x` and possibly **yank** any older `1.0.0` from PyPI to reduce confusion. In short, **if you see `1.0.0`** in the wild, upgrade directly to `1.1.x` or later, because the final stable version has renamed flags and structured config.

---

# Summary of Breaking Changes

1. **`vars.dbt-osmosis`** must nest sources under `sources:`.
2. **Renamed CLI flags**:
   - `--char-length` → `--string-length`
   - `--numeric-precision` → `--numeric-precision-and-scale`
   - `--catalog-file` → `--catalog-path`
3. **`organize`/`refactor`** now prompt for file moves unless `--auto-apply` is used.
4. **Seeds** require a `+dbt-osmosis: <path>` config.
5. **Child descriptions** are **not** overwritten unless `--force-inherit-descriptions` is specified (old `osmosis_keep_description` is gone).
6. **Meta merges** for child/parent are more additive (less overwriting).
7. **New plugin system** for fuzzy matching logic.

---

## Recommended Upgrade Steps

1. **Update your `dbt_project.yml`:**
   - Move source definitions under `vars.dbt-osmosis.sources`.
   - Add `+dbt-osmosis: <path>` to your `seeds:` section.

2. **Scan for Old Flags** in scripts or docs:
   - Replace `--char-length`, `--numeric-precision`, `--catalog-file` with the new equivalents.
   - If you rely on no-prompt file moves, add `--auto-apply`.

3. **Decide on Overwrite Strategy**:
   - If you want to preserve old behavior of forcing all child columns to adopt ancestor descriptions, pass `--force-inherit-descriptions`.
   - Otherwise, enjoy the new default: child docs remain if present.

4. **Check Your Options**:
   - Migrate any old `dbt-osmosis` config keys (like prefix usage, skip-add-data-types, skip-merge-meta) into folder-level or node-level overrides as needed.

5. **Explore the New Plugin System**:
   - If you have a complex naming strategy or want to adapt the built-in fuzzy match, you can write a **pluggy** plugin.

6. **Verify**:
   - Run `dbt-osmosis yaml refactor --dry-run` in your project. Check the changes it would make.
   - If everything looks good, run without `--dry-run`.

---

# Conclusion

With **dbt-osmosis 1.x.x**, the YAML management flow becomes **more declarative** and more **extensible**. The changes can require some minor updates to your `dbt_project.yml` and any scripts using the older flags. However, once migrated:

- You gain **safer** merges (less overwriting child docs without intent),
- A **cleaner** config approach for sources and ignoring columns,
- And a **plugin system** for advanced rename logic.

We hope this guide clarifies each step, so you can confidently move to **dbt-osmosis 1.x.x** and enjoy the new features and stability. If you encounter any issues, feel free to open a GitHub issue or consult the updated docs for additional help!
