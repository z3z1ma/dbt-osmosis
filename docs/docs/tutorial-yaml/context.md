---
sidebar_position: 2
---
# Context Variables

dbt-osmosis provides three primary variables—`{model}`, `{node}`, and `{parent}`—that can be referenced in your `+dbt-osmosis:` path configurations. These variables let you build **powerful** and **dynamic** rules for where your YAML files should live, all while staying **DRY** (don’t repeat yourself).

## `{model}`

This variable expands to the **model name** being processed. If your model file is named `stg_marketo__leads.sql`, `{model}` will be `stg_marketo__leads`.

**Usage Example**

```yaml title="dbt_project.yml"
models:
  your_project_name:
    # A default configuration that places each model's docs in a file named after the model,
    # prefixed with an underscore
    +dbt-osmosis: "_{model}.yml"

    intermediate:
      # Overrides the default in the 'intermediate' folder:
      # places YAMLs in a nested folder path, grouping them in "some/deeply/nested/path/"
      +dbt-osmosis: "some/deeply/nested/path/{model}.yml"
```

### Why Use `{model}`?

- **One-file-per-model** strategy: `_{model}.yml` => `_stg_marketo__leads.yml`
- **Direct mapping** of model name to YAML file, making it easy to find
- **Simple** approach when you want each model’s metadata stored separately

## `{node}`

`{node}` is a **powerful** placeholder giving you the entire node object as it appears in the manifest. This object includes details like:

- `node.fqn`: A list describing the folder structure (e.g., `["my_project", "staging", "salesforce", "contacts"]`)
- `node.resource_type`: `model`, `source`, or `seed`
- `node.language`: Typically `"sql"`
- `node.config[materialized]`: The model’s materialization (e.g. `"table"`, `"view"`, `"incremental"`)
- `node.tags`: A list of tags you assigned in your model config
- `node.name`: The name of the node (same as `{model}`, but you get it as `node.name`)

With this variable, you can reference **any** node attribute directly in your file path.

**Usage Example**

```yaml title="dbt_project.yml"
models:
  jaffle_shop:
    # We have a default config somewhere higher up. Now we override for intermediate or marts subfolders.

    intermediate:
      # advanced usage: use a combination of node.fqn, resource_type, language, and name
      +dbt-osmosis: "node.fqn[-2]/{node.resource_type}_{node.language}/{node.name}.yml"

    marts:
      # more advanced: nest YAML by materialization, then by the first tag.
      +dbt-osmosis: "node.config[materialized]/node.tags[0]/schema.yml"
```

### Creative Use Cases

1. **Sort YAML by materialization**

   ```yaml
   +dbt-osmosis: "{node.config[materialized]}/{model}.yml"
   ```

   If your model is a `table`, the file path might become `table/stg_customers.yml`.

2. **Sort YAML by a specific tag** (you could use `meta` as well)

   ```yaml
   +dbt-osmosis: "node.tags[0]/{model}.yml"
   ```

   If the first tag is `finance`, you’d get `finance/my_model.yml`.

3. **Split by subfolders**

   ```yaml
   +dbt-osmosis: "{node.fqn[-2]}/{model}.yml"
   ```

   This references the “second-last” element in your FQN array, often the subfolder name.

4. **Multi-level grouping**

   ```yaml
   +dbt-osmosis: "{node.resource_type}/{node.config[materialized]}/{node.name}.yml"
   ```

   Group by whether it’s a model, source, or seed, *and* by its materialization.

In short, `{node}` is extremely flexible if you want to tailor your YAML file structure to reflect deeper aspects of the model’s metadata.

## `{parent}`

This variable represents the **immediate parent directory** of the **YAML file** that’s being generated, which typically aligns with the folder containing the `.sql` model file. For example, if you have:

```
models/
  staging/
    salesforce/
      opportunities.sql
```

The `{parent}` for `opportunities.sql` is `salesforce`. Thus if you do `+dbt-osmosis: "{parent}.yml"`, you’ll end up with a single `salesforce.yml` in the `staging/salesforce/` folder (lumping all models in that folder together).

**Usage Example**

```yaml title="dbt_project.yml"
models:
  jaffle_shop:
    staging:
      # So models in staging/salesforce => salesforce.yml
      # models in staging/marketo => marketo.yml
      # etc.
      +dbt-osmosis: "{parent}.yml"
```

### Why Use `{parent}`?

- **Consolidated** YAML: All models in a given folder share a single YAML. For example, `staging/salesforce/salesforce.yml` for 2–3 “salesforce” models.
- Great for **folder-based** org structures—like `staging/facebook_ads`, `staging/google_ads`—and you want a single file for each source’s staging models.

---

## Putting It All Together

You can mix and match these variables for **fine-grained** control. Here’s a complex example that merges all:

```yaml
models:
  my_project:
    super_warehouse:
      +dbt-osmosis: "{parent}/{node.config[materialized]}/{node.tags[0]}_{model}.yml"
```

1. **`{parent}`** => Name of the immediate subfolder under `super_warehouse`.
2. **`{node.config[materialized]}`** => Another subfolder named after the model’s materialization.
3. **`{node.tags[0]}`** => A prefix in the filename, e.g. `marketing_` or `analytics_`.
4. **`{model}`** => The actual model name for clarity.

So if you have a model `super_warehouse/snapshots/payment_stats.sql` with `materialized='table'` and a first tag of `'billing'`, it might produce:

```
super_warehouse/models/table/billing_payment_stats.yml
```

This approach ensures your YAML files reflect **both** how your code is organized (folder structure) **and** the model’s metadata (materialization, tags, etc.), with minimal manual overhead.

---

**In summary**, **context variables** are the backbone of dbt-osmosis’s dynamic file routing system. With `{model}`, `{node}`, and `{parent}`, you can define a wide range of file layout patterns and rely on dbt-osmosis to keep everything consistent. Whether you choose a single YAML per model, a single YAML per folder, or a more exotic arrangement that depends on tags, materializations, or your node’s FQN, dbt-osmosis will automatically **organize** and **update** your YAMLs to match your declared config.
