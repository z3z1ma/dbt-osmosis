---
sidebar_position: 4
---
# Selection

When you run **dbt-osmosis** commands like `yaml refactor`, `yaml organize`, or `yaml document`, you typically want to narrow the scope to **some** subset of models or sources in your project. This helps you:

- Focus on a subset of changes rather than the entire project.
- Speed up refactoring or documentation runs.
- Reduce noise or risk when iterating incrementally.

dbt-osmosis provides **two** major strategies for doing this:

1. **Positional Selectors** (recommended)
2. **`--fqn` Flag** (for advanced users or special cases)

## 1. Positional Selectors

**Positional selectors** are the **non-flag arguments** you provide after the command. They’re interpreted as **paths or model names**. The underlying logic tries to match each positional argument to:

- A model name (like `stg_customers`)
- A file path (like `models/staging/stg_customers.sql`)
- A directory path (like `models/staging`)
- A file glob (like `marts/**/*.sql`)

In other words, anything that isn’t prefixed with `--` is treated as a *positional path or node name* that dbt-osmosis will attempt to match against your project.

### Example Commands

```bash
# Select all models in the models/staging directory
dbt-osmosis yaml refactor models/staging
```

In this case, dbt-osmosis processes **all** `.sql` files recognized as dbt models in `models/staging`. Similarly:

```bash
# Select only one model if the name is stg_customers and it exists
dbt-osmosis yaml refactor stg_customers
```

dbt-osmosis looks for a node with the **exact** name `stg_customers`. If found, it processes **just** that single model. If the name doesn’t match a known node, dbt-osmosis checks if there’s a path or file called `stg_customers`. If that fails, no models are selected.

#### Using Globs

If your shell supports wildcards or recursive globs:

```bash
# Recursively select all .sql models in marts/ subdirectories
dbt-osmosis yaml refactor marts/**/*.sql
```

This is equivalent to selecting every `.sql` file under `marts/` at **any** nested level.

#### Absolute Paths

You can also supply **absolute** or relative paths. For example:

```bash
dbt-osmosis yaml refactor /full/path/to/my_project/models/staging/*.sql
```

If dbt-osmosis recognizes those `.sql` files as part of your current dbt project, it will include them.

### How dbt-osmosis Interprets Positional Selectors

1. **Exact Node Name Check**: If the positional argument **directly matches** a known model name (like `stg_customers`), dbt-osmosis picks that node.
2. **File or Directory Check**: If the argument is a valid path (relative or absolute), dbt-osmosis includes all recognized `.sql` models beneath it (or the file itself if it’s a single `.sql`).
3. **Glob Expansion**: If your shell expands the glob, dbt-osmosis picks each resulting path that maps to a dbt model.

This approach is **intuitive**, typically what you’d want for partial refactors, and is less error-prone than advanced flags.

---

## 2. The `--fqn` Flag

:::caution Caution
This may be **deprecated** in the future. We recommend positional selectors first.
:::

The **`--fqn`** flag provides an alternative approach. An **FQN** (fully qualified name) in dbt typically includes:

- The project name
- The resource type (`model`, `source`, or `seed`)
- Subfolders or packages leading to the node
- The final node name

In dbt-osmosis, we **omit** the project name and resource type segments, focusing only on the latter parts of the FQN. For instance, if `dbt ls` returns:

```
my_project.model.staging.salesforce.contacts
```

You could specify:

```bash
dbt-osmosis yaml refactor --fqn=staging.salesforce.contacts
```

And dbt-osmosis would match precisely that node. Or:

```bash
dbt-osmosis yaml refactor --fqn=staging.salesforce
```

This would select **all** nodes under `staging.salesforce.*`, effectively everything in the `staging/salesforce` sub-tree.

### Why Use `--fqn`?

- **Precise FQN-based** selection.
- If you’re comfortable with dbt’s concept of FQNs, it can be simpler to copy/paste from `dbt ls`.
- **Partial segments**: You might not remember the exact file path, but you know the dbt FQN you want.

### Example

```bash
dbt-osmosis yaml refactor --fqn=marts.sales.customers
```

If your project is named `my_project`, dbt-osmosis internally interprets that as `my_project.model.marts.sales.customers`—**only** that single model. Or:

```bash
dbt-osmosis yaml refactor --fqn=staging.salesforce
```

It selects **any** model where the FQN starts with `staging.salesforce`, capturing all models in your `staging/salesforce/` subfolder.

### Edge Cases / Limitations

- We assume you’re only dealing with models or sources from the **current** project (not upstream packages).
- If multiple subfolders share the same partial FQN, you might get more matches than expected—though that’s relatively uncommon if your dbt naming is well-structured.

---

## Which Should I Use?

**For 90% of use cases**, **positional selectors** are easiest and more future-proof, since `--fqn` may be deprecated. Simply specifying the path or the node name is generally enough. However, if you have advanced use cases or find it more convenient to copy/paste FQNs from `dbt ls`, then `--fqn` remains a viable option.

---

## Putting It All Together

Here’s a quick run-down of typical usage patterns:

- **One folder at a time**:

  ```bash
  dbt-osmosis yaml refactor models/staging
  ```

- **One specific model**:

  ```bash
  dbt-osmosis yaml document stg_customers
  ```

- **All .sql files in `marts`**:

  ```bash
  dbt-osmosis yaml organize marts/*.sql
  ```

- **A partial FQN for multiple subfolders**:

  ```bash
  dbt-osmosis yaml refactor --fqn=staging
  ```

  (selects *all* staging models)

Regardless of your approach, dbt-osmosis will do its usual work of **refactoring** or **documenting** or **organizing** whichever subset of models (and sources, if relevant) match your selection criteria.

---

**In summary**, the **Selection** mechanism in dbt-osmosis is flexible enough to handle both straightforward file-based filters and advanced FQN-based filters. Use **positional selectors** for most tasks, and consider `--fqn` if you have a specific workflow that benefits from it. This ensures you only run dbt-osmosis on **exactly** the nodes you care about.
