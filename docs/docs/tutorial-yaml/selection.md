---
sidebar_position: 4
---

# Selection

Use selection to limit dbt-osmosis to a subset of models or sources.

## Positional selectors (recommended)

Positional arguments can be:

- Model names (`stg_customers`)
- Paths (`models/staging`)
- Globs (`marts/**/*.sql`)

Examples:

```bash
# All models in a folder
dbt-osmosis yaml refactor models/staging

# A single model by name
dbt-osmosis yaml document stg_customers

# A glob of SQL files
dbt-osmosis yaml organize marts/**/*.sql
```

## FQN selectors (`--fqn`)

`--fqn` matches fully-qualified names (without the project prefix):

```bash
dbt-osmosis yaml refactor --fqn=staging.salesforce.contacts
```

Use `--fqn` when you already have an FQN from `dbt ls`, or when you need a precise subtree selection.

## Tips

- Prefer positional selectors for most workflows.
- Combine `--dry-run` with selection to validate your scope before applying changes.
