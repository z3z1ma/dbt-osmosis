---
sidebar_position: 2
---

# Your first refactor

This tutorial walks through a safe first run of dbt-osmosis on a dbt project.

## 1. Ensure your project compiles

From the dbt project root:

```bash
dbt parse
```

## 2. Add routing rules

Add a `+dbt-osmosis` rule for models and seeds:

```yaml title="dbt_project.yml"
models:
  your_project_name:
    +dbt-osmosis: "_{model}.yml"
seeds:
  your_project_name:
    +dbt-osmosis: "_schema.yml"
```

## 3. Dry run

```bash
dbt-osmosis yaml refactor --dry-run --check
```

Review the console output and confirm the planned file moves and doc changes.

## 4. Apply changes

```bash
dbt-osmosis yaml refactor --auto-apply
```

This will:

- Create missing YAML files
- Move or merge existing YAML to the desired paths
- Inherit upstream column descriptions and metadata

## 5. Review the result

```bash
git status
```

Inspect the generated YAML, adjust descriptions as needed, and commit the changes.
