---
sidebar_position: 3
---

# YAML routing and file placement

At the core of dbt-osmosis is a path template that tells it where each node’s YAML should live. That template is the value of `+dbt-osmosis`.

## How routing works

When dbt-osmosis processes a model or seed, it evaluates the `+dbt-osmosis` template with the current node context, then writes or moves the YAML file to the resulting path.

Templates can include:

- `{model}` for the model name
- `{parent}` for the parent folder
- `{node.*}` for full node metadata

## Example templates

```yaml
+dbt-osmosis: "_{model}.yml"
```

```yaml
+dbt-osmosis: "{parent}.yml"
```

```yaml
+dbt-osmosis: "{node.config[materialized]}/{model}.yml"
```

## Why this matters

A single routing rule can keep YAML files organized across a large project. When you change a rule, dbt-osmosis computes the new target paths and offers to move or merge existing YAML files into the correct location.

## Related topics

- [Context variables](../tutorial-yaml/context)
- [Configuration](../tutorial-yaml/configuration)
