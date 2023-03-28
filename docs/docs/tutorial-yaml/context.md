---
sidebar_position: 2
---
# Context Variables

This section describes the context variables available in dbt-osmosis. These variables are used to derive **dynamic** naming rules for your yaml files.

## `{model}`

This variable is the name of the model that is currently being processed. This variable is useful when you want to generate a yaml file for each model in a directory. For a model named `stg_marketo__leads.sql`, the model variable would be `stg_marketo__leads` and can be used however you see fit.

**Usage**

```yaml title="dbt_project.yml"
models:
  jaffle_shop:
    # this can be thought of as a default configuration, this must always be set by the user
    +dbt-osmosis: "_{model}.yml"

    intermediate:
      # override the default configuration for the intermediate directory
      +dbt-osmosis: "some/deeply/nested/path/{model}.yml"
```

## `{node}`

This variable is the actual node that dbt-osmosis is processing. This node object has all of the contents of a node as you might see in a `manifest.json` file. There a many ways to use this but it is recommended you are familiar with the underlying data structure before using this variable.

**Usage**

```yaml title="dbt_project.yml"
models:
  jaffle_shop:
    # don't forget to have a default, we have just omitted it for brevity

    intermediate:
      # advanced usage
      +dbt-osmosis: "node.fqn[-2]/{node.resource_type}_{node.language}/{node.name}.yml"

    marts:
      # more advanced examples
      +dbt-osmosis: "node.config[materialized]/node.tags[0]/schema.yml"
```

## `{parent}`

This variable is the name of the parent directory of the YAML file that is currently being processed. This variable is useful if you want to generate a single yaml file for all the models in a directory dynamically based on the parent directory name. This should be equivalent to `node.fqn[-2]` but is more concise. The fqn prop lets you step further up the hierarchy but is not recommended except in advanced use cases.

**Usage**

```yaml title="dbt_project.yml"
models:
  jaffle_shop:
    # don't forget to have a default, we have just omitted it for brevity

    staging:
      # make it so models in staging/salesforce, staging/marketo, etc. all route docs into
      # files named salesforce.yml, marketo.yml, etc. in their respective directories
      +dbt-osmosis: "{parent}.yml"
```