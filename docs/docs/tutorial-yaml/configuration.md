---
sidebar_position: 1
---
# Configuration

## Configuring dbt-osmosis

### Models

dbt-osmosis' primary purpose is to automatically generate and manage YAML files for your dbt models. We opt for explicitness over implicitness. Thus the following configuration is required to even run dbt-osmosis. By specifying this configuration at the top-level beneath your project key, you are specifying a default configuration for all models in your project. You can override this configuration for individual models by specifying the `+dbt-osmosis` configuration at various levels of the hierarchy in the configuration file. These levels match your folder structure exactly.

```yaml title="dbt_project.yml"
models:
  <your_project_name>:
    +dbt-osmosis: <path>
```

- `<your_project_name>` is the name of your dbt project.
- `<path>` is the path to the YAML file that will be generated for the model. This path is **relative to the model's (sql file) directory.**

#### Examples

```yaml title="dbt_project.yml"
models:
  your_project_name:
    # a default blanket rule
    +dbt-osmosis: "_{model}.yml"

    staging:
      # nest docs in subfolder relative to model
      +dbt-osmosis: "schema/{model}.yml"

    intermediate:
      # separate docs based on materialization
      +dbt-osmosis: "{node.config[materialized]}/{model}.yml"

    marts:
      # static paths are perfectly fine!
      +dbt-osmosis: "prod.yml"
```

### Sources

dbt-osmosis can be configured to automatically generate YAML files for your dbt sources. To enable this feature, add the following to your `dbt_project.yml` file.

```yaml title="dbt_project.yml"
vars:
  dbt-osmosis:
    <source_name>: <path>
    <source_name>:
      path: <path>
      database: <database>
      schema: <schema>
    _blacklist: <blacklist>
```

- `<source_name>` is the name of a source in your `dbt_project.yml` file.
- `<path>` is the path to the YAML file that will be generated for the source. This path is relative to the root of your dbt project models directory.
- `<database>` is the database that will be used for the source. If not specified, the database will default to the one in your profiles.yml file.
- `<schema>` is the schema that will be used for the source. If not specified, the source name is assumed to be the schema which matches dbt's default behavior.
- `<blacklist>` is the columns to be ignored. You can use regular expressions to specify which columns you'd like to exclude.

#### Examples

```yaml title="dbt_project.yml"
vars:
  dbt-osmosis:
    # a source with a different schema
    salesforce:
      path: "staging/salesforce/source.yml"
      schema: "salesforce_v2"

    # a source with the same schema as the source name
    marketo: "staging/customer/marketo.yml"

    # a special variable interpolated at runtime
    jira: "staging/project_mgmt/{parent}.yml"

    # a dedicated directory for all sources
    github: "all_sources/github.yml"

  _blacklist:
    - "_FIVETRAN_SYNCED"
    - ".*__key__.namespace"
```

Notice the use of the `{parent}` variable in the `jira` source configuration. This variable is a special variable that will be replaced with the name of the parent directory of the YAML file. The other special variables are `{node}` and `{model}`. We will discuss these variables in more detail in the next section.
