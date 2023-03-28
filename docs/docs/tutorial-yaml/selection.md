---
sidebar_position: 4
---
# Selection

## Selecting models

The `dbt-osmosis yaml` commands have two methods to selecting files to execute on. 

### Positional selectors

The first method is positional selectors. Positional selectors consist of all of the arguments that are not flags. For example, `dbt-osmosis yaml refactor models/staging` would select all models in the `models/staging` directory. `dbt-osmosis yaml refactor models/staging/base` would select all models in the `models/staging/base` directory. You can use globs if they are supported by your shell. For example, `dbt-osmosis yaml refactor marts/*.sql` would select all models in the `models/marts` directory but not its subdirectories. `dbt-osmosis yaml refactor marts/**/*.sql` would select all models in the `models/marts` directory and its subdirectories via a recursive glob. Absolute paths are also supported.

Positional selectors will also short circuit if the positional selector matches a node name exactly. For example, `dbt-osmosis yaml refactor stg_customers` would select `models/staging/stg_customers.sql` if it exists. This is useful as a convenience to run a specific model since all nodes have unique names. File paths are more explicit but you can use both if you want.

This should all be fairly intuitive. This is the preferred method of selecting models.

### `--fqn` flag

:::caution Caution

This may be deprecated in the future. Please use model positional selectors instead.

:::

The `--fqn` flag allows you to specify a list of fully qualified names (FQNs) to select models. This is a fairly explicit way to select models. A dbt fqn is comprised of segments. Each segment corresponds to a unit of information.

- The first segment is the project name.
- The second segment is the resource type.
  - `model` for models
  - `source` for sources
- The following segments are the path to the resource from the /models directory.

In our implementation of the `fqn` selector, we omit the project name and resource type. This is because we assert that the user is only working with models in the current project (ie not packages) and that the user is only working with models and sources as that is all that is supported by the `dbt-osmosis yaml` commands.

:::tip Tip

When you run `dbt ls` you will see the fully qualified name of each model. If you omit the project name and resource type, you can copy and paste the FQN from the `dbt ls` output.

:::

So for example, using `--fqn=staging.customers` in the `my_project` project context would select `my_project.model.staging.customers`. The utility of this feature is that you can use only a few segments of the FQN to select a large number of models. For example, `--fqn=staging` would select all models in the `staging` directory. `--fqn=staging.base` would select all models in the `staging/base` directory.
