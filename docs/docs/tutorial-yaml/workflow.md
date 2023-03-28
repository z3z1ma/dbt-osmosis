---
sidebar_position: 5
---
# Workflow

## YAML Files

### Sources

dbt-osmosis will manage synchronizing your sources regardless of if you specify them in the vars.dbt-osmosis key your `dbt_project.yml` or not. That key, as seen in the example below, only serves to **declaratively** tell dbt-osmosis where the source file _should_ live. 

The advantage of this approach is that you can use dbt-osmosis to manage your sources without having to scaffold the YAML file yourself. You simply add a key value to the dictionary in the vars.dbt-osmosis key and dbt-osmosis will create the YAML file for you on next execution. It also hardens it against changes that violate the declarative nature of dbt-osmosis since it will simply migrate the file back to its original state on next execution unless you explicitly change it.

As a reminder, the vars.dbt-osmosis key is a dictionary where the key is the name of the source and the value is the path to the YAML file. The path is relative to the root of your dbt project models directory.

```yaml title="dbt_project.yml"
vars:
  dbt-osmosis:
    <source_name>: <path>
    <source_name>:
      path: <path>
      schema: <schema>
```

There is no need to specify the schema if it matches the source name. dbt-osmosis will assume that the schema is the same as the source name if the schema key is not specified. You can have multiple sources pointing at the same file if you like if you want them to all live there, but you cannot have multiple sources with the same name. The dictionary implicitly enforces this.

### Models

dbt-osmosis will manage synchronizing your models, but it is required it **knows** what to do with them. This is done by specifying the `+dbt-osmosis` configuration at various levels of the hierarchy in the configuration file. These levels match your folder structure exactly. It helps to be familiar with how dbt configuration works. It makes grokking this section extremely easy.

```yaml title="dbt_project.yml"
models:
  <your_project_name>:
    +dbt-osmosis: <path>
    <staging_dir>:
      +dbt-osmosis: <path>
    <intermediate_dir>:
      +dbt-osmosis: <path>

    <some_other_dir>:
      +dbt-osmosis: <path>

      <some_nested_dir>:
        +dbt-osmosis: <path>
```

## Running dbt-osmosis

I will step through 3 ways to run dbt-osmosis. These are not mutually exclusive. You can use any combination of these approaches to get the most out of dbt-osmosis. They are ordered based on the amount of effort required to get started and by the overall scalability as model count increases. 

### On-demand ⭐️

The easiest way to take advantage of dbt-osmosis is to run it periodically. This simply takes aligment from the development team on the configuration / rules and then a single developer can run it and commit the changes if they like them. This can be done weekly, monthly, quarterly, etc. It is up to the team to decide how often they want to run it. This is by far the simplest as a single execution provides significant value. (this is what I do today)

### Pre-commit hook ⭐️⭐️

We now include a pre-commit hook for dbt-osmosis! This is a great way to keep documentation up to date. It will only run on models that have changed.

```yaml title=".pre-commit-config.yaml"
repos:
  - repo: https://github.com/z3z1ma/dbt-osmosis
    rev: v0.11.10 # verify the latest version
    hooks:
      - id: dbt-osmosis
        files: ^models/
        # you'd normally run this against your prod target, you can use any target though
        args: [--target=prod]
```

### CI/CD ⭐️⭐️⭐️

You can also run dbt-osmosis as part of your CI/CD pipeline. The best way to do this is to simply clone the repo, run dbt-osmosis, and then commit the changes. Preferably, you would do this in a separate branch and then open a PR. This is the most robust approach since it ensures that the changes are reviewed and approved by a human before they are merged into the main branch whilst taking the load off of developer machines. 

```bash title="example.sh"
# this only exists to provide color, in the future we may add a preconfigured GHA to do this
git clone https://github.com/my-org/my-dbt-project.git
git checkout -b dbt-osmosis-refactor

dbt-osmosis yaml refactor --target=prod

git commit -am “✨ Refactor yaml files”
git push origin -f
gh pr create
```
