# dbt-osmosis

<!--![GitHub Actions](https://github.com/z3z1ma/dbt-osmosis/actions/workflows/master.yml/badge.svg)-->

![PyPI](https://img.shields.io/pypi/v/dbt-osmosis)
![Downloads](https://pepy.tech/badge/dbt-osmosis)
![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)
![black](https://img.shields.io/badge/code%20style-black-000000.svg)

## Primary Objectives

Hello and welcome to the project! dbt-osmosis serves to enhance the developer experience significantly. We do this by automating the most of the management of schema yml files, we synchronize inheritable column level documentation which permits a write-it-once principle in a DAG oriented way, and we expose a workbench which allows you to interactively develop in dbt. The workbench allows you to develop and instantly compile models side by side (incredibly quickly), document model columns, test the query against your data warehouse, inspect row level diffs and diff metric as you modify SQL, run tests, and more. 


Here are some of the foundational pillars:

First and foremost, we want dbt documentation to retain a DRY principle. Every time we repeat ourselves, we waste our time. 80% of documentation is often a matter of inheritance and continued passing down of columns from parent models to children. They need not be redocumented if there has been no mutation. 

Second, we want to standardize ways that we all organize our schema files which hold the fruits of our documentation. We should be able to enforce a standard on a per directory basis and jump between layouts at will as certain folders scale up the number of models or scale down. 

Lastly, and tangential to the first objective, we want to understand column level lineage, streamline impact analysis, and audit our documentation.


## How to Use

dbt-osmosis is ready to use as-is. To get familiar, you should run it on a fresh branch and ensure everything is backed in source control. Enjoy!

You should set a base config in your dbt_project.yml and ensure any models within the scope of your execution plan will inherit a config/preference. Example below.

```yaml
models:

    your_dbt_project:

        # This config will apply to your whole project
        +dbt-osmosis: "schema/model.yml"

        staging:

            # This config will apply to your staging directory
            +dbt-osmosis: "folder.yml"

            +tags: 
                - "staged"

            +materialized: view

            monday:
                intermediate:
                    +materialized: ephemeral

        marts:

            +tags: 
                - "mart"

            supply_chain: 
```

To use dbt-osmosis, simply run the following:

```bash
# Install
pip install dbt-osmosis
# Alternatively
pipx install dbt-osmosis


# This command executes all tasks in preferred order and is usually all you need

dbt-osmosis run --project-dir /path/to/dbt/project --target prod


# Inherit documentation in staging/salesforce/ & sync 
# schema yaml columns with database columns

dbt-osmosis document --project-dir /path/to/dbt/project --target prod --fqn staging.salesforce


# Reorganize marts/operations/ & inject undocumented models 
# into schema files or create new schema files as needed

dbt-osmosis compose --project-dir /path/to/dbt/project --target prod --fqn marts.operations
```

## Roadmap

These features are being actively developed and will be merged into the next few minor releases

1. Complete build out of `sources` tools.
2. Add `--min-cov` flag to audit task
3. Add interactive documentation flag that engages user to documents ONLY progenitors and novel columns for a subset of models (the most optimized path to full documentation coverage feasible)
4. Add `impact` command that allows us to leverage our resolved column level progenitors for ad hoc impact analysis

## Features

### Standardize organization of schema files (and provide ability to define and conform with code)

- Config can be set on per directory basis if desired utilizing `dbt_project.yml`, all models which are processed require direct or inherited config `+dbt-osmosis:`. If even one dir is missing the config, we close gracefully and inform user to update dbt_project.yml. No assumed defaults. Placing our config under your dbt project name in `models:` is enough to set a default for the project since the config applies to all subdirectories. 

    Note: You can **change these configs as often as you like** or try them all, dbt-osmosis will take care of restructuring your project schema files-- _no human effort required_. 

    A directory can be configured to conform to any one of the following standards:

    - Can be one schema file to one model file sharing the same name and directory ie. 

            staging/
                stg_order.sql
                stg_order.yml
                stg_customer.sql
                stg_customer.yml

        - `+dbt-osmosis: "model.yml"`

    - Can be one schema file per directory wherever model files reside named schema.yml, ie.

            staging/
                stg_order.sql
                stg_customer.sql
                schema.yml

        - `+dbt-osmosis: "schema.yml"`
    - Can be one schema file per directory wherever model files reside named after its containing folder, ie. 

            staging/
                stg_order.sql
                stg_customer.sql
                staging.yml

        - `+dbt-osmosis: "folder.yml"`

    - Can be one schema file to one model file sharing the same name _nested_ in a schema subdir wherever model files reside, ie. 

            staging/
                stg_order.sql
                stg_customer.sql
                schema/
                    stg_order.yml
                    stg_customer.yml

        - `+dbt-osmosis: "schema/model.yml"`

### Build and Inject Non-documented models

- Injected models will automatically conform to above config per directory based on location of model file. 

- This means you can focus fully on modelling; and documentation, including yaml updates or creation, will automatically follow at any time with simple invocation of dbt-osmosis

### Propagate existing column level documentation downward to children

- Build column level knowledge graph accumulated and updated from furthest identifiable origin (ancestors) to immediate parents

- Will automatically populate undocumented columns of the same name with passed down knowledge accumulated within the context of the models upstream dependency tree

- This means you can freely generate models and all columns you pull into the models SQL that already have been documented will be automatically learned/propagated. Again the focus for analysts is almost fully on modelling and yaml work is an afterthought / less heavy of a manual lift.

### Order Matters

In a full run [ `dbt-osmosis run` ] we will:

1. Conform dbt project
    - Configuration lives in `dbt_project.yml` --> we require our config to run, can be at root level of `models:` to apply a default convention to a project 
    or can be folder by folder, follows dbt config resolution where config is overridden by scope. 
    Config is called `+dbt-osmosis: "folder.yml" | "schema.yml" | "model.yml" | "schema/model.yml"`
2. Bootstrap models to ensure all models exist
3. Recompile Manifest
4. Propagate definitions downstream to undocumented models solely within the context of each models dependency tree


## New workflows enabled!

1. Build one dbt model or a __bunch__ of them without documenting anything (gasp)

    Run `dbt-osmosis run` or `dbt-osmosis compose && dbt-osmosis document`
    
    Sit back and watch as:

    Automatically constructed/updated schema yamls are built with as much of the definitions pre-populated as possible from upstream dependencies 
    
    Schema yaml(s) are automatically organized in exactly the right directories / style that conform to the easily configurable standard upheld and enforced across your dbt project on a directory by directory basis 
    
    boom, mic drop

2. Problem reported by stakeholder with data **(WIP)**
    
    Identify column
    
    Run `dbt-osmosis impact --model orders --column price`
    
    Find the originating model and action

3. Need to score our documentation **(WIP)**

    Run `dbt-osmosis audit --docs --min-cov 80`

    Get a curated list of all the documentation to update in your pre-bootstrapped dbt project

    Sip coffee and engage in documentation

4. Add dbt-osmosis to a pre-commit hook to ensure all your analysts are passing down column level documentation & reaching your designated min-coverage