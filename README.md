# dbt-osmosis
First and foremost, we want dbt documentation to retain a DRY principle. Every time we repeat ourselves, we waste our time. Second, we want to understand column level lineage and automate impact analysis.



## Primary Objectives

Standardize organization of schema files (and provide ability to define and conform with code)

- Config can be set on per directory basis if desired utilizing `dbt_project.yml`, all models require direct or inherited config `+dbt-osmosis:`. If even one dir is missing the config, we close gracefully and inform user to update dbt_project.yml. No assumed defaults. Placing our config under your dbt project name in `models:` is enough to set a default for the project since the config applies to all subdirectories. 

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

Build and Inject Non-documented models

- Injected models will automatically conform to above config per directory based on location of model file. 

- This means you can focus fully on modelling and documentation (yaml updates/yaml creation depending on your config) will automatically follow with simple CLI invocation

Propagate existing column level documentation downward to children

- Build column level knowledge graph accumulated and updated from furthest identifiable origin (ancestors) to immediate parents

- Will automatically populate undocumented columns of the same name with passed down knowledge accumulated within the context of the models upstream dependency tree

- This means you can freely generate models and all columns you pull in that already have been documented will be automatically learned/documented. Again the focus is fully on modelling and any yaml work is an afterthought.

### Order Matters

In a full run we will:

1. Conform dbt project
    - Configuration lives in `dbt_project.yml` --> we require our config to run, can be at root level of `models:` to apply a default convention to a project 
    or can be folder by folder, follows dbt config resolution where config is overridden by scope. 
    Config is called `+dbt-osmosis: "folder.yml" | "schema.yml" | "model.yml" | "schema/model.yml"`
2. Bootstrap models to ensure all models exist
3. Recompile Manifest
4. Propagate definitions downstream to undocumented models solely within the context of the models dependency tree


## New workflows enabled!

1. Build one dbt model or a __bunch__ of them without documenting anything (gasp)

    Run `dbt-osmosis`
    
    Automatically construct/update your schema yamls built with as much of the definitions pre-populated as possible from upstream dependencies 
    
    Schema yaml is automatically built in exactly the right directories / style that conform to the configured standard upheld and enforced across your dbt project on a dir by dir basis automatically
        
    Configured using just the dbt_project.yml and `+dbt-osmosis:` configs
    
    boom, mic drop

2. Problem reported by stakeholder with data (WIP)
    
    Identify column
    
    Run `dbt-osmosis impact --model orders --column price`
    
    Find the originating model and action

3. Need to score our documentation (WIP)

    Run `dbt-osmosis coverage --docs --min-cov 80`

    Get a curated list of all the documentation to update in your pre-bootstrapped dbt project

    Sip coffee and engage in documentation
