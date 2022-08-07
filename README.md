# dbt-osmosis

<!--![GitHub Actions](https://github.com/z3z1ma/dbt-osmosis/actions/workflows/master.yml/badge.svg)-->

![PyPI](https://img.shields.io/pypi/v/dbt-osmosis)
![Downloads](https://pepy.tech/badge/dbt-osmosis)
![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)
![black](https://img.shields.io/badge/code%20style-black-000000.svg)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://z3z1ma-dbt-osmosis-srcdbt-osmosisapp-4y67qs.streamlitapp.com/)


## Primary Objectives

Hello and welcome to the project! [dbt-osmosis](https://github.com/z3z1ma/dbt-osmosis) 🌊 serves to enhance the developer experience significantly. We do this through providing 3 core features:

1. Automated schema YAML management (minimize developers repetitive tasks)

2. Workbench for dbt Jinja SQL (maximize developers dbt SQL authoring efficiency + learning + testing)

3. Diffs for data model outputs to model outputs across git revisions (optimize developers observability during iteration)


When combined with an IDE such as VS Code, developers can work with renewed efficiency, enjoyment, and effectiveness throughout their days. 


[Workbench Reference](#Workbench)

[CLI Reference](#CLI)

____


## Workbench

Demo the workbench 👇 

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://z3z1ma-dbt-osmosis-srcdbt-osmosisapp-4y67qs.streamlitapp.com/)

 
```sh
# Command to start server
dbt-osmosis workbench
```

Press "r" to reload the workbench at any time.



✔️ dbt Editor with instant dbt compilation side-by-side or pivoted

✔️ Full control over model and workbench theme, light and dark mode

✔️ Query Tester, test the model you are working on for instant feedback

✔️ Data Profiler (leverages pandas-profiling)


**Editor** 

The editor is able to compile models with control+enter or dynamically as you type. Its speedy! You can choose any target defined in your profiles yml for compilation and execution.

![editor](/screenshots/osmosis_editor.png?raw=true "dbt-osmosis Workbench")

You can pivot the editor for a fuller view while workbenching some dbt SQL.

![pivot](/screenshots/osmosis_editor_pivot.png?raw=true "dbt-osmosis Pivot Layout")


**Test Query**

Test dbt models as you work against whatever profile you have selected and inspect the results. This allows very fast iterative feedback loops not possible with VS Code alone.

![test-model](/screenshots/osmosis_tester.png?raw=true "dbt-osmosis Test Model")

**Profile Model Results**

Profile your datasets on the fly while you develop without switching context. Allows for more refined interactive data modelling when dataset fits in memory.

![profile-data](/screenshots/osmosis_profile_data.png?raw=true "dbt-osmosis Profile Data")


**Useful Links and RSS Feed**

Some useful links and RSS feeds at the bottom. 🤓

![profile-data](/screenshots/osmosis_links.png?raw=true "dbt-osmosis Profile Data")

____


## CLI

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

            # Underscore prefixed model name as recommended in dbt best practices
            +dbt-osmosis: "_model.yml"

            +tags: 
                - "mart"

            supply_chain: 
```

To use dbt-osmosis, simply run the following:

```bash
# Install
pip install dbt-osmosis
# Alternatively
pipx install dbt-osmosis dbt-<adapter>


# This command executes all tasks in preferred order and is usually all you need

dbt-osmosis run --project-dir /path/to/dbt/project --target prod


# Inherit documentation in staging/salesforce/ & sync 
# schema yaml columns with database columns

dbt-osmosis document --project-dir /path/to/dbt/project --target prod --fqn staging.salesforce


# Reorganize marts/operations/ & inject undocumented models 
# into schema files or create new schema files as needed

dbt-osmosis compose --project-dir /path/to/dbt/project --target prod --fqn marts.operations


# Open the dbt-osmosis workbench

dbt-osmosis workbench


# Diff a model from git HEAD to revision on disk

dbt-osmosis diff -m int_account_events --pk 'concat(account_id, date_day)' --output bar
```

## Features

### Standardize organization of schema files (and provide ability to define and conform with code)

- Config can be set on per directory basis if desired utilizing `dbt_project.yml`, all models which are processed require direct or inherited config `+dbt-osmosis:`. If even one dir is missing the config, we close gracefully and inform user to update dbt_project.yml. No assumed defaults. Placing our config under your dbt project name in `models:` is enough to set a default for the project since the config applies to all subdirectories. 

    Note: You can **change these configs as often as you like** or try them all, dbt-osmosis will take care of restructuring your project schema files-- _no human effort required_. 

    A directory can be configured to conform to any one of the following standards:

    - Can be one schema file to one model file sharing the same name and directory ie. 

            staging/
                stg_customer.sql
                stg_customer.yml
                stg_order.sql
                stg_order.yml

        - `+dbt-osmosis: "model.yml"`

    - Can be one schema file per directory wherever model files reside named schema.yml, ie.

            staging/
                schema.yml
                stg_customer.sql
                stg_order.sql

        - `+dbt-osmosis: "schema.yml"`
    - Can be one schema file per directory wherever model files reside named after its containing folder, ie. 

            staging/
                stg_customer.sql
                stg_order.sql
                staging.yml

        - `+dbt-osmosis: "folder.yml"`

    - Can be one schema file to one model file sharing the same name _nested_ in a schema subdir wherever model files reside, ie. 

            staging/
                stg_order.sql
                stg_customer.sql
                schema/
                    stg_customer.yml
                    stg_order.yml

        - `+dbt-osmosis: "schema/model.yml"`

    - Can be one schema file to one model file sharing the same name and directory, models prefixed with underscore for IDE sorting ie. 

            staging/
                _stg_customer.yml
                _stg_order.yml
                stg_customer.sql
                stg_order.sql

        - `+dbt-osmosis: "_model.yml"`

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
    Config is called `+dbt-osmosis: "folder.yml" | "schema.yml" | "model.yml" | "schema/model.yml" | "_model.yml"`
2. Bootstrap models to ensure all models exist
3. Recompile Manifest
4. Propagate definitions downstream to undocumented models solely within the context of each models dependency tree


## Python API

Though each core function is useful enough to stand as its own package, dbt osmosis sits as a unified interface primarily because all of these functions are built off of the same core API structures in the dbt osmosis package. dbt osmosis provides one of the cleanest interfaces to interacting with dbt if you aren't keen to play with dbt on-the-rails (like me) or you want to extend what osmosis can do.

```python
# Programmatic Examples:
from dbt_osmosis.core import DbtOsmosis
from dbt_osmosis.diff import diff_and_print_to_console

runner = DbtOsmosis(
    project_dir="/Users/alexanderbutler/Documents/harness/analytics-pipelines/projects/meltano/harness/transform",
    dry_run=True,
    target="prod",
)


# Some dbt osmosis YAML management 📜

# review the generated plan
runner.pretty_print_restructure_plan(runner.draft_project_structure_update_plan())

# organize your dbt project based on declarative config
runner.commit_project_restructure_to_disk()

# propagate column level documentation down the DAG
runner.propagate_documentation_downstream()


# Console utilities 📺

# leverage git to diff the OUTPUT of a model from git HEAD 
# to your revision on disk to safely audit changes as you work
diff_and_print_to_console("fct_sales", pk="order_id", runner=runner)  


# Massively simplified dbt interfaces you likely won't find elsewhere 👏

# execute macros through a simple interface without subprocesses
runner.execute_macro(
    "create_schema",
    kwargs={"relation": relation},
)

# compile SQL as easy as this 🤟
runner.compile_sql("select * from {{ ref('stg_salesforce__users') }}")

# run SQL too
adapter_resp, table = runner.execute_sql(
    "select * from {{ ref('stg_salesforce__users') }}", 
    compile=True, 
    fetch=True,
)
```

## Roadmap

These features are being actively developed and will be merged into the next few minor releases

1. Extend git diff functionality to pin revisions in the warehouse  
2. Complete build out of `sources` tools.
3. Add `--min-cov` flag to audit task and to workbench
4. Add interactive documentation flag that engages user to documents ONLY progenitors and novel columns for a subset of models (the most optimized path to full documentation coverage feasible)
5. Add `impact` command that allows us to leverage our resolved column level progenitors for ad hoc impact analysis
