# dbt-osmosis

<!--![GitHub Actions](https://github.com/z3z1ma/dbt-osmosis/actions/workflows/master.yml/badge.svg)-->

![PyPI](https://img.shields.io/pypi/v/dbt-osmosis)
![Downloads](https://pepy.tech/badge/dbt-osmosis)
![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)
![black](https://img.shields.io/badge/code%20style-black-000000.svg)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://z3z1ma-dbt-osmosis-srcdbt-osmosisapp-4y67qs.streamlitapp.com/)


## Primary Objectives

Hello and welcome to the project! [dbt-osmosis](https://github.com/z3z1ma/dbt-osmosis) 🌊 serves to enhance the developer experience significantly. We do this through providing 4 core features:

1. Automated schema YAML management.
    
    1a. `dbt-osmosis yaml refactor --project-dir ... --profiles-dir ...`

    > Automatically generate documentation based on upstream documented columns, organize yaml files based on configurable rules defined in dbt_project.yml, scaffold new yaml files based on the same rules, inject columns from data warehouse schema if missing in yaml and remove columns no longer present in data warehouse (organize -> document)

    1b. `dbt-osmosis yaml organize --project-dir ... --profiles-dir ...`

    > Organize yaml files based on configurable rules defined in dbt_project.yml, scaffold new yaml files based on the same rules

    1c. `dbt-osmosis yaml document --project-dir ... --profiles-dir ...`

    > Automatically generate documentation based on upstream documented columns

2. A highly performant dbt server which integrates with tools such as dbt-power-user for VS Code to enable interactive querying + realtime compilation from your IDE

    2a. `dbt-osmosis server serve --project-dir ... --profiles-dir ...`

    > Spins up a FastAPI server. Can be passed --register-project to automatically register your local project. API documentation is available at /docs endpoint where interestingly enough, you can query your data warehouse or compile SQL via the Try It function

3. Workbench for dbt Jinja SQL. This workbench is powered by streamlit and the badge at the top of the readme will take you to a demo on streamlit cloud with jaffle_shop loaded (requires extra `pip install dbt-osmosis[workbench]`). 

    3a. `dbt-osmosis workbench --project-dir ... --profiles-dir ...`

    > Spins up a streamlit app. This workbench offers similar functionality to the osmosis server + power-user combo without a reliance on VS code. Realtime compilation, query execution, pandas profiling all via copying and pasting whatever you are working on into the workbenchat your leisure. Spin it up and down as needed.

4. Diffs for data model outputs to model outputs across git revisions 

    4a. `dbt-osmosis diff -m some_model  --project-dir ... --profiles-dir ...`

    > Run diffs on models dynamically. This pulls the state of the model before changes from your git history, injects it as a node to the dbt manifest, compiles the old and modified nodes, and diffs their query results optionally writing nodes to temp tables before running the diff query for warehouses with performance or query complexity limits (👀 bigquery)
    
## References

[Server Reference](#server)

[Workbench Reference](#workbench)

[YAML Reference](#yaml-management)

[Python API Reference](#python-api)

____

## Server

```sh
# Command to start server
dbt-osmosis server serve --host ... --port ...
```

The server is self documenting via open API. From the open API docs you can compile SQL or run it to get an idea of the requests and responses. Furthermore the server supports multiple dbt projects out of the box. This means the server can `/register` 10s to 100s of projects and selectively compile or run against a specific one via an `X-dbt-Project` header. It is stress tested at high loads and volumes, higher than its ever likely to be put through as primarily a dev accelerator but it could be used in a production application too and is the focus of much of the development in the repo. It is Apache 2.0 licensed which differentiates it from dbt-core server. Furthermore it is more focused on SQL than "models" as it is not a replacement for the CLI nor does it aspire to be. Instead it is more of a database adapter/interface of sorts which lets it be really good at one thing.

![server-docs](/screenshots/osmosis_server_docs.png)

Starting the server is easy. Its most interesting and impactful integration is through [dbt-power-user](https://github.com/innoverio/vscode-dbt-power-user) which in the near term will hide away the details of starting or managing the server and simply provide a high quality developer experience out-of-the-box.

![server-start](/screenshots/osmosis_server_startup.png)

____

## Workbench

Demo the workbench 👇 

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://z3z1ma-dbt-osmosis-srcdbt-osmosisapp-4y67qs.streamlitapp.com/)

 
```sh
pip install dbt-osmosis[workbench]

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

![editor](/screenshots/osmosis_editor_main.png?raw=true "dbt-osmosis Workbench")

You can pivot the editor for a fuller view while workbenching some dbt SQL.

![pivot](/screenshots/osmosis_editor_pivot.png?raw=true "dbt-osmosis Pivot Layout")


**Test Query**

Test dbt models as you work against whatever profile you have selected and inspect the results. This allows very fast iterative feedback loops not possible with VS Code alone.

![test-model](/screenshots/osmosis_tester.png?raw=true "dbt-osmosis Test Model")

**Profile Model Results**

Profile your datasets on the fly while you develop without switching context. Allows for more refined interactive data modelling when dataset fits in memory.

![profile-data](/screenshots/osmosis_profile.png?raw=true "dbt-osmosis Profile Data")


**Useful Links and RSS Feed**

Some useful links and RSS feeds at the bottom. 🤓

![profile-data](/screenshots/osmosis_links.png?raw=true "dbt-osmosis Profile Data")

____


## YAML Management

dbt-osmosis yaml management is extremely powerful and ready to use as-is. To get familiar, you should run it on a fresh branch and ensure everything is backed in source control. You'll wonder why its not in dbt-core. Enjoy!

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

            # Underscore prefixed model name as recommended in dbt best practices for everything in "marts" folder
            +dbt-osmosis: "_model.yml"

            +tags: 
                - "mart"

            supply_chain: 
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

In a full run [ `dbt-osmosis yaml refactor` ] we will:

1. Conform dbt project
    - Configuration lives in `dbt_project.yml` --> we require our config to run, can be at root level of `models:` to apply a default convention to a project 
    or can be folder by folder, follows dbt config resolution where config is overridden by scope. 
    Config is called `+dbt-osmosis: "folder.yml" | "schema.yml" | "model.yml" | "schema/model.yml" | "_model.yml"`
2. Bootstrap models to ensure all models exist
3. Recompile Manifest
4. Propagate definitions downstream to undocumented models solely within the context of each models dependency tree


## Python API

Though each core function is useful enough to stand as its own package, dbt osmosis sits as a unified interface primarily because all of these functions are built off of the same core API structures in the dbt osmosis package. dbt osmosis provides one of the cleanest interfaces to interacting with dbt if you aren't keen to play with dbt on-the-rails (like me) or you want to extend what osmosis can do, see below examples for how to interface with it from Python.

```python
# Programmatic Examples:
from dbt_osmosis.core import DbtProject, DbtYamlManager
from dbt_osmosis.diff import diff_and_print_to_console

# Some dbt osmosis YAML management 📜
dbt_yaml_manager = DbtYamlManager(
    project_dir="/Users/alexanderbutler/Documents/harness/analytics-pipelines/projects/meltano/harness/transform",
    target="prod",
)

# review the generated plan
dbt_yaml_manager.pretty_print_restructure_plan(dbt_yaml_manager.draft_project_structure_update_plan())

# organize your dbt project based on declarative config
dbt_yaml_manager.commit_project_restructure_to_disk()

# propagate column level documentation down the DAG
dbt_yaml_manager.propagate_documentation_downstream()



# Massively simplified dbt interfaces you likely won't find elsewhere 👏

runner = DbtProject(
    project_dir="/Users/alexanderbutler/Documents/harness/analytics-pipelines/projects/meltano/harness/transform",
    target="prod",
)

# execute macros through a simple interface without subprocesses
runner.execute_macro(
    "create_schema",
    kwargs={"relation": relation},
)

# compile SQL as easy as this 🤟
runner.compile_sql("select * from {{ ref('stg_salesforce__users') }}")

# run SQL too
result = runner.execute_sql("select * from {{ ref('stg_salesforce__users') }}")
result.table.print_csv()

# leverage git to diff the OUTPUT of a model from git HEAD 
# to your revision on disk to safely audit changes as you work
diff_and_print_to_console("fct_sales", pk="order_id", runner=runner)  
```

## Roadmap

These features are being actively developed and will be merged into the next few minor releases

1. Complete high performance dbt server solution for running & compiling dbt SQL statements
2. Extend git diff functionality to pin revisions of models in dedicated schema(s) in the warehouse  
3. Complete build out of `sources` tools

![graph](https://repobeats.axiom.co/api/embed/df37714aa5780fc79871c60e6fc623f8f8e45c35.svg "Repobeats analytics image")
