import os
import time
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import pandas_profiling
import streamlit as st
from dbt.adapters.base.relation import BaseRelation
# The app does two things we do not scope ourselves to in the CLI:
# Build Models, Compile Models
# So include these imports here
from dbt.contracts.graph import compiled, parsed
from dbt.exceptions import CompilationException, DatabaseException
from dbt.flags import set_from_args
from dbt.task.run import ModelRunner
from streamlit_ace import THEMES, st_ace
from streamlit_pandas_profiling import st_profile_report

import dbt_osmosis.main

# ----------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------

st.set_page_config(page_title="dbt-osmosis Workbench", page_icon="🌊", layout="wide")
st.title("dbt-osmosis 🌊")
if "target_profile" not in st.session_state:
    st.session_state["target_profile"] = None
    # Str
    # Target profile, can be changed freely which invokes refresh_dbt() rebuilding manifest
if "profiles_dir" not in st.session_state:
    st.session_state["profiles_dir"] = str(dbt_osmosis.main.DEFAULT_PROFILES_DIR)
    # Str
    # dbt profile dir saved from bootup form
if "project_dir" not in st.session_state:
    cwd = Path().cwd()
    root_path = Path("/")
    while cwd != root_path:
        project_file = cwd / "dbt_project.yml"
        if project_file.exists():
            break
        cwd = cwd.parent
    else:
        cwd = Path().cwd()
    st.session_state["project_dir"] = str(cwd)
    # str
    # dbt project directory saved from bootup form
if "model_rev" not in st.session_state:
    st.session_state["model_rev"] = {}
    # MutableMapping
    # Stores model SQL when initially loaded and can ONLY be mutated by commiting a change
    # This is used as the basis for reverting changes
if "this_sql" not in st.session_state:
    st.session_state["this_sql"] = {}
    # MutableMapping
    # Key stores current model name, when selected model != key then we know model has changed
    # Value stores current SQL from workbench as its updated
if "failed_relation" not in st.session_state:
    st.session_state["failed_relation"] = None
    # None or DatabaseException
if "failed_temp_relation" not in st.session_state:
    st.session_state["failed_temp_relation"] = False
    # Bool
    # We use temp relations to determine output columns from a modified query
    # If that op fails, we use this state to fork logic
if "iter_state" not in st.session_state:
    st.session_state["iter_state"] = 0
    # Int
    # Iterate this to remount components forcing reload - used in editor during revert
    # Only remounts components which interpolate this state var in their `key`
if "new_model_name" not in st.session_state:
    st.session_state["new_model_name"] = ""
    # str
    # Stores the name of a created model in the state so it can be automatically selected
    # on refresh
if "schema_map" not in st.session_state:
    st.session_state["schema_map"] = {}
    # dict
    # Stores a mapping of model unique id to schema yml current location and target location


def refresh_dbt():
    (
        st.session_state["project"],
        st.session_state["profile"],
        st.session_state["config"],
        st.session_state["adapter"],
    ) = dbt_osmosis.main.load_dbt(
        project_dir=st.session_state["project_dir"],
        profiles_dir=st.session_state["profiles_dir"],
        target=st.session_state["target_profile"],
    )
    st.session_state["manifest"] = dbt_osmosis.main.compile_project_load_manifest(
        cfg=st.session_state["config"],
        flat=False,
    )
    st.session_state["schema_map"] = dbt_osmosis.main.build_schema_folder_map(
        st.session_state["project"].project_name,
        st.session_state["project"].project_root,
        st.session_state["manifest"].flat_graph,
    )
    return True


# ----------------------------------------------------------------
# INIT STATE
# ----------------------------------------------------------------


def sync_bootup_state():
    args = dbt_osmosis.main.PseudoArgs(
        profiles_dir=st.session_state["profiles_dir"],
        project_dir=st.session_state["project_dir"],
    )
    set_from_args(args, args)
    st.session_state["project_dir"] = st.session_state["proj_select"]
    st.session_state["profiles_dir"] = st.session_state["prof_select"]


if (
    "project" not in st.session_state
    or "profile" not in st.session_state
    or "config" not in st.session_state
    or "adapter" not in st.session_state
    or "manifest" not in st.session_state
):
    init_profile_val = st.session_state["profiles_dir"]
    init_project_val = st.session_state["project_dir"]
    config_toast = st.empty()
    config_container = st.empty()
    bootup_form = config_container.form("boot_up", clear_on_submit=True)
    st.session_state["profiles_dir"] = bootup_form.text_input(
        "Enter path to profiles.yml",
        value=init_profile_val,
        key=f"prof_select",
    )
    st.session_state["project_dir"] = bootup_form.text_input(
        "Enter path to dbt_project.yml",
        value=init_project_val,
        key="proj_select",
    )
    do_proceed = bootup_form.form_submit_button("Load Project", on_click=sync_bootup_state)
    if not do_proceed or not (Path(st.session_state["project_dir"]) / "dbt_project.yml").exists():
        if do_proceed:
            config_toast.error("Invalid path")
        st.stop()
    config_container.empty()
    config_toast.empty()
    os.chdir(st.session_state["project_dir"])
    with st.spinner(text="Parsing profile and reading your dbt project 🦸"):
        LOADED = refresh_dbt()


# ----------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------

st.sidebar.header("Profiles")

st.sidebar.write(
    "Select a profile used for materializing, compiling, and testing models. Can be updated at any time."
)
profile_data = dbt_osmosis.main.get_raw_profiles(st.session_state["profiles_dir"])
st.session_state["target_profile"] = st.sidebar.radio(
    f"Loaded profiles from {st.session_state['project'].profile_name}",
    [targ for targ in profile_data[st.session_state["project"].profile_name].get("outputs", [])],
    key="profile_selector",
)
st.sidebar.markdown(f"Current Target: **{st.session_state['target_profile']}**")
st.sidebar.write("")
st.sidebar.write("Utility")
st.sidebar.button("Reload dbt project", key="refresh_dbt_ctx")
if (
    st.session_state["profile"].target_name != st.session_state["target_profile"]
    or st.session_state["refresh_dbt_ctx"]
):
    refresh_dbt()
st.sidebar.caption(
    "Use this if any updated assets in your project have not yet reflected in the workbench, for example: you add some generic tests to some models while osmosis workbench is running."
)
st.sidebar.write("")
editor_theme = st.sidebar.selectbox("Editor Theme", THEMES, key="theme_picker")
editor_lang = st.sidebar.selectbox(
    "Editor Language", ("pgsql", "mysql", "sql", "sqlserver"), key="lang_picker"
)

# ----------------------------------------------------------------
# MODEL IDE
# ----------------------------------------------------------------

st.write("")

modeller_header, modeller_creator = st.columns(2)
modeller_header.subheader("Osmosis IDE 🛠️")


def recurse_dirs(
    curr_dir: Path = Path().cwd(), opts: Optional[Sequence] = None, depth: Optional[int] = 0
) -> Sequence[Path]:
    """Resursively build a list of directories starting from input

    Args:
        curr_dir (Path, optional): Initial input directory. Defaults to current working dir
        opts (Optional[Sequence], optional): Container list which to accumulates values. Defaults to None which instantiates new list.
        depth (Optional[int], optional): Can be used in recursion to render interesting list vals using "."*depth for example. Defaults to 0.

    Returns:
        Sequence[Path]: List of directory paths
    """
    if opts is None:
        opts = []
    for path in list(filter(lambda f: f.is_dir(), Path(curr_dir).iterdir())):
        opts.append(path.relative_to(Path(st.session_state["project"].project_root)))
        opts = recurse_dirs(path, opts, depth + 1)
    return opts


# MODEL CREATE/DELETE
with modeller_creator.expander("Create or delete model"):
    model_creator_status = st.empty()

    # Build opts accounting for dbt change in model path
    try:
        model_dirs = st.session_state["project"].source_paths
    except AttributeError:
        model_dirs = st.session_state["project"].model_paths
    dir_opts = []
    for dir_ in model_dirs:
        dir_opts.extend(recurse_dirs(Path(st.session_state["project"].project_root) / dir_))

    # Show opts
    new_model_dir: Sequence[Path] = st.selectbox(
        "Model Directory", dir_opts, key=f"new_model_dir_{st.session_state['iter_state']}"
    )
    new_model_name = st.text_input(
        "Model Name (exclude .sql)", key=f"new_model_name_{st.session_state['iter_state']}"
    )
    if st.button("Create Model", key="new_model_create"):
        if not new_model_name:
            model_creator_status.error("Please provide a model name")
        else:
            new_model: Path = new_model_dir / f"{new_model_name}.sql"
            if not new_model.exists():
                new_model.touch()
                f = new_model.open("w", encoding="utf-8")
                f.write("SELECT 1")
                f.close()
                model_creator_status.success("Successfully created model")
                st.session_state["new_model_name"] = new_model_name
                with st.spinner(text="Reloading your dbt project 🦸"):
                    LOADED = refresh_dbt()
                model_creator_status.empty()
                st.session_state["iter_state"] += 1
            else:
                model_creator_status.warning("Model already exists")
    if st.button("Delete Model", key="delete_model"):
        if not new_model_name:
            model_creator_status.error("Please provide a model name")
        else:
            new_model: Path = new_model_dir / f"{new_model_name}.sql"
            if not new_model.exists():
                model_creator_status.warning("Model does not exist")
            else:
                new_model.unlink(missing_ok=True)
                model_creator_status.success("Successfully deleted model")
                with st.spinner(text="Reloading your dbt project 🦸"):
                    LOADED = refresh_dbt()
                model_creator_status.empty()
                st.session_state["iter_state"] += 1

st.write("")

if "hide_viewer" not in st.session_state:
    st.session_state["hide_viewer"] = False
    # Bool
    # This is used to pivot the editor layout


def toggle_viewer():
    st.session_state["hide_viewer"] = not st.session_state["hide_viewer"]


if "reset_model" not in st.session_state:
    st.session_state["reset_model"] = False
    # Bool
    # Set to true when `revert_model` is called; when editor component requests
    # its body value, a switch iterates iter_state causing remount and updating display
    # to the correctly reverted value


def toggle_reset():
    st.session_state["reset_model"] = not st.session_state["reset_model"]


# MODEL SELECTOR VALID OPTIONS
opts = []
init_model_ix = 0
iter_model_ix = 0
for NODE, config in st.session_state["manifest"].flat_graph["nodes"].items():
    if config["resource_type"] == "model":
        opts.append(NODE)
        if config["name"] == st.session_state["new_model_name"]:
            init_model_ix = iter_model_ix
            st.session_state["new_model_name"] = ""
        iter_model_ix += 1


# MODEL SELECTOR
with st.container():
    choice = st.selectbox("Select a model to edit", options=opts, index=init_model_ix)
    model = st.session_state["manifest"].flat_graph["nodes"].get(choice)


# DYNAMIC COMPILATION OPTION
st.write("")
auto_compile = st.checkbox("Dynamic Compilation", key="auto_compile_flag")
if auto_compile:
    st.caption("Compiling SQL on change")
else:
    st.caption("Compiling SQL with control + enter")


if choice not in st.session_state["model_rev"]:
    # Here is our LTS
    st.session_state["model_rev"][choice] = model.copy()
    UNMODIFIED_NODE = parsed.ParsedModelNode._deserialize(st.session_state["model_rev"][choice])


if st.session_state["this_sql"].get(choice) is None:
    # MODEL CHANGED
    st.session_state["this_sql"] = {}
    st.session_state["sql_data"] = pd.DataFrame()
    st.session_state["iter_state"] += 1


def revert_model(target_model: dict) -> None:
    """Reverts the editor to the last known value of the model prior to any uncommitted changes,
    also sets reset_model state to True which is processed by `get_model_sql` to help force update the editor during remount

    Args:
        target_model (dict): Input model dict containing at least the raw_sql key
    """
    target_model["raw_sql"] = st.session_state["model_rev"][choice]["raw_sql"]
    st.session_state["this_sql"][choice] = ""
    st.session_state["reset_model"] = True


def save_model(target_model: dict) -> None:
    """Saves an updated model committing the changes to file and to the last know representation of the model

    Args:
        target_model (dict): Input model dict containing at least the raw_sql, root_path, and original_file_path keys
    """
    path = Path(target_model["root_path"]) / Path(target_model["original_file_path"])
    path.touch()
    with open(path, "w", encoding="utf-8") as f:
        f.write(target_model["raw_sql"])
    st.session_state["model_rev"][choice]["raw_sql"] = target_model["raw_sql"]


def build_model(
    target_node: compiled.CompiledModelNode, is_temp: bool = False
) -> Optional[BaseRelation]:
    """Builds model in database, this will error if the model exists -- we catch the error but cannot distinguish it from any other
    `DatabaseException`. This function is only invoked in one place, the model action button

    Args:
        target_node (compiled.CompiledModelNode): The compiled NODE to manifest
        is_temp (bool, optional): A flag that tells us to update a state var for rendering a specific error. Defaults to False.

    Returns:
        Optional[BaseRelation]: Return on error is none, otherwise a Relation
    """
    # Copy needed to avoid mutation
    _target_node = deepcopy(target_node)
    runner = ModelRunner(
        st.session_state["config"], st.session_state["adapter"], _target_node, 1, 1
    )
    with st.session_state["adapter"].connection_named("dbt-osmosis"):
        runner.before_execute()
        try:
            result = runner.execute(_target_node, st.session_state["manifest"])
        except DatabaseException as err:
            print(err)
            st.session_state["failed_relation"] = err
            if is_temp:
                st.session_state["failed_temp_relation"] = True
        else:
            st.session_state["failed_relation"] = None
            st.session_state["failed_temp_relation"] = False
            runner.after_execute(result)
            return st.session_state["adapter"].Relation.create_from(
                st.session_state["config"], _target_node, type=_target_node.config.materialized
            )


def model_action(exists: bool) -> str:
    """Switch function to return string that is represented in model action button,
    I used this sort of proxy function because of inconsistent button render behavior when performing the switch in the main loop

    Args:
        exists (bool): Boolean indicator of if model exists or not as determined by `get_relation_if_exists`

    Returns:
        str: returns string to be rendered in component
    """
    if exists:
        return "Update dbt Model in Database"
    else:
        return "Build dbt Model in Database"


def get_relation_if_exists(target_model: dict) -> Tuple[BaseRelation, bool]:
    """Check if table exists in database using adapter get_relation

    Args:
        target_model (dict): Model dictionary containing at least the keys database, schema, and name

    Returns:
        Tuple[BaseRelation, bool]: Returns a tuple of the relation and True or None and False
    """
    try:
        with st.session_state["adapter"].connection_named("dbt-osmosis"):
            table = st.session_state["adapter"].get_relation(
                target_model["database"], target_model["schema"], target_model["name"]
            )
    except DatabaseException:
        table, table_exists = None, False
        return table, table_exists
    else:
        table_exists = table is not None
        return table, table_exists


def get_model_sql(target_model: dict) -> str:
    """Extracts SQL from model, forcibly extracts previous state SQL when reset_model toggle is True

    Args:
        target_model (dict): Input model dictionary containing at least the key raw_sql

    Returns:
        str: Output SQL
    """
    if st.session_state["reset_model"]:
        st.session_state["iter_state"] += 1
        st.session_state["reset_model"] = False
        return st.session_state["model_rev"][choice]["raw_sql"]
    return target_model["raw_sql"]


@st.cache
def get_database_columns(relation: Optional[BaseRelation]) -> Sequence:
    """Get columns for table from the database. If relation is None, we will return an empty list, if the query fails
    we will also return an empty list

    Args:
        table (BaseRelation, optional): Relation to query for typically built by `get_relation_if_exists`

    Returns:
        Sequence: List of database columns, empty list if unable to satisfy
    """
    if not relation:
        return []
    try:
        with st.session_state["adapter"].connection_named("dbt-osmosis"):
            database_columns = [
                c.name for c in st.session_state["adapter"].get_columns_in_relation(relation)
            ]
    except DatabaseException:
        database_columns = []
    return database_columns


@st.cache
def build_node(current_model: dict) -> parsed.ParsedModelNode:
    """Takes a dict representing the model and deserializes it into a parsed NODE

    Args:
        current_model (dict): Dict representing the model

    Returns:
        parsed.ParsedModelNode: Parsed NODE
    """
    return parsed.ParsedModelNode._deserialize(current_model)


@st.cache
def compile_node(current_node: parsed.ParsedModelNode) -> compiled.CompiledModelNode:
    """Compiles a NODE, this is agnostic to the fact of whether the SQL is valid but is stringent on valid
    Jinja, therefore a None return value after catching the error affirms invalid jinja from user (which is fine since they might be compiling as they type)
    Caching here is immensely valuable since we know a NODE which hashes the same as a prior input will have the same output and this fact gives us a big speedup

    Args:
        current_node (parsed.ParsedModelNode): Input model NODE with SQL in sync with editor

    Returns:
        compiled.CompiledModelNode: Compiled NODE
    """
    try:
        with st.session_state["adapter"].connection_named("dbt-osmosis"):
            return (
                st.session_state["adapter"]
                .get_compiler()
                .compile_node(current_node, st.session_state["manifest"])
            )
    except CompilationException:
        return None


def update_manifest_node(current_node: parsed.ParsedModelNode) -> bool:
    """Updates NODE in stateful manifest so it is preserved even as we traverse models unless dbt is refreshed

    Args:
        current_node (parsed.ParsedModelNode): Current NODE in sync with editor

    Returns:
        bool: True when complete
    """
    st.session_state["manifest"].update_node(current_node)
    st.session_state["manifest"].build_flat_graph()
    return True


st.write("")

# EDITOR CONTROLS CONTAINER
btn_container = st.container()

# EDITOR ERROR CONTAINER
db_res = st.session_state["failed_relation"]
if isinstance(db_res, DatabaseException):
    show_err_1 = st.error(f"Model materialization failed: {db_res}")
    time.sleep(4.20)
    st.session_state["failed_relation"] = None
    show_err_1.empty()

# EDITOR / VIEWER CONTAINER
with st.container():

    if not st.session_state["hide_viewer"]:
        editor, viewer = st.columns(2)
        with editor:
            model_editor = st.expander("Edit Model")
        with viewer:
            compiled_viewer = st.expander("View Compiled SQL")

    else:

        model_editor = st.expander("Edit Model")
        compiled_viewer = st.expander("View Compiled SQL")


# MODEL EDITOR CONTENTS
with model_editor:
    model["raw_sql"] = st_ace(
        value=get_model_sql(model),
        theme=editor_theme,
        language=editor_lang,
        auto_update=auto_compile,
        key=f"dbt_ide_{st.session_state['iter_state']}",
    )
    if model["raw_sql"] != st.session_state["this_sql"].get(choice, ""):
        NODE = build_node(model)
        manifest_node_current = update_manifest_node(NODE)
    st.session_state["this_sql"][choice] = model["raw_sql"]

# CACHE OPTIMIZED GLOBALLY USED VALS
UNMODIFIED_NODE = build_node(st.session_state["model_rev"][choice])
NODE = build_node(model)
COMPILED_UNMODIFIED_NODE = compile_node(UNMODIFIED_NODE)
COMPILED_NODE = compile_node(NODE)
THIS, EXISTS = get_relation_if_exists(model)

# COMPILED SQL VIEWER CONTENTS
with compiled_viewer:
    if COMPILED_NODE:
        st.code(COMPILED_NODE.compiled_sql, language="sql")
    else:
        st.warning("Invalid Jinja")

# BUTTON & NOTIFICATION CONTAINER CONTENTS
with btn_container:
    pivot_layout_btn, build_model_btn, commit_changes_btn, revert_changes_btn = st.columns(4)
    with pivot_layout_btn:
        st.button("Pivot Layout", on_click=toggle_viewer)
    with build_model_btn:
        if COMPILED_NODE:
            run_model = st.button(
                label=model_action(EXISTS),
            )
        else:
            run_model = False
    if not NODE.same_body(UNMODIFIED_NODE):
        with commit_changes_btn:
            st.button("Commit changes to file", on_click=save_model, kwargs={"target_model": model})
        with revert_changes_btn:
            st.button("Revert changes", on_click=revert_model, kwargs={"target_model": model})
        st.info("Uncommitted changes detected in model")

    if run_model:
        with st.spinner("Running model against target... ⚙️"):
            build_model(COMPILED_NODE)
        with st.spinner("Model ran against target! 🧑‍🏭"):
            time.sleep(2)
        run_model = False

# ----------------------------------------------------------------
# QUERY RESULT INSPECTOR
# ----------------------------------------------------------------

if "sql_data" not in st.session_state:
    st.session_state["sql_data"] = pd.DataFrame()
    # pd.DataFrame
    # Stores query viewer output from either test or diff

if "sql_query_info" not in st.session_state:
    st.session_state["sql_query_info"] = ""
    # AdapterRepsonse
    # Stores response from dbt adapter with contains minimal metadata

if "sql_query_mode" not in st.session_state:
    st.session_state["sql_query_mode"] = "test"
    # Str
    # Track query viewer as having test query results or diff query results


st.write("")
st.subheader("Osmosis Query Result Inspector 🔬")
st.write("")


def highlight_string(cell: Any, string: str, color: str):
    return np.where(cell == string, f"color: {color};", None)


def build_style(new: pd.DataFrame, old: pd.DataFrame):
    # Highlight added rows green
    style = new.where(new == old, "background-color: green")
    # Other cells as is
    style = style.where(new != old, "background-color: white")
    return style


@st.cache
def get_column_set(select_sql: str) -> set:
    """Adapted from dbt get_columns_in_query macro, return a set of the columns that would be returned from a SQL select statement"""
    try:
        with st.session_state["adapter"].connection_named("dbt-osmosis"):
            cols = set(
                map(
                    lambda col: col.name,
                    st.session_state["adapter"]
                    .execute(
                        f"""
            select * from (
                { select_sql }
            ) as __dbt_sbq
            where false
            limit 0
            """,
                        auto_begin=True,
                        fetch=True,
                    )[1]
                    .columns,
                )
            )
    except DatabaseException:
        cols = set()
    return cols


def test_query() -> None:
    """This function queries the database using the editor SQL and stores results in the state variables:
    sql_data which should always contain a DataFrame
    sql_query_info which contains an adapter response object or empty str, a truthy value should always indicate a valid reponse
    sql_query_mode is set to "test" which acts as a switch for the components rendered post function execution
    """

    with st.session_state["adapter"].connection_named("dbt-osmosis"):
        query = st.session_state["sql_data"] = st.session_state["adapter"].execute(
            f"select * from ({COMPILED_NODE.compiled_sql}) __all_data limit {QUERY_LIMIT}",
            fetch=True,
        )

    # I WISH THERE WAS A BETTER WAY TO WORK WITH AGATE IN STREAMLIT
    # This is admittedly terrible efficiency-wise IMO, the user defined LIMIT alleviates this somewhat
    table = query[1]
    output = []
    json_funcs = [c.jsonify for c in table._column_types]
    for row in table._rows:
        values = tuple(json_funcs[i](d) for i, d in enumerate(row))
        output.append(OrderedDict(zip(row.keys(), values)))
    st.session_state["sql_data"] = pd.DataFrame(output)
    st.session_state["sql_query_info"] = query[0]
    st.session_state["sql_query_mode"] = "test"


def diff_query(
    primary_keys: Sequence,
    diff_compute_engine: str = "database",
    aggregate_result: bool = False,
    prev_cols: set = set(),
    curr_cols: set = set(),
) -> None:

    with st.session_state["adapter"].connection_named("dbt-osmosis"):

        # Execute Query
        if diff_compute_engine == "database":

            # Account for this slight semantic difference in BigQuery
            except_ = "except"
            if st.session_state["adapter"].type == "bigquery":
                except_ = "except distinct"

            # These are chosen from an intersection of columns from the unmodified and current nodes
            compare_cols = ",".join(primary_keys)
            superset = prev_cols.union(curr_cols)

            # Build final selects
            prev_final_select = ""
            curr_final_select = ""
            for col in list(superset):
                prev_final_select += f", old.{col}" if col in prev_cols else f", '' AS {col}"
                curr_final_select += f", new.{col}" if col in curr_cols else f", '' AS {col}"

            # Build the join conditions
            join_cond_added = ""
            join_cond_removed = ""
            for k in primary_keys:
                join_cond_added += f"unioned.{k} = new.{k} and "
                join_cond_removed += f"unioned.{k} = old.{k} and "
            else:
                join_cond_added = join_cond_added[:-4]
                join_cond_removed = join_cond_removed[:-4]

            query = f"""
    with new as (

        SELECT * FROM ({COMPILED_NODE.compiled_sql}) AS __cte_1
        
    )
    
    , old AS (

        SELECT * FROM  ({COMPILED_UNMODIFIED_NODE.compiled_sql}) AS __cte_2

    )
    
    , new_minus_old AS (

        SELECT {compare_cols} FROM new
        {except_}
        SELECT {compare_cols} FROM old

    )
    
    , old_minus_new AS (

        SELECT {compare_cols} FROM old
        {except_}
        SELECT {compare_cols} FROM new

    )
    
    , unioned AS (

        SELECT 
            'ADDED' AS __diff
            , * 
        FROM 
            new_minus_old

        UNION ALL

        SELECT 
            'REMOVED' AS __diff
            , * 
        FROM 
            old_minus_new

    )

    SELECT
        __diff
        {curr_final_select}
    FROM 
        new
    INNER JOIN 
        unioned 
            ON ({join_cond_added})

    UNION ALL

    SELECT 
        __diff
        {prev_final_select}
    FROM 
        old
    INNER JOIN 
        unioned 
            ON ({join_cond_removed})

                """

            # We must either aggregate or limit since we aren't in the quantum computing age of laptops yet
            if aggregate_result:
                query = f"""
                    SELECT 
                        __diff
                        , count(*) as records
                    FROM 
                        ({query}) __agg_res
                    GROUP BY 
                        __diff
                """
            else:
                query += f"LIMIT {QUERY_LIMIT}"

            # Execute Query
            print(query)
            result = st.session_state["sql_data"] = st.session_state["adapter"].execute(
                query,
                fetch=True,
            )

            # I WISH THERE WAS A BETTER WAY TO WORK WITH AGATE IN STREAMLIT
            # This is admittedly terrible efficiency-wise IMO, the user defined LIMIT alleviates this somewhat
            table = result[1]
            output = []
            json_funcs = [c.jsonify for c in table._column_types]
            for row in table._rows:
                values = tuple(json_funcs[i](d) for i, d in enumerate(row))
                output.append(OrderedDict(zip(row.keys(), values)))
            st.session_state["sql_data"] = pd.DataFrame(output)
            st.session_state["sql_query_info"] = result[0]
            st.session_state["sql_query_mode"] = "diff"

        elif diff_compute_engine == "pandas":
            pass  # Not implemented yet, in theory slightly more robust for small datasets - cell level diffs vs row level
        else:
            # We can consider alternative diff methods or engines like Dask perhaps
            raise NotImplemented("Only valid comput engines are SQL and Pandas")


# DATABASE COLS
database_columns = []
if EXISTS:
    database_columns = get_database_columns(THIS)

# CURRENT COLUMNS
previous_columns = get_column_set(COMPILED_UNMODIFIED_NODE.compiled_sql)
current_columns = get_column_set(COMPILED_NODE.compiled_sql)


# TEST BENCH
if NODE.same_body(UNMODIFIED_NODE):
    col_test_control, col_limit_control, _ = st.columns([1, 1, 3])

    with col_test_control:
        if COMPILED_NODE:
            # Hide test control if compiled node is invalid/None
            st.button("Test Compiled Query", on_click=test_query)
            st.caption("This will run the compiled SQL against your data warehouse")
    with col_limit_control:
        limit_container = st.empty()

else:
    col_test_control, col_diff_control, col_limit_control = st.columns([1, 1, 3])

    with col_diff_control:
        diff_engine = st.selectbox(
            "Diff Compute Engine", ("SQL", "Python"), index=0, key="engine_selector"
        )
        agg_diff_results = st.checkbox("Aggregate Diff Results", key="agg_diff_results")
        st.caption("Use this to get aggregate diff results when data does not fit in memory")

    if COMPILED_NODE:
        # Hide test/diff controls if compiled node is invalid/None
        with col_test_control:
            st.button("Test Compiled Query", on_click=test_query)
            st.caption("This will run the compiled SQL against your data warehouse")
        with col_limit_control:
            valid_diff_col_opts = list(current_columns.intersection(previous_columns))
            diff_cols = st.multiselect(
                "Diff Primary Key(s)",
                valid_diff_col_opts,
                default=[valid_diff_col_opts[0]] if valid_diff_col_opts else [],
                key="pk_selector",
            )
        with col_test_control:
            st.button(
                "Calculate Row Level Diff",
                on_click=diff_query,
                kwargs={
                    "primary_keys": diff_cols,
                    "aggregate_result": agg_diff_results,
                    "prev_cols": previous_columns,
                    "curr_cols": current_columns,
                },
            )
            st.caption("This will output the rows added or removed by changes made to the query")

    with col_limit_control:
        limit_container = st.empty()

QUERY_LIMIT = limit_container.number_input(
    "Limit Results", min_value=1, max_value=50_000, value=2_000, step=1, key="query_limiter"
)

# TEST RESULT RENDERER
if not st.session_state["sql_data"].empty:

    if st.session_state["sql_query_mode"] == "test":
        st.write("Compiled SQL query results")

    elif st.session_state["sql_query_mode"] == "diff":
        st.write("Compiled SQL query row-level diff")

    st.write(st.session_state["sql_query_info"])

    st.dataframe(
        st.session_state["sql_data"]
        .style.apply(
            highlight_string,
            string="REMOVED",
            color="red",
        )
        .apply(
            highlight_string,
            string="ADDED",
            color="green",
        )
    )

else:

    st.write("")
    st.markdown(
        "> The results of your workbench query will show up here. Click `Test Compiled Query` to see the results. "
    )
    st.write("")


@st.cache
def convert_df_to_csv(sql_data: pd.DataFrame):
    return sql_data.to_csv().encode("utf-8")


@st.cache(
    hash_funcs={
        pandas_profiling.report.presentation.core.container.Container: lambda _: COMPILED_NODE
        or NODE,
        pandas_profiling.report.presentation.core.html.HTML: lambda _: COMPILED_NODE or NODE,
    },
    allow_output_mutation=True,
)
def build_profile_report(sql_data: pd.DataFrame, minimal: bool) -> pandas_profiling.ProfileReport:
    """Another example of a massively improved function thanks to simple caching"""
    return sql_data.profile_report(minimal=minimal)


@st.cache(
    hash_funcs={
        pandas_profiling.report.presentation.core.container.Container: lambda _: COMPILED_NODE
        or NODE,
        pandas_profiling.report.presentation.core.html.HTML: lambda _: COMPILED_NODE or NODE,
    }
)
def convert_profile_report_to_html(profiled_data: pandas_profiling.ProfileReport) -> str:
    return profiled_data.to_html()


st.write("")

# QUERY RESULT CONTROLS
col_download, col_profile, col_profile_opt = st.columns([1, 1, 3])

with col_download:
    st.download_button(
        label="Download data as CSV",
        data=convert_df_to_csv(st.session_state["sql_data"]),
        file_name=f"{choice}.csv",
        mime="text/csv",
    )

with col_profile:
    st.button("Profile Data", key="do_profile")

with col_profile_opt:
    st.checkbox("Basic Profiler", key="do_profile_minimal")
    st.caption(
        "Useful for larger datasets, use the minimal pandas-profiling option for a simpler report"
    )

# PROFILER OUTPUT
if st.session_state["do_profile"]:
    pr = build_profile_report(st.session_state["sql_data"], st.session_state["do_profile_minimal"])
    st_profile_report(pr, height=500)
    st.download_button(
        label="Download profile report",
        data=convert_profile_report_to_html(pr),
        file_name=f"{choice}_profile.html",
        mime="text/html",
        key="query_result_downloader",
    )
    st.write("")

# ----------------------------------------------------------------
# Documentation IDE
# ----------------------------------------------------------------

st.write("")
st.subheader("Osmosis Docs Editor ✏️")
st.write("")

SCHEMA_FILE = st.session_state["schema_map"].get(choice)
KNOWLEDGE = dbt_osmosis.main.pass_down_knowledge(
    dbt_osmosis.main.build_ancestor_tree(model, st.session_state["manifest"].flat_graph),
    st.session_state["manifest"].flat_graph,
)


def fix_documentation() -> bool:
    # Migrates model, injects model (requires model in database!)
    # Essentially the CLI `compose` task scoped to a single model
    schema_map = {choice: SCHEMA_FILE}
    was_restructured = dbt_osmosis.main.commit_project_restructure(
        dbt_osmosis.main.build_project_structure_update_plan(
            schema_map, st.session_state["manifest"].flat_graph, st.session_state["adapter"]
        )
    )
    return was_restructured and refresh_dbt()


doc_btns = []
doc_actions = []
n_doc_note = 1
if SCHEMA_FILE.current is None:
    st.caption(f"{n_doc_note}. This model is currently not documented")
    n_doc_note += 1
    if SCHEMA_FILE.target.exists():
        doc_btns.append("Update schema file")  # fix_documentation
        doc_actions.append(fix_documentation)
        st.caption(f"{n_doc_note}. An appropriate target schema yml exists")
        n_doc_note += 1
    else:
        doc_btns.append("Build schema file")  # fix_documentation
        doc_actions.append(fix_documentation)
        st.caption(f"{n_doc_note}. The target schema yml does not exist")
        n_doc_note += 1
else:
    st.caption(f"{n_doc_note}. This model is currently documented")
    if not SCHEMA_FILE.is_valid:
        st.caption(
            f"{n_doc_note}. The current location of the schema file is invalid. This model should be migrated to target schema file"
        )
        doc_btns.append("Migrate schema file model")  # fix_documentation
        doc_actions.append(fix_documentation)
        n_doc_note += 1
    else:
        st.caption(f"{n_doc_note}. Schema file location is valid")
        # only actions are inherit + commit
        n_doc_note += 1
if not EXISTS:
    st.caption(
        f"{n_doc_note}. Table must exist in database to action. Use the `Build dbt Model in Database` button to build the model"
    )
elif len(doc_btns) > 0:
    st.write("")
    for doc_btn, doc_col, doc_action in zip(doc_btns, st.columns(len(doc_btns)), doc_actions):
        doc_col.button(doc_btn, on_click=doc_action)

# DOC EDITOR
with st.expander("Edit documentation"):
    for column in database_columns:
        st.text_input(
            column,
            value=(
                model["columns"]
                .get(column, {"description": "Not in schema file"})
                .get("description")
                or "Not documented"
            ),
        )
        progenitor = KNOWLEDGE.get(column)
        if progenitor:
            st.write(progenitor)
        else:
            st.caption("This is the column's origin")

# ----------------------------------------------------------------
# Data Test Runner
# ----------------------------------------------------------------

st.subheader("Test Runner")
st.write("Execute configured tests and validate results")

if "test_results" not in st.session_state:
    st.session_state["test_results"] = pd.DataFrame()
    # pd.DataFrame
    # Stores data from executed test

if "test_meta" not in st.session_state:
    st.session_state["test_meta"] = ""
    # AdapterResponse
    # Stores response from executed test query
    # The truthiness of this is the switch which renders Test runner results section


def run_test(
    test_node: Union[
        compiled.CompiledSingularTestNode,
        compiled.CompiledGenericTestNode,
        parsed.ParsedSingularTestNode,
        parsed.ParsedGenericTestNode,
    ]
):
    with st.session_state["adapter"].connection_named("dbt-osmosis-tester"):
        # Grab the compiled_sql attribute
        sql_select_test = getattr(test_node, "compiled_sql", None)
        if not sql_select_test:
            compiled_test = (
                st.session_state["adapter"]
                .get_compiler()
                .compile_node(test_node, st.session_state["manifest"])
            )
            sql_select_test = compiled_test.compiled_sql
        test_results = st.session_state["adapter"].execute(
            sql_select_test,
            fetch=True,
        )

    # I WISH THERE WAS A BETTER WAY TO WORK WITH AGATE IN STREAMLIT
    # This is admittedly terrible efficiency-wise IMO, the user defined LIMIT alleviates this somewhat
    table = test_results[1]
    output = []
    json_funcs = [c.jsonify for c in table._column_types]
    for row in table._rows:
        values = tuple(json_funcs[i](d) for i, d in enumerate(row))
        output.append(OrderedDict(zip(row.keys(), values)))
    st.session_state["test_meta"] = test_results[0]
    st.session_state["test_results"] = pd.DataFrame(output)


# DECLARED TESTS
test_opts = {}
for node_config in st.session_state["manifest"].nodes.values():
    if node_config.resource_type == "test" and choice in node_config.depends_on.nodes:
        test_opts[node_config.name] = node_config

test_pick_col, test_result_col, test_kpi_col = st.columns([2, 1, 1])

# RENDER TEST RUNNER IF WE HAVE OPTS
if not EXISTS:
    st.markdown(f" > Model is not materialized in database")
    st.write("")

elif test_opts:

    with test_pick_col:
        selected_test = st.selectbox("Model Test", list(test_opts.keys()), key="test_picker")
        st.button(
            "Run Test",
            key="test_runner",
            on_click=run_test,
            kwargs={"test_node": test_opts[selected_test]},
        )
    test_record_count = len(st.session_state["test_results"].index)
    with test_kpi_col:
        if st.session_state["test_meta"]:
            st.metric(label="Failing Records", value=test_record_count)
            st.markdown("#### PASSED" if test_record_count == 0 else "#### FAILED")
    with test_result_col:
        if st.session_state["test_meta"]:
            st.write("Completed Test Metadata")
            st.caption(f"Model Name: {model['name']}")
            st.caption(f"Test Name: {selected_test}")
            if st.session_state["test_results"].empty:
                st.caption("Test Result: **Passed**! No failing records detected")
            else:
                st.caption(f"Test Result: **Failed**! {test_record_count} failing records detected")
    if st.session_state["test_meta"]:
        with st.expander("Test Results", expanded=True):
            st.write("Adapter Response")
            st.write(st.session_state["test_meta"].__dict__)
            st.write("")
            st.write("Returned Data")
            st.dataframe(st.session_state["test_results"])
            st.write("")
            st.download_button(
                label="Download test results as CSV",
                data=convert_df_to_csv(st.session_state["test_results"]),
                file_name=f"{selected_test}.csv",
                mime="text/csv",
                key="test_result_downloader",
            )
            st.write("")

else:
    st.markdown(f" > No tests found for model `{choice}` or model")
    st.write("")

# ----------------------------------------------------------------
# MANIFEST INSPECTOR
# ----------------------------------------------------------------

st.subheader("Manifest Representation")
with st.expander("View Manifest"):
    st.json(model)
