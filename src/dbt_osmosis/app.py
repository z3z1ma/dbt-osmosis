import argparse
import hashlib
import sys
import time
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import pandas_profiling
import streamlit as st
from dbt.adapters.base.relation import BaseRelation
from dbt.contracts.graph import compiled, manifest, parsed
from dbt.exceptions import CompilationException, DatabaseException
from dbt.task.run import ModelRunner
from streamlit_ace import THEMES, st_ace
from streamlit_pandas_profiling import st_profile_report

from dbt_osmosis.core.osmosis import DEFAULT_PROFILES_DIR, DbtOsmosis, SchemaFile, get_raw_profiles

st.set_page_config(page_title="dbt-osmosis Workbench", page_icon="🌊", layout="wide")

parser = argparse.ArgumentParser(description="dbt osmosis workbench")
parser.add_argument("--profiles-dir", help="dbt profile directory")
parser.add_argument("--project-dir", help="dbt project directory")
parser.add_argument("-m", "--model", required=True, help="dbt model")
args = vars(parser.parse_args(sys.argv[1:]))

# GLOBAL STATE VARS
DBT = "dbt_osmosis_controller_interface"
MAP = "dbt_osmosis_folder_mapping"
st.session_state.setdefault(
    (PROJ_DIR := "dbt_osmosis_project_dir"), args["project_dir"] or str(Path.cwd())
)
st.session_state.setdefault(
    (PROF_DIR := "dbt_osmosis_profiles_dir"), args["profiles_dir"] or DEFAULT_PROFILES_DIR
)

# COMPONENT KEYS
PROFILE_SELECTOR = "dbt_osmosis_profile_selector"
THEME_PICKER = "dbt_osmosis_editor_theme"
DIALECT_PICKER = "dbt_osmosis_dialect"

# COMPONENT OPTIONS
DIALECTS = ("pgsql", "mysql", "sql", "sqlserver")

# TRIGGERS
st.session_state.setdefault((EDITOR_STATE_ITER_1 := "EDITOR_STATE_ITER_1"), 1)
DBT_DO_RELOAD = "DBT_DO_RELOAD"
st.session_state.setdefault((PIVOT_LAYOUT := "PIVOT_LAYOUT"), False)
st.session_state.setdefault((FAILED_RELATION := "FAILED_RELATION"), None)
st.session_state.setdefault((REVERT_MODEL := "REVERT_MODEL"), False)


def hash_parsed_node(node: parsed.ParsedModelNode) -> str:
    return hashlib.md5(node.raw_sql.encode("utf-8")).hexdigest()


def hash_compiled_node(node: parsed.ParsedModelNode):
    return hashlib.md5(node.raw_sql.encode("utf-8")).hexdigest()


hash_funcs = {
    parsed.ParsedModelNode: hash_parsed_node,
    compiled.CompiledModelNode: hash_compiled_node,
    manifest.Manifest: lambda _: None,
    DbtOsmosis: lambda _: None,
}


def inject_dbt():
    if DBT not in st.session_state:
        dbt_ctx = DbtOsmosis(
            project_dir=st.session_state[PROJ_DIR],
            profiles_dir=st.session_state[PROF_DIR],
        )
    else:
        dbt_ctx: DbtOsmosis = st.session_state[DBT]
        dbt_ctx.rebuild_dbt_manifest()

    st.session_state[DBT] = dbt_ctx
    st.session_state[MAP] = dbt_ctx.build_schema_folder_mapping()
    return True


if DBT not in st.session_state:
    inject_dbt()
ctx: DbtOsmosis = st.session_state[DBT]
schema_map: Dict[str, SchemaFile] = st.session_state[MAP]
st.session_state.setdefault((TARGET_PROFILE := "TARGET_PROFILE"), ctx.profile.target_name)


def toggle_viewer() -> None:
    st.session_state[PIVOT_LAYOUT] = not st.session_state[PIVOT_LAYOUT]


def revert_model(node: manifest.ManifestNode) -> None:
    """Reverts the editor to the last known value of the model prior to any uncommitted changes"""
    node.raw_sql = ""
    st.session_state[REVERT_MODEL] = True


def save_model(node: manifest.ManifestNode) -> None:
    """Saves an updated model committing the changes to file"""
    path = Path(node.root_path) / Path(node.original_file_path)
    # with open(path, "w", encoding="utf-8") as sql_file:
    #    sql_file.write(node.raw_sql)


def run_model(node: manifest.ManifestNode) -> Optional[BaseRelation]:
    """Builds model in database"""
    isolated_node = deepcopy(node)
    runner = ModelRunner(ctx.config, ctx.adapter, isolated_node, 1, 1)
    with st.session_state["adapter"].connection_named("dbt-osmosis"):
        runner.before_execute()
        try:
            result = runner.execute(isolated_node, ctx.dbt)
        except DatabaseException as error:
            st.session_state[FAILED_RELATION] = error
        else:
            st.session_state[FAILED_RELATION] = None
            runner.after_execute(result)
            return ctx.adapter.Relation.create_from(
                st.session_state["config"], isolated_node, type=isolated_node.config.materialized
            )


def get_model_action_text(exists: bool) -> str:
    if exists:
        return "Update dbt Model in Database"
    else:
        return "Build dbt Model in Database"


@st.experimental_memo
def get_relation_if_exists(_node: manifest.ManifestNode) -> Tuple[BaseRelation, bool]:
    """Check if table exists in database using adapter get_relation"""
    try:
        with ctx.adapter.connection_named("dbt-osmosis"):
            table = ctx.adapter.get_relation(_node.database, _node.schema, _node.name)
    except DatabaseException:
        table, table_exists = None, False
        return table, table_exists
    else:
        table_exists = table is not None
        return table, table_exists


def get_model_sql(node: manifest.ManifestNode) -> str:
    """DEPRECATED: Left here for reference in case we need to force trigger state change
    Extracts SQL from model, forcibly extracts previous state SQL when revert_model toggle is True"""
    if st.session_state[REVERT_MODEL]:
        st.session_state[EDITOR_STATE_ITER_1] += 1
        st.session_state[REVERT_MODEL] = False
        return st.session_state[BASE_NODE].raw_sql
    return node.raw_sql


@st.cache(hash_funcs=hash_funcs)
def get_database_columns(node: manifest.ManifestNode) -> List[str]:
    """Get columns for table from the database. If relation is None, we will return an empty list, if the query fails
    we will also return an empty list"""
    try:
        with ctx.adapter.connection_named("dbt-osmosis"):
            database_columns = ctx.get_columns(node)
    except DatabaseException:
        database_columns = []
    return database_columns


def update_manifest_node(node: manifest.ManifestNode) -> bool:
    """Updates NODE in stateful manifest so it is preserved even as we traverse models unless dbt is refreshed"""
    ctx.dbt.update_node(node)
    # ctx.dbt.build_flat_graph()
    return True


@st.cache(hash_funcs=hash_funcs)
def compile_model(node: manifest.ManifestNode) -> Optional[manifest.ManifestNode]:
    """Compiles a NODE, this is agnostic to the fact of whether the SQL is valid but is stringent on valid
    Jinja, therefore a None return value after catching the error affirms invalid jinja from user (which is fine since they might be compiling as they type)
    Caching here is immensely valuable since we know a NODE which hashes the same as a prior input will have the same output and this fact gives us a big speedup"""
    try:
        node = parsed.ParsedModelNode.from_dict(node.to_dict())
        ctx.dbt.update_node(node)
        with ctx.adapter.connection_named(f"dbt-osmosis"):
            compiled_node = ctx.adapter.get_compiler().compile_node(node, ctx.dbt)
        return compiled_node
    except CompilationException:
        return None


@st.experimental_singleton
def singleton_node_finder(model_name: str) -> Optional[manifest.ManifestNode]:
    """Finds a singleton node by name"""
    return ctx.dbt.ref_lookup.find(model_name, package=None, manifest=ctx.dbt)


# Singleton Base Node
st.session_state.setdefault((BASE_NODE := "BASE_NODE"), singleton_node_finder(args["model"]))

# Deepcopy a Node for Mutating
st.session_state.setdefault(
    (COMPILED_BASE_NODE := "COMPILED_BASE_NODE"), compile_model(st.session_state[BASE_NODE])
)
THIS, EXISTS = get_relation_if_exists(st.session_state[BASE_NODE])

# Deepcopy a Node for Mutating
st.session_state.setdefault((MUT_NODE := "MUT_NODE"), deepcopy(st.session_state[BASE_NODE]))

# Initial Compilation
st.session_state.setdefault(
    (COMPILED_MUT_NODE := "COMPILED_MUT_NODE"), compile_model(st.session_state[MUT_NODE])
)

# Raw Profiles So User Can Select
if "profiles" not in locals():
    profiles = get_raw_profiles(st.session_state[PROF_DIR])

# IDE Starting Text
if not locals().get("IDE"):
    IDE = st.session_state[MUT_NODE].raw_sql

st.title("dbt-osmosis 🌊")

st.sidebar.header("Profiles")
st.sidebar.write(
    "Select a profile used for materializing, compiling, and testing models. Can be updated at any time."
)
st.session_state[TARGET_PROFILE] = st.sidebar.radio(
    f"Loaded profiles from {ctx.project.profile_name}",
    [target for target in profiles[ctx.profile.profile_name].get("outputs", [])],
    key=PROFILE_SELECTOR,
)
st.sidebar.markdown(f"Current Target: **{st.session_state[TARGET_PROFILE]}**")
st.sidebar.write("")
st.sidebar.write("Utility")
st.sidebar.button("Reload dbt project", key=DBT_DO_RELOAD)
st.sidebar.caption(
    "Use this if any updated assets in your project have not yet reflected in the workbench, for example: you add some generic tests to some models while osmosis workbench is running."
)
st.sidebar.write("")
st.sidebar.selectbox("Editor Theme", THEMES, index=8, key=THEME_PICKER)
st.sidebar.selectbox("Editor Language", DIALECTS, key=DIALECT_PICKER)

# IDE LAYOUT
compileOptionContainer = st.container()
ideContainer = st.container()
if not st.session_state[PIVOT_LAYOUT]:
    idePart1, idePart2 = ideContainer.columns(2)
else:
    idePart1 = ideContainer.container()
    idePart2 = ideContainer.container()
controlsContainer = st.container()


with compileOptionContainer:
    st.write("")
    auto_compile = st.checkbox("Dynamic Compilation [Experimental]", key="auto_compile_flag")
    if auto_compile:
        st.caption("Compiling SQL on change")
    else:
        st.caption("Compiling SQL with control + enter")

with idePart1:
    IDE = st_ace(
        value=IDE,
        theme=st.session_state[THEME_PICKER],
        language=st.session_state[DIALECT_PICKER],
        auto_update=auto_compile,
        key="AceEditorGoGoGo",
    )

if st.session_state[MUT_NODE].raw_sql != IDE:
    st.session_state[
        MUT_NODE
    ].raw_sql = IDE  # update the st.session_state[MUT_NODE] with the new SQL
    st.session_state[COMPILED_MUT_NODE] = compile_model(st.session_state[MUT_NODE])

with idePart2:
    with st.expander("Compiled SQL", expanded=True):
        st.code(
            st.session_state[COMPILED_MUT_NODE].compiled_sql
            if st.session_state[COMPILED_MUT_NODE]
            else " -- Invalid Jinja, awaiting model to become valid",
            language="sql",
        )

with controlsContainer:
    pivot_layout_btn, build_model_btn, commit_changes_btn, revert_changes_btn = st.columns(4)
    with pivot_layout_btn:
        st.button("Pivot Layout", on_click=toggle_viewer)
    with build_model_btn:
        do_run_model = st.button(
            label=get_model_action_text(EXISTS),
        )
    with commit_changes_btn:
        if not st.session_state[MUT_NODE].same_body(st.session_state[BASE_NODE]):
            st.button(
                "Commit changes to file",
                on_click=save_model,
                kwargs={"node": st.session_state[MUT_NODE]},
            )
    with revert_changes_btn:
        if not st.session_state[MUT_NODE].same_body(st.session_state[BASE_NODE]):
            st.button(
                "Revert changes", on_click=revert_model, kwargs={"node": st.session_state[MUT_NODE]}
            )
    st.success("Model successfully loaded, started hacking away!")

    if do_run_model and st.session_state[COMPILED_MUT_NODE]:
        with st.spinner("Running model against target... ⚙️"):
            run_model(st.session_state[MUT_NODE])
        with st.spinner("Model ran against target! 🧑‍🏭"):
            time.sleep(2)
        do_run_model = False


# ----------------------------------------------------------------
# QUERY RESULT INSPECTOR
# ----------------------------------------------------------------

if ctx.profile.target_name != st.session_state[TARGET_PROFILE] or st.session_state[DBT_DO_RELOAD]:
    print("RELOADING DBT")
    inject_dbt()

st.stop()

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

SCHEMA_FILE = st.session_state[MAP].get(choice)
KNOWLEDGE = dbt_osmosis.main.pass_down_knowledge(
    dbt_osmosis.main.build_ancestor_tree(model, st.session_state[DBT].manifest),
    st.session_state[DBT].manifest,
)


def fix_documentation() -> bool:
    # Migrates model, injects model (requires model in database!)
    # Essentially the CLI `compose` task scoped to a single model
    schema_map = {choice: SCHEMA_FILE}
    was_restructured = dbt_osmosis.main.commit_project_restructure(
        dbt_osmosis.main.build_project_structure_update_plan(
            schema_map, st.session_state[DBT].manifest, st.session_state["adapter"]
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
                .compile_node(test_node, st.session_state[DBT].dbt)
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
for node_config in st.session_state[DBT].dbt.nodes.values():
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
