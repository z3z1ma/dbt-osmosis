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
from dbt_osmosis.core.diff import diff_queries

st.set_page_config(page_title="dbt-osmosis Workbench", page_icon="🌊", layout="wide")

parser = argparse.ArgumentParser(description="dbt osmosis workbench")
parser.add_argument("--profiles-dir", help="dbt profile directory")
parser.add_argument("--project-dir", help="dbt project directory")
parser.add_argument("-m", "--model", required=True, help="dbt model")
args = vars(parser.parse_args(sys.argv[1:]))

# GLOBAL STATE VARS
DBT = "DBT"
MAP = "MAP"
st.session_state.setdefault((PROJ_DIR := "PROJ_DIR"), args["project_dir"] or str(Path.cwd()))
st.session_state.setdefault((PROF_DIR := "PROF_DIR"), args["profiles_dir"] or DEFAULT_PROFILES_DIR)
st.session_state.setdefault((SQL_RESULT := "SQL_RESULT"), pd.DataFrame())
st.session_state.setdefault((SQL_RESP_INFO := "SQL_RESP_INFO"), None)
st.session_state.setdefault((SQL_QUERY_MODE := "SQL_QUERY_MODE"), "test")
st.session_state.setdefault((MODEL_NAME := "MODEL_NAME"), args["model"])

# COMPONENT KEYS
PROFILE_SELECTOR = "PROFILE_SELECTOR"
THEME_PICKER = "THEME_PICKER"
DIALECT_PICKER = "DIALECT_PICKER"
QUERY_LIMITER = "QUERY_LIMITER"
DIFF_PK_SELECTOR = "DIFF_PK_SELECTOR"
DIFF_AGG_CHECKBOX = "DIFF_AGG_CHECKBOX"
BASIC_PROFILE_OPT = "BASIC_PROFILE_OPT"
PROFILE_DOWNLOADER = "PROFILE_DOWNLOADER"
DYNAMIC_COMPILATION = "DYNAMIC_COMPILATION"

# COMPONENT OPTIONS
DIALECTS = ("pgsql", "mysql", "sql", "sqlserver")

# TRIGGERS
st.session_state.setdefault((EDITOR_STATE_ITER_1 := "EDITOR_STATE_ITER_1"), 1)
DBT_DO_RELOAD = "DBT_DO_RELOAD"
st.session_state.setdefault((PIVOT_LAYOUT := "PIVOT_LAYOUT"), False)
st.session_state.setdefault((FAILED_RELATION := "FAILED_RELATION"), None)
st.session_state.setdefault((REVERT_MODEL := "REVERT_MODEL"), False)
DO_PROFILE = "DO_PROFILE"


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
        return "▶️ Run Model"
    else:
        return "▶️ Run Model"


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


def _highlight_string(cell: Any, string: str, color: str):
    return np.where(cell == string, f"color: {color};", None)


def _build_style(new: pd.DataFrame, old: pd.DataFrame):
    return new.where(
        new == old,
        "background-color: green",
    ).where(new != old, "background-color: white")


@st.cache(hash_funcs=hash_funcs)
def get_column_set(select_stmt: str) -> set:
    """Adapted from dbt get_columns_in_query macro, return a set of the columns that would be returned from a SQL select statement"""
    try:
        with ctx.adapter.connection_named("dbt-osmosis"):
            columns = set(
                col.name
                for col in ctx.adapter.execute(
                    f"select * from ({select_stmt}) as __dbt_sbq where false limit 0",
                    auto_begin=True,
                    fetch=True,
                )[1].columns
            )
    except DatabaseException:
        columns = set()
    return columns


def test_query(node: manifest.ManifestNode, limit: int = 2000) -> None:
    """This function queries the database using the editor SQL and stores results in the state variables:
    sql_data which should always contain a DataFrame
    sql_query_info which contains an adapter response object or empty str, a truthy value should always indicate a valid reponse
    sql_query_mode is set to "test" which acts as a switch for the components rendered post function execution
    """

    with ctx.adapter.connection_named("dbt-osmosis"):
        query = ctx.adapter.execute(
            f"select * from ({node.compiled_sql}) as __all_data limit {limit}",
            fetch=True,
        )

    table = query[1]
    output = []
    json_funcs = [c.jsonify for c in table._column_types]
    for row in table._rows:
        values = tuple(json_funcs[i](d) for i, d in enumerate(row))
        output.append(OrderedDict(zip(row.keys(), values)))

    st.session_state[SQL_RESULT] = pd.DataFrame(output)
    st.session_state[SQL_RESP_INFO] = query[0]
    st.session_state[SQL_QUERY_MODE] = "test"


@st.cache
def convert_df_to_csv(dataframe: pd.DataFrame):
    return dataframe.to_csv().encode("utf-8")


@st.cache(
    hash_funcs={
        pandas_profiling.report.presentation.core.container.Container: lambda _: st.session_state[
            COMPILED_MUT_NODE
        ]
        or st.session_state[MUT_NODE],
        pandas_profiling.report.presentation.core.html.HTML: lambda _: st.session_state[
            COMPILED_MUT_NODE
        ]
        or st.session_state[MUT_NODE],
    },
    allow_output_mutation=True,
)
def build_profile_report(
    dataframe: pd.DataFrame, minimal: bool = True
) -> pandas_profiling.ProfileReport:
    return dataframe.profile_report(minimal=minimal)


@st.cache(
    hash_funcs={
        pandas_profiling.report.presentation.core.container.Container: lambda _: st.session_state[
            COMPILED_MUT_NODE
        ]
        or st.session_state[MUT_NODE],
        pandas_profiling.report.presentation.core.html.HTML: lambda _: st.session_state[
            COMPILED_MUT_NODE
        ]
        or st.session_state[MUT_NODE],
    },
    allow_output_mutation=True,
)
def convert_profile_report_to_html(profile: pandas_profiling.ProfileReport) -> str:
    return profile.to_html()


# Singleton Base Node
st.session_state.setdefault((BASE_NODE := "BASE_NODE"), singleton_node_finder(args["model"]))

# Deepcopy a Node for Mutating
st.session_state.setdefault(
    (COMPILED_BASE_NODE := "COMPILED_BASE_NODE"), compile_model(st.session_state[BASE_NODE])
)
st.session_state.setdefault(
    (BASE_COLUMNS := "BASE_COLUMNS"),
    get_column_set(st.session_state[COMPILED_BASE_NODE].compiled_sql),
)
st.session_state.setdefault(
    (EXISTS := "EXISTS"), get_relation_if_exists(st.session_state[BASE_NODE])
)

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
    auto_compile = st.checkbox("Dynamic Compilation", key=DYNAMIC_COMPILATION, value=True)
    if auto_compile:
        st.caption("👉 Compiling SQL on change")
    else:
        st.caption("👉 Compiling SQL with control + enter")

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
    with st.expander("📝 Compiled SQL", expanded=True):
        st.code(
            st.session_state[COMPILED_MUT_NODE].compiled_sql
            if st.session_state[COMPILED_MUT_NODE]
            else " -- Invalid Jinja, awaiting model to become valid",
            language="sql",
        )

with controlsContainer:
    pivot_layout_btn, build_model_btn, commit_changes_btn, revert_changes_btn = st.columns(
        [3, 1, 1, 1]
    )
    with pivot_layout_btn:
        st.button("📏 Pivot Layout", on_click=toggle_viewer)
    with build_model_btn:
        do_run_model = st.button(
            label=get_model_action_text(st.session_state[EXISTS]),
        )
    with commit_changes_btn:
        if not st.session_state[MUT_NODE].same_body(st.session_state[BASE_NODE]):
            st.button(
                "💾 Save Changes",
                on_click=save_model,
                kwargs={"node": st.session_state[MUT_NODE]},
            )
    with revert_changes_btn:
        if not st.session_state[MUT_NODE].same_body(st.session_state[BASE_NODE]):
            st.button(
                "🔙 Revert changes",
                on_click=revert_model,
                kwargs={"node": st.session_state[MUT_NODE]},
            )

    if do_run_model and st.session_state[COMPILED_MUT_NODE]:
        with st.spinner("Running model against target... ⚙️"):
            run_model(st.session_state[MUT_NODE])
        with st.spinner("Model ran against target! 🧑‍🏭"):
            time.sleep(2)
        do_run_model = False

if ctx.profile.target_name != st.session_state[TARGET_PROFILE] or st.session_state[DBT_DO_RELOAD]:
    print("RELOADING DBT")
    inject_dbt()

# TEST LAYOUT
testHeaderContainer = st.container()
test_column_1, test_column_2, test_column_3 = st.columns([1, 1, 3])
testContainer = st.container()
downloadBtnContainer, profileBtnContainer, profileOptContainer = st.columns([1, 1, 3])
profilerContainer = st.container()

with testHeaderContainer:
    st.write("")
    st.subheader("Osmosis Query Result Inspector 🔬")
    st.write("")

query_limit = test_column_3.number_input(
    "Limit Results", min_value=1, max_value=50_000, value=2_000, step=1, key=QUERY_LIMITER
)
if st.session_state[COMPILED_MUT_NODE]:
    test_column_1.button(
        "Test Compiled Query",
        on_click=test_query,
        kwargs={"node": st.session_state[COMPILED_MUT_NODE], "limit": query_limit},
    )
    test_column_1.caption("This will run the compiled SQL against your data warehouse")

projected_columns = get_column_set(st.session_state[COMPILED_MUT_NODE].compiled_sql)
pk_options = list(projected_columns.intersection(st.session_state[BASE_COLUMNS]))
diff_cols = test_column_3.multiselect(
    "Diff Primary Key(s)",
    pk_options,
    default=[pk_options[0]] if pk_options else [],
    key=DIFF_PK_SELECTOR,
)
agg_diff_results = test_column_2.checkbox("Aggregate Diff Results", key=DIFF_AGG_CHECKBOX)
test_column_2.caption("Use this to get aggregate diff results when data does not fit in memory")
test_column_1.button(
    "Calculate Row Level Diff",
    on_click=lambda **_: None,
    kwargs={
        "primary_keys": diff_cols,
        "aggregate_result": agg_diff_results,
        "prev_cols": st.session_state[BASE_COLUMNS],
        "curr_cols": projected_columns,
    },
)
test_column_1.caption(
    "(⚠️ being refactored) This will output the rows added or removed by changes made to the query"
)

with testContainer:
    st.write("\n\n\n\n\n")
    if st.session_state[SQL_QUERY_MODE] == "test":
        st.write("Compiled SQL query results")
    elif st.session_state[SQL_QUERY_MODE] == "diff":
        st.write("Compiled SQL query row-level diff")
    if not st.session_state[SQL_RESULT].empty:
        st.write(st.session_state[SQL_RESP_INFO])
        st.dataframe(
            st.session_state[SQL_RESULT]
            .style.apply(
                _highlight_string,
                string="REMOVED",
                color="red",
            )
            .apply(
                _highlight_string,
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
    st.write("")

with downloadBtnContainer:
    st.download_button(
        label="Download data as CSV",
        data=convert_df_to_csv(st.session_state[SQL_RESULT]),
        file_name=f"{st.session_state[MODEL_NAME]}.csv",
        mime="text/csv",
    )

with profileBtnContainer:
    st.button("Profile Data", key=DO_PROFILE)

with profileOptContainer:
    st.checkbox("Basic Profiler", key=BASIC_PROFILE_OPT, value=True)
    st.caption(
        "Useful for larger datasets, use the minimal pandas-profiling option for a simpler report"
    )

if st.session_state[DO_PROFILE]:
    pr = build_profile_report(st.session_state[SQL_RESULT], st.session_state[BASIC_PROFILE_OPT])
    with profilerContainer:
        st_profile_report(pr, height=650)
        st.download_button(
            label="Download profile report",
            data=convert_profile_report_to_html(pr),
            file_name=f"{st.session_state[MODEL_NAME]}_profile.html",
            mime="text/html",
            key=PROFILE_DOWNLOADER,
        )
        st.write("")

manifestViewer = st.container()
with manifestViewer:
    st.write("")
    st.subheader("Manifest Representation")
    st.json(st.session_state[MUT_NODE].to_dict(), expanded=False)

st.stop()

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
