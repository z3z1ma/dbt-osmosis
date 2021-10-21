import streamlit as st
from streamlit_ace import st_ace
from streamlit_pandas_profiling import st_profile_report

from collections import OrderedDict
from typing import Sequence
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pandas_profiling

# The app does two things we do not scope ourselves to in the CLI:
# Build Models, Compile Models
# So include these imports here
from dbt.contracts.graph import parsed
from dbt.task.run import ModelRunner

import dbt_osmosis.main

# ----------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------

st.set_page_config(layout="wide")
st.title("dbt-osmosis 🌊")
profile_data = dbt_osmosis.main.get_raw_profiles()
if "target_profile" not in st.session_state:
    st.session_state["target_profile"] = None
if "profiles_dir" not in st.session_state:
    st.session_state["profiles_dir"] = str(dbt_osmosis.main.DEFAULT_PROFILES_DIR)
if "project_dir" not in st.session_state:
    st.session_state["project_dir"] = str(Path().cwd())
if "verified_project" not in st.session_state:
    st.session_state["verified_project"] = False
if "celebrate_good_times_oh_yeah" not in st.session_state:
    st.session_state["celebrate_good_times_oh_yeah"] = False
if "model_rev" not in st.session_state:
    st.session_state["model_rev"] = {}
if "this_sql" not in st.session_state:
    st.session_state["this_sql"] = {}
if "iter_state" not in st.session_state:
    st.session_state["iter_state"] = 0


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
    return True


# ----------------------------------------------------------------
# INIT STATE
# ----------------------------------------------------------------

if (
    "project" not in st.session_state
    or "profile" not in st.session_state
    or "config" not in st.session_state
    or "adapter" not in st.session_state
    or "manifest" not in st.session_state
):
    init_profile_val = st.session_state["profiles_dir"]
    init_project_val = st.session_state["project_dir"]
    config_container = st.empty()
    bootup_form = config_container.form("boot_up")
    st.session_state["profiles_dir"] = bootup_form.text_input(
        "Enter path to profiles.yml",
        value=init_profile_val,
        key="prof_select",
    )
    st.session_state["project_dir"] = bootup_form.text_input(
        "Enter path to dbt_project.yml",
        value=init_project_val,
        key="proj_select",
    )
    proceed = bootup_form.form_submit_button("Load Project")
    if not proceed:
        st.stop()
    config_container.empty()
    with st.spinner(text="Parsing profile and reading your dbt project 🦸"):
        LOADED = refresh_dbt()


# ----------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------

st.sidebar.header("Profiles")

st.sidebar.write(
    "Select a target profile used for materializing, compiling, and testing models. Can be updated at any time."
)
for prof in profile_data:
    st.session_state["target_profile"] = st.sidebar.radio(
        f"Loaded profiles from {prof}",
        [targ for targ in profile_data[prof]["outputs"]],
        key="profile_selector",
    )
    break
st.sidebar.markdown(f"Current Target: **{st.session_state['target_profile']}**")
if st.session_state["profile"].target_name != st.session_state["target_profile"]:
    refresh_dbt()

# ----------------------------------------------------------------
# MODEL IDE
# ----------------------------------------------------------------

st.write("")
st.subheader("Osmosis IDE 🛠️")

if "hide_viewer" not in st.session_state:
    st.session_state["hide_viewer"] = False


def toggle_viewer():
    st.session_state["hide_viewer"] = not st.session_state["hide_viewer"]


if "reset_model" not in st.session_state:
    st.session_state["reset_model"] = False


def toggle_reset():
    st.session_state["reset_model"] = not st.session_state["reset_model"]


# Model Selector For IDE
opts = []
for node, config in st.session_state["manifest"].flat_graph["nodes"].items():
    if config["resource_type"] == "model":
        opts.append(node)

with st.container():
    choice = st.selectbox("Select a model", options=opts)
    model = st.session_state["manifest"].flat_graph["nodes"].get(choice)

st.write("")
auto_compile = st.checkbox("Dynamic Compilation", key="auto_compile_flag")
if auto_compile:
    st.caption("Compiling SQL on change")
else:
    st.caption("Compiling SQL with control + enter")

if choice not in st.session_state["model_rev"]:
    # Here is our LTS
    st.session_state["model_rev"][choice] = model.copy()

if st.session_state["this_sql"].get(choice) is None:
    st.session_state["this_sql"] = {}
    st.session_state["sql_data"] = pd.DataFrame()
    st.session_state["iter_state"] += 1


def revert_model(target_model):
    target_model["raw_sql"] = st.session_state["model_rev"][choice]["raw_sql"]
    st.session_state["this_sql"][choice] = ""
    st.session_state["reset_model"] = True


def save_model(target_model):
    path = Path(target_model["root_path"]) / Path(target_model["original_file_path"])
    path.touch()
    with open(path, "w", encoding="utf-8") as f:
        f.write(target_model["raw_sql"])
    st.session_state["model_rev"][choice]["raw_sql"] = target_model["raw_sql"]


def build_model(target_node):
    runner = ModelRunner(st.session_state["config"], st.session_state["adapter"], target_node, 1, 1)
    runner.before_execute()
    result = runner.execute(target_node, st.session_state["manifest"])
    runner.after_execute(result)
    return st.session_state["adapter"].Relation.create_from(
        st.session_state["config"], target_node, type=target_node.config.materialized
    )


def model_action(exists: bool) -> str:
    if exists:
        return "Sync Database Model"
    else:
        return "Build Database Model"


def prepare_model(target_model):
    # Check for dbt.exceptions.FailedToConnectException(str(e)) here
    with st.session_state["adapter"].connection_named("dbt-osmosis"):
        table = st.session_state["adapter"].get_relation(
            target_model["database"], target_model["schema"], target_model["name"]
        )
    table_exists_in_db = table is not None
    columns_synced_with_db = False
    return table, table_exists_in_db, columns_synced_with_db


def get_model_sql(target_model):
    if st.session_state["reset_model"]:
        st.session_state["iter_state"] += 1
        st.session_state["reset_model"] = False
        return st.session_state["model_rev"][choice]["raw_sql"]
    return target_model["raw_sql"]


st.write("")
with st.container():
    table, table_exists_in_db, columns_synced_with_db = prepare_model(model)
    if not st.session_state["hide_viewer"]:
        editor, viewer = st.columns(2)
        with editor:
            st.button("Pivot Layout", on_click=toggle_viewer)
            with st.expander("Edit Model"):
                model["raw_sql"] = st_ace(
                    value=get_model_sql(model),
                    theme="twilight",
                    language="pgsql",
                    auto_update=auto_compile,
                    key=f"dbt_ide_{st.session_state['iter_state']}",
                )
                st.session_state["this_sql"][choice] = model["raw_sql"]
        unmodified_node = parsed.ParsedModelNode._deserialize(st.session_state["model_rev"][choice])
        node = parsed.ParsedModelNode._deserialize(model)
        st.session_state["manifest"].update_node(node)
        st.session_state["manifest"].build_flat_graph()
        with st.session_state["adapter"].connection_named("dbt-osmosis"):
            compiled_node = (
                st.session_state["adapter"]
                .get_compiler()
                .compile_node(node, st.session_state["manifest"])
            )
            if table_exists_in_db:
                database_columns = [
                    c.name for c in st.session_state["adapter"].get_columns_in_relation(table)
                ]
                columns_synced_with_db = True
            else:
                database_columns = list(model["columns"].keys())

        with viewer:
            if not node.same_body(unmodified_node):
                st.info("Uncommitted changes detected in model")
                st.button(
                    "Commit changes to file", on_click=save_model, kwargs={"target_model": model}
                )
                st.button("Revert changes", on_click=revert_model, kwargs={"target_model": model})
            st.button(
                label=model_action(table_exists_in_db),
                on_click=build_model,
                kwargs={"target_node": compiled_node},
            )
            with st.expander("View Compiled SQL"):
                st.code(compiled_node.compiled_sql, language="sql")

    else:
        show_compile_btn, build_model_btn, commit_changes_btn, revert_changes_btn = st.columns(4)
        with show_compile_btn:
            st.button("Pivot Layout", on_click=toggle_viewer)

        unmodified_node = parsed.ParsedModelNode._deserialize(st.session_state["model_rev"][choice])
        node = parsed.ParsedModelNode._deserialize(model)
        st.session_state["manifest"].update_node(node)
        with st.session_state["adapter"].connection_named("dbt-osmosis"):
            compiled_node = (
                st.session_state["adapter"]
                .get_compiler()
                .compile_node(node, st.session_state["manifest"])
            )
            if table_exists_in_db:
                database_columns = [
                    c.name for c in st.session_state["adapter"].get_columns_in_relation(table)
                ]
                columns_synced_with_db = True
            else:
                database_columns = list(model["columns"].keys())

        with build_model_btn:
            st.button(
                label=model_action(table_exists_in_db),
                on_click=build_model,
                kwargs={"target_node": compiled_node},
            )

        if not node.same_body(unmodified_node):
            st.info("Uncommitted changes detected in model")
            with commit_changes_btn:
                st.button(
                    "Commit changes to file", on_click=save_model, kwargs={"target_model": model}
                )
            with revert_changes_btn:
                st.button("Revert changes", on_click=revert_model, kwargs={"target_model": model})

        with st.expander("Edit Model"):
            model["raw_sql"] = st_ace(
                value=get_model_sql(model),
                theme="twilight",
                language="pgsql",
                auto_update=auto_compile,
                key=f"dbt_ide_full_{st.session_state['iter_state']}",
            )
            st.session_state["this_sql"][choice] = model["raw_sql"]

        with st.expander("View Compiled SQL"):
            st.code(compiled_node.compiled_sql, language="sql")


# ----------------------------------------------------------------
# QUERY RESULT INSPECTOR
# ----------------------------------------------------------------

if "sql_data" not in st.session_state:
    st.session_state["sql_data"] = pd.DataFrame()

if "sql_query_info" not in st.session_state:
    st.session_state["sql_query_info"] = ""

if "sql_query_mode" not in st.session_state:
    st.session_state["sql_query_mode"] = "test"


st.write("")
st.subheader("Osmosis Query Result Inspector 🔬")
st.write("")


def highlight_added(cell, color):
    return np.where(cell == "ADDED", f"color: {color};", None)


def highlight_removed(cell, color):
    return np.where(cell == "REMOVED", f"color: {color};", None)


def build_style(new, old):
    # Highlight added rows green
    style = new.where(new == old, "background-color: green")
    # Other cells as is
    style = style.where(new != old, "background-color: white")
    return style


@st.cache
def determine_modified_columns(modified_compiled_node) -> Sequence:
    """Leverage the database to determine output columns of a query vs reinventing the wheel
    with the, trust me, big complexities of ambiguous sql"""
    cache_materialization = modified_compiled_node.config.materialized
    cache_alias = modified_compiled_node.alias
    modified_compiled_node.config.materialized = "view"
    modified_compiled_node.alias = modified_compiled_node.alias + "__osmosis_temp"
    with st.session_state["adapter"].connection_named("dbt-osmosis"):
        temp_rel = build_model(modified_compiled_node)
        temp_database_columns = [
            c.name for c in st.session_state["adapter"].get_columns_in_relation(temp_rel)
        ]
    with st.session_state["adapter"].connection_named("dbt-osmosis-cleaner"):
        st.session_state["adapter"].drop_relation(temp_rel)
    modified_compiled_node.config.materialized = cache_materialization
    modified_compiled_node.alias = cache_alias
    return temp_database_columns


def test_query():
    if not table_exists_in_db:
        with st.warning("Table does not exist in database"):
            time.sleep(3)
        return

    with st.session_state["adapter"].connection_named("dbt-osmosis"):
        query = st.session_state["sql_data"] = st.session_state["adapter"].execute(
            f"select * from ({compiled_node.compiled_sql}) __all_data limit {limit}",
            fetch=True,
        )

    table = query[1]
    output = []
    json_funcs = [c.jsonify for c in table._column_types]
    for row in table._rows:
        values = tuple(json_funcs[i](d) for i, d in enumerate(row))
        output.append(OrderedDict(zip(row.keys(), values)))
    st.session_state["sql_data"] = pd.DataFrame(output)
    st.session_state["sql_query_info"] = query[0]
    st.session_state["sql_query_mode"] = "test"


def diff_query(primary_keys: Sequence, diff_compute_engine: str = "database"):
    if not table_exists_in_db:
        with st.warning("Table does not exist in database"):
            time.sleep(3)
        return

    # df1.merge(df2, on='Name', how='outer', suffixes=['', '_'], indicator=True)

    with st.session_state["adapter"].connection_named("dbt-osmosis"):
        compiled_unmodified_node = (
            st.session_state["adapter"]
            .get_compiler()
            .compile_node(unmodified_node, st.session_state["manifest"])
        )
        except_ = "except"
        if st.session_state["adapter"].type == "bigquery":
            except_ = "except distinct"
        compare_cols = ",".join(primary_keys)
        join_cond_added = ""
        join_cond_removed = ""
        for k in primary_keys:
            join_cond_added += f"unioned.{k} = new.{k} and "
            join_cond_removed += f"unioned.{k} = old.{k} and "
        else:
            join_cond_added = join_cond_added[:-4]
            join_cond_removed = join_cond_removed[:-4]
        compare_cols = ",".join(primary_keys)
        if diff_compute_engine == "database":
            query = st.session_state["sql_data"] = st.session_state["adapter"].execute(
                f"""
                with new as (
                    select * from ({compiled_node.compiled_sql}) AS __cte_1
                ), old as (
                    select * from  ({compiled_unmodified_node.compiled_sql}) AS __cte_2
                ), new_minus_old as (
                    select {compare_cols} from new
                    {except_}
                    select {compare_cols} from old
                ), old_minus_new as (
                    select {compare_cols} from old
                    {except_}
                    select {compare_cols} from new
                ), unioned as (
                    select 'ADDED' as __diff, * from new_minus_old
                    union all
                    select 'REMOVED' as __diff, * from old_minus_new
                )
                select __diff, new.*
                from new
                inner join unioned on ({join_cond_added})
                union all
                select __diff, old.*
                from old
                inner join unioned on ({join_cond_removed})
                """,
                fetch=True,
            )
            table = query[1]
            output = []
            json_funcs = [c.jsonify for c in table._column_types]
            for row in table._rows:
                values = tuple(json_funcs[i](d) for i, d in enumerate(row))
                output.append(OrderedDict(zip(row.keys(), values)))
            st.session_state["sql_data"] = pd.DataFrame(output)
            st.session_state["sql_query_info"] = query[0]
            st.session_state["sql_query_mode"] = "diff"
        elif diff_compute_engine == "pandas":
            ...
        else:
            raise NotImplemented("Only valid comput engines are SQL and Pandas")


if node.same_body(unmodified_node):
    col_test, col_select_1, _ = st.columns([1, 2, 2])

    with col_test:
        st.button("Test Compiled Query", on_click=test_query)
        st.caption("This will run the compiled SQL against your data warehouse")

    with col_select_1:
        limit = st.number_input(
            "Limit Results", min_value=1, value=2000, step=1, key="query_limiter"
        )

else:
    col_test, col_select_1, col_select_2 = st.columns([1, 1, 3])

    with col_test:
        st.button("Test Compiled Query", on_click=test_query)
        st.caption("This will run the compiled SQL against your data warehouse")

    with col_select_2:
        pk_opts = database_columns or list(model["columns"].keys())
        pks = st.multiselect(
            "Diff Primary Key(s)", pk_opts, default=[pk_opts[0]], key="pk_selector_m"
        )
        limit = st.number_input(
            "Limit Results", min_value=1, value=2000, step=1, key="query_limiter_m"
        )

    with col_test:
        st.button("Calculate Row Level Diff", on_click=diff_query, kwargs={"primary_keys": pks})
        st.caption("This will output the rows added or removed by changes made to the query")

    with col_select_1:
        diff_engine = st.selectbox(
            "Diff Compute Engine", ("SQL", "Python"), index=0, key="engine_selector"
        )

if not st.session_state["sql_data"].empty:
    if st.session_state["sql_query_mode"] == "test":
        st.write("Compiled SQL query results")
    elif st.session_state["sql_query_mode"] == "diff":
        st.write("Compiled SQL query row-level diff")
    st.write(st.session_state["sql_query_info"])
    st.dataframe(
        st.session_state["sql_data"]
        .style.apply(highlight_removed, color="red")
        .apply(highlight_added, color="green")
    )
else:
    st.write("")
    st.markdown(
        "> The results of your workbench query will show up here. Click `Test Compiled Query` to see the results. "
    )
    st.write("")


@st.cache
def convert_df_to_csv(sql_data):
    return sql_data.to_csv().encode("utf-8")


@st.cache(
    hash_funcs={
        pandas_profiling.report.presentation.core.container.Container: lambda _: compiled_node,
        pandas_profiling.report.presentation.core.html.HTML: lambda _: compiled_node,
    },
    allow_output_mutation=True,
)
def build_profile_report(sql_data: pd.DataFrame, minimal: bool):
    return sql_data.profile_report(minimal=minimal)


@st.cache(
    hash_funcs={
        pandas_profiling.report.presentation.core.container.Container: lambda _: compiled_node,
        pandas_profiling.report.presentation.core.html.HTML: lambda _: compiled_node,
    }
)
def convert_profile_report_to_html(profiled_data: pandas_profiling.ProfileReport) -> str:
    return profiled_data.to_html()


st.write("")
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

with st.expander("Edit documentation"):
    for column in database_columns:
        st.text_input(
            column,
            value=model["columns"]
            .get(column, {"description": "Not present in yaml"})
            .get("description", "Not documented")
            or "Not documented",
        )

# ----------------------------------------------------------------
# Data Test Runner
# ----------------------------------------------------------------

st.subheader("Test Runner")
st.write("Execute configured tests and validate results")

if "test_results" not in st.session_state:
    st.session_state["test_results"] = pd.DataFrame()

if "test_meta" not in st.session_state:
    st.session_state["test_meta"] = ""


def run_test(test_node):
    with st.session_state["adapter"].connection_named("dbt-osmosis-tester"):
        test_sql = getattr(test_node, "compiled_sql", None)
        if not test_sql:
            compiled_test = (
                st.session_state["adapter"]
                .get_compiler()
                .compile_node(test_node, st.session_state["manifest"])
            )
            test_sql = compiled_test.compiled_sql
        test_results = st.session_state["adapter"].execute(
            test_sql,
            fetch=True,
        )
    table = test_results[1]
    output = []
    json_funcs = [c.jsonify for c in table._column_types]
    for row in table._rows:
        values = tuple(json_funcs[i](d) for i, d in enumerate(row))
        output.append(OrderedDict(zip(row.keys(), values)))
    st.session_state["test_meta"] = test_results[0]
    st.session_state["test_results"] = pd.DataFrame(output)


test_opts = {}
for node, config in st.session_state["manifest"].nodes.items():
    if config.resource_type == "test" and choice in config.depends_on.nodes:
        test_opts[config.name] = config

test_pick_col, test_result_col, test_kpi_col = st.columns([2, 1, 1])

if test_opts:
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
    st.write(f"No tests found for model {choice}")

# ----------------------------------------------------------------
# MANIFEST INSPECTOR
# ----------------------------------------------------------------

st.subheader("Manifest Representation")
with st.expander("View Manifest"):
    st.json(model)
