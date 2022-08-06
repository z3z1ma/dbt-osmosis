import argparse
import hashlib
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import pandas_profiling
import streamlit as st
from dbt.contracts.graph import compiled, manifest, parsed
from dbt.exceptions import CompilationException, DatabaseException, RuntimeException
from streamlit_ace import THEMES, st_ace
from streamlit_pandas_profiling import st_profile_report

from dbt_osmosis.core.osmosis import DEFAULT_PROFILES_DIR, DbtOsmosis, get_raw_profiles

st.set_page_config(page_title="dbt-osmosis Workbench", page_icon="🌊", layout="wide")
state = st.session_state

try:
    parser = argparse.ArgumentParser(description="dbt osmosis workbench")
    parser.add_argument("--profiles-dir", help="dbt profile directory")
    parser.add_argument("--project-dir", help="dbt project directory")
    args = vars(parser.parse_args(sys.argv[1:]))
except:
    args = {}

root_path = Path(__file__).parent
demo_dir = root_path / "demo"

# GLOBAL STATE VARS
DBT = "DBT"
"""DbtOsmosis object"""
PROJ_DIR = "PROJ_DIR"
"""dbt project directory"""
PROF_DIR = "PROJ_DIR"
"""dbt profile directory"""

_proj_dir = args.get("project_dir")
state.setdefault(PROJ_DIR, _proj_dir or os.getenv("DBT_PROJECT_DIR", str(Path.cwd())))
_prof_dir = args.get("profiles_dir")
state.setdefault(PROF_DIR, _prof_dir or os.getenv("DBT_PROFILES_DIR", DEFAULT_PROFILES_DIR))


RAW_PROFILES = "RAW_PROFILES"
"""All profiles as parsed from raw profiles yaml"""
state.setdefault(RAW_PROFILES, get_raw_profiles(state[PROF_DIR]))

# SQL WORKBENCH VARS
SQL_RESULT = "SQL_RESULT"
"""SQL result as a pandas dataframe"""
SQL_ADAPTER_RESP = "SQL_ADAPTER_RESP"
"""Adapter response from dbt"""
SQL_QUERY_STATE = "SQL_QUERY_STATE"
"""SQL query state tracking if it is successful or failed"""

state.setdefault(SQL_RESULT, pd.DataFrame())
state.setdefault(SQL_ADAPTER_RESP, None)
state.setdefault(SQL_QUERY_STATE, "test")

# PRIMARY SQL CONTAINERS
COMPILED_SQL = "COMPILED_SQL"
"""Compiled sql container"""
state.setdefault(COMPILED_SQL, "")
RAW_SQL = "RAW_SQL"
"""Raw sql container"""

if "demo" in state[PROJ_DIR]:
    state.setdefault(
        RAW_SQL,
        """
{% set payment_methods = ['credit_card', 'coupon', 'bank_transfer', 'gift_card'] %}

with orders as (

    select * from {{ ref('stg_orders') }}

),

payments as (

    select * from {{ ref('stg_payments') }}

),

order_payments as (

    select
        order_id,

        {% for payment_method in payment_methods -%}
        sum(case when payment_method = '{{ payment_method }}' then amount else 0 end) as {{ payment_method }}_amount,
        {% endfor -%}

        sum(amount) as total_amount

    from payments

    group by order_id

),

final as (

    select
        orders.order_id,
        orders.customer_id,
        orders.order_date,
        orders.status,

        {% for payment_method in payment_methods -%}

        order_payments.{{ payment_method }}_amount,

        {% endfor -%}

        order_payments.total_amount as amount

    from orders


    left join order_payments
        on orders.order_id = order_payments.order_id

)

select * from final
    """,
    )
else:
    state.setdefault(RAW_SQL, "")

# COMPONENT KEYS
PROFILE_SELECTOR = "PROFILE_SELECTOR"
"""Selected profile"""
THEME_PICKER = "THEME_PICKER"
"""Selected theme for workbench"""
DIALECT_PICKER = "DIALECT_PICKER"
"""Selected SQL dialect for workbench"""
QUERY_LIMITER = "QUERY_LIMITER"
"""Limit results returned in SQL runner"""
BASIC_PROFILE_OPT = "BASIC_PROFILE_OPT"
"""Use basic profiling for pandas-profiling"""
PROFILE_DOWNLOADER = "PROFILE_DOWNLOADER"
"""Controller for downloading HTML results of pandas-profiler"""
DYNAMIC_COMPILATION = "DYNAMIC_COMPILATION"
"""Toggle to compile on-type or compile on control+enter"""

# COMPONENT OPTIONS
DIALECTS = ("pgsql", "mysql", "sql", "sqlserver")
"""Tuple of SQL dialects usable in ace editor"""

# TRIGGERS
DBT_DO_RELOAD = "DBT_DO_RELOAD"
"""This triggers dbt to reparse the project"""
RUN_PROFILER = "RUN_PROFILER"
"""Run pandas profiler on test bench result set"""
PIVOT_LAYOUT = "PIVOT_LAYOUT"
"""Pivot the editor layout from side-by-side to top-bottom"""

state.setdefault(PIVOT_LAYOUT, False)


def hash_parsed_node(node: parsed.ParsedModelNode) -> str:
    return hashlib.md5(node.raw_sql.encode("utf-8")).hexdigest()


def hash_compiled_node(node: parsed.ParsedModelNode):
    return hashlib.md5(node.raw_sql.encode("utf-8")).hexdigest()


hash_funcs = {
    parsed.ParsedModelNode: hash_parsed_node,
    compiled.CompiledModelNode: lambda _: None,
    manifest.Manifest: lambda _: None,
    DbtOsmosis: lambda _: None,
}


def inject_dbt(change_target: Optional[str] = None):
    """Parse dbt project and load context var"""
    if DBT not in state:
        dbt_ctx = DbtOsmosis(
            project_dir=state[PROJ_DIR],
            profiles_dir=state[PROF_DIR],
            target=change_target,
        )
    else:
        dbt_ctx: DbtOsmosis = state[DBT]
        dbt_ctx.rebuild_dbt_manifest(reset=True)

    state[DBT] = dbt_ctx
    return True


if DBT not in state:
    inject_dbt()
ctx: DbtOsmosis = state[DBT]

TARGET_PROFILE = "TARGET_PROFILE"
"""Target profile for dbt to execute against"""

state.setdefault(TARGET_PROFILE, ctx.profile.target_name)


def toggle_viewer() -> None:
    state[PIVOT_LAYOUT] = not state[PIVOT_LAYOUT]


@st.cache(hash_funcs=hash_funcs)
def compile_model(node: manifest.ManifestNode) -> Optional[manifest.ManifestNode]:
    """Compiles a NODE, this is agnostic to the fact of whether the SQL is valid but is stringent on valid
    Jinja, therefore a None return value after catching the error affirms invalid jinja from user (which is fine since they might be compiling as they type)
    Caching here is immensely valuable since we know a NODE which hashes the same as a prior input will have the same output and this fact gives us a big speedup"""
    try:
        node = parsed.ParsedModelNode.from_dict(node.to_dict())
        ctx.dbt.update_node(node)
        with ctx.adapter.connection_named("dbt-osmosis"):
            compiled_node = ctx.adapter.get_compiler().compile_node(node, ctx.dbt)
        return compiled_node
    except CompilationException:
        return None


# @st.cache
def compile_sql(sql: str) -> str:
    try:
        return ctx.compile_sql(sql).compiled_sql
    except CompilationException:
        return None


def run_query(sql: str, limit: int = 2000) -> None:
    try:
        resp, table = ctx.execute_sql(
            f"select * from ({sql}) as __all_data limit {limit}", compile=True, fetch=True
        )
    except DatabaseException as error:
        state[SQL_QUERY_STATE] = "error"
        state[SQL_ADAPTER_RESP] = str(error)
    else:
        output = []
        json_funcs = [c.jsonify for c in table._column_types]
        for row in table._rows:
            values = tuple(json_funcs[i](d) for i, d in enumerate(row))
            output.append(OrderedDict(zip(row.keys(), values)))

        state[SQL_RESULT] = pd.DataFrame(output)
        state[SQL_ADAPTER_RESP] = resp
        state[SQL_QUERY_STATE] = "success"


@st.cache
def convert_df_to_csv(dataframe: pd.DataFrame):
    return dataframe.to_csv().encode("utf-8")


@st.cache(
    hash_funcs={
        pandas_profiling.report.presentation.core.container.Container: lambda _: state[
            COMPILED_SQL
        ],
        pandas_profiling.report.presentation.core.html.HTML: lambda _: state[COMPILED_SQL],
    },
    allow_output_mutation=True,
)
def build_profile_report(
    dataframe: pd.DataFrame, minimal: bool = True
) -> pandas_profiling.ProfileReport:
    return dataframe.profile_report(minimal=minimal)


@st.cache(
    hash_funcs={
        pandas_profiling.report.presentation.core.container.Container: lambda _: state[
            COMPILED_SQL
        ],
        pandas_profiling.report.presentation.core.html.HTML: lambda _: state[COMPILED_SQL],
    },
    allow_output_mutation=True,
)
def convert_profile_report_to_html(profile: pandas_profiling.ProfileReport) -> str:
    return profile.to_html()


st.title("dbt-osmosis 🌊")

st.sidebar.header("Profiles")
st.sidebar.write(
    "Select a profile used for materializing, compiling, and testing models. Can be updated at any time."
)
state[TARGET_PROFILE] = st.sidebar.radio(
    f"Loaded profiles from {ctx.project.profile_name}",
    [target for target in state[RAW_PROFILES][ctx.profile.profile_name].get("outputs", [])],
    key=PROFILE_SELECTOR,
)
st.sidebar.markdown(f"Current Target: **{state[TARGET_PROFILE]}**")
st.sidebar.write("")
st.sidebar.write("Utility")
# st.sidebar.button("Reload dbt project", key=DBT_DO_RELOAD)
st.sidebar.caption(
    "Refresh the page to reparse dbt. This is useful if any updated models or macros in your physical project \
    on disk have changed and are not yet reflected in the workbench as refable or updated."
)
st.sidebar.write("")
st.sidebar.selectbox("Editor Theme", THEMES, index=8, key=THEME_PICKER)
st.sidebar.selectbox("Editor Language", DIALECTS, key=DIALECT_PICKER)

# IDE LAYOUT
notificationContainer = st.empty()
descriptionContainer = st.container()
compileOptionContainer = st.container()
ideContainer = st.container()

descriptionContainer.markdown(
    """
Welcome to the [dbt-osmosis](https://github.com/z3z1ma/dbt-osmosis) workbench 👋. 
The workbench serves as a no fuss way to spin up 
an environment where you can very quickly iterate on dbt models. In an ideal flow, a developer
can spin up the workbench and use it as a _complement_ to their IDE, not a replacement. This means
copying and pasting over a model you are really digging into 🧑‍💻 OR it is just as valid to use 
the workbench as a scratchpad 👷‍♀️. In a full day of development, you may never spin down the workbench.
Refreshing the page is enough to reparse the physical dbt project on disk. The instantaneous feedback
rarely experienced with jinja + ability to execute the SQL both synergize to supercharge ⚡️ productivity!
"""
)

if not state[PIVOT_LAYOUT]:
    idePart1, idePart2 = ideContainer.columns(2)
else:
    idePart1 = ideContainer.container()
    idePart2 = ideContainer.container()


compileOptionContainer.write("")
compileOpt1, compileOpt2 = compileOptionContainer.columns(2)
auto_update = compileOpt1.checkbox("Dynamic Compilation", key=DYNAMIC_COMPILATION, value=True)
if auto_update:
    compileOpt1.caption("👉 Compiling SQL on change")
else:
    compileOpt1.caption("👉 Compiling SQL with control + enter")
compileOpt2.button("Pivot Layout", on_click=toggle_viewer)

with idePart1:
    state[RAW_SQL] = st_ace(
        value=state[RAW_SQL],
        theme=state[THEME_PICKER],
        language=state[DIALECT_PICKER],
        auto_update=auto_update,
        key=f"AceEditor",
        max_lines=35,
        min_lines=20,
        height=500,
    )

with idePart2:
    with st.expander("📝 Compiled SQL", expanded=True):
        st.code(
            state[COMPILED_SQL]
            if state[COMPILED_SQL]
            else " --> Invalid Jinja, awaiting model to become valid",
            language="sql",
        )

if compile_sql(state[RAW_SQL]) != state[COMPILED_SQL]:
    state[COMPILED_SQL] = compile_sql(state[RAW_SQL])
    st.experimental_rerun()  # This eager re-run speeds up the app


if ctx.profile.target_name != state[TARGET_PROFILE]:  # or state[DBT_DO_RELOAD]:
    # TODO: vet the effectiveness of this re-injection
    print("Reloading dbt project...")
    with notificationContainer:
        ctx.profile.target_name = state[TARGET_PROFILE]
        ctx.config.target_name = state[TARGET_PROFILE]
        with st.spinner("Reloading dbt... ⚙️"):
            inject_dbt(state[TARGET_PROFILE])
            state[COMPILED_SQL] = compile_sql(state[RAW_SQL])
    st.experimental_rerun()


# TEST LAYOUT
testHeaderContainer = st.container()
test_column_1, _, test_column_2 = st.columns([1, 2, 1])
testContainer = st.container()
testContainerViewer = testContainer.expander("Result Viewer 🔎", expanded=True)
test_view_1, _, test_view_2 = testContainerViewer.columns([1, 2, 1])

downloadBtnContainer, profileBtnContainer, profileOptContainer = st.columns([1, 1, 3])
profilerContainer = st.container()

with testHeaderContainer:
    st.write("")
    st.subheader("Osmosis Query Result Inspector 🔬")
    st.write("")
    st.markdown(
        """Run queries against your datawarehouse leveraging the selected target profile. This is a critical step in
    developer productivity 📈 and dbt-osmosis workbench aims to keep it a click away. Additionally, you can leverage the 
    profiling functionality to get an idea of the dataset you have in memory."""
    ),
    st.write(""), st.write("")

query_limit = test_column_2.number_input(
    "Limit Results", min_value=1, max_value=50_000, value=2_000, step=1, key=QUERY_LIMITER
)
test_column_2.caption(
    "Limit the number of results returned by the query, the maximum value is 50,000"
)

if state[COMPILED_SQL]:
    test_column_1.button(
        "Test Compiled Query",
        on_click=run_query,
        kwargs={"sql": state[COMPILED_SQL], "limit": query_limit},
    )
    test_column_1.caption("This will run the compiled SQL against your data warehouse")

with testContainerViewer:
    st.write("\n\n\n\n\n")

    if state[SQL_QUERY_STATE] == "success":
        test_view_1.write("#### Compiled SQL query results")
    elif state[SQL_QUERY_STATE] == "error":
        test_view_1.warning(f"SQL query error: {state[SQL_ADAPTER_RESP]}")
    if not state[SQL_RESULT].empty:
        test_view_2.info(f"Adapter Response: {state[SQL_ADAPTER_RESP]}")
        st.dataframe(state[SQL_RESULT])
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
        data=convert_df_to_csv(state[SQL_RESULT]),
        file_name=f"dbt_osmosis_workbench.csv",
        mime="text/csv",
    )

with profileBtnContainer:
    st.button("Profile Data", key=RUN_PROFILER)

with profileOptContainer:
    st.checkbox("Basic Profiler", key=BASIC_PROFILE_OPT, value=True)
    st.caption(
        "Useful for larger datasets, use the minimal pandas-profiling option for a simpler report"
    )

if state[RUN_PROFILER]:
    pr = build_profile_report(state[SQL_RESULT], state[BASIC_PROFILE_OPT])
    with profilerContainer:
        st_profile_report(pr, height=650)
        st.download_button(
            label="Download profile report",
            data=convert_profile_report_to_html(pr),
            file_name=f"dbt_osmosis_workbench_profile.html",
            mime="text/html",
            key=PROFILE_DOWNLOADER,
        )
        st.write("")

st.write(""), st.write("")
st.header("Useful Links 🧐")
st.write("")
st.markdown(
    """
##### dbt docs
- [docs.getdbt.com](https://docs.getdbt.com/)

##### dbt core repo
- [github.com/dbt-labs/dbt-core](https://github.com/dbt-labs/dbt-core/)

##### data team reference material

- [Gitlab Data Team Wiki](https://about.gitlab.com/handbook/business-technology/data-team/)
- [dbt Best Practices](https://docs.getdbt.com/guides/best-practices/how-we-structure/1-guide-overview)

"""
)
