import argparse
import decimal
import os
import sys
import time
from collections import OrderedDict
from datetime import datetime, date
from textwrap import dedent
from types import SimpleNamespace

import dbt.config.profile as dbt_profile
import pandas as pd
import ydata_profiling
import streamlit as st
from streamlit import session_state as state
from streamlit_elements_fluence import elements, event, sync

from dbt_osmosis.components.editor import Editor, Tabs as EditorTabs
from dbt_osmosis.components.dashboard import Dashboard
from dbt_osmosis.components.preview import Preview
from dbt_osmosis.components.profiler import Profiler
from dbt_osmosis.components.renderer import Renderer
from dbt_osmosis.components.feed import RssFeed
from dbt_osmosis.vendored.dbt_core_interface import (
    DbtProject,
    default_profiles_dir,
    default_project_dir,
)

default_prompt = (
    "-- This is a scratch model\n-- it will not persist if you jump to another model\n-- you can"
    " use this to test your dbt SQL queries\n\nselect 1 as id, 'hello' as name"
)


def _get_demo_query() -> str:
    return dedent("""
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
            sum(case when payment_method = '{{ payment_method }}' then amount else 0 end) 
            as {{ payment_method }}_amount,
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
        """)


def _parse_args() -> dict:
    """Parse command line arguments"""
    try:
        parser = argparse.ArgumentParser(description="dbt osmosis workbench")
        parser.add_argument("--profiles-dir", help="dbt profile directory")
        parser.add_argument("--project-dir", help="dbt project directory")
        args = vars(parser.parse_args(sys.argv[1:]))
    except Exception:
        args = {}
    return args


def change_target() -> None:
    """Change the target profile"""
    ctx: DbtProject = state.w.ctx
    if ctx.config.target_name != state.target_profile:
        print(f"Changing target to {state.target_profile}")
        ctx.base_config.target = state.target_profile
        ctx.safe_parse_project(reinit=True)
        state.w.raw_sql += " "  # invalidate cache on next compile?
        state.w.cache_version += 1


def inject_model() -> None:
    """Inject model into editor"""
    ctx: DbtProject = state.w.ctx
    if state.model is not None and state.model != "SCRATCH":
        path = os.path.join(ctx.project_root, state.model.original_file_path)
        with open(path, "r") as f:
            state.w.raw_sql = f.read()
        state.w.editor.update_content("SQL", state.w.raw_sql)
    elif state.model == "SCRATCH":
        state.w.editor.update_content("SQL", default_prompt)
    state.w.cache_version += 1


def save_model() -> None:
    """Save model to disk"""
    ctx: DbtProject = state.w.ctx
    if state.model is not None and state.model != "SCRATCH":
        path = os.path.join(ctx.project_root, state.model.original_file_path)
        with open(path, "w") as f:
            f.write(state.w.editor.get_content("SQL"))
        print(f"Saved model to {path}")


def sidebar(ctx: DbtProject) -> None:
    # Model selector
    with st.sidebar.expander("💡 Models", expanded=True):
        st.caption(
            "Select a model to use as a starting point for your query. The filter supports"
            " typeahead. All changes are ephemeral unless you save the model."
        )
        state.w.model = st.selectbox(
            "Select a model",
            options=state.w.model_opts,
            format_func=lambda x: getattr(x, "name", x),
            on_change=inject_model,
            key="model",
        )
        btn1, btn2 = st.columns(2)
        if state.w.model != "SCRATCH":
            btn1.button("💾 - Save", on_click=save_model, key="save_model")
        btn2.button("⏮ - Revert", on_click=inject_model, key="reset_model")

    # Profile selector
    with st.sidebar.expander("💁 Profiles", expanded=True):
        st.caption(
            "Select a profile used for materializing, compiling, and testing models.\n\nIf you"
            " change profiles, you may need to modify the workbench query to invalidate the cache."
        )
        state.w.target_profile = st.radio(
            f"Loaded profiles from {ctx.config.profile_name}",
            [target for target in state.w.raw_profiles[ctx.config.profile_name].get("outputs", [])],
            on_change=change_target,
            key="target_profile",
        )
        st.markdown(f"Current Target: **{state.w.target_profile}**")

    # Query template
    with st.sidebar.expander("📝 Query Template"):
        st.caption(
            "This is a template query that will be used when executing SQL. The {sql} variable will"
            " be replaced with the compiled SQL."
        )
        state.w.sql_template = st.text_area(
            "SQL Template",
            value=state.w.sql_template,
            height=100,
        )

    # Refresh instructions
    st.sidebar.write("Notes")
    st.sidebar.caption(
        "Refresh the page to reparse dbt. This is useful if any updated models or macros in your"
        " physical project     on disk have changed and are not yet reflected in the workbench as"
        " refable or updated."
    )


def ttl_cache(wait):
    """Debounce a function call."""

    def decorator(func):
        last_called = time.time()
        cached_result = None

        def cached(*args, **kwargs):
            nonlocal last_called, cached_result
            elapsed = time.time() - last_called
            if elapsed < wait and cached_result is not None:
                return cached_result
            cached_result = func(*args, **kwargs)
            last_called = time.time()
            return cached_result

        return cached

    return decorator


@ttl_cache(3.0)
def compile(ctx: DbtProject, sql: str) -> str:
    """Compile SQL using dbt context."""
    try:
        with ctx.adapter.connection_named("__sql_workbench__"):
            return ctx.compile_code(sql).compiled_code
    except Exception as e:
        return str(e)


def ser(x):
    if isinstance(x, decimal.Decimal):
        return float(x)
    if isinstance(x, date):
        return x.isoformat()
    if isinstance(x, datetime):
        return x.isoformat()
    return x


def run_query() -> None:
    """Run SQL query using dbt context.

    This mutates the state of the app.
    """
    ctx: DbtProject = state.w.ctx
    sql = state.w.compiled_sql
    try:
        state.w.sql_query_state = "running"
        with ctx.adapter.connection_named("__sql_workbench__"):
            result = ctx.execute_code(state.w.sql_template.format(sql=sql))
    except Exception as error:
        state.w.sql_query_state = "error"
        state.w.sql_adapter_resp = str(error)
        state.w.sql_result_columns = []
    else:
        state.w.sql_query_state = "success"
        state.w.sql_adapter_resp = result.adapter_response
        output = [
            OrderedDict(zip(result.table.column_names, (ser(v) for v in row)))
            for row in result.table.rows
        ]
        state.w.sql_result_df = pd.DataFrame(output)
        state.w.sql_result_columns = [
            {"field": c, "headerName": c.upper()} for c in result.table.column_names
        ]
        state.w.sql_result_rows = output


@st.cache
def convert_df_to_csv(dataframe: pd.DataFrame) -> bytes:
    """Convert a dataframe to a CSV file."""
    return state.w.sql_result_df.to_csv().encode("utf-8")


def build_profile_report(minimal: bool = True) -> ydata_profiling.ProfileReport:
    """Build a profile report for a given dataframe.

    This is a wrapper around the ydata_profiling library. It is cached to avoid
    re-running the report every time the user changes the SQL query.
    """
    return state.w.sql_result_df.profile_report(minimal=minimal)


def convert_profile_report_to_html(profile: ydata_profiling.ProfileReport) -> str:
    """Convert a profile report to HTML."""
    return profile.to_html()


def run_profile(minimal: bool = True) -> str:
    """Run a profile report and return the HTML report."""
    if not state.w.sql_result_df.empty:
        state.w.profile_html = convert_profile_report_to_html(build_profile_report(minimal))


def main():
    args = _parse_args()

    st.title("dbt-osmosis 🌊")

    if "w" not in state:
        # Initialize state
        board = Dashboard()
        w = SimpleNamespace(
            # Components
            dashboard=board,
            editor=Editor(board, 0, 0, 6, 11, minW=3, minH=3, sql_compiler=compile),
            renderer=Renderer(board, 6, 0, 6, 11, minW=3, minH=3),
            preview=Preview(board, 0, 11, 12, 9, minW=3, minH=3),
            profiler=Profiler(board, 0, 20, 8, 9, minW=3, minH=3),
            feed=RssFeed(board, 8, 20, 4, 9, minW=3, minH=3),
            # Base Args
            project_dir=args.get("project_dir") or str(default_project_dir()),
            profiles_dir=args.get("profiles_dir") or str(default_profiles_dir()),
            # SQL Editor
            compiled_sql="",
            raw_sql="",
            theme="dark",
            lang="sql",
            # Query Runner
            sql_result_df=pd.DataFrame(),
            sql_result_columns=[],
            sql_result_rows=[],
            sql_adapter_resp=None,
            sql_query_state="test",
            sql_template="select * from ({sql}) as _query limit 200",
            # Profiler
            profile_minimal=True,
            profile_html="",
            # Model
            model="SCRATCH",
            cache_version=0,
        )
        # Load raw profiles
        w.raw_profiles = dbt_profile.read_profile(w.profiles_dir)
        # Seed demo project query
        if w.project_dir.rstrip(os.path.sep).endswith(("demo_sqlite", "demo_duckdb")):
            w.raw_sql = _get_demo_query()
        else:
            w.raw_sql = default_prompt
        # Initialize dbt context
        w.ctx = DbtProject(
            project_dir=w.project_dir,
            profiles_dir=w.profiles_dir,
        )
        w.target_profile = w.ctx.config.target_name
        # Demo compilation hook + seed editor
        w.editor.tabs[EditorTabs.SQL]["content"] = w.raw_sql
        w.compiled_sql = compile(w.ctx, w.raw_sql) if w.raw_sql else ""
        # Grab nodes
        model_nodes = []
        for node in w.ctx.manifest.nodes.values():
            if node.resource_type == "model" and node.package_name == w.ctx.config.project_name:
                model_nodes.append(node)
        w.model_nodes = model_nodes
        w.model_opts = ["SCRATCH"] + [node for node in model_nodes]
        # Save state
        state.w = w
        w.editor.update_content("SQL", w.raw_sql)
    else:
        # Load state
        w = state.w

    ctx: DbtProject = w.ctx

    # Render Sidebar
    sidebar(ctx)

    # Render Interface
    with elements("dashboard"):
        # Bind hotkeys, maybe one day we can figure out how to override Monaco's cmd+enter binding
        event.Hotkey("ctrl+enter", sync(), bindInputs=True, overrideDefault=True)
        event.Hotkey("command+s", sync(), bindInputs=True, overrideDefault=True)
        event.Hotkey("ctrl+shift+enter", lambda: run_query(), bindInputs=True, overrideDefault=True)
        event.Hotkey("command+shift+s", lambda: run_query(), bindInputs=True, overrideDefault=True)

        # Render Dashboard
        with w.dashboard(rowHeight=57):
            w.editor()
            w.renderer()
            w.preview(query_run_fn=run_query)
            w.profiler(run_profile_fn=run_profile)
            w.feed()


if __name__ == "__main__":
    st.set_page_config(page_title="dbt-osmosis Workbench", page_icon="🌊", layout="wide")
    main()
