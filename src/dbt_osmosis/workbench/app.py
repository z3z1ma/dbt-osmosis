# pyright: reportMissingTypeStubs=false, reportAny=false, reportUnusedCallResult=false, reportUnknownMemberType=false, reportUntypedFunctionDecorator=false
import argparse
import decimal
import os
import sys
import typing as t
from collections import OrderedDict
from datetime import date, datetime
from textwrap import dedent
from types import SimpleNamespace

import dbt.config.profile as dbt_profile
import feedparser
import pandas as pd
import streamlit as st
import ydata_profiling
from streamlit import session_state as state
from streamlit_elements_fluence import elements, event, sync

from dbt_osmosis.core.osmosis import (
    DbtConfiguration,
    _reload_manifest,  # pyright: ignore[reportPrivateUsage]
    compile_sql_code,
    create_dbt_project_context,
    discover_profiles_dir,
    discover_project_dir,
    execute_sql_code,
)
from dbt_osmosis.core.osmosis import (
    DbtProjectContext as DbtProject,
)
from dbt_osmosis.workbench.components.dashboard import Dashboard
from dbt_osmosis.workbench.components.editor import Editor
from dbt_osmosis.workbench.components.editor import TabName as EditorTab
from dbt_osmosis.workbench.components.feed import RssFeed
from dbt_osmosis.workbench.components.preview import Preview
from dbt_osmosis.workbench.components.profiler import Profiler
from dbt_osmosis.workbench.components.renderer import Renderer

st.set_page_config(page_title="dbt-osmosis Workbench", page_icon="🌊", layout="wide")

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


def _parse_args() -> dict[str, t.Any]:
    """Parse command line arguments"""
    try:
        parser = argparse.ArgumentParser(description="dbt osmosis workbench")
        _ = parser.add_argument("--profiles-dir", help="dbt profile directory")
        _ = parser.add_argument("--project-dir", help="dbt project directory")
        args = vars(parser.parse_args(sys.argv[1:]))
    except Exception:
        args = {}
    return args


def change_target() -> None:
    """Change the target profile"""
    ctx: DbtProject = state.app.ctx
    if ctx.runtime_cfg.target_name != state.app.target_name:
        print(f"Changing target to {state.app.target_name}")
        ctx.runtime_cfg.target_name = state.app.target_name
        _reload_manifest(ctx)
        state.app.compiled_query = compile(state.app.query)


def inject_model() -> None:
    """Inject model into editor"""
    ctx: DbtProject = state.app.ctx
    if state.model is not None and state.model != "SCRATCH":
        path = os.path.join(ctx.runtime_cfg.project_root, state.model.original_file_path)
        with open(path, "r") as f:
            state.app.query = f.read()
        state.app.editor.update_content("SQL", state.app.query)
    elif state.model == "SCRATCH":
        state.app.editor.update_content("SQL", default_prompt)


def save_model() -> None:
    """Save model to disk"""
    ctx: DbtProject = state.app.ctx
    if state.model is not None and state.model != "SCRATCH":
        path = os.path.join(ctx.runtime_cfg.project_root, state.model.original_file_path)
        with open(path, "w") as f:
            _ = f.write(state.app.editor.get_content("SQL"))
        print(f"Saved model to {path}")


def sidebar(ctx: DbtProject) -> None:
    """Render the sidebar"""

    with st.sidebar.expander("💡 Models", expanded=True):
        st.caption(
            "Select a model to use as a starting point for your query. The filter supports typeahead. All changes are ephemeral unless you save the model."
        )
        state.app.model = st.selectbox(
            "Select a model",
            options=["SCRATCH", *state.app.model_nodes],
            format_func=lambda node_or_str: getattr(node_or_str, "name", node_or_str),
            on_change=inject_model,
            key="model",
        )
        save, revert = st.columns(2)
        if state.app.model != "SCRATCH":
            save.button("💾 - Save", on_click=save_model, key="save_model")
        revert.button("⏮ - Revert", on_click=inject_model, key="reset_model")

    with st.sidebar.expander("💁 Profiles", expanded=True):
        st.caption(
            "Select a target used for materializing, compiling, and testing models.",
        )
        state.app.target_name = st.radio(
            f"Loaded targets from {ctx.runtime_cfg.profile_name}",
            [
                target
                for target in state.app.all_profiles[ctx.runtime_cfg.profile_name].get(
                    "outputs", []
                )
            ],
            on_change=change_target,
            key="target_name",
        )
        st.markdown(f"Current Target: **{state.app.target_name}**")

    with st.sidebar.expander("📝 Query Template"):
        st.caption(
            "This is a template query that will be used when executing SQL. The {sql} variable will be replaced with the compiled SQL."
        )
        state.app.query_template = st.text_area(
            "SQL Template",
            value=state.app.query_template,
            height=100,
        )

    st.sidebar.write("Notes")
    st.sidebar.caption(
        "Refresh the page to reparse dbt. This is useful if any updated models or macros in your physical project on disk have changed and are not yet reflected in the workbench as refable or updated."
    )


def compile(sql: str) -> str:
    """Compile SQL using dbt context."""
    ctx: DbtProject = state.app.ctx
    try:
        return compile_sql_code(ctx, sql).compiled_code or ""
    except Exception as e:
        return str(e)


def make_json_compat(v: t.Any) -> t.Any:
    """Convert a value to be safe for JSON serialization."""
    if isinstance(v, decimal.Decimal):
        return float(v)
    if isinstance(v, date):
        return v.isoformat()
    if isinstance(v, datetime):
        return v.isoformat()
    return v


def run_query() -> None:
    """Run SQL query using dbt context.

    This mutates the state of the app.
    """
    ctx: DbtProject = state.app.ctx
    sql = state.app.compiled_query
    try:
        state.app.query_state = "running"
        resp, table = execute_sql_code(ctx, state.app.query_template.format(sql=sql))
    except Exception as error:
        state.app.query_state = "error"
        state.app.query_adapter_resp = str(error)
        state.app.query_result_columns = []
    else:
        state.app.query_state = "success"
        state.app.query_adapter_resp = resp
        output = [
            OrderedDict(zip(table.column_names, (make_json_compat(v) for v in row)))  # pyright: ignore[reportUnknownVariableType,reportUnknownArgumentType]
            for row in table.rows  # pyright: ignore[reportUnknownVariableType]
        ]
        state.app.query_result_df = pd.DataFrame(output)
        state.app.query_result_columns = [
            {"field": c, "headerName": c.upper()} for c in t.cast(tuple[str], table.column_names)
        ]
        state.app.query_result_rows = output


def build_profile_report(minimal: bool = True) -> ydata_profiling.ProfileReport:
    """Build a profile report for a given dataframe.

    This is a wrapper around the ydata_profiling library. It is cached to avoid
    re-running the report every time the user changes the SQL query.
    """
    return state.app.query_result_df.profile_report(minimal=minimal)


def convert_profile_report_to_html(profile: ydata_profiling.ProfileReport) -> str:
    """Convert a profile report to HTML."""
    return profile.to_html()


def run_profile(minimal: bool = True) -> None:
    """Run a profile report and return the HTML report."""
    if not state.app.query_result_df.empty:
        state.app.profile_html = convert_profile_report_to_html(build_profile_report(minimal))


def main():
    args = _parse_args()

    st.title("dbt-osmosis 🌊")

    if "app" not in state:
        # Initialize state
        board = Dashboard()

        app = SimpleNamespace(
            model="SCRATCH",
            dashboard=board,
            editor=Editor(board, 0, 0, 6, 11, minW=3, minH=3, compile_action=compile),
            renderer=Renderer(board, 6, 0, 6, 11, minW=3, minH=3),
            preview=Preview(board, 0, 11, 12, 9, minW=3, minH=3, query_action=run_query),
            profiler=Profiler(board, 0, 20, 8, 9, minW=3, minH=3, prof_action=run_profile),
            feed=RssFeed(board, 8, 20, 4, 9, minW=3, minH=3),
        )
        for v in vars(app).copy().values():
            if isinstance(v, Dashboard.Item):
                for k, v in v.initial_state().items():
                    setattr(app, k, v)

        state.app = app

        proj_dir = args.get("project_dir") or discover_project_dir()
        prof_dir = args.get("profiles_dir") or discover_profiles_dir()

        app.all_profiles = dbt_profile.read_profile(prof_dir)

        if proj_dir.rstrip(os.path.sep).endswith(("demo_sqlite", "demo_duckdb")):
            app.query = _get_demo_query()
        else:
            app.query = default_prompt

        app.ctx = create_dbt_project_context(
            config=DbtConfiguration(project_dir=proj_dir, profiles_dir=prof_dir)
        )
        app.target_name = app.ctx.runtime_cfg.target_name

        app.editor.tabs[EditorTab.SQL]["content"] = app.query
        app.compiled_query = compile(app.query) if app.query else ""

        model_nodes: list[t.Any] = []
        for node in app.ctx.manifest.nodes.values():
            if (
                node.resource_type == "model"
                and node.package_name == app.ctx.runtime_cfg.project_name
            ):
                model_nodes.append(node)
        app.model_nodes = model_nodes

        app.editor.update_content("SQL", app.query)

        hackernews_rss = t.cast(t.Any, feedparser.parse("https://news.ycombinator.com/rss"))
        feed_html = []
        for entry in hackernews_rss.entries:
            feed_html.append(
                dedent(
                    f"""
                <div style="padding: 10px 5px 10px 5px; border-bottom: 1px solid #e0e0e0;">
                    <a href="{entry.link}" target="_blank" style="font-size: 16px; font-weight: bold; color: #FF4136; text-decoration: none;">{entry.title}</a>
                    <div style="font-size: 12px; color: #9e9e9e; padding-top: 3px;">{entry.published}
                    <span style="color: #FF4136;">|</span>
                    <a href="{entry.comments}" target="_blank" style="color: #FF4136; text-decoration: none;">Comments</a>
                    </div>
                </div>
            """
                )
            )
        app.feed_html = "".join(t.cast(list[str], feed_html))
    else:
        app = state.app

    ctx: DbtProject = app.ctx

    sidebar(ctx)

    with elements("dashboard"):  # pyright: ignore[reportGeneralTypeIssues]
        event.Hotkey("ctrl+enter", sync(), bindInputs=True, overrideDefault=True)
        event.Hotkey("command+s", sync(), bindInputs=True, overrideDefault=True)
        event.Hotkey("ctrl+shift+enter", lambda: run_query(), bindInputs=True, overrideDefault=True)
        event.Hotkey("command+shift+s", lambda: run_query(), bindInputs=True, overrideDefault=True)

        with app.dashboard(rowHeight=57):
            app.editor()
            app.renderer()
            app.preview()
            app.profiler()
            app.feed()


if __name__ == "__main__":
    main()
