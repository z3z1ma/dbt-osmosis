import hashlib
from pathlib import Path
from typing import Tuple

import agate
from dbt.adapters.base.relation import BaseRelation
from git import Repo

from dbt_osmosis.core.log_controller import logger
from dbt_osmosis.vendored.dbt_core_interface.project import DbtProject


def build_diff_queries(model: str, runner: DbtProject) -> Tuple[str, str]:
    """Leverage git to build two temporary tables for diffing the results of a query
    throughout a change
    """
    # Resolve git node
    node = runner.get_ref_node(model)
    dbt_path = Path(node.root_path)
    repo = Repo(dbt_path, search_parent_directories=True)
    t = next(Path(repo.working_dir).rglob(node.original_file_path)).relative_to(repo.working_dir)
    sha = repo.head.object.hexsha
    target = repo.head.object.tree[str(t)]

    # Create original node
    git_node_name = "z_" + sha[-7:]
    original_node = runner.get_server_node(target.data_stream.read().decode("utf-8"), git_node_name)

    # Alias changed node
    changed_node = node

    # Compile models
    original_node = runner.compile_node(original_node)
    changed_node = runner.compile_node(changed_node)

    return original_node.compiled_sql, changed_node.compiled_sql


def build_diff_tables(model: str, runner: DbtProject) -> Tuple[BaseRelation, BaseRelation]:
    """Leverage git to build two temporary tables for diffing the results of a query throughout a change"""
    # Resolve git node
    node = runner.get_ref_node(model)
    dbt_path = Path(node.root_path)
    repo = Repo(dbt_path, search_parent_directories=True)
    t = next(Path(repo.working_dir).rglob(node.original_file_path)).relative_to(repo.working_dir)
    sha = repo.head.object.hexsha
    target = repo.head.object.tree[str(t)]

    # Create original node
    git_node_name = "z_" + sha[-7:]
    original_node = runner.get_server_node(target.data_stream.read().decode("utf-8"), git_node_name)

    # Alias changed node
    changed_node = node

    # Compile models
    original_node = runner.compile_node(original_node).node
    changed_node = runner.compile_node(changed_node).node

    # Lookup and resolve original ref based on git sha
    git_node_parts = original_node.database, "dbt_diff", git_node_name
    ref_A, did_exist = runner.get_or_create_relation(*git_node_parts)
    if not did_exist:
        logger().info("Creating new relation for %s", ref_A)
        with runner.adapter.connection_named("dbt-osmosis"):
            runner.execute_macro(
                "create_schema",
                kwargs={"relation": ref_A},
            )
            runner.execute_macro(
                "create_table_as",
                kwargs={
                    "sql": original_node.compiled_sql,
                    "relation": ref_A,
                    "temporary": True,
                },
                run_compiled_sql=True,
            )

    # Resolve modified fake ref based on hash of it compiled SQL
    temp_node_name = "z_" + hashlib.md5(changed_node.compiled_sql.encode("utf-8")).hexdigest()[-7:]
    git_node_parts = original_node.database, "dbt_diff", temp_node_name
    ref_B, did_exist = runner.get_or_create_relation(*git_node_parts)
    if not did_exist:
        ref_B = runner.adapter.Relation.create(*git_node_parts)
        logger().info("Creating new relation for %s", ref_B)
        with runner.adapter.connection_named("dbt-osmosis"):
            runner.execute_macro(
                "create_schema",
                kwargs={"relation": ref_B},
            )
            runner.execute_macro(
                "create_table_as",
                kwargs={
                    "sql": original_node.compiled_sql,
                    "relation": ref_B,
                    "temporary": True,
                },
                run_compiled_sql=True,
            )

    return ref_A, ref_B


def diff_tables(
    ref_A: BaseRelation,
    ref_B: BaseRelation,
    pk: str,
    runner: DbtProject,
    aggregate: bool = True,
) -> agate.Table:
    logger().info("Running diff")
    _, table = runner.adapter_execute(
        runner.execute_macro(
            "_dbt_osmosis_compare_relations_agg" if aggregate else "_dbt_osmosis_compare_relations",
            kwargs={
                "a_relation": ref_A,
                "b_relation": ref_B,
                "primary_key": pk,
            },
        ),
        auto_begin=True,
        fetch=True,
    )
    return table


def diff_queries(
    sql_A: str, sql_B: str, pk: str, runner: DbtProject, aggregate: bool = True
) -> agate.Table:
    logger().info("Running diff")
    _, table = runner.adapter_execute(
        runner.execute_macro(
            "_dbt_osmosis_compare_queries_agg" if aggregate else "_dbt_osmosis_compare_queries",
            kwargs={
                "a_query": sql_A,
                "b_query": sql_B,
                "primary_key": pk,
            },
        ),
        auto_begin=True,
        fetch=True,
    )
    return table


def diff_and_print_to_console(
    model: str,
    pk: str,
    runner: DbtProject,
    make_temp_tables: bool = False,
    agg: bool = True,
    output: str = "table",
) -> None:
    """
    Compare two tables and print the results to the console
    """
    if make_temp_tables:
        table = diff_tables(*build_diff_tables(model, runner), pk, runner, agg)
    else:
        table = diff_queries(*build_diff_queries(model, runner), pk, runner, agg)
    print("")
    output = output.lower()
    if output == "table":
        table.print_table()
    elif output in ("chart", "bar"):
        if not agg:
            logger().warn(
                "Cannot render output format chart with --no-agg option, defaulting to table"
            )
            table.print_table()
        else:
            _table = table.compute(
                [
                    (
                        "in_original, in_changed",
                        agate.Formula(agate.Text(), lambda r: "%(in_a)s, %(in_b)s" % r),
                    )
                ]
            )
            _table.print_bars(
                label_column_name="in_original, in_changed", value_column_name="count"
            )
    elif output == "csv":
        table.to_csv("dbt-osmosis-diff.csv")
    else:
        logger().warn("No such output format %s, defaulting to table", output)
        table.print_table()
