import hashlib
import tempfile
from pathlib import Path
from typing import Tuple

import agate
from dbt.adapters.base.relation import BaseRelation
from git import Repo

from dbt_osmosis.core.logging import logger
from dbt_osmosis.core.osmosis import DbtOsmosis


def build_diff_queries(model: str, runner: DbtOsmosis) -> Tuple[str, str]:
    """Leverage git to build two temporary tables for diffing the results of a query throughout a change"""
    node = runner.dbt.ref_lookup.find(model, package=None, manifest=runner.dbt)
    dbt_path = Path(node.root_path)
    abs_path = dbt_path / node.original_file_path
    repo = Repo(dbt_path, search_parent_directories=True)
    t = next(Path(repo.working_dir).rglob(node.original_file_path)).relative_to(repo.working_dir)
    target = repo.head.object.tree[str(t)]
    with tempfile.NamedTemporaryFile(delete=True, dir=str(abs_path.parent), suffix=".sql") as f:
        f.write(target.data_stream.read())
        f.flush()
        runner.rebuild_dbt_manifest()
        original_node = runner.dbt.ref_lookup.find(
            Path(f.name).stem, package=None, manifest=runner.dbt
        )  # Injected Original Temp Node --> Identifier resolved ot commit hash at runtime
        changed_node = runner.dbt.ref_lookup.find(model, package=None, manifest=runner.dbt)
        with runner.adapter.connection_named("dbt-osmosis"):
            # Original Node Resolution
            original_node = runner.adapter.get_compiler().compile_node(original_node, runner.dbt)
            # Diff Node Resolution
            changed_node = runner.adapter.get_compiler().compile_node(changed_node, runner.dbt)

    return original_node.compiled_sql, changed_node.compiled_sql


def build_diff_tables(model: str, runner: DbtOsmosis) -> Tuple[BaseRelation, BaseRelation]:
    """Leverage git to build two temporary tables for diffing the results of a query throughout a change"""
    node = runner.dbt.ref_lookup.find(model, package=None, manifest=runner.dbt)
    dbt_path = Path(node.root_path)
    abs_path = dbt_path / node.original_file_path
    repo = Repo(dbt_path, search_parent_directories=True)
    t = next(Path(repo.working_dir).rglob(node.original_file_path)).relative_to(repo.working_dir)
    sha = repo.head.object.hexsha
    target = repo.head.object.tree[str(t)]
    with tempfile.NamedTemporaryFile(delete=True, dir=str(abs_path.parent), suffix=".sql") as f:
        f.write(target.data_stream.read())
        f.flush()
        runner.rebuild_dbt_manifest()
        original_node = runner.dbt.ref_lookup.find(
            Path(f.name).stem, package=None, manifest=runner.dbt
        )  # Injected Original Temp Node --> Identifier resolved ot commit hash at runtime
        changed_node = runner.dbt.ref_lookup.find(model, package=None, manifest=runner.dbt)
        with runner.adapter.connection_named("dbt-osmosis"):
            # Original Node Resolution
            original_node = runner.adapter.get_compiler().compile_node(original_node, runner.dbt)
            _ref = runner.adapter.get_relation(original_node.database, original_node.schema, sha)
            if not _ref:
                ref_A = runner.adapter.Relation.create(
                    original_node.database, original_node.schema, sha
                )
                logger().info("Creating new relation for %s", ref_A)
                runner.execute_macro(
                    "create_table_as",
                    kwargs={
                        "sql": original_node.compiled_sql,
                        "relation": ref_A,
                        "temporary": True,
                    },
                    run_compiled_sql=True,
                )
            else:
                ref_A = _ref
                logger().info("Found existing relation for %s", ref_A)
            # Diff Node Resolution
            changed_node = runner.adapter.get_compiler().compile_node(changed_node, runner.dbt)
            ref_B = runner.adapter.Relation.create(
                changed_node.database,
                changed_node.schema,
                hashlib.md5(changed_node.identifier.encode("utf-8")).hexdigest(),
            )
            logger().info("Creating new relation for %s", ref_B)
            runner.execute_macro(
                "create_table_as",
                kwargs={
                    "sql": changed_node.compiled_sql,
                    "relation": ref_B,
                    "temporary": True,
                },
                run_compiled_sql=True,
            )

    return ref_A, ref_B


def diff_tables(
    ref_A: BaseRelation, ref_B: BaseRelation, pk: str, runner: DbtOsmosis
) -> agate.Table:

    logger().info("Running diff")
    _, _, table = runner.execute_macro(
        "compare_relations",
        kwargs={
            "a_relation": ref_A,
            "b_relation": ref_B,
            "primary_key": pk,
        },
        run_compiled_sql=True,
        fetch=True,
    )
    return table


def diff_queries(
    sql_A: str, sql_B: str, pk: str, runner: DbtOsmosis, aggregate: bool = True
) -> agate.Table:

    logger().info("Running diff")
    _, _, table = runner.execute_macro(
        "compare_queries" if aggregate else "dbt_osmosis_compare",
        kwargs={
            "a_query": sql_A,
            "b_query": sql_B,
            "primary_key": pk,
        },
        run_compiled_sql=True,
        fetch=True,
    )
    return table


def diff_and_print_to_console(
    model: str, pk: str, runner: DbtOsmosis, make_temp_tables: bool = False
) -> None:
    """
    Compare two tables and print the results to the console
    """
    if make_temp_tables:
        table = diff_tables(*build_diff_tables(model, runner), pk, runner)
    else:
        table = diff_queries(*build_diff_queries(model, runner), pk, runner)
    print("")
    table.print_table()
