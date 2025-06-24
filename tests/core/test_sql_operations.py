# pyright: reportPrivateImportUsage=false, reportPrivateUsage=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportAny=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportArgumentType=false, reportFunctionMemberAccess=false, reportUnknownVariableType=false, reportUnusedParameter=false

from unittest import mock

import pytest

from dbt_osmosis.core.config import DbtConfiguration, create_dbt_project_context
from dbt_osmosis.core.settings import YamlRefactorContext, YamlRefactorSettings
from dbt_osmosis.core.sql_operations import compile_sql_code, execute_sql_code


@pytest.fixture(scope="module")
def yaml_context() -> YamlRefactorContext:
    """
    Creates a YamlRefactorContext for the real 'demo_duckdb' project.
    """
    cfg = DbtConfiguration(project_dir="demo_duckdb", profiles_dir="demo_duckdb")
    cfg.vars = {"dbt-osmosis": {}}

    project_context = create_dbt_project_context(cfg)
    context = YamlRefactorContext(
        project_context,
        settings=YamlRefactorSettings(
            dry_run=True,
            use_unrendered_descriptions=True,
        ),
    )
    return context


def test_compile_sql_code_no_jinja(yaml_context: YamlRefactorContext):
    """
    Check compile_sql_code with a plain SELECT (no Jinja).
    We should skip calling the 'process_node' logic and the returned node
    should have the raw SQL as is.
    """
    raw_sql = "SELECT 1 AS mycol"
    with mock.patch("dbt_osmosis.core.sql_operations.process_node") as mock_process:
        node = compile_sql_code(yaml_context.project, raw_sql)
        mock_process.assert_not_called()
    assert node.raw_code == raw_sql
    assert node.compiled_code is None


def test_compile_sql_code_with_jinja(yaml_context: YamlRefactorContext):
    """
    Compile SQL that has Jinja statements, ensuring 'process_node' is invoked and
    we get a compiled node.
    """
    raw_sql = "SELECT {{ 1 + 1 }} AS mycol"
    with (
        mock.patch("dbt_osmosis.core.sql_operations.process_node") as mock_process,
        mock.patch("dbt_osmosis.core.sql_operations.SqlCompileRunner.compile") as mock_compile,
    ):
        node_mock = mock.Mock()
        node_mock.raw_code = raw_sql
        node_mock.compiled_code = "SELECT 2 AS mycol"
        mock_compile.return_value = node_mock

        compiled_node = compile_sql_code(yaml_context.project, raw_sql)
        mock_process.assert_called_once()
        mock_compile.assert_called_once()
        assert compiled_node.compiled_code == "SELECT 2 AS mycol"


def test_execute_sql_code_no_jinja(yaml_context: YamlRefactorContext):
    """
    If there's no jinja, 'execute_sql_code' calls adapter.execute directly with raw_sql.
    """
    raw_sql = "SELECT 42 AS meaning"
    with mock.patch.object(yaml_context.project._adapter, "execute") as mock_execute:
        mock_execute.return_value = ("OK", mock.Mock(rows=[(42,)]))
        resp, table = execute_sql_code(yaml_context.project, raw_sql)
        mock_execute.assert_called_with(raw_sql, auto_begin=False, fetch=True)
    assert resp == "OK"
    assert table.rows[0] == (42,)


def test_execute_sql_code_with_jinja(yaml_context: YamlRefactorContext):
    """
    If there's Jinja, we compile first, then execute the compiled code.
    """
    raw_sql = "SELECT {{ 2 + 2 }} AS four"
    with (
        mock.patch.object(yaml_context.project._adapter, "execute") as mock_execute,
        mock.patch("dbt_osmosis.core.sql_operations.compile_sql_code") as mock_compile,
    ):
        mock_execute.return_value = ("OK", mock.Mock(rows=[(4,)]))

        node_mock = mock.Mock()
        node_mock.compiled_code = "SELECT 4 AS four"
        mock_compile.return_value = node_mock

        resp, table = execute_sql_code(yaml_context.project, raw_sql)
        mock_compile.assert_called_once()
        assert resp == "OK"
        assert table.rows[0] == (4,)
