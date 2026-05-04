# pyright: reportPrivateImportUsage=false, reportPrivateUsage=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportAny=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportArgumentType=false, reportFunctionMemberAccess=false, reportUnknownVariableType=false, reportUnusedParameter=false

from unittest import mock

import pytest
from dbt.artifacts.resources.types import NodeType

import dbt_osmosis.core.sql_operations as sql_operations
from dbt_osmosis.core.settings import YamlRefactorContext
from dbt_osmosis.core.sql_operations import compile_sql_code, execute_sql_code


def _sql_operation_keys(yaml_context: YamlRefactorContext) -> set[str]:
    return {
        key
        for key in yaml_context.project.manifest.nodes
        if key.startswith(
            f"{NodeType.SqlOperation}.{yaml_context.project.runtime_cfg.project_name}."
        )
    }


def test_compile_sql_code_no_jinja(yaml_context: YamlRefactorContext):
    """Check compile_sql_code with a plain SELECT (no Jinja).
    We should skip calling the 'process_node' logic and the returned node
    should have the raw SQL as is.
    """
    raw_sql = "SELECT 1 AS mycol"
    before_node_count = len(yaml_context.project.manifest.nodes)
    with mock.patch("dbt_osmosis.core.sql_operations.process_node") as mock_process:
        node = compile_sql_code(yaml_context.project, raw_sql)
        mock_process.assert_not_called()
    after_node_count = len(yaml_context.project.manifest.nodes)
    assert node.raw_code == raw_sql
    assert node.compiled_code == raw_sql
    assert after_node_count == before_node_count


def test_compile_sql_code_with_ref_uses_real_fixture(yaml_context: YamlRefactorContext):
    """Compile ref-based SQL through dbt's real parser/runner path."""
    raw_sql = "select * from {{ ref('customers') }} limit 1"
    before_operation_keys = _sql_operation_keys(yaml_context)

    node = compile_sql_code(yaml_context.project, raw_sql)

    assert node.raw_code == raw_sql
    assert node.compiled_code is not None
    assert "{{" not in node.compiled_code
    assert "customers" in node.compiled_code.lower()
    assert _sql_operation_keys(yaml_context) == before_operation_keys


def test_execute_sql_code_select_1_real_fixture(yaml_context: YamlRefactorContext):
    """Execute plain SQL against the real DuckDB fixture."""
    _, table = execute_sql_code(yaml_context.project, "select 1 as value")

    assert table.column_names == ("value",)
    assert len(table.rows) == 1
    assert table.rows[0][0] == 1


def test_compile_sql_code_cleans_temp_node_after_compile_failure(
    yaml_context: YamlRefactorContext, monkeypatch: pytest.MonkeyPatch
):
    """Temporary manifest SQL operation nodes are cleaned up when compile fails."""
    tmp_id = "00000000-0000-0000-0000-000000000001"
    key = f"{NodeType.SqlOperation}.{yaml_context.project.runtime_cfg.project_name}.{tmp_id}"

    def fail_process(runtime_cfg, manifest, node):
        manifest.nodes[key] = node
        raise RuntimeError("forced compile failure")

    monkeypatch.setattr(sql_operations.uuid, "uuid4", lambda: tmp_id)
    monkeypatch.setattr(sql_operations, "process_node", fail_process)

    with pytest.raises(RuntimeError, match="forced compile failure"):
        compile_sql_code(yaml_context.project, "select {{ 1 }} as value")

    assert key not in yaml_context.project.manifest.nodes


def test_compile_sql_code_with_jinja(yaml_context: YamlRefactorContext):
    """Compile SQL that has Jinja statements, ensuring 'process_node' is invoked and
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
    """If there's no jinja, 'execute_sql_code' calls adapter.execute directly with raw_sql."""
    raw_sql = "SELECT 42 AS meaning"
    with mock.patch.object(yaml_context.project.adapter, "execute") as mock_execute:
        mock_execute.return_value = ("OK", mock.Mock(rows=[(42,)]))
        resp, table = execute_sql_code(yaml_context.project, raw_sql)
        mock_execute.assert_called_with(raw_sql, auto_begin=False, fetch=True)
    assert resp == "OK"
    assert table.rows[0] == (42,)


def test_execute_sql_code_with_jinja(yaml_context: YamlRefactorContext):
    """If there's Jinja, we compile first, then execute the compiled code."""
    raw_sql = "SELECT {{ 2 + 2 }} AS four"
    with (
        mock.patch.object(yaml_context.project.adapter, "execute") as mock_execute,
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
