# pyright: reportPrivateImportUsage=false, reportPrivateUsage=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportAny=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportArgumentType=false, reportFunctionMemberAccess=false, reportUnknownVariableType=false, reportUnusedParameter=false

from unittest import mock

from dbt_osmosis.core.node_filters import _topological_sort


def test_topological_sort():
    """Test the topological sort functionality."""
    # We'll simulate a trivial adjacency-like approach:
    node_a = mock.MagicMock()
    node_b = mock.MagicMock()
    node_c = mock.MagicMock()
    node_a.depends_on_nodes = ["node_b"]  # a depends on b
    node_b.depends_on_nodes = ["node_c"]  # b depends on c
    node_c.depends_on_nodes = []
    input_list = [
        ("node_a", node_a),
        ("node_b", node_b),
        ("node_c", node_c),
    ]
    sorted_nodes = _topological_sort(input_list)
    # We expect node_c -> node_b -> node_a
    assert [uid for uid, _ in sorted_nodes] == ["node_c", "node_b", "node_a"]
