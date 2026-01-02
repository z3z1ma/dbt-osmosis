# pyright: reportPrivateImportUsage=false, reportPrivateUsage=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportAny=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportArgumentType=false, reportFunctionMemberAccess=false, reportUnknownVariableType=false, reportUnusedParameter=false

"""Tests for complex inheritance scenarios including circular dependencies and multiple inheritance paths.

These tests use mocks because setting up real dbt projects with these edge cases would be
extremely complex. The tests validate important behaviors:

- Circular dependencies don't cause infinite loops
- Multiple inheritance paths (diamond pattern) are handled correctly
- Depth limits prevent unbounded recursion
- Only model/seed/source dependencies are included

These are behavior tests that validate the observable behavior of the ancestor tree algorithm
under edge case conditions, using mocks only where necessary to set up those conditions.
"""

from unittest import mock

import pytest
from dbt.contracts.graph.manifest import Manifest
from dbt.contracts.graph.nodes import ModelNode, SourceDefinition

from dbt_osmosis.core.inheritance import _build_node_ancestor_tree


@pytest.fixture
def mock_manifest() -> Manifest:
    """Create a mock manifest with complex dependency relationships."""
    manifest = mock.Mock(spec=Manifest)

    # Helper function to create column mocks
    def _make_column(
        name: str, description: str = "", tags: list | None = None, meta: dict | None = None
    ):
        col = mock.Mock()
        col.name = name
        col.description = description
        col.tags = tags or []
        col.meta = meta or {}
        col.data_type = None
        col.to_dict = mock.Mock(
            return_value={
                "name": name,
                "description": description,
                "tags": tags or [],
                "meta": meta or {},
            }
        )
        return col

    # Create mock nodes with different relationships
    source_node = mock.Mock(spec=SourceDefinition)
    source_node.unique_id = "source.my_project.raw_users"
    source_node.resource_type = "source"
    source_node.name = "raw_users"
    source_node.schema = "raw"
    source_node.columns = {
        "id": _make_column("id", "Source ID"),
        "name": _make_column("name", "Source name"),
    }
    source_node.depends_on = mock.Mock(nodes=[])
    source_node.meta = {}
    source_node.config = mock.Mock(extra={})

    stg_users = mock.Mock(spec=ModelNode)
    stg_users.unique_id = "model.my_project.stg_users"
    stg_users.resource_type = "model"
    stg_users.name = "stg_users"
    stg_users.schema = "staging"
    stg_users.columns = {
        "id": _make_column("id"),
        "name": _make_column("name", tags=["pii"]),
        "email": _make_column("email", "Email address"),
    }
    stg_users.depends_on = mock.Mock(nodes=["source.my_project.raw_users"])
    stg_users.depends_on_nodes = ["source.my_project.raw_users"]
    stg_users.description = "Staging users table"
    stg_users.meta = {}
    stg_users.config = mock.Mock(extra={})
    stg_users.relation_name = "stg_users"

    int_users = mock.Mock(spec=ModelNode)
    int_users.unique_id = "model.my_project.int_users"
    int_users.resource_type = "model"
    int_users.name = "int_users"
    int_users.schema = "intermediate"
    int_users.columns = {
        "id": _make_column("id"),
        "name": _make_column("name"),
        "email": _make_column("email"),
        "created_at": _make_column("created_at", "Creation timestamp"),
    }
    int_users.depends_on = mock.Mock(nodes=["model.my_project.stg_users"])
    int_users.depends_on_nodes = ["model.my_project.stg_users"]
    int_users.description = "Intermediate users table"
    int_users.meta = {}
    int_users.config = mock.Mock(extra={})
    int_users.relation_name = "int_users"

    # Create a node with multiple inheritance paths
    fct_orders = mock.Mock(spec=ModelNode)
    fct_orders.unique_id = "model.my_project.fct_orders"
    fct_orders.resource_type = "model"
    fct_orders.name = "fct_orders"
    fct_orders.schema = "marts"
    fct_orders.columns = {
        "order_id": _make_column("order_id"),
        "user_id": _make_column("user_id"),
        "order_date": _make_column("order_date"),
    }
    fct_orders.depends_on = mock.Mock(
        nodes=["model.my_project.int_users", "model.my_project.stg_orders"]
    )
    fct_orders.depends_on_nodes = ["model.my_project.int_users", "model.my_project.stg_orders"]
    fct_orders.description = "Orders fact table"
    fct_orders.meta = {}
    fct_orders.config = mock.Mock(extra={})
    fct_orders.relation_name = "fct_orders"

    stg_orders = mock.Mock(spec=ModelNode)
    stg_orders.unique_id = "model.my_project.stg_orders"
    stg_orders.resource_type = "model"
    stg_orders.name = "stg_orders"
    stg_orders.schema = "staging"
    stg_orders.columns = {
        "order_id": _make_column("order_id", "Order ID"),
        "user_id": _make_column("user_id", "User FK"),
    }
    stg_orders.depends_on = mock.Mock(nodes=["source.my_project.raw_orders"])
    stg_orders.depends_on_nodes = ["source.my_project.raw_orders"]
    stg_orders.description = "Staging orders"
    stg_orders.meta = {}
    stg_orders.config = mock.Mock(extra={})
    stg_orders.relation_name = "stg_orders"

    raw_orders = mock.Mock(spec=SourceDefinition)
    raw_orders.unique_id = "source.my_project.raw_orders"
    raw_orders.resource_type = "source"
    raw_orders.name = "raw_orders"
    raw_orders.schema = "raw"
    raw_orders.columns = {
        "order_id": _make_column("order_id", "Source Order ID"),
        "user_id": _make_column("user_id", "Source User FK"),
    }
    raw_orders.depends_on = mock.Mock(nodes=[])
    raw_orders.meta = {}
    raw_orders.config = mock.Mock(extra={})

    # Create a circular dependency node for testing cycle detection
    circular_a = mock.Mock(spec=ModelNode)
    circular_a.unique_id = "model.my_project.circular_a"
    circular_a.resource_type = "model"
    circular_a.name = "circular_a"
    circular_a.schema = "test"
    circular_a.columns = {"id": _make_column("id")}
    circular_a.depends_on = mock.Mock(nodes=["model.my_project.circular_b"])
    circular_a.depends_on_nodes = ["model.my_project.circular_b"]
    circular_a.meta = {}
    circular_a.config = mock.Mock(extra={})
    circular_a.relation_name = "circular_a"

    circular_b = mock.Mock(spec=ModelNode)
    circular_b.unique_id = "model.my_project.circular_b"
    circular_b.resource_type = "model"
    circular_b.name = "circular_b"
    circular_b.schema = "test"
    circular_b.columns = {"id": _make_column("id")}
    circular_b.depends_on = mock.Mock(nodes=["model.my_project.circular_a"])
    circular_b.depends_on_nodes = ["model.my_project.circular_a"]
    circular_b.meta = {}
    circular_b.config = mock.Mock(extra={})
    circular_b.relation_name = "circular_b"

    # Set up manifest
    manifest.nodes = {
        "model.my_project.stg_users": stg_users,
        "model.my_project.int_users": int_users,
        "model.my_project.fct_orders": fct_orders,
        "model.my_project.stg_orders": stg_orders,
        "model.my_project.circular_a": circular_a,
        "model.my_project.circular_b": circular_b,
    }
    manifest.sources = {
        "source.my_project.raw_users": source_node,
        "source.my_project.raw_orders": raw_orders,
    }

    return manifest


def test_build_ancestor_tree_simple(mock_manifest: Manifest) -> None:
    """Test building ancestor tree for a simple dependency chain."""
    node = mock_manifest.nodes["model.my_project.stg_users"]
    tree = _build_node_ancestor_tree(mock_manifest, node)

    assert "generation_0" in tree
    assert "model.my_project.stg_users" in tree["generation_0"]
    assert "generation_1" in tree
    assert "source.my_project.raw_users" in tree["generation_1"]


def test_build_ancestor_tree_multiple_generations(mock_manifest: Manifest) -> None:
    """Test building ancestor tree across multiple generations."""
    node = mock_manifest.nodes["model.my_project.int_users"]
    tree = _build_node_ancestor_tree(mock_manifest, node)

    assert "generation_0" in tree
    assert "model.my_project.int_users" in tree["generation_0"]
    assert "generation_1" in tree
    assert "model.my_project.stg_users" in tree["generation_1"]
    assert "generation_2" in tree
    assert "source.my_project.raw_users" in tree["generation_2"]


def test_build_ancestor_tree_multiple_parents(mock_manifest: Manifest) -> None:
    """Test building ancestor tree with multiple parents (diamond pattern)."""
    node = mock_manifest.nodes["model.my_project.fct_orders"]
    tree = _build_node_ancestor_tree(mock_manifest, node)

    assert "generation_0" in tree
    assert "model.my_project.fct_orders" in tree["generation_0"]
    assert "generation_1" in tree
    # Should have both parents
    assert "model.my_project.int_users" in tree["generation_1"]
    assert "model.my_project.stg_orders" in tree["generation_1"]


def test_build_ancestor_tree_circular_dependency(
    mock_manifest: Manifest, caplog: pytest.LogCaptureFixture
) -> None:
    """Test that circular dependencies are detected and handled gracefully."""
    import logging

    caplog.set_level(logging.WARNING)

    node = mock_manifest.nodes["model.my_project.circular_a"]
    tree = _build_node_ancestor_tree(mock_manifest, node)

    # Should not infinite loop - this is the main test
    assert "generation_0" in tree
    assert "model.my_project.circular_a" in tree["generation_0"]
    assert "generation_1" in tree
    assert "model.my_project.circular_b" in tree["generation_1"]

    # The tree should be finite (e.g., less than 10 generations)
    # If there was an infinite loop, this would fail or hang
    assert len(tree) < 10


def test_build_ancestor_tree_max_depth(
    mock_manifest: Manifest, caplog: pytest.LogCaptureFixture
) -> None:
    """Test that max depth limit prevents unbounded recursion."""
    import logging

    caplog.set_level(logging.WARNING)

    # Create a very deep chain
    node = mock_manifest.nodes["model.my_project.stg_users"]
    tree = _build_node_ancestor_tree(mock_manifest, node, max_depth=1)

    # Should respect max depth
    assert "generation_0" in tree
    assert "generation_1" in tree
    # generation_2 should not exist due to max_depth=1
    assert "generation_2" not in tree


def test_build_ancestor_tree_sorted_generations(mock_manifest: Manifest) -> None:
    """Test that generations are sorted for deterministic ordering."""
    node = mock_manifest.nodes["model.my_project.fct_orders"]
    tree = _build_node_ancestor_tree(mock_manifest, node)

    # Each generation should be sorted
    for generation in tree.values():
        assert generation == sorted(generation)


def test_build_ancestor_tree_filters_non_model_deps(mock_manifest: Manifest) -> None:
    """Test that only model/seed/source dependencies are included."""
    # Create a node with various dependency types
    node = mock.Mock(spec=ModelNode)
    node.unique_id = "model.my_project.test_model"
    node.resource_type = "model"
    node.name = "test_model"
    node.schema = "test"
    node.columns = {}
    node.depends_on = mock.Mock(
        nodes=[
            "model.my_project.stg_users",  # Should be included
            "test.my_project.some_test",  # Should be filtered out
        ]
    )
    node.meta = {}
    node.config = mock.Mock(extra={})

    tree = _build_node_ancestor_tree(mock_manifest, node)

    # Should only include model and source dependencies
    assert "generation_1" in tree
    assert "model.my_project.stg_users" in tree["generation_1"]
    # test should not be in generation_1
    assert "test.my_project.some_test" not in tree.get("generation_1", [])


def test_build_ancestor_tree_empty_depends_on(mock_manifest: Manifest) -> None:
    """Test building ancestor tree for node with no dependencies."""
    source_node = mock_manifest.sources["source.my_project.raw_users"]
    tree = _build_node_ancestor_tree(mock_manifest, source_node)

    # Should only have generation_0
    assert "generation_0" in tree
    assert "source.my_project.raw_users" in tree["generation_0"]
    assert len(tree) == 1
