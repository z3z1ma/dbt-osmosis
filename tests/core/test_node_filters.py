# pyright: reportPrivateImportUsage=false, reportPrivateUsage=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportAny=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportArgumentType=false, reportFunctionMemberAccess=false, reportUnknownVariableType=false, reportUnusedParameter=false

from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from dbt.artifacts.resources.types import NodeType

from dbt_osmosis.core.node_filters import _iter_candidate_nodes, _topological_sort


def _make_node(
    unique_id: str,
    name: str,
    *,
    package_name: str,
    resource_type: NodeType = NodeType.Model,
    materialized: str = "table",
) -> mock.MagicMock:
    node = mock.MagicMock()
    node.unique_id = unique_id
    node.name = name
    node.resource_type = resource_type
    node.package_name = package_name
    node.config = SimpleNamespace(materialized=materialized)
    node.original_file_path = f"models/{name}.sql"
    node.patch_path = None
    node.fqn = [package_name, "models", name]
    node.depends_on_nodes = []
    return node


def _make_context(
    *nodes: mock.MagicMock, include_external: bool = False, models: list[Path | str] | None = None
) -> SimpleNamespace:
    return SimpleNamespace(
        settings=SimpleNamespace(include_external=include_external, models=models or [], fqn=[]),
        project=SimpleNamespace(
            runtime_cfg=SimpleNamespace(project_name="my_project", project_root="/repo"),
            manifest=SimpleNamespace(
                nodes={
                    node.unique_id: node for node in nodes if node.resource_type != NodeType.Source
                },
                sources={
                    node.unique_id: node for node in nodes if node.resource_type == NodeType.Source
                },
            ),
        ),
    )


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


def test_iter_candidate_nodes_excludes_external_packages_by_default():
    local_node = _make_node(
        "model.my_project.local_model", "local_model", package_name="my_project"
    )
    external_node = _make_node(
        "model.package.external_model",
        "external_model",
        package_name="external_package",
    )
    context = _make_context(local_node, external_node)

    candidate_ids = [uid for uid, _ in _iter_candidate_nodes(context)]

    assert candidate_ids == ["model.my_project.local_model"]


def test_iter_candidate_nodes_uses_context_include_external_setting():
    local_node = _make_node(
        "model.my_project.local_model", "local_model", package_name="my_project"
    )
    external_node = _make_node(
        "source.package.external_source",
        "external_source",
        package_name="external_package",
        resource_type=NodeType.Source,
    )
    context = _make_context(local_node, external_node, include_external=True)

    candidate_ids = {uid for uid, _ in _iter_candidate_nodes(context)}

    assert candidate_ids == {
        "model.my_project.local_model",
        "source.package.external_source",
    }


def test_iter_candidate_nodes_keeps_model_filters_when_external_included():
    selected_node = _make_node(
        "model.my_project.selected_model",
        "selected_model",
        package_name="my_project",
    )
    external_node = _make_node(
        "model.package.external_model",
        "external_model",
        package_name="external_package",
    )
    context = _make_context(
        selected_node,
        external_node,
        include_external=True,
        models=[Path("selected_model.sql")],
    )

    candidate_ids = [uid for uid, _ in _iter_candidate_nodes(context)]

    assert candidate_ids == ["model.my_project.selected_model"]
