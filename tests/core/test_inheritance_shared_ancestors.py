"""Shared lineage is not a cycle, and must not change inheritance precedence."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from dbt_osmosis.core.inheritance import _build_node_ancestor_tree
from dbt_osmosis.core.transforms import inherit_upstream_column_knowledge


def _manifest(edges):
    nodes = {
        name: SimpleNamespace(unique_id=name, depends_on=SimpleNamespace(nodes=parents))
        for name, parents in edges.items()
    }
    return SimpleNamespace(nodes=nodes, sources={})


@pytest.mark.parametrize("shared_kind", ["model", "seed", "source"])
def test_diamond_has_no_cycle_warning(shared_kind):
    root, left, right = (f"model.pkg.{name}" for name in ("root", "left", "right"))
    shared = f"{shared_kind}.pkg.shared"
    manifest = _manifest({root: [right, left], left: [shared], right: [shared], shared: []})
    if shared_kind == "source":
        manifest.sources[shared] = manifest.nodes.pop(shared)

    with patch("dbt_osmosis.core.inheritance.logger.warning") as warning:
        tree = _build_node_ancestor_tree(manifest, manifest.nodes[root])

    assert tree == {
        "generation_0": [root],
        "generation_1": [left, right],
        "generation_2": [shared],
    }
    warning.assert_not_called()


@pytest.mark.parametrize("short_path_first", [False, True])
def test_unequal_paths_preserve_first_discovery_depth(short_path_first):
    root, branch, shared, upstream = (
        f"model.pkg.{name}" for name in ("root", "branch", "shared", "upstream")
    )
    parents = [shared, branch] if short_path_first else [branch, shared]
    manifest = _manifest({root: parents, branch: [shared], shared: [upstream], upstream: []})
    with patch("dbt_osmosis.core.inheritance.logger.warning") as warning:
        tree = _build_node_ancestor_tree(manifest, manifest.nodes[root])

    expected = (
        {"generation_0": [root], "generation_1": [branch, shared], "generation_2": [upstream]}
        if short_path_first
        else {
            "generation_0": [root],
            "generation_1": [branch],
            "generation_2": [shared],
            "generation_3": [upstream],
        }
    )
    assert tree == expected
    warning.assert_not_called()


@pytest.mark.parametrize("cycle_target", ["root", "parent", "child"])
def test_real_cycles_warn_and_terminate(cycle_target):
    root, parent, child = (f"model.pkg.{name}" for name in ("root", "parent", "child"))
    target = f"model.pkg.{cycle_target}"
    manifest = _manifest({root: [parent], parent: [child], child: [target]})
    with patch("dbt_osmosis.core.inheritance.logger.warning") as warning:
        tree = _build_node_ancestor_tree(manifest, manifest.nodes[root])

    assert tree == {
        "generation_0": [root],
        "generation_1": [parent],
        "generation_2": [child],
    }
    warning.assert_called_once_with(
        ":rotating_light: Circular dependency detected: %s -> %s", child, target
    )


@pytest.mark.parametrize("reverse_parents", [False, True])
@pytest.mark.parametrize("parent_descriptions", [False, True])
def test_diamond_inherits_shared_documentation_and_preserves_precedence(
    yaml_context, reverse_parents, parent_descriptions
):
    manifest = yaml_context.project.manifest
    template = manifest.nodes["model.jaffle_shop_duckdb.stg_customers.v1"]
    nodes = {}
    for name in ("root", "left", "right", "shared"):
        node = deepcopy(template)
        node.unique_id = f"model.jaffle_shop_duckdb.{name}"
        node.name = name
        node.patch_path = None
        node.columns = {"first_name": deepcopy(template.columns["first_name"])}
        node.columns["first_name"].description = ""
        nodes[name] = node
        manifest.nodes[node.unique_id] = node
    nodes["shared"].depends_on.nodes = []
    nodes["shared"].columns["first_name"].description = "Shared lookup documentation"
    for name in ("left", "right"):
        nodes[name].depends_on.nodes = [nodes["shared"].unique_id]
        if parent_descriptions:
            nodes[name].columns["first_name"].description = f"{name} documentation"
    parents = [nodes["left"].unique_id, nodes["right"].unique_id]
    nodes["root"].depends_on.nodes = parents[::-1] if reverse_parents else parents
    yaml_context.settings.force_inherit_descriptions = True
    yaml_context.settings.use_unrendered_descriptions = False

    with patch("dbt_osmosis.core.inheritance.logger.warning") as warning:
        inherit_upstream_column_knowledge(yaml_context, nodes["root"])

    expected = "left documentation" if parent_descriptions else "Shared lookup documentation"
    assert nodes["root"].columns["first_name"].description == expected
    warning.assert_not_called()
