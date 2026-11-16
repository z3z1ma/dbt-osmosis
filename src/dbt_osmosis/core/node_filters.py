from __future__ import annotations

import typing as t
from collections import defaultdict, deque
from itertools import chain
from pathlib import Path

from dbt.artifacts.resources.types import NodeType
from dbt.contracts.graph.nodes import ResultNode

from dbt_osmosis.core import logger

__all__ = [
    "_is_file_match",
    "_is_fqn_match",
    "_iter_candidate_nodes",
    "_topological_sort",
]


def _is_fqn_match(node: ResultNode, fqns: list[str]) -> bool:
    """Filter models based on the provided fully qualified name matching on partial segments."""
    logger.debug(":mag_right: Checking if node => %s matches any FQNs => %s", node.unique_id, fqns)
    for fqn_str in fqns:
        parts = fqn_str.split(".")
        segment_match = len(node.fqn[1:]) >= len(parts) and all(
            left == right for left, right in zip(parts, node.fqn[1:])
        )
        if segment_match:
            logger.debug(":white_check_mark: FQN matched => %s", fqn_str)
            return True
    return False


def _node_sql_path(node: ResultNode, root: Path | str) -> Path:
    return Path(root, node.original_file_path).resolve()


def _node_yaml_path(node: ResultNode, root: Path | str) -> Path | None:
    if not node.patch_path:
        return None

    absolute_patch_path = Path(root, node.patch_path.partition("://")[-1]).resolve()
    return absolute_patch_path if absolute_patch_path.exists() else None


def _is_directory_match(candidate: Path, node_path: Path, yaml_path: Path | None) -> bool:
    return candidate.is_dir() and (
        candidate in node_path.parents or (yaml_path is not None and candidate in yaml_path.parents)
    )


def _is_file_path_match(candidate: Path, node_path: Path, yaml_path: Path | None) -> bool:
    return candidate.is_file() and (
        candidate.samefile(node_path) or (yaml_path is not None and candidate.samefile(yaml_path))
    )


def _is_path_candidate_match(
    node: ResultNode,
    candidate: Path,
    node_path: Path,
    yaml_path: Path | None,
) -> bool:
    if node.name == candidate.stem:
        logger.debug(":white_check_mark: Name match => %s", candidate)
        return True
    if _is_directory_match(candidate, node_path, yaml_path):
        logger.debug(":white_check_mark: Directory path match => %s", candidate)
        return True
    if _is_file_path_match(candidate, node_path, yaml_path):
        logger.debug(":white_check_mark: File path match => %s", candidate)
        return True
    return False


def _is_file_match(node: ResultNode, paths: list[Path | str], root: Path | str) -> bool:
    """Check if a node's file path matches any of the provided file paths or names."""
    node_path = _node_sql_path(node, root)
    yaml_path = _node_yaml_path(node, root)
    for model_or_dir in paths:
        candidate = Path(model_or_dir).resolve()
        if _is_path_candidate_match(node, candidate, node_path, yaml_path):
            return True
    return False


def _initial_in_degrees(candidate_nodes: list[tuple[str, ResultNode]]) -> defaultdict[str, int]:
    in_degree: defaultdict[str, int] = defaultdict(int)
    for uid, _ in candidate_nodes:
        in_degree[uid] = 0
    return in_degree


def _dependency_graph(
    candidate_nodes: list[tuple[str, ResultNode]],
) -> tuple[defaultdict[str, set[str]], defaultdict[str, int]]:
    adjacency: defaultdict[str, set[str]] = defaultdict(set)
    in_degree = _initial_in_degrees(candidate_nodes)
    all_uids = {uid for uid, _ in candidate_nodes}

    for uid, node in candidate_nodes:
        for dep_uid in node.depends_on_nodes:
            if dep_uid in all_uids:
                adjacency[dep_uid].add(uid)
                in_degree[uid] += 1
    return adjacency, in_degree


def _visit_topological_nodes(
    adjacency: defaultdict[str, set[str]],
    in_degree: defaultdict[str, int],
) -> list[str]:
    queue: deque[str] = deque([uid for uid, deg in in_degree.items() if deg == 0])
    sorted_uids: list[str] = []
    while queue:
        parent_uid = queue.popleft()
        sorted_uids.append(parent_uid)

        for child_uid in adjacency[parent_uid]:
            in_degree[child_uid] -= 1
            if in_degree[child_uid] == 0:
                queue.append(child_uid)
    return sorted_uids


def _topological_sort(
    candidate_nodes: list[tuple[str, ResultNode]],
) -> list[tuple[str, ResultNode]]:
    """Perform a topological sort on the given candidate_nodes (uid, node) pairs
    based on their dependencies. If a cycle is detected, raise a ValueError.

    Kahn's Algorithm:
      1) Build adjacency list: parent -> {child, child, ...}
         (Because if node 'child' depends on 'parent', we have an edge parent->child).
      2) Compute in-degrees for all nodes.
      3) Collect all nodes with in-degree == 0 into a queue.
      4) Repeatedly pop from queue and 'visit' that node,
         then decrement the in-degree of its children.
         If any child's in-degree becomes 0, push it into the queue.
      5) If we visited all nodes, we have a valid topological order.
         Otherwise, a cycle exists.
    """
    adjacency, in_degree = _dependency_graph(candidate_nodes)
    sorted_uids = _visit_topological_nodes(adjacency, in_degree)

    if len(sorted_uids) < len(candidate_nodes):
        raise ValueError(
            "Cycle detected in node dependencies. Cannot produce a valid topological order.",
        )

    uid_to_node = dict(candidate_nodes)
    return [(uid, uid_to_node[uid]) for uid in sorted_uids]


def _is_candidate_node(
    context: t.Any, node: ResultNode, *, include_external: bool, exclude_external_patches: bool
) -> bool:
    if node.resource_type not in (NodeType.Model, NodeType.Source, NodeType.Seed):
        return False
    if node.package_name != context.project.runtime_cfg.project_name and not include_external:
        return False
    if (
        exclude_external_patches
        and not include_external
        and node.resource_type in (NodeType.Model, NodeType.Seed)
    ):
        patch_path = node.patch_path
        if patch_path:
            patch_package, separator, _ = patch_path.partition("://")
            if separator and patch_package != context.project.runtime_cfg.project_name:
                # A local override can be documented by an installed package. Treat its
                # patch as external too, or root-relative writes create duplicate patches.
                return False
    if node.resource_type == NodeType.Model and node.config.materialized == "ephemeral":
        return False
    if context.settings.models and not _is_file_match(
        node,
        context.settings.models,
        context.project.runtime_cfg.project_root,
    ):
        return False
    return not (context.settings.fqn and not _is_fqn_match(node, context.settings.fqn))


def _manifest_nodes(context: t.Any) -> t.Iterator[tuple[str, ResultNode]]:
    yield from chain(
        context.project.manifest.nodes.items(), context.project.manifest.sources.items()
    )


def _iter_candidate_nodes(
    context: t.Any,  # YamlRefactorContext type will be imported
    *,
    exclude_external_patches: bool = False,
) -> t.Iterator[tuple[str, ResultNode]]:
    """Select nodes, optionally excluding package-owned patches for YAML management.

    Read-only consumers such as SQL lint still need locally owned nodes regardless
    of which package documents them.
    """
    logger.debug(
        ":mag: Filtering nodes (models/sources/seeds) with user-specified settings => %s",
        context.settings,
    )

    include_external = context.settings.include_external

    candidate_nodes: list[t.Any] = []
    for uid, dbt_node in _manifest_nodes(context):
        if _is_candidate_node(
            context,
            dbt_node,
            include_external=include_external,
            exclude_external_patches=exclude_external_patches,
        ):
            logger.debug(
                ":white_check_mark: Node => %s passed filtering logic.", dbt_node.unique_id
            )
            candidate_nodes.append((uid, dbt_node))

    for uid, node in _topological_sort(candidate_nodes):
        yield uid, node
