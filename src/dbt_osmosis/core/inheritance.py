from __future__ import annotations

import typing as t
from types import MappingProxyType

from dbt.contracts.graph.manifest import Manifest
from dbt.contracts.graph.nodes import ModelNode, ResultNode, SeedNode, SourceDefinition

import dbt_osmosis.core.logger as logger

__all__ = [
    "_build_node_ancestor_tree",
    "_get_node_yaml",
    "_build_column_knowledge_graph",
]


def _build_node_ancestor_tree(
    manifest: Manifest,
    node: ResultNode,
    tree: dict[str, list[str]] | None = None,
    visited: set[str] | None = None,
    depth: int = 1,
) -> dict[str, list[str]]:
    """Build a flat graph of a node and it's ancestors."""
    logger.debug(":seedling: Building ancestor tree/branch for => %s", node.unique_id)
    if tree is None or visited is None:
        visited = set(node.unique_id)
        tree = {"generation_0": [node.unique_id]}
        depth = 1

    if not hasattr(node, "depends_on"):
        return tree

    for dep in getattr(node.depends_on, "nodes", []):
        if not dep.startswith(("model.", "seed.", "source.")):
            continue
        if dep not in visited:
            visited.add(dep)
            member = manifest.nodes.get(dep, manifest.sources.get(dep))
            if member:
                tree.setdefault(f"generation_{depth}", []).append(dep)
                _ = _build_node_ancestor_tree(manifest, member, tree, visited, depth + 1)

    for generation in tree.values():
        generation.sort()  # For deterministic ordering

    return tree


def _get_node_yaml(context: t.Any, member: ResultNode) -> MappingProxyType[str, t.Any] | None:
    """Get a read-only view of the parsed YAML for a dbt model or source node."""
    from pathlib import Path

    from dbt_osmosis.core.introspection import _find_first
    from dbt_osmosis.core.schema.reader import _read_yaml

    project_dir = Path(context.project.runtime_cfg.project_root)

    if isinstance(member, SourceDefinition):
        if not member.original_file_path:
            return None
        path = project_dir.joinpath(member.original_file_path)
        sources = t.cast(
            list[dict[str, t.Any]],
            _read_yaml(context.yaml_handler, context.yaml_handler_lock, path).get("sources", []),
        )
        source = _find_first(sources, lambda s: s["name"] == member.source_name, {})
        tables = source.get("tables", [])
        maybe_doc = _find_first(tables, lambda tbl: tbl["name"] == member.name)
        if maybe_doc is not None:
            return MappingProxyType(maybe_doc)

    elif isinstance(member, (ModelNode, SeedNode)):
        if not member.patch_path:
            return None
        path = project_dir.joinpath(member.patch_path.split("://")[-1])
        section = f"{member.resource_type}s"
        models = t.cast(
            list[dict[str, t.Any]],
            _read_yaml(context.yaml_handler, context.yaml_handler_lock, path).get(section, []),
        )
        maybe_doc = _find_first(models, lambda model: model["name"] == member.name)
        if maybe_doc is not None:
            return MappingProxyType(maybe_doc)

    return None


def _build_column_knowledge_graph(context: t.Any, node: ResultNode) -> dict[str, dict[str, t.Any]]:
    """Generate a column knowledge graph for a dbt model or source node."""
    tree = _build_node_ancestor_tree(context.project.manifest, node)
    logger.debug(":family_tree: Node ancestor tree => %s", tree)

    # Get the default_progenitor from the node's meta, if it exists
    node_yaml = _get_node_yaml(context, node)
    default_progenitor = None
    if node_yaml:
        default_progenitor = node_yaml.get("meta", {}).get("default_progenitor")

    from dbt_osmosis.core.plugins import get_plugin_manager

    pm = get_plugin_manager()
    node_column_variants: dict[str, list[str]] = {}
    for column_name, _ in node.columns.items():
        variants = node_column_variants.setdefault(column_name, [column_name])
        for v in pm.hook.get_candidates(name=column_name, node=node, context=context.project):
            variants.extend(t.cast(list[str], v))

    def _get_unrendered(k: str, ancestor: ResultNode) -> t.Any:
        raw_yaml: t.Mapping[str, t.Any] = _get_node_yaml(context, ancestor) or {}
        raw_columns = t.cast(list[dict[str, t.Any]], raw_yaml.get("columns", []))
        from dbt_osmosis.core.introspection import _find_first, normalize_column_name

        raw_column_metadata = _find_first(
            raw_columns,
            lambda c: normalize_column_name(c["name"], context.project.runtime_cfg.credentials.type)
            in node_column_variants[name],
            {},
        )
        return raw_column_metadata.get(k)

    column_knowledge_graph: dict[str, dict[str, t.Any]] = {}
    column_progenitor_alternatives: dict[str, list[str]] = {}
    for generation in reversed(sorted(tree.keys())):
        ancestors = tree[generation]
        for ancestor_uid in ancestors:
            ancestor = context.project.manifest.nodes.get(
                ancestor_uid, context.project.manifest.sources.get(ancestor_uid)
            )
            if not isinstance(ancestor, (SourceDefinition, SeedNode, ModelNode)):
                continue

            for name, _ in node.columns.items():
                graph_node = column_knowledge_graph.setdefault(name, {})
                alternatives = column_progenitor_alternatives.setdefault(name, [])


                for variant in node_column_variants[name]:
                    incoming = ancestor.columns.get(variant)
                    if incoming is not None:
                        # Track this ancestor as a potential progenitor (but exclude self-reference)
                        if ancestor.unique_id != node.unique_id and ancestor.unique_id not in alternatives:
                            alternatives.append(ancestor.unique_id)
                        break
                else:
                    continue

                # Default: use current ancestor
                selected_ancestor = ancestor

                graph_edge = incoming.to_dict(omit_none=True)

                from dbt_osmosis.core.introspection import _get_setting_for_node

                if _get_setting_for_node(
                    "add-progenitor-to-meta",
                    node,
                    name,
                    fallback=context.settings.add_progenitor_to_meta,
                ):
                    # Only set the progenitor if it's not already set (first match wins)
                    if "osmosis_progenitor" not in graph_node.get("meta", {}):
                        graph_node.setdefault("meta", {}).setdefault(
                            "osmosis_progenitor", selected_ancestor.unique_id
                        )
                    # Always add the alternatives array
                    graph_node.setdefault("meta", {})["osmosis_progenitor_alternatives"] = alternatives.copy()

                if _get_setting_for_node(
                    "use-unrendered-descriptions",
                    node,
                    name,
                    fallback=context.settings.use_unrendered_descriptions,
                ):
                    if unrendered_description := _get_unrendered("description", selected_ancestor):
                        graph_edge["description"] = unrendered_description

                current_tags = graph_node.get("tags", [])
                if merged_tags := (set(graph_edge.pop("tags", [])) | set(current_tags)):
                    graph_edge["tags"] = list(merged_tags)

                current_meta = graph_node.get("meta", {})
                incoming_meta = graph_edge.pop("meta", {})

                # Preserve specific metadata keys that should not be overwritten by inheritance
                preserved_keys = ["column_default_progenitor", "osmosis_progenitor", "osmosis_progenitor_alternatives"]
                preserved_meta = {k: v for k, v in current_meta.items() if k in preserved_keys}

                # Merge with preserved keys taking precedence
                if merged_meta := {**current_meta, **incoming_meta, **preserved_meta}:
                    graph_edge["meta"] = merged_meta

                for inheritable in _get_setting_for_node(
                    "add-inheritance-for-specified-keys",
                    node,
                    name,
                    fallback=context.settings.add_inheritance_for_specified_keys,
                ):
                    current_val = graph_node.get(inheritable)
                    if incoming_unrendered_val := _get_unrendered(inheritable, selected_ancestor):
                        graph_edge[inheritable] = incoming_unrendered_val
                    elif incoming_val := graph_edge.pop(inheritable, current_val):
                        graph_edge[inheritable] = incoming_val

                from dbt_osmosis.core.settings import EMPTY_STRING

                if graph_edge.get("description", EMPTY_STRING) in context.placeholders or (
                    generation == "generation_0"
                    and _get_setting_for_node(
                        "force_inherit_descriptions",
                        node,
                        name,
                        fallback=context.settings.force_inherit_descriptions,
                    )
                ):
                    graph_edge.pop("description", None)
                if graph_edge.get("tags") == []:
                    del graph_edge["tags"]
                if graph_edge.get("meta") == {}:
                    del graph_edge["meta"]
                for k in list(graph_edge.keys()):
                    if graph_edge[k] is None:
                        graph_edge.pop(k)

                graph_node.update(graph_edge)

    # Second pass: Apply default_progenitor and column_default_progenitor overrides
    for name, graph_node in column_knowledge_graph.items():
        if "osmosis_progenitor" not in graph_node.get("meta", {}):
            continue

        current_progenitor = graph_node["meta"]["osmosis_progenitor"]
        alternatives = graph_node["meta"].get("osmosis_progenitor_alternatives", [])

        # Check for column_default_progenitor (highest priority)
        column_default_progenitor = None
        if node_yaml:
            columns = node_yaml.get("columns", [])
            from dbt_osmosis.core.introspection import _find_first
            column_meta = _find_first(columns, lambda c: c.get("name") == name, {})
            column_default_progenitor = column_meta.get("meta", {}).get("column_default_progenitor")

        new_progenitor = current_progenitor

        if column_default_progenitor:
            # Use column default progenitor if specified
            new_progenitor = column_default_progenitor
        elif (default_progenitor and
              len(alternatives) > 1 and
              default_progenitor in alternatives and
              current_progenitor != default_progenitor):
            # Use model-level default progenitor if available and different from current
            new_progenitor = default_progenitor

        # Update the progenitor if it changed
        if new_progenitor != current_progenitor:
            # Find the actual progenitor that the new progenitor inherits from
            new_ancestor = context.project.manifest.nodes.get(
                new_progenitor, context.project.manifest.sources.get(new_progenitor)
            )
            if new_ancestor and isinstance(new_ancestor, (SourceDefinition, SeedNode, ModelNode)):
                for variant in node_column_variants[name]:
                    new_incoming = new_ancestor.columns.get(variant)
                    if new_incoming is not None:
                        # Get the actual progenitor from the new ancestor's metadata
                        new_column_meta = new_incoming.to_dict(omit_none=True).get("meta", {})
                        actual_progenitor = new_column_meta.get("osmosis_progenitor")

                        if actual_progenitor:
                            # Use the actual progenitor (what the new ancestor inherits from)
                            graph_node["meta"]["osmosis_progenitor"] = actual_progenitor

                            # Inherit from the actual progenitor
                            actual_ancestor = context.project.manifest.nodes.get(
                                actual_progenitor, context.project.manifest.sources.get(actual_progenitor)
                            )
                            if actual_ancestor and isinstance(actual_ancestor, (SourceDefinition, SeedNode, ModelNode)):
                                for actual_variant in node_column_variants[name]:
                                    actual_incoming = actual_ancestor.columns.get(actual_variant)
                                    if actual_incoming is not None:
                                        actual_graph_edge = actual_incoming.to_dict(omit_none=True)

                                        # Get unrendered description if available
                                        if _get_setting_for_node(
                                            "use-unrendered-descriptions",
                                            node,
                                            name,
                                            fallback=context.settings.use_unrendered_descriptions,
                                        ):
                                            if unrendered_description := _get_unrendered("description", actual_ancestor):
                                                actual_graph_edge["description"] = unrendered_description

                                        # Update description and other inherited properties
                                        for key in ["description", "tags"]:
                                            if key in actual_graph_edge:
                                                graph_node[key] = actual_graph_edge[key]
                                        break
                        else:
                            # No osmosis_progenitor in the new ancestor, use it directly
                            graph_node["meta"]["osmosis_progenitor"] = new_progenitor
                            new_graph_edge = new_incoming.to_dict(omit_none=True)

                            # Get unrendered description if available
                            if _get_setting_for_node(
                                "use-unrendered-descriptions",
                                node,
                                name,
                                fallback=context.settings.use_unrendered_descriptions,
                            ):
                                if unrendered_description := _get_unrendered("description", new_ancestor):
                                    new_graph_edge["description"] = unrendered_description

                            # Update description and other inherited properties
                            for key in ["description", "tags"]:
                                if key in new_graph_edge:
                                    graph_node[key] = new_graph_edge[key]

                        # Preserve column_default_progenitor if it was the reason for this change
                        if column_default_progenitor:
                            graph_node.setdefault("meta", {})["column_default_progenitor"] = column_default_progenitor
                        break

    return column_knowledge_graph
