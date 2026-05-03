from __future__ import annotations

from copy import deepcopy
import typing as t
from importlib import import_module
from types import MappingProxyType

from dbt.contracts.graph.nodes import ModelNode, ResultNode, SeedNode, SourceDefinition

if t.TYPE_CHECKING:
    from dbt_osmosis.core.dbt_protocols import YamlRefactorContextProtocol

from dbt_osmosis.core import logger

__all__ = [
    "_apply_progenitor_overrides",
    "_build_column_knowledge_graph",
    "_build_graph_edge",
    "_build_node_ancestor_tree",
    "_clean_graph_edge",
    "_collect_column_variants",
    "_column_to_dict",
    "_find_matching_column",
    "_get_inherited_metadata_from_progenitor",
    "_get_node_yaml",
    "_get_progenitor_override",
    "_get_unrendered",
    "_merge_graph_node_data",
]


def _column_to_dict(column: t.Any, **kwargs: t.Any) -> dict[str, t.Any]:
    """Convert a ColumnInfo to dict, handling missing config attribute in dbt-core 1.10+.

    In dbt-core 1.10+, ColumnInfo objects may not have a 'config' attribute set,
    which causes mashumaro serialization to fail. This helper ensures the attribute
    exists before calling to_dict().

    Args:
        column: A ColumnInfo object from dbt.artifacts.resources
        **kwargs: Additional arguments to pass to to_dict() (e.g., omit_none=True)

    Returns:
        Dictionary representation of the column

    """
    if not hasattr(column, "config"):
        try:
            module = import_module("dbt.artifacts.resources.v1.components")
            column_config = getattr(module, "ColumnConfig", None)
            if column_config is not None:
                t.cast("t.Any", column).config = column_config()
        except (ImportError, AttributeError):
            pass  # Older dbt version, attribute should already exist
    return column.to_dict(**kwargs)


def _apply_effective_column_metadata(column_data: dict[str, t.Any]) -> None:
    """Expose config.meta/config.tags through the inherited meta/tags fields."""
    from dbt_osmosis.core.introspection import (
        _get_effective_column_meta,
        _get_effective_column_tags,
    )

    if effective_meta := _get_effective_column_meta(column_data):
        column_data["meta"] = effective_meta
    if effective_tags := _get_effective_column_tags(column_data):
        column_data["tags"] = effective_tags


def _initialize_column_knowledge(column: t.Any, node: ResultNode) -> dict[str, t.Any]:
    """Normalize one local column into the knowledge-graph representation."""
    column_data = _column_to_dict(column, omit_none=True)
    _apply_effective_column_metadata(column_data)

    # Clear out self-referential progenitors left behind by prior runs.
    if column_data.get("meta", {}).get("osmosis_progenitor") == node.unique_id:
        column_data["meta"].pop("osmosis_progenitor", None)
        if not column_data["meta"]:
            column_data.pop("meta", None)

    # Handle the config.meta path used by fusion_compat output.
    if isinstance(column_data.get("config"), dict):
        config_meta = column_data["config"].get("meta", {})
        if config_meta.get("osmosis_progenitor") == node.unique_id:
            config_meta.pop("osmosis_progenitor", None)
            if not config_meta:
                column_data["config"].pop("meta", None)
            if not column_data["config"]:
                column_data.pop("config", None)

    # Match the existing graph-builder behavior by dropping empty scalars/collections.
    return {k: v for k, v in column_data.items() if v not in ("", [], ())}


def _strip_progenitor_override_controls(column_data: dict[str, t.Any]) -> None:
    """Remove local override-control metadata before inheriting from another node."""
    meta = column_data.get("meta")
    if isinstance(meta, dict):
        meta.pop("column_default_progenitor", None)
        if not meta:
            column_data.pop("meta", None)

    config = column_data.get("config")
    if isinstance(config, dict):
        config_meta = config.get("meta")
        if isinstance(config_meta, dict):
            config_meta.pop("column_default_progenitor", None)
            if not config_meta:
                config.pop("meta", None)
        if not config:
            column_data.pop("config", None)


def _preserve_column_progenitor_override(
    graph_node: dict[str, t.Any],
    node_yaml: t.Mapping[str, t.Any] | None,
    column_name: str,
) -> None:
    """Keep the target column's override metadata so sync does not erase it."""
    from dbt_osmosis.core.introspection import _find_first

    if not node_yaml:
        return

    column_meta = _find_first(
        node_yaml.get("columns", []),
        lambda c: c.get("name") == column_name,
        {},
    )
    override_value = column_meta.get("meta", {}).get("column_default_progenitor")
    if override_value:
        graph_node.setdefault("meta", {})["column_default_progenitor"] = override_value

    config_override_value = (
        column_meta.get("config", {}).get("meta", {}).get("column_default_progenitor")
    )
    if config_override_value:
        graph_node.setdefault("config", {}).setdefault("meta", {})["column_default_progenitor"] = (
            config_override_value
        )


def _build_node_ancestor_tree(
    manifest: t.Any,
    node: ResultNode,
    tree: dict[str, list[str]] | None = None,
    visited: set[str] | None = None,
    depth: int = 1,
    max_depth: int = 100,
) -> dict[str, list[str]]:
    """Build a flat graph of a node and it's ancestors."""
    logger.debug(":seedling: Building ancestor tree/branch for => %s", node.unique_id)
    if tree is None or visited is None:
        visited = set(node.unique_id)
        tree = {"generation_0": [node.unique_id]}
        depth = 1

    if not hasattr(node, "depends_on"):
        return tree

    # Prevent unbounded recursion
    if depth > max_depth:
        logger.warning(
            ":rotating_light: Max depth %d exceeded for node %s, possible circular dependency",
            max_depth,
            node.unique_id,
        )
        return tree

    for dep in getattr(node.depends_on, "nodes", []):
        if not dep.startswith(("model.", "seed.", "source.")):
            continue

        # Cycle detection: skip if already visited
        if dep in visited:
            logger.warning(
                ":rotating_light: Circular dependency detected: %s -> %s",
                node.unique_id,
                dep,
            )
            continue

        visited.add(dep)
        member = manifest.nodes.get(dep, manifest.sources.get(dep))
        if member:
            tree.setdefault(f"generation_{depth}", []).append(dep)
            _ = _build_node_ancestor_tree(manifest, member, tree, visited, depth + 1, max_depth)

    for generation in tree.values():
        generation.sort()  # For deterministic ordering

    return tree


def _get_node_yaml(
    context: YamlRefactorContextProtocol,
    member: ResultNode,
) -> MappingProxyType[str, t.Any] | None:
    """Get a read-only view of the parsed YAML for a dbt model or source node."""
    from pathlib import Path

    from dbt_osmosis.core.introspection import _find_first
    from dbt_osmosis.core.schema.reader import _read_yaml

    project_root = context.project.runtime_cfg.project_root
    if not project_root:
        return None
    project_dir = Path(project_root)

    if isinstance(member, SourceDefinition):
        if not member.original_file_path:
            return None
        path = project_dir.joinpath(member.original_file_path)
        sources = t.cast(
            "list[dict[str, t.Any]]",
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
            "list[dict[str, t.Any]]",
            _read_yaml(context.yaml_handler, context.yaml_handler_lock, path).get(section, []),
        )
        maybe_doc = _find_first(models, lambda model: model["name"] == member.name)
        if maybe_doc is not None:
            return MappingProxyType(maybe_doc)

    return None


def _collect_column_variants(
    context: YamlRefactorContextProtocol,
    node: ResultNode,
) -> dict[str, list[str]]:
    """Collect column variants from node columns and plugins."""
    from dbt_osmosis.core.plugins import get_plugin_manager

    pm = get_plugin_manager()
    node_column_variants: dict[str, list[str]] = {}
    for column_name, _ in node.columns.items():
        variants = node_column_variants.setdefault(column_name, [column_name])
        for v in pm.hook.get_candidates(name=column_name, node=node, context=context.project):
            variants.extend(t.cast("list[str]", v))

    return node_column_variants


def _get_unrendered(
    context: YamlRefactorContextProtocol,
    k: str,
    name: str,
    ancestor: ResultNode,
    node_column_variants: dict[str, list[str]],
) -> t.Any:
    """Get unrendered value for a column from ancestor YAML."""
    raw_yaml: t.Mapping[str, t.Any] = _get_node_yaml(context, ancestor) or {}
    raw_columns = t.cast("list[dict[str, t.Any]]", raw_yaml.get("columns", []))
    from dbt_osmosis.core.introspection import _find_first, normalize_column_name

    raw_column_metadata = _find_first(
        raw_columns,
        lambda c: (
            normalize_column_name(c["name"], context.project.runtime_cfg.credentials.type)
            in node_column_variants[name]
        ),
        {},
    )
    return raw_column_metadata.get(k)


def _build_graph_edge(
    context: YamlRefactorContextProtocol,
    node: ResultNode,
    name: str,
    incoming: t.Any,
    ancestor: ResultNode,
    node_column_variants: dict[str, list[str]],
) -> dict[str, t.Any]:
    """Build a graph edge from incoming column with inheritance applied."""
    graph_edge = _column_to_dict(incoming, omit_none=True)
    _apply_effective_column_metadata(graph_edge)

    from dbt_osmosis.core.introspection import _get_setting_for_node

    # Add progenitor to meta if configured
    if _get_setting_for_node(
        "add-progenitor-to-meta",
        node,
        name,
        fallback=context.settings.add_progenitor_to_meta,
    ):
        graph_edge.setdefault("meta", {})["osmosis_progenitor"] = ancestor.unique_id

    # Use unrendered descriptions if configured
    if _get_setting_for_node(
        "use-unrendered-descriptions",
        node,
        name,
        fallback=context.settings.use_unrendered_descriptions,
    ):
        if unrendered_description := _get_unrendered(
            context,
            "description",
            name,
            ancestor,
            node_column_variants,
        ):
            graph_edge["description"] = unrendered_description

    # Handle inheritance for specified keys
    for inheritable in _get_setting_for_node(
        "add-inheritance-for-specified-keys",
        node,
        name,
        fallback=context.settings.add_inheritance_for_specified_keys,
    ):
        current_val = graph_edge.get(inheritable)
        if incoming_unrendered_val := _get_unrendered(
            context,
            inheritable,
            name,
            ancestor,
            node_column_variants,
        ):
            graph_edge[inheritable] = incoming_unrendered_val
        elif incoming_val := graph_edge.pop(inheritable, current_val):
            graph_edge[inheritable] = incoming_val

    return graph_edge


def _clean_graph_edge(
    context: YamlRefactorContextProtocol,
    graph_edge: dict[str, t.Any],
    generation: str,
    node: ResultNode,
    name: str,
) -> None:
    """Clean up empty values and placeholder descriptions from graph edge."""
    from dbt_osmosis.core.introspection import _get_setting_for_node
    from dbt_osmosis.core.settings import EMPTY_STRING

    # Remove placeholder descriptions or force inherit if direct ancestor
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

    # Remove empty descriptions (that weren't caught by placeholder check)
    if graph_edge.get("description") == "":
        graph_edge.pop("description", None)

    # Remove empty tags and meta objects
    if graph_edge.get("tags") == []:
        del graph_edge["tags"]
    if graph_edge.get("meta") == {}:
        del graph_edge["meta"]

    # Clean up empty nested config entries (handles data from fusion_compat mode)
    if isinstance(graph_edge.get("config"), dict):
        config = graph_edge["config"]
        if config.get("meta", {}) == {}:
            config.pop("meta", None)
        if config.get("tags", []) == []:
            config.pop("tags", None)
        if not config:
            graph_edge.pop("config", None)

    # Remove None values
    for k in list(graph_edge.keys()):
        if graph_edge[k] is None:
            graph_edge.pop(k)


def _find_matching_column(ancestor: ResultNode, column_variants: list[str]) -> t.Any | None:
    """Find a matching column in ancestor from the given variants."""
    for variant in column_variants:
        incoming = ancestor.columns.get(variant)
        if incoming is not None:
            return incoming
    return None


def _merge_graph_node_data(
    graph_node: dict[str, t.Any],
    graph_edge: dict[str, t.Any],
) -> None:
    """Merge graph edge data into existing graph node, handling tags and meta merging."""
    # Merge top-level tags
    current_tags = graph_node.get("tags", [])
    if merged_tags := (set(graph_edge.pop("tags", [])) | set(current_tags)):
        graph_edge["tags"] = list(merged_tags)

    # Merge top-level meta, but preserve osmosis_progenitor from the first (farthest) generation
    # The osmosis_progenitor should always point to the original source, not intermediate sources
    current_meta = graph_node.get("meta", {})
    edge_meta = graph_edge.pop("meta", {})

    # Preserve existing osmosis_progenitor if it exists in current_meta
    progenitor = current_meta.get("osmosis_progenitor")
    if merged_meta := {**current_meta, **edge_meta}:
        graph_edge["meta"] = merged_meta
        # Restore the original progenitor if it existed
        if progenitor:
            graph_edge["meta"]["osmosis_progenitor"] = progenitor

    # Merge config-level meta and tags (handles data from fusion_compat mode)
    current_config = graph_node.get("config")
    edge_config = graph_edge.pop("config", None)
    if isinstance(current_config, dict) or isinstance(edge_config, dict):
        current_config = current_config if isinstance(current_config, dict) else {}
        edge_config = edge_config if isinstance(edge_config, dict) else {}
        # Merge config.meta
        current_config_meta = current_config.get("meta", {})
        edge_config_meta = edge_config.pop("meta", {})
        config_progenitor = current_config_meta.get("osmosis_progenitor")
        merged_config_meta = {**current_config_meta, **edge_config_meta}
        if config_progenitor:
            merged_config_meta["osmosis_progenitor"] = config_progenitor
        if merged_config_meta:
            edge_config["meta"] = merged_config_meta
        # Merge config.tags
        current_config_tags = current_config.get("tags", [])
        edge_config_tags = edge_config.pop("tags", [])
        if merged_config_tags := (set(edge_config_tags) | set(current_config_tags)):
            edge_config["tags"] = list(merged_config_tags)
        # Merge remaining config keys
        for k, v in current_config.items():
            if k not in edge_config:
                edge_config[k] = v
        if edge_config:
            graph_edge["config"] = edge_config

    # Update graph node with merged data
    graph_node.update(graph_edge)


def _get_progenitor_override(
    column_name: str,
    node_yaml: t.Mapping[str, t.Any] | None,
) -> str | None:
    """Get the progenitor override for a column.

    Checks for column-level override first (column_default_progenitor), then
    model-level default (default_progenitor).

    Args:
        column_name: Name of the column
        node_yaml: The parsed YAML for the node

    Returns:
        The unique_id of the override progenitor, or None

    """
    from dbt_osmosis.core.introspection import _find_first

    # Check for column-level override first (highest priority)
    # Check both top-level and config-nested meta (handles fusion_compat mode)
    if node_yaml:
        columns = t.cast("list[dict[str, t.Any]]", node_yaml.get("columns", []))
        column_meta = _find_first(columns, lambda c: c.get("name") == column_name, {})
        column_default_progenitor = column_meta.get("meta", {}).get(
            "column_default_progenitor"
        ) or column_meta.get("config", {}).get("meta", {}).get("column_default_progenitor")
        if column_default_progenitor:
            return column_default_progenitor

    # Check for model-level default
    if node_yaml:
        default_progenitor = node_yaml.get("meta", {}).get("default_progenitor")
        if default_progenitor:
            return default_progenitor

    return None


def _get_inherited_metadata_from_progenitor(
    context: YamlRefactorContextProtocol,
    column_name: str,
    progenitor_id: str,
    node_column_variants: dict[str, list[str]],
    progenitor_knowledge_cache: dict[str, dict[str, dict[str, t.Any]]] | None = None,
) -> dict[str, t.Any] | None:
    """Get inherited metadata from a specific progenitor.

    Args:
        context: The refactor context
        column_name: Name of the column
        progenitor_id: The unique_id of the progenitor
        node_column_variants: Column name variants
        progenitor_knowledge_cache: Optional cache of previously-built knowledge graphs

    Returns:
        Dictionary of inherited metadata, or None if progenitor not found

    """
    progenitor = context.project.manifest.nodes.get(
        progenitor_id,
        context.project.manifest.sources.get(progenitor_id),
    )
    if not isinstance(progenitor, (SourceDefinition, SeedNode, ModelNode)):
        return None

    progenitor_knowledge = None
    if progenitor_knowledge_cache is not None:
        progenitor_knowledge = progenitor_knowledge_cache.get(progenitor_id)
    if progenitor_knowledge is None:
        progenitor_knowledge = _build_column_knowledge_graph(context, progenitor)
        if progenitor_knowledge_cache is not None:
            progenitor_knowledge_cache[progenitor_id] = progenitor_knowledge

    # Find the concrete column name in the progenitor, then reuse that node's
    # already-merged knowledge graph so overrides inherit the same contract that
    # downstream lineage would see from that node.
    incoming = _find_matching_column(progenitor, node_column_variants[column_name])
    if incoming is None:
        return None

    graph_edge = progenitor_knowledge.get(getattr(incoming, "name", column_name))
    if graph_edge is None:
        return None

    graph_edge = deepcopy(graph_edge)
    _strip_progenitor_override_controls(graph_edge)
    return graph_edge


def _apply_progenitor_overrides(
    context: YamlRefactorContextProtocol,
    node: ResultNode,
    column_knowledge_graph: dict[str, dict[str, t.Any]],
    progenitor_alternatives: dict[str, list[str]],
    node_yaml: t.Mapping[str, t.Any] | None,
    node_column_variants: dict[str, list[str]],
) -> None:
    """Apply progenitor overrides based on column_default_progenitor and default_progenitor.

    This is a second pass that allows users to override the automatically selected
    progenitor with a specific ancestor. This is useful when you want to inherit
    from a specific upstream source rather than the closest one.

    Args:
        context: The refactor context
        node: The dbt node
        column_knowledge_graph: The column knowledge graph (modified in-place)
        progenitor_alternatives: Map of column name to list of potential progenitors
        node_yaml: The parsed YAML for the node
        node_column_variants: Column name variants

    """
    progenitor_knowledge_cache: dict[str, dict[str, dict[str, t.Any]]] = {}

    for column_name, graph_node in column_knowledge_graph.items():
        # Check both top-level and config-nested meta for progenitor
        current_progenitor = graph_node.get("meta", {}).get("osmosis_progenitor") or graph_node.get(
            "config", {}
        ).get("meta", {}).get("osmosis_progenitor")
        alternatives = progenitor_alternatives.get(column_name, [])

        # Get the override progenitor (column-level takes precedence over model-level)
        override_progenitor = _get_progenitor_override(column_name, node_yaml)

        if not override_progenitor:
            continue

        # Only apply the override when it points at a real ancestor. When progenitor
        # tracking is disabled there may be no current_progenitor, but the override
        # still needs to select a different inheritance source.
        if override_progenitor not in alternatives or override_progenitor == current_progenitor:
            continue

        logger.debug(
            ":fork_and_knife: Applying progenitor override for column %s: %s -> %s",
            column_name,
            current_progenitor,
            override_progenitor,
        )

        # Get inherited metadata from the override progenitor
        inherited = _get_inherited_metadata_from_progenitor(
            context,
            column_name,
            override_progenitor,
            node_column_variants,
            progenitor_knowledge_cache,
        )

        if not inherited:
            logger.warning(
                ":warning: Could not find progenitor %s for column %s",
                override_progenitor,
                column_name,
            )
            continue

        overridden_graph_node = _initialize_column_knowledge(node.columns[column_name], node)
        _merge_graph_node_data(overridden_graph_node, inherited)
        _preserve_column_progenitor_override(overridden_graph_node, node_yaml, column_name)

        graph_node.clear()
        graph_node.update(overridden_graph_node)


def _build_column_knowledge_graph(
    context: YamlRefactorContextProtocol,
    node: ResultNode,
) -> dict[str, dict[str, t.Any]]:
    """Generate a column knowledge graph for a dbt model or source node."""
    tree = _build_node_ancestor_tree(context.project.manifest, node)
    logger.debug(":family_tree: Node ancestor tree => %s", tree)

    node_yaml = _get_node_yaml(context, node)
    node_column_variants = _collect_column_variants(context, node)

    # Initialize the column knowledge graph with the local node's column data
    # This ensures local metadata is preserved and merged with inherited metadata
    column_knowledge_graph: dict[str, dict[str, t.Any]] = {}
    for name, column in node.columns.items():
        column_knowledge_graph[name] = _initialize_column_knowledge(column, node)

    # Track which columns have been processed in each generation to avoid
    # multiple ancestors in the same generation from overwriting each other
    processed_columns_in_generation: dict[str, set[str]] = {}

    # Track potential progenitor alternatives for each column
    # This allows for column-level and model-level progenitor overrides
    progenitor_alternatives: dict[str, list[str]] = {}

    # Process ancestors from farthest to closest
    for generation in reversed(sorted(tree.keys())):
        ancestors = tree[generation]
        processed_columns_in_generation[generation] = set()

        for ancestor_uid in ancestors:
            ancestor = context.project.manifest.nodes.get(
                ancestor_uid,
                context.project.manifest.sources.get(ancestor_uid),
            )
            if not isinstance(ancestor, (SourceDefinition, SeedNode, ModelNode)):
                continue

            # Special handling for the target node itself in generation_0:
            # The target node should only be processed for columns that don't exist
            # in any upstream source (i.e., columns that originate in this model).
            if ancestor_uid == node.unique_id:
                # Only process columns that haven't been found in any upstream ancestor yet
                for name in node.columns.keys():
                    if name in processed_columns_in_generation[generation]:
                        continue
                    # Only process if this column hasn't been processed in ANY generation
                    # (meaning it doesn't exist in any upstream source)
                    if not any(name in cols for cols in processed_columns_in_generation.values()):
                        # For columns originating in the target node, set it as the progenitor
                        # This provides useful information for tracking column lineage
                        from dbt_osmosis.core.introspection import _get_setting_for_node

                        if _get_setting_for_node(
                            "add-progenitor-to-meta",
                            node,
                            name,
                            fallback=context.settings.add_progenitor_to_meta,
                        ):
                            # Get the current column data to build the edge
                            incoming = node.columns[name]
                            graph_edge = _column_to_dict(incoming, omit_none=True)
                            _apply_effective_column_metadata(graph_edge)
                            # Set osmosis_progenitor to the target node itself
                            graph_edge.setdefault("meta", {})["osmosis_progenitor"] = node.unique_id

                            # Clean up empty values (like empty descriptions)
                            _clean_graph_edge(context, graph_edge, generation, node, name)

                            # Mark as processed
                            processed_columns_in_generation[generation].add(name)

                            # Merge with existing graph node
                            graph_node = column_knowledge_graph.setdefault(name, {})
                            _merge_graph_node_data(graph_node, graph_edge)
                continue

            # Process each column in the target node
            for name, _ in node.columns.items():
                # Skip if this column was already processed in this generation
                if name in processed_columns_in_generation[generation]:
                    continue

                # Find matching column in ancestor
                incoming = _find_matching_column(ancestor, node_column_variants[name])
                if incoming is None:
                    continue

                # Track this ancestor as a potential progenitor alternative
                # (excluding self-reference which happens in generation_0 above)
                if ancestor_uid != node.unique_id:
                    alternatives = progenitor_alternatives.setdefault(name, [])
                    if ancestor_uid not in alternatives:
                        alternatives.append(ancestor_uid)

                # Mark this column as processed in this generation
                processed_columns_in_generation[generation].add(name)

                # Build graph edge with inheritance applied
                graph_edge = _build_graph_edge(
                    context,
                    node,
                    name,
                    incoming,
                    ancestor,
                    node_column_variants,
                )

                # Clean up empty values and placeholders
                _clean_graph_edge(context, graph_edge, generation, node, name)

                # Merge with existing graph node (which already has local column data)
                graph_node = column_knowledge_graph.setdefault(name, {})
                _merge_graph_node_data(graph_node, graph_edge)

    # Apply progenitor overrides based on column_default_progenitor and default_progenitor
    # This is a second pass that allows users to override the automatically selected
    # progenitor with a specific ancestor
    _apply_progenitor_overrides(
        context,
        node,
        column_knowledge_graph,
        progenitor_alternatives,
        node_yaml,
        node_column_variants,
    )

    return column_knowledge_graph
