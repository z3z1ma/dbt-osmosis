from __future__ import annotations

import typing as t
from functools import partial
from pathlib import Path

from dbt.contracts.graph.nodes import ModelNode, ResultNode
from dbt.node_types import NodeType

if t.TYPE_CHECKING:
    from dbt_osmosis.core.dbt_protocols import YamlRefactorContextProtocol

import dbt_osmosis.core.logger as logger

__all__ = [
    "_sync_doc_section",
    "sync_node_to_yaml",
]


def _sync_doc_section(
    context: YamlRefactorContextProtocol, node: ResultNode, doc_section: dict[str, t.Any]
) -> None:
    """Helper function that overwrites 'doc_section' with data from 'node'.

    This includes columns, description, meta, tags, etc.
    We assume node is the single source of truth, so doc_section is replaced.

    If a catalog is available (via --catalog-path), data_type from the catalog
    takes precedence over the manifest's data_type.
    """
    logger.debug(":arrows_counterclockwise: Syncing doc_section with node => %s", node.unique_id)
    if node.description and not doc_section.get("description"):
        doc_section["description"] = node.description

    current_columns: list[dict[str, t.Any]] = doc_section.setdefault("columns", [])
    incoming_columns: list[dict[str, t.Any]] = []

    current_map = {}
    for c in current_columns:
        from dbt_osmosis.core.introspection import normalize_column_name

        norm_name = normalize_column_name(c["name"], context.project.runtime_cfg.credentials.type)
        current_map[norm_name] = c

    # Build a map of catalog column types if catalog is available
    catalog_column_types: dict[str, str] = {}
    if catalog := context.read_catalog():
        from itertools import chain

        from dbt_osmosis.core.introspection import _find_first, normalize_column_name

        catalog_entry = _find_first(
            chain(catalog.nodes.values(), catalog.sources.values()),
            lambda c: (
                c.metadata.name == node.name
                and c.metadata.schema == node.schema
                and (not hasattr(node, "source_name") or c.metadata.source == node.source_name)
            ),
        )
        if catalog_entry:
            for col_name, col_meta in catalog_entry.columns.items():
                norm_name = normalize_column_name(
                    col_name, context.project.runtime_cfg.credentials.type
                )
                catalog_column_types[norm_name] = col_meta.type

    for name, meta in node.columns.items():
        # Null check: validate meta exists before calling to_dict
        if meta is None:
            logger.warning(
                ":warning: Column %s has None metadata in node %s, skipping",
                name,
                node.unique_id,
            )
            continue
        cdict = meta.to_dict(omit_none=True)
        # Filter out 'config' and 'doc_blocks' fields added in dbt-core >= 1.9.6
        # These contain redundant meta/tags that duplicate top-level fields
        cdict = {k: v for k, v in cdict.items() if k not in ("config", "doc_blocks")}
        cdict["name"] = name
        from dbt_osmosis.core.introspection import _get_setting_for_node, normalize_column_name

        norm_name = normalize_column_name(name, context.project.runtime_cfg.credentials.type)

        # Use catalog data_type if available, otherwise use manifest's data_type
        if norm_name in catalog_column_types:
            cdict["data_type"] = catalog_column_types[norm_name]

        current_yaml = t.cast(dict[str, t.Any], current_map.get(norm_name, {}))
        merged = dict(current_yaml)

        skip_add_types = _get_setting_for_node(
            "skip-add-data-types", node, name, fallback=context.settings.skip_add_data_types
        )

        # Check if we should preserve unrendered descriptions from current YAML
        # When use_unrendered_descriptions is True and the current description contains
        # doc blocks (e.g., {{ doc(...) }} or {% docs %}{% enddocs %}), we should
        # preserve the unrendered version instead of using the rendered version from manifest
        current_description = current_yaml.get("description")
        use_unrendered = _get_setting_for_node(
            "use-unrendered-descriptions",
            node,
            name,
            fallback=context.settings.use_unrendered_descriptions,
        )
        preserve_current_description = False
        if use_unrendered and current_description:
            # Check if current description contains unrendered doc blocks
            if "{{ doc(" in current_description or "{% docs " in current_description:
                preserve_current_description = True

        for k, v in cdict.items():
            if k == "data_type" and skip_add_types:
                # don't add data types if told not to
                continue
            if k == "constraints" and "constraints" in merged:
                # keep constraints as is if present, mashumaro dumps too much info :shrug:
                continue
            if k == "description" and preserve_current_description:
                # Preserve the unrendered description from current YAML
                continue
            merged[k] = v

        if merged.get("description") is None:
            merged.pop("description", None)
        if merged.get("tags", []) == []:
            merged.pop("tags", None)
        if merged.get("meta", {}) == {}:
            merged.pop("meta", None)

        for k in set(merged.keys()) - {"name", "description", "tags", "meta"}:
            if merged[k] in (None, [], {}):
                merged.pop(k)

        if _get_setting_for_node(
            "output-to-lower", node, name, fallback=context.settings.output_to_lower
        ):
            merged["name"] = merged["name"].lower()

        incoming_columns.append(merged)

    doc_section["columns"] = incoming_columns


def _get_resource_type_key(node: ResultNode) -> str:
    """Get the resource type key for a node (e.g., 'models', 'sources', 'seeds')."""
    if node.resource_type == NodeType.Source:
        return "sources"
    if node.resource_type == NodeType.Seed:
        return "seeds"
    return "models"


def _prepare_yaml_document(
    context: YamlRefactorContextProtocol, node: ResultNode, current_path: t.Optional[Path]
) -> dict[str, t.Any]:
    """Prepare or load the YAML document for a node.

    Returns the document dict, creating a minimal structure if needed.
    """
    from dbt_osmosis.core.path_management import get_target_yaml_path
    from dbt_osmosis.core.schema.reader import _read_yaml

    # Determine the path to use
    if not current_path or not current_path.exists():
        logger.debug(
            ":warning: Current path does not exist => %s. Using target path instead.",
            current_path,
        )
        current_path = get_target_yaml_path(context, node)

    doc: dict[str, t.Any] = _read_yaml(
        context.yaml_handler, context.yaml_handler_lock, current_path
    )
    if not doc:
        doc = {"version": 2}
    return doc


def _get_or_create_source(
    doc: dict[str, t.Any], source_name: str, table_name: str | None = None
) -> dict[str, t.Any]:
    """Find or create a source entry in the YAML document.

    Handles the case where the YAML contains a Jinja template for the source name
    (e.g., "{{ var('some_source')[target.name] }}") but the manifest contains the
    resolved value (e.g., "some_source_dev"). In such cases, we try to find the
    source by checking if any existing source already contains the table we're looking for.
    """
    sources = doc.setdefault("sources", [])

    # First, try exact match
    for source in sources:
        if source.get("name") == source_name:
            return source

    # If no exact match and we have a table name, check if any existing source
    # already contains this table (which could indicate a Jinja template source name)
    if table_name:
        for source in sources:
            for table in source.get("tables", []):
                if table.get("name") == table_name:
                    # Found the table in an existing source with a different name
                    # This is likely a Jinja template source name situation
                    logger.debug(
                        ":link: Found table %s in source %s (node source_name is %s), "
                        "likely Jinja template source name - using existing source",
                        table_name,
                        source.get("name"),
                        source_name,
                    )
                    return source

    # Create new source
    new_source = {"name": source_name, "tables": []}
    sources.append(new_source)
    return new_source


def _get_or_create_source_table(doc_source: dict[str, t.Any], table_name: str) -> dict[str, t.Any]:
    """Find or create a table entry within a source."""
    for table in doc_source["tables"]:
        if table.get("name") == table_name:
            return table
    # Create new table
    new_table = {"name": table_name, "columns": []}
    doc_source["tables"].append(new_table)
    return new_table


def _sync_source_node(
    context: YamlRefactorContextProtocol,
    node: ResultNode,
    doc: dict[str, t.Any],
    resource_key: str,
) -> None:
    """Sync a source node to its YAML representation.

    Sources have a nested structure: sources -> [source_name] -> tables -> [table_name]
    """
    doc_source = _get_or_create_source(doc, node.source_name, table_name=node.name)
    doc_table = _get_or_create_source_table(doc_source, node.name)
    _sync_doc_section(context, node, doc_table)


def _deduplicate_model_entries(
    doc_list: list[dict[str, t.Any]], model_name: str
) -> dict[str, t.Any] | None:
    """Find and deduplicate model entries by name.

    Returns the first model entry if found, or None if not found.
    Removes duplicate entries if they exist.
    """
    model_indices: list[int] = []
    for i, item in enumerate(doc_list):
        if item.get("name") == model_name:
            model_indices.append(i)

    if len(model_indices) > 1:
        logger.warning(":warning: Found duplicate entries for model => %s", model_name)
        doc_model = doc_list[model_indices[0]]
        # Remove duplicates in reverse order to avoid index shifting
        for idx in sorted(model_indices[1:], reverse=True):
            doc_list.pop(idx)
        return doc_model
    elif model_indices:
        return doc_list[model_indices[0]]
    return None


def _get_or_create_model(doc_list: list[dict[str, t.Any]], model_name: str) -> dict[str, t.Any]:
    """Find or create a model entry in the YAML document."""
    doc_model = _deduplicate_model_entries(doc_list, model_name)
    if not doc_model:
        doc_model = {"name": model_name, "columns": []}
        doc_list.append(doc_model)
    return doc_model


def _deduplicate_versions(
    doc_model: dict[str, t.Any],
) -> dict[t.Union[int, str, float], dict[str, t.Any]]:
    """Deduplicate version entries by version number.

    Returns a dict mapping version numbers to version dicts.
    """
    version_by_v: dict[t.Union[int, str, float], dict[str, t.Any]] = {}
    for version in doc_model.get("versions", []):
        v_value = version.get("v")
        if v_value is not None:
            version_by_v[v_value] = version
    # Replace versions list with deduplicated versions
    doc_model["versions"] = list(version_by_v.values())
    return version_by_v


def _get_or_create_version(
    doc_model: dict[str, t.Any],
    version: t.Union[int, str, float],
) -> dict[str, t.Any]:
    """Find or create a version entry within a model."""
    version_by_v = _deduplicate_versions(doc_model)
    doc_version = version_by_v.get(version)
    if not doc_version:
        doc_version = {"v": version, "columns": []}
        doc_model["versions"].append(doc_version)
    return doc_version


def _sync_versioned_model(
    context: YamlRefactorContextProtocol,
    node: ModelNode,
    doc_model: dict[str, t.Any],
) -> None:
    """Sync a versioned model to its YAML representation."""
    if "versions" not in doc_model:
        doc_model["versions"] = []

    doc_version = _get_or_create_version(doc_model, node.version)

    # Ensure latest_version is set
    if "latest_version" not in doc_model:
        doc_model["latest_version"] = node.version

    # Sync data to the version object
    _sync_doc_section(context, node, doc_version)


def _sync_non_versioned_node(
    context: YamlRefactorContextProtocol,
    node: ResultNode,
    doc_model: dict[str, t.Any],
) -> None:
    """Sync a non-versioned model or seed to its YAML representation."""
    _sync_doc_section(context, node, doc_model)


def _sync_model_or_seed_node(
    context: YamlRefactorContextProtocol,
    node: ResultNode,
    doc: dict[str, t.Any],
    resource_key: str,
) -> None:
    """Sync a model or seed node to its YAML representation."""
    doc_list = doc.setdefault(resource_key, [])
    doc_model = _get_or_create_model(doc_list, node.name)

    # Handle versioned models differently
    if isinstance(node, ModelNode) and node.version is not None:
        _sync_versioned_model(context, node, doc_model)
    else:
        _sync_non_versioned_node(context, node, doc_model)


def _cleanup_empty_sections(doc: dict[str, t.Any]) -> None:
    """Remove empty sections from the YAML document."""
    for k in ("models", "sources", "seeds"):
        if len(doc.get(k, [])) == 0:
            _ = doc.pop(k, None)


def _sync_single_node_to_yaml(
    context: YamlRefactorContextProtocol,
    node: ResultNode,
    *,
    commit: bool = True,
) -> None:
    """Synchronize a single node's columns, description, tags, meta, etc.

    This is an internal helper that processes one node.

    Args:
        context: The YAML refactor context
        node: The node to sync
        commit: Whether to commit changes to disk
    """
    # Skip package models (models from dbt packages) as they don't have writable YAML files
    if node.package_name != context.project.runtime_cfg.project_name:
        logger.debug(
            ":package: Skipping package model => %s from package => %s",
            node.unique_id,
            node.package_name,
        )
        return

    from dbt_osmosis.core.path_management import get_current_yaml_path, get_target_yaml_path

    current_path = get_current_yaml_path(context, node)
    doc = _prepare_yaml_document(context, node, current_path)
    resource_key = _get_resource_type_key(node)

    if node.resource_type == NodeType.Source:
        _sync_source_node(context, node, doc, resource_key)
    else:
        _sync_model_or_seed_node(context, node, doc, resource_key)

    _cleanup_empty_sections(doc)

    if commit:
        # Commit the changes
        logger.info(":inbox_tray: Committing YAML doc changes for => %s", node.unique_id)
        from dbt_osmosis.core.schema.writer import _write_yaml

        _write_yaml(
            context.yaml_handler,
            context.yaml_handler_lock,
            current_path or get_target_yaml_path(context, node),
            doc,
            context.settings.dry_run,
            context.register_mutations,
        )


def _deduplicated_version_nodes(context: YamlRefactorContextProtocol) -> t.Iterator[ResultNode]:
    """Yield nodes, deduplicating versioned models by base name."""
    from dbt_osmosis.core.node_filters import _iter_candidate_nodes

    processed_models = set()
    for _, n in _iter_candidate_nodes(context):
        # For versioned models, only process each base model name once
        if n.resource_type == NodeType.Model and n.name in processed_models:
            continue
        processed_models.add(n.name) if n.resource_type == NodeType.Model else None
        yield n


def sync_node_to_yaml(
    context: YamlRefactorContextProtocol,
    node: t.Optional[ResultNode] = None,
    *,
    commit: bool = True,
) -> None:
    """Synchronize a single node's columns, description, tags, meta, etc. from the manifest into its corresponding YAML file.

    We assume the manifest node is the single source of truth, so the YAML file is overwritten to match.

    - If the YAML file doesn't exist yet, we create it with minimal structure.
    - If the YAML file exists, we read it from the file/ cache, locate the node's section,
      and then overwrite that section to match the node's current columns, meta, etc.

    This is a one-way sync:
        Manifest Node => YAML

    All changes to the Node (columns, metadata, etc.) should happen before calling this function.

    Args:
        context: The YAML refactor context
        node: The node to sync. If None, syncs all matched nodes.
        commit: Whether to commit changes to disk (default: True).
               When False, only performs the sync in memory without writing.
    """
    if node is None:
        logger.info(":wave: No single node specified; synchronizing all matched nodes.")
        for _ in context.pool.map(
            partial(sync_node_to_yaml, context, commit=commit),
            _deduplicated_version_nodes(context),
        ):
            ...
        return

    # Sync the single node
    _sync_single_node_to_yaml(context, node, commit=commit)
