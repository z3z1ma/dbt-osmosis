from __future__ import annotations

import typing as t
from functools import partial

from dbt.contracts.graph.nodes import ModelNode, ResultNode
from dbt.node_types import NodeType

import dbt_osmosis.core.logger as logger

__all__ = [
    "_sync_doc_section",
    "sync_node_to_yaml",
]


def _sync_doc_section(context: t.Any, node: ResultNode, doc_section: dict[str, t.Any]) -> None:
    """Helper function that overwrites 'doc_section' with data from 'node'.

    This includes columns, description, meta, tags, etc.
    We assume node is the single source of truth, so doc_section is replaced.
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

    for name, meta in node.columns.items():
        cdict = meta.to_dict(omit_none=True)
        cdict["name"] = name
        from dbt_osmosis.core.introspection import _get_setting_for_node, normalize_column_name

        norm_name = normalize_column_name(name, context.project.runtime_cfg.credentials.type)

        current_yaml = t.cast(dict[str, t.Any], current_map.get(norm_name, {}))
        merged = dict(current_yaml)

        skip_add_types = _get_setting_for_node(
            "skip-add-data-types", node, name, fallback=context.settings.skip_add_data_types
        )
        for k, v in cdict.items():
            if k == "data_type" and skip_add_types:
                # don't add data types if told not to
                continue
            if k == "constraints" and "constraints" in merged:
                # keep constraints as is if present, mashumaro dumps too much info :shrug:
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


def sync_node_to_yaml(
    context: t.Any, node: t.Optional[ResultNode] = None, *, commit: bool = True
) -> None:
    """Synchronize a single node's columns, description, tags, meta, etc. from the manifest into its corresponding YAML file.

    We assume the manifest node is the single source of truth, so the YAML file is overwritten to match.

    - If the YAML file doesn't exist yet, we create it with minimal structure.
    - If the YAML file exists, we read it from the file/ cache, locate the node's section,
      and then overwrite that section to match the node's current columns, meta, etc.

    This is a one-way sync:
        Manifest Node => YAML

    All changes to the Node (columns, metadata, etc.) should happen before calling this function.
    """
    if node is None:
        logger.info(":wave: No single node specified; synchronizing all matched nodes.")
        processed_models = set()

        # Closure to filter nodes to avoid duplicates for versioned models
        def _deduplicated_version_nodes():
            from dbt_osmosis.core.node_filters import _iter_candidate_nodes

            for _, n in _iter_candidate_nodes(context):
                # for versioned models, only process each base model name once
                if n.resource_type == NodeType.Model and n.name in processed_models:
                    continue
                processed_models.add(n.name) if n.resource_type == NodeType.Model else None
                yield n

        for _ in context.pool.map(
            partial(sync_node_to_yaml, context, commit=commit),
            _deduplicated_version_nodes(),
        ):
            ...
        return

    from dbt_osmosis.core.path_management import get_current_yaml_path, get_target_yaml_path
    from dbt_osmosis.core.schema.reader import _read_yaml
    from dbt_osmosis.core.schema.writer import _write_yaml

    current_path = get_current_yaml_path(context, node)
    if not current_path or not current_path.exists():
        logger.debug(
            ":warning: Current path does not exist => %s. Using target path instead.", current_path
        )
        current_path = get_target_yaml_path(context, node)

    doc: dict[str, t.Any] = _read_yaml(
        context.yaml_handler, context.yaml_handler_lock, current_path
    )
    if not doc:
        doc = {"version": 2}

    if node.resource_type == NodeType.Source:
        resource_k = "sources"
    elif node.resource_type == NodeType.Seed:
        resource_k = "seeds"
    else:
        resource_k = "models"

    if node.resource_type == NodeType.Source:
        # The doc structure => sources: [ { "name": <source_name>, "tables": [...]}, ... ]
        # Step A: find or create the source
        doc_source: t.Optional[dict[str, t.Any]] = None
        for s in doc.setdefault(resource_k, []):
            if s.get("name") == node.source_name:
                doc_source = s
                break
        if not doc_source:
            doc_source = {
                "name": node.source_name,
                "tables": [],
            }
            doc["sources"].append(doc_source)

        # Step B: find or create the table
        doc_table: t.Optional[dict[str, t.Any]] = None
        for t_ in doc_source["tables"]:
            if t_.get("name") == node.name:
                doc_table = t_
                break
        if not doc_table:
            doc_table = {
                "name": node.name,
                "columns": [],
            }
            doc_source["tables"].append(doc_table)

        # We'll store the columns & description on "doc_table"
        # For source, "description" is stored at table-level in the Node
        _sync_doc_section(context, node, doc_table)

    else:
        # Models or Seeds => doc[ "models" ] or doc[ "seeds" ] is a list of { "name", "description", "columns", ... }
        doc_list = doc.setdefault(resource_k, [])
        doc_model: t.Optional[dict[str, t.Any]] = None
        doc_version: t.Optional[dict[str, t.Any]] = None

        # First, check for duplicate model entries and remove them
        model_indices: list[int] = []
        for i, item in enumerate(doc_list):
            if item.get("name") == node.name:
                model_indices.append(i)

        # Keep only the first instance and remove others if there are duplicates
        if len(model_indices) > 1:
            logger.warning(":warning: Found duplicate entries for model => %s", node.name)
            # Keep the first one and remove the rest
            doc_model = doc_list[model_indices[0]]
            # Remove duplicates in reverse order to avoid index shifting
            for idx in sorted(model_indices[1:], reverse=True):
                doc_list.pop(idx)
        elif model_indices:
            doc_model = doc_list[model_indices[0]]

        # If the model doesn't exist in the document, create it
        if not doc_model:
            doc_model = {
                "name": node.name,
                "columns": [],
            }
            doc_list.append(doc_model)

        # Handle versioned models differently
        if isinstance(node, ModelNode) and node.version is not None:
            # Ensure versions array exists
            if "versions" not in doc_model:
                doc_model["versions"] = []

            # Deduplicate versions with the same 'v' value
            version_by_v: dict[t.Union[int, str, float], dict[str, t.Any]] = {}
            for version in doc_model.get("versions", []):
                v_value = version.get("v")
                if v_value is not None:
                    version_by_v[v_value] = version

            # Replace versions list with deduplicated versions
            doc_model["versions"] = list(version_by_v.values())

            # Try to find the specific version
            doc_version = version_by_v.get(node.version)

            # If version doesn't exist, create it
            if not doc_version:
                doc_version = {"v": node.version, "columns": []}
                doc_model["versions"].append(doc_version)

            # Ensure latest_version is set
            if "latest_version" not in doc_model:
                doc_model["latest_version"] = node.version

            # Sync data to the version object
            _sync_doc_section(context, node, doc_version)
        else:
            # For non-versioned models, sync directly to the model object
            _sync_doc_section(context, node, doc_model)

    for k in ("models", "sources", "seeds"):
        if len(doc.get(k, [])) == 0:
            _ = doc.pop(k, None)

    if commit:
        logger.info(":inbox_tray: Committing YAML doc changes for => %s", node.unique_id)
        _write_yaml(
            context.yaml_handler,
            context.yaml_handler_lock,
            current_path,
            doc,
            context.settings.dry_run,
            context.register_mutations,
        )
