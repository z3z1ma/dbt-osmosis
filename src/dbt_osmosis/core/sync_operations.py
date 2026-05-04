from __future__ import annotations

import typing as t
from pathlib import Path

from dbt.artifacts.resources.types import NodeType
from dbt.contracts.graph.nodes import ModelNode, ResultNode, SourceDefinition

if t.TYPE_CHECKING:
    from dbt_osmosis.core.dbt_protocols import YamlRefactorContextProtocol

from dbt_osmosis.core import logger
from dbt_osmosis.core.exceptions import YamlValidationError

__all__ = [
    "_sync_doc_section",
    "sync_node_to_yaml",
]


def _sync_doc_section(
    context: YamlRefactorContextProtocol,
    node: ResultNode,
    doc_section: dict[str, t.Any],
) -> None:
    """Helper function that overwrites 'doc_section' with data from 'node'.

    This includes columns, description, meta, tags, etc.
    We assume node is the single source of truth, so doc_section is replaced.

    If a catalog is available (via --catalog-path), data_type from the catalog
    takes precedence over the manifest's data_type.
    """
    logger.debug(":arrows_counterclockwise: Syncing doc_section with node => %s", node.unique_id)
    if node.description and not doc_section.get("description"):
        if context.settings.scaffold_empty_configs or node.description not in context.placeholders:
            doc_section["description"] = node.description

    current_columns: list[dict[str, t.Any]] = doc_section.setdefault("columns", [])
    preserved_column_entries: list[dict[str, t.Any]] = []
    incoming_columns: list[dict[str, t.Any]] = []

    current_map = {}
    for c in current_columns:
        from dbt_osmosis.core.introspection import normalize_column_name

        if not isinstance(c, t.Mapping):
            continue
        column_name = c.get("name")
        if not isinstance(column_name, str):
            if "v" in doc_section and ("include" in c or "exclude" in c):
                preserved_column_entries.append(dict(c))
            continue

        norm_name = normalize_column_name(column_name, context.project.runtime_cfg.credentials.type)
        current_map[norm_name] = c

    # Build a map of catalog column types if catalog is available
    catalog_column_types: dict[str, str] = {}
    if catalog := context.read_catalog():
        from dbt_osmosis.core.introspection import _find_first, normalize_column_name

        # For source nodes, search catalog.sources; for model nodes, search catalog.nodes
        # CatalogTable metadata doesn't have a 'source' field, so we match based on which dict we're searching
        if hasattr(node, "source_name"):
            catalog_entries = catalog.sources.values()
        else:
            catalog_entries = catalog.nodes.values()

        catalog_entry = _find_first(
            catalog_entries,
            lambda c: c.metadata.name == node.name and c.metadata.schema == node.schema,
        )
        if catalog_entry:
            for col_name, col_meta in catalog_entry.columns.items():
                norm_name = normalize_column_name(
                    col_name,
                    context.project.runtime_cfg.credentials.type,
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
        # Filter out fields added in dbt-core >= 1.9.6.
        # In fusion_compat mode, preserve 'config' (meta/tags will be nested inside it).
        # In classic mode, strip 'config' (meta/tags stay at top level).
        if context.fusion_compat:
            cdict = {k: v for k, v in cdict.items() if k != "doc_blocks"}
        else:
            cdict = {k: v for k, v in cdict.items() if k not in ("config", "doc_blocks")}
        cdict["name"] = name
        from dbt_osmosis.core.introspection import normalize_column_name, resolve_setting

        norm_name = normalize_column_name(name, context.project.runtime_cfg.credentials.type)

        # Use catalog data_type if available, otherwise use manifest's data_type
        if norm_name in catalog_column_types:
            cdict["data_type"] = catalog_column_types[norm_name]

        current_yaml = t.cast("dict[str, t.Any]", current_map.get(norm_name, {}))
        merged = dict(current_yaml)

        skip_add_types = resolve_setting(
            context,
            "skip-add-data-types",
            node,
            name,
            fallback=context.settings.skip_add_data_types,
        )
        skip_merge_meta = resolve_setting(
            context,
            "skip-merge-meta",
            node,
            name,
            fallback=context.settings.skip_merge_meta,
        )

        # Check if we should preserve unrendered descriptions from current YAML
        # When use_unrendered_descriptions is True and the current description contains
        # doc blocks (e.g., {{ doc(...) }} or {% docs %}{% enddocs %}), we should
        # preserve the unrendered version instead of using the rendered version from manifest
        current_description = current_yaml.get("description")
        use_unrendered = resolve_setting(
            context,
            "use-unrendered-descriptions",
            node,
            name,
            fallback=context.settings.use_unrendered_descriptions,
        )
        prefer_yaml = resolve_setting(
            context,
            "prefer-yaml-values",
            node,
            name,
            fallback=context.settings.prefer_yaml_values,
        )
        preserve_current_description = False
        if use_unrendered and current_description:
            # Check if current description contains unrendered doc blocks
            if "{{ doc(" in current_description or "{% docs " in current_description:
                preserve_current_description = True

        # Fields to preserve from current YAML when prefer_yaml_values is enabled
        # This includes ANY field that has unrendered jinja templates
        preserved_yaml_fields: set[str] = set()
        if prefer_yaml:
            from dbt_osmosis.core.introspection import PropertyAccessor

            accessor = PropertyAccessor(context=context)
            for field_key, field_value in current_yaml.items():
                if field_key == "name":
                    continue
                # Check if the field value contains unrendered jinja
                # Handles strings, lists (e.g., policy_tags), and nested dicts
                if accessor._has_unrendered_jinja(field_value):
                    preserved_yaml_fields.add(field_key)
                    logger.debug(
                        ":magic_wand: Preserving unrendered YAML field '%s' for column %s "
                        "(prefer-yaml-values enabled)",
                        field_key,
                        name,
                    )

        for k, v in cdict.items():
            if k == "data_type" and skip_add_types:
                # don't add data types if told not to
                continue
            if k == "meta" and skip_merge_meta and current_yaml.get("meta") not in (None, {}):
                continue
            if k == "constraints" and "constraints" in merged:
                # keep constraints as is if present, mashumaro dumps too much info :shrug:
                continue
            if k == "description" and preserve_current_description:
                # Preserve the unrendered description from current YAML
                continue
            if k in preserved_yaml_fields:
                # Preserve unrendered jinja templates from current YAML (prefer-yaml-values)
                continue
            merged[k] = v

        if context.fusion_compat:
            # Fusion mode: push top-level meta/tags INTO config block
            config_value = merged.get("config")
            if not isinstance(config_value, dict):
                config_value = {}
            meta_value = merged.pop("meta", None)
            tags_value = merged.pop("tags", None)
            if isinstance(meta_value, dict) and meta_value:
                existing_config_meta = config_value.get("meta", {})
                config_value["meta"] = {**existing_config_meta, **meta_value}
            if isinstance(tags_value, list) and tags_value:
                existing_config_tags = config_value.get("tags", [])
                seen = set(existing_config_tags)
                merged_tags = list(existing_config_tags)
                for tag in tags_value:
                    if tag not in seen:
                        merged_tags.append(tag)
                        seen.add(tag)
                config_value["tags"] = merged_tags
            if config_value:
                merged["config"] = config_value
        else:
            # Classic mode: keep meta/tags at top level and strip config wrappers.
            config_value = merged.get("config")
            if isinstance(config_value, dict):
                config_meta = config_value.get("meta")
                if isinstance(config_meta, dict) and config_meta:
                    existing_meta = merged.get("meta")
                    if isinstance(existing_meta, dict):
                        merged["meta"] = {**config_meta, **existing_meta}
                    else:
                        merged["meta"] = config_meta

                config_tags = config_value.get("tags")
                if isinstance(config_tags, list) and config_tags:
                    existing_tags = merged.get("tags")
                    if isinstance(existing_tags, list):
                        seen = set(existing_tags)
                        merged_tags = list(existing_tags)
                        for tag in config_tags:
                            if tag not in seen:
                                merged_tags.append(tag)
                                seen.add(tag)
                        merged["tags"] = merged_tags
                    else:
                        merged["tags"] = config_tags

                merged.pop("config", None)

        # Clean up empty nested config entries (e.g., config: {meta: {}, tags: []})
        if isinstance(merged.get("config"), dict):
            config = merged["config"]
            if config.get("meta", {}) == {}:
                config.pop("meta", None)
            if config.get("tags", []) == []:
                config.pop("tags", None)
            if not config:
                merged.pop("config", None)

        if merged.get("description") is None:
            merged.pop("description", None)
        if merged.get("tags", []) == []:
            merged.pop("tags", None)
        if merged.get("meta", {}) == {}:
            merged.pop("meta", None)

        for k in set(merged.keys()) - {"name", "description", "tags", "meta"}:
            if merged[k] in (None, [], {}):
                merged.pop(k)

        if not context.settings.scaffold_empty_configs:
            if merged.get("description") == "":
                merged.pop("description", None)
            if merged.get("tags", []) == []:
                merged.pop("tags", None)
            if merged.get("meta", {}) == {}:
                merged.pop("meta", None)
            if merged.get("config", {}) == {}:
                merged.pop("config", None)

        if resolve_setting(
            context,
            "output-to-upper",
            node,
            name,
            fallback=context.settings.output_to_upper,
        ):
            merged["name"] = merged["name"].upper()
        elif resolve_setting(
            context,
            "output-to-lower",
            node,
            name,
            fallback=context.settings.output_to_lower,
        ):
            merged["name"] = merged["name"].lower()

        incoming_columns.append(merged)

    synced_columns = preserved_column_entries + incoming_columns
    if synced_columns:
        doc_section["columns"] = synced_columns
    else:
        doc_section.pop("columns", None)


def _get_resource_type_key(node: ResultNode) -> str:
    """Get the resource type key for a node (e.g., 'models', 'sources', 'seeds')."""
    if node.resource_type == NodeType.Source:
        return "sources"
    if node.resource_type == NodeType.Seed:
        return "seeds"
    return "models"


def _prepare_yaml_document(
    context: YamlRefactorContextProtocol,
    node: ResultNode,
    current_path: Path | None,
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
        context.yaml_handler,
        context.yaml_handler_lock,
        current_path,
    )
    if not doc:
        doc = {"version": 2}
    return doc


def _finalize_synced_document(
    context: YamlRefactorContextProtocol,
    target_path: Path,
    doc: dict[str, t.Any],
    *,
    commit: bool,
) -> None:
    """Persist or pin a synchronized YAML document after in-memory updates."""
    _cleanup_empty_sections(doc)

    from dbt_osmosis.core.schema.reader import (
        _YAML_BUFFER_CACHE_LOCK,
        _discard_yaml_caches,
        _mark_yaml_caches_dirty,
    )

    # sync_node_to_yaml(commit=False) mutates the shared buffer in place, so pin it
    # until a real disk outcome clears the cache entry.
    _mark_yaml_caches_dirty(target_path)

    if not commit:
        if getattr(context.settings, "dry_run", False):
            with _YAML_BUFFER_CACHE_LOCK:
                _discard_yaml_caches(target_path)
        return

    logger.info(":inbox_tray: Committing YAML doc changes for => %s", target_path)

    from dbt_osmosis.core.schema.writer import _write_yaml

    _write_yaml(
        context.yaml_handler,
        context.yaml_handler_lock,
        target_path,
        doc,
        dry_run=context.settings.dry_run,
        mutation_tracker=context.register_mutations,
        strip_eof_blank_lines=context.settings.strip_eof_blank_lines,
        written_file_tracker=getattr(context, "register_written_file", None),
    )


def _get_or_create_source(
    doc: dict[str, t.Any],
    source_name: str,
    table_name: str | None = None,
    *,
    table_identifier: str | None = None,
    schema_name: str | None = None,
    database_name: str | None = None,
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

    def _tables_for_source_scan(source: dict[str, t.Any]) -> list[t.Any]:
        tables = source.get("tables", [])
        if not isinstance(tables, list):
            source_label = source.get("name", "<unknown>")
            raise YamlValidationError(
                f"Invalid YAML source '{source_label}': expected 'tables' to be a list before "
                "matching source tables. Fix the source tables structure before syncing."
            )
        return tables

    def _matching_sources(match_field: str, match_value: str) -> list[dict[str, t.Any]]:
        return [
            source
            for source in sources
            if any(
                table.get(match_field) == match_value for table in _tables_for_source_scan(source)
            )
        ]

    def _disambiguate(candidates: list[dict[str, t.Any]]) -> dict[str, t.Any] | None:
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]

        narrowed = candidates
        if schema_name is not None:
            schema_matches = [source for source in narrowed if source.get("schema") == schema_name]
            if len(schema_matches) == 1:
                return schema_matches[0]
            if schema_matches:
                narrowed = schema_matches

        if database_name is not None:
            database_matches = [
                source for source in narrowed if source.get("database") == database_name
            ]
            if len(database_matches) == 1:
                return database_matches[0]
            if database_matches:
                narrowed = database_matches

        return narrowed[0] if len(narrowed) == 1 else None

    # If no exact match and we have a table name, check for an existing source that can be
    # truthfully identified by table identifier/name plus optional schema/database narrowing.
    if table_name:
        candidates = _matching_sources("identifier", table_identifier) if table_identifier else []
        if not candidates:
            candidates = _matching_sources("name", table_name)

        matched_source = _disambiguate(candidates)
        if matched_source is not None:
            logger.debug(
                ":link: Reusing source %s for table %s (node source_name is %s)",
                matched_source.get("name"),
                table_name,
                source_name,
            )
            return matched_source

        if candidates:
            logger.warning(
                ":warning: Ambiguous source match for %s.%s; creating a new source entry instead of guessing",
                source_name,
                table_name,
            )

    # Create new source
    new_source = {"name": source_name, "tables": []}
    sources.append(new_source)
    return new_source


def _get_or_create_source_table(doc_source: dict[str, t.Any], table_name: str) -> dict[str, t.Any]:
    """Find or create a table entry within a source."""
    tables = doc_source.setdefault("tables", [])
    if not isinstance(tables, list):
        source_label = doc_source.get("name", "<unknown>")
        raise YamlValidationError(
            f"Invalid YAML source '{source_label}': expected 'tables' to be a list before "
            "syncing source tables. Fix the source tables structure before syncing."
        )

    for table in tables:
        if table.get("name") == table_name:
            return table
    # Create new table
    new_table = {"name": table_name, "columns": []}
    tables.append(new_table)
    return new_table


def _sync_source_node(
    context: YamlRefactorContextProtocol,
    node: SourceDefinition,
    doc: dict[str, t.Any],
    resource_key: str,
) -> None:
    """Sync a source node to its YAML representation.

    Sources have a nested structure: sources -> [source_name] -> tables -> [table_name]
    """
    doc_source = _get_or_create_source(
        doc,
        node.source_name,
        table_name=node.name,
        table_identifier=getattr(node, "identifier", None),
        schema_name=getattr(node, "schema", None),
        database_name=getattr(node, "database", None),
    )
    doc_table = _get_or_create_source_table(doc_source, node.name)
    _sync_doc_section(context, node, doc_table)


def _deduplicate_model_entries(
    doc_list: list[dict[str, t.Any]],
    model_name: str,
) -> dict[str, t.Any] | None:
    """Find model entries by name and fail closed on duplicates.

    Returns the first model entry if found, or None if not found.
    """
    model_indices: list[int] = []
    for i, item in enumerate(doc_list):
        if item.get("name") == model_name:
            model_indices.append(i)

    if len(model_indices) > 1:
        index_list = ", ".join(str(index) for index in model_indices)
        raise YamlValidationError(
            f"Duplicate YAML model entries for '{model_name}' at models indexes {index_list}. "
            "dbt-osmosis refuses to sync because choosing one entry would delete "
            "user-authored YAML content. Consolidate duplicate models[] entries before syncing."
        )
    if model_indices:
        return doc_list[model_indices[0]]
    return None


def _get_or_create_model(doc_list: list[dict[str, t.Any]], model_name: str) -> dict[str, t.Any]:
    """Find or create a model entry in the YAML document."""
    doc_model = _deduplicate_model_entries(doc_list, model_name)
    if not doc_model:
        doc_model = {"name": model_name, "columns": []}
        doc_list.append(doc_model)
    return doc_model


def _deduplicate_versions(doc_model: dict[str, t.Any]) -> dict[str, dict[str, t.Any]]:
    """Index version entries by version number and fail closed on duplicates.

    Returns a dict mapping version numbers to version dicts.
    """
    from dbt_osmosis.core.inheritance import _raw_model_version_value, _version_values_match

    model_name = doc_model.get("name", "<unknown>")
    valid_version_entries: list[tuple[int, int | float | str]] = []
    version_by_v: dict[str, dict[str, t.Any]] = {}
    for version_idx, version in enumerate(doc_model.get("versions", [])):
        if not isinstance(version, t.Mapping):
            continue
        v_value = version.get("v")
        v_key = _raw_model_version_value(v_value)
        if v_key is not None:
            duplicate_entry = next(
                (
                    (seen_idx, seen_value)
                    for seen_idx, seen_value in valid_version_entries
                    if _version_values_match(seen_value, v_value)
                ),
                None,
            )
            if duplicate_entry is not None:
                raise YamlValidationError(
                    f"Duplicate YAML version entries for model '{model_name}' at versions indexes "
                    f"{duplicate_entry[0]} and {version_idx} identify the same version "
                    f"({duplicate_entry[1]!r} and {v_value!r}). dbt-osmosis refuses to sync "
                    "because choosing one entry would delete user-authored YAML content. "
                    "Consolidate duplicate models[].versions[] entries before syncing."
                )
            if not isinstance(v_value, bool) and isinstance(v_value, (int, float, str)):
                valid_version_entries.append((version_idx, v_value))
            version_by_v[v_key] = t.cast("dict[str, t.Any]", version)
    return version_by_v


def _get_or_create_version(
    doc_model: dict[str, t.Any],
    version: int | float | str,
) -> dict[str, t.Any]:
    """Find or create a version entry within a model."""
    from dbt_osmosis.core.inheritance import _raw_model_version_value, _version_values_match

    version_by_v = _deduplicate_versions(doc_model)
    version_key = _raw_model_version_value(version)
    doc_version = version_by_v.get(version_key) if version_key is not None else None
    if doc_version is None:
        doc_version = next(
            (
                doc_version
                for doc_version in version_by_v.values()
                if _version_values_match(doc_version.get("v"), version)
            ),
            None,
        )
    if doc_version is None:
        doc_version = {"v": version, "columns": []}
        doc_model["versions"].append(doc_version)
    return doc_version


def _sync_versioned_model(
    context: YamlRefactorContextProtocol,
    node: ModelNode,
    doc_model: dict[str, t.Any],
) -> None:
    """Sync a versioned model to its YAML representation."""
    # This function is only called when node.version is not None (see line 418)
    version: int | float | str = t.cast("int | float | str", node.version)

    if "versions" not in doc_model:
        doc_model["versions"] = []

    doc_version = _get_or_create_version(doc_model, version)

    # Keep the top-level version metadata aligned with the manifest rather than preserving
    # stale YAML values when a newer latest_version is introduced.
    latest_version = getattr(node, "latest_version", None)
    if latest_version is not None:
        doc_model["latest_version"] = latest_version
    elif "latest_version" not in doc_model:
        doc_model["latest_version"] = version

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


def _validate_no_duplicate_sync_entries(doc: dict[str, t.Any], target_path: Path) -> None:
    """Fail before syncing when one YAML document has duplicate writable entries."""
    for resource_key in ("models", "seeds"):
        entries = doc.get(resource_key, [])
        if not isinstance(entries, list):
            continue

        seen_names: dict[str, int] = {}
        for idx, entry in enumerate(entries):
            if not isinstance(entry, t.Mapping):
                continue
            name = entry.get("name")
            if not isinstance(name, str):
                continue
            if name in seen_names:
                raise YamlValidationError(
                    f"Duplicate YAML {resource_key} entries for '{name}' in {target_path} at "
                    f"{resource_key} indexes {seen_names[name]} and {idx}. dbt-osmosis refuses "
                    "to sync because choosing one entry would delete user-authored YAML content. "
                    f"Consolidate duplicate {resource_key} entries before syncing."
                )
            seen_names[name] = idx

            if resource_key == "models":
                _deduplicate_versions(t.cast("dict[str, t.Any]", entry))


def _preflight_sync_group(context: YamlRefactorContextProtocol, nodes: list[ResultNode]) -> None:
    """Read one target document and fail on duplicates before any sync writes occur."""
    if not nodes:
        return

    representative = nodes[0]
    current_path, target_path = _resolve_sync_yaml_paths(context, representative)
    doc = _prepare_yaml_document(context, representative, current_path)
    _validate_no_duplicate_sync_entries(doc, target_path)


def _cleanup_empty_sections(doc: dict[str, t.Any]) -> None:
    """Remove empty sections from the YAML document."""
    for k in ("models", "sources", "seeds"):
        if len(doc.get(k, [])) == 0:
            _ = doc.pop(k, None)


def _resolve_sync_yaml_paths(
    context: YamlRefactorContextProtocol,
    node: ResultNode,
) -> tuple[Path | None, Path]:
    """Resolve the current and effective target YAML paths for a sync node."""
    from dbt_osmosis.core.path_management import get_current_yaml_path, get_target_yaml_path

    current_path = get_current_yaml_path(context, node)
    target_path = current_path
    if not target_path or not target_path.exists():
        target_path = get_target_yaml_path(context, node)
    return current_path, target_path


def _sync_node_group_to_yaml(
    context: YamlRefactorContextProtocol,
    nodes: list[ResultNode],
    *,
    commit: bool = True,
) -> None:
    """Synchronize all nodes for one YAML path through a single document update."""
    if not nodes:
        return

    representative = nodes[0]

    current_path, target_path = _resolve_sync_yaml_paths(context, representative)

    doc = _prepare_yaml_document(context, representative, current_path)

    for grouped_node in _order_sync_group_nodes(nodes):
        resource_key = _get_resource_type_key(grouped_node)
        if grouped_node.resource_type == NodeType.Source:
            _sync_source_node(context, t.cast("SourceDefinition", grouped_node), doc, resource_key)
        else:
            _sync_model_or_seed_node(context, grouped_node, doc, resource_key)

    _finalize_synced_document(context, target_path, doc, commit=commit)


def _order_sync_group_nodes(nodes: list[ResultNode]) -> list[ResultNode]:
    """Keep same-path group processing deterministic, especially for versioned models."""

    def sync_order(node: ResultNode) -> tuple[str, str, int, str, str]:
        version = getattr(node, "version", None)
        return (
            str(node.resource_type),
            node.name,
            0 if version is not None else 1,
            str(version or ""),
            node.unique_id,
        )

    return sorted(nodes, key=sync_order)


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

    current_path, target_path = _resolve_sync_yaml_paths(context, node)

    doc = _prepare_yaml_document(context, node, current_path)
    resource_key = _get_resource_type_key(node)

    if node.resource_type == NodeType.Source:
        _sync_source_node(context, node, doc, resource_key)
    else:
        _sync_model_or_seed_node(context, node, doc, resource_key)

    _finalize_synced_document(context, target_path, doc, commit=commit)


def _group_sync_nodes(context: YamlRefactorContextProtocol) -> list[list[ResultNode]]:
    """Group candidate nodes so each YAML document is updated truthfully once per work item."""
    from dbt_osmosis.core.node_filters import _iter_candidate_nodes

    groups_by_path: dict[Path, list[ResultNode]] = {}

    for _, n in _iter_candidate_nodes(context):
        if n.package_name != context.project.runtime_cfg.project_name:
            logger.debug(
                ":package: Skipping package model => %s from package => %s",
                n.unique_id,
                n.package_name,
            )
            continue

        _, target_path = _resolve_sync_yaml_paths(context, n)
        groups_by_path.setdefault(target_path.resolve(), []).append(n)

    return list(groups_by_path.values())


def sync_node_to_yaml(
    context: YamlRefactorContextProtocol,
    node: ResultNode | None = None,
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
        groups = _group_sync_nodes(context)

        for group in groups:
            _preflight_sync_group(context, group)

        def _sync_group(group: list[ResultNode]) -> None:
            if len(group) == 1:
                sync_node_to_yaml(context, group[0], commit=commit)
                return

            _sync_node_group_to_yaml(context, group, commit=commit)

        for _ in context.pool.map(
            _sync_group,
            groups,
        ):
            ...
        return

    # Sync the single node
    _sync_single_node_to_yaml(context, node, commit=commit)
