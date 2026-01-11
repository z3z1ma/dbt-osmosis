from __future__ import annotations

import os
import typing as t
from dataclasses import dataclass, field
from pathlib import Path

from dbt.artifacts.resources.types import NodeType
from dbt.contracts.graph.nodes import ResultNode

if t.TYPE_CHECKING:
    from dbt_osmosis.core.dbt_protocols import YamlRefactorContextProtocol

from dbt_osmosis.core import logger
from dbt_osmosis.core.exceptions import MissingOsmosisConfig, PathResolutionError

__all__ = [
    "MissingOsmosisConfig",
    "SchemaFileLocation",
    "SchemaFileMigration",
    "_get_yaml_path_template",
    "build_yaml_file_mapping",
    "create_missing_source_yamls",
    "get_current_yaml_path",
    "get_target_yaml_path",
]


@dataclass
class SchemaFileLocation:
    """Describes the current and target location of a schema file."""

    target: Path
    current: Path | None = None
    node_type: NodeType = NodeType.Model

    @property
    def is_valid(self) -> bool:
        """Check if the current and target locations are valid."""
        valid = self.current == self.target
        logger.debug(":white_check_mark: Checking if schema file location is valid => %s", valid)
        return valid


@dataclass
class SchemaFileMigration:
    """Describes a schema file migration operation."""

    output: dict[str, t.Any] = field(
        default_factory=lambda: {"version": 2, "models": [], "sources": []},
    )
    supersede: dict[Path, list[ResultNode]] = field(default_factory=dict)


def _get_yaml_path_template(context: YamlRefactorContextProtocol, node: ResultNode) -> str | None:
    """Get the yaml path template for a dbt model or source node.

    First checks for a model-specific `+dbt-osmosis` config via SettingsResolver,
    then falls back to the global `dbt_osmosis_default_path` var from dbt_project.yml.
    """
    from dbt_osmosis.core.introspection import SettingsResolver

    if node.resource_type == NodeType.Source:
        def_or_path = context.source_definitions.get(node.source_name)
        if isinstance(def_or_path, dict):
            return def_or_path.get("path")
        return def_or_path

    # Use SettingsResolver to get the path template from config sources
    resolver = SettingsResolver()
    path_template = resolver.get_yaml_path_template(node)

    # If no model-specific config, check for global var
    if not path_template:
        try:
            project_vars = context.project.runtime_cfg.vars.to_dict()
            path_template = project_vars.get("dbt_osmosis_default_path")
            if path_template:
                logger.debug(
                    ":earth_americas: Using global var 'dbt_osmosis_default_path': %s",
                    path_template,
                )
        except Exception as e:
            logger.debug(":warning: Failed to read global var: %s", e)
            path_template = None

    if not path_template:
        raise MissingOsmosisConfig(
            f"Config key `+dbt-osmosis:` or var `dbt_osmosis_default_path` not set for model {node.name}",
        )
    logger.debug(":gear: Resolved YAML path template => %s", path_template)
    return path_template


def get_current_yaml_path(
    context: YamlRefactorContextProtocol,
    node: ResultNode,
) -> Path | None:
    """Get the current yaml path for a dbt model or source node."""
    if node.resource_type in (NodeType.Model, NodeType.Seed) and getattr(node, "patch_path", None):
        path = Path(context.project.runtime_cfg.project_root).joinpath(
            t.cast("str", node.patch_path).partition("://")[-1],
        )
        logger.debug(":page_facing_up: Current YAML path => %s", path)
        return path
    if node.resource_type == NodeType.Source:
        path = Path(context.project.runtime_cfg.project_root, node.path)
        logger.debug(":page_facing_up: Current YAML path => %s", path)
        return path
    return None


def get_target_yaml_path(context: YamlRefactorContextProtocol, node: ResultNode) -> Path:
    """Get the target yaml path for a dbt model or source node."""
    tpl = _get_yaml_path_template(context, node)
    if not tpl:
        logger.warning(":warning: No path template found for => %s", node.unique_id)
        return Path(context.project.runtime_cfg.project_root, node.original_file_path)

    # Use local copies to avoid TOCTOU race conditions from mutating node objects
    # Build a safe format dict with only immutable/copy data
    path = Path(context.project.runtime_cfg.project_root, node.original_file_path)

    # Create a simple node object with common attributes for format strings
    # Avoid exposing fqn/tags as indexed dicts to prevent TOCTOU issues
    node_attrs = {
        "name": node.name,
        "schema": node.schema,
        "database": node.database,
        "package": node.package_name,
    }
    # Add source_name only for Source nodes (non-source nodes don't have this attribute)
    if node.resource_type == NodeType.Source:
        node_attrs["source_name"] = node.source_name

    format_dict = {
        "model": node.name,
        "parent": path.parent.name,
        "schema": node.schema,
        "node": type("obj", (object,), node_attrs)(),
    }

    rendered = tpl.format(**format_dict)

    segments: list[Path | str] = []

    if node.resource_type == NodeType.Source:
        segments.append(context.project.runtime_cfg.model_paths[0])
    elif rendered.startswith("/"):
        segments.append(context.project.runtime_cfg.model_paths[0])
        # SECURITY: Remove only the first leading slash, not all slashes (prevents path traversal)
        rendered = rendered.removeprefix("/")
    else:
        segments.append(path.parent)

    if not (rendered.endswith(".yml") or rendered.endswith(".yaml")):
        rendered += ".yml"
    segments.append(rendered)

    path = Path(context.project.runtime_cfg.project_root, *segments)
    # SECURITY: Validate path is within project root to prevent directory traversal
    resolved_path = path.resolve()
    project_root = Path(context.project.runtime_cfg.project_root).resolve()
    if not resolved_path.is_relative_to(project_root):
        raise PathResolutionError(
            f"Security violation: Target YAML path '{resolved_path}' is outside project root '{project_root}'",
        )
    logger.debug(":star2: Target YAML path => %s", path)
    return path


def build_yaml_file_mapping(
    context: t.Any,
    create_missing_sources: bool = False,
) -> dict[str, SchemaFileLocation]:
    """Build a mapping of dbt model and source nodes to their current and target yaml paths."""
    logger.info(":globe_with_meridians: Building YAML file mapping...")

    if create_missing_sources:
        create_missing_source_yamls(context)

    out_map: dict[str, SchemaFileLocation] = {}
    from dbt_osmosis.core.node_filters import _iter_candidate_nodes

    for uid, node in _iter_candidate_nodes(context):
        current_path = get_current_yaml_path(context, node)
        out_map[uid] = SchemaFileLocation(
            target=get_target_yaml_path(context, node).resolve(),
            current=current_path.resolve() if current_path else None,
            node_type=node.resource_type,
        )

    logger.debug(":card_index_dividers: Built YAML file mapping => %s", out_map)
    return out_map


def create_missing_source_yamls(context: t.Any) -> None:
    """Create source files for sources defined in the dbt_project.yml dbt-osmosis var which don't exist as nodes.

    This is a useful preprocessing step to ensure that all sources are represented in the dbt project manifest. We
    do not have rich node information for non-existent sources, hence the alternative codepath here to bootstrap them.

    For existing sources, this function will also discover and add new tables from the database that are not yet
    defined in the source YAML file (fixes #217).
    """
    from dbt_osmosis.core.config import _reload_manifest
    from dbt_osmosis.core.introspection import _find_first, get_columns
    from dbt_osmosis.core.schema.reader import (
        _YAML_BUFFER_CACHE,
        _YAML_BUFFER_CACHE_LOCK,
        _read_yaml,
    )
    from dbt_osmosis.core.schema.writer import _write_yaml

    if context.project.config.disable_introspection:
        logger.warning(":warning: Introspection is disabled, cannot create missing source YAMLs.")
        return
    logger.info(":factory: Creating missing source YAMLs and updating existing sources (if any).")
    database: str = context.project.runtime_cfg.credentials.database
    lowercase: bool = context.settings.output_to_lower
    uppercase: bool = context.settings.output_to_upper

    did_side_effect: bool = False
    for source, spec in context.source_definitions.items():
        if isinstance(spec, str):
            schema = source
            src_yaml_path = spec
        elif isinstance(spec, dict):
            database = t.cast("str", spec.get("database", database))
            schema = t.cast("str", spec.get("schema", source))
            src_yaml_path = t.cast("str", spec["path"])
        else:
            continue

        # Check if source already exists in the manifest
        existing_source_node = _find_first(
            context.project.manifest.sources.values(),
            lambda s: s.source_name == source,
        )

        if existing_source_node:
            # Source already exists - check for new tables to add
            # Get the set of tables already defined in the manifest/YAML
            manifest_tables = {
                s.name for s in context.project.manifest.sources.values() if s.source_name == source
            }
            logger.debug(
                ":white_check_mark: Source => %s already exists in the manifest with %d tables.",
                source,
                len(manifest_tables),
            )

        # SECURITY: Remove only the first leading separator, not all (prevents path traversal)
        cleaned_path = src_yaml_path[1:] if src_yaml_path.startswith(os.sep) else src_yaml_path
        src_yaml_path_obj = Path(
            context.project.runtime_cfg.project_root,
            context.project.runtime_cfg.model_paths[0],
            cleaned_path,
        )
        # SECURITY: Validate path is within project root to prevent directory traversal
        resolved_path = src_yaml_path_obj.resolve()
        project_root = Path(context.project.runtime_cfg.project_root).resolve()
        if not resolved_path.is_relative_to(project_root):
            raise PathResolutionError(
                f"Security violation: Source YAML path '{resolved_path}' is outside project root '{project_root}'",
            )

        def _describe(relation: t.Any) -> dict[str, t.Any]:
            assert relation.identifier, "No identifier found for relation."
            identifier_name = relation.identifier
            if uppercase:
                identifier_name = identifier_name.upper()
            elif lowercase:
                identifier_name = identifier_name.lower()

            s = {
                "name": identifier_name,
                "description": "",
                "columns": [
                    {
                        "name": name.upper()
                        if uppercase
                        else (name.lower() if lowercase else name),
                        "description": meta.comment or "",
                        "data_type": meta.type.upper()
                        if uppercase
                        else (meta.type.lower() if lowercase else meta.type),
                    }
                    for name, meta in get_columns(context, relation).items()
                ],
            }
            if context.settings.skip_add_data_types:
                for col in t.cast("list[dict[str, t.Any]]", s["columns"]):
                    _ = col.pop("data_type", None)
            return s

        # Get all tables from the database
        db_relations = list(
            context.project.adapter.list_relations(database=database, schema=schema),
        )
        db_tables = {rel.identifier: _describe(rel) for rel in db_relations}

        if existing_source_node:
            # For existing sources, check if there are new tables in the database
            new_table_names = set(db_tables.keys()) - manifest_tables
            if new_table_names:
                logger.info(
                    ":sparkles: Found %d new tables for source => %s: %s",
                    len(new_table_names),
                    source,
                    sorted(new_table_names),
                )
                # Read the existing YAML file
                existing_doc = _read_yaml(
                    context.yaml_handler,
                    context.yaml_handler_lock,
                    src_yaml_path_obj,
                )
                # Find the source entry and add new tables
                for src_entry in existing_doc.get("sources", []):
                    if src_entry.get("name") == source:
                        # Add new tables to the existing source
                        existing_tables_set = {t["name"] for t in src_entry.get("tables", [])}
                        for table_name in sorted(new_table_names):
                            if table_name not in existing_tables_set:
                                src_entry.setdefault("tables", []).append(db_tables[table_name])
                                logger.info(
                                    ":plus: Adding new table => %s to source => %s",
                                    table_name,
                                    source,
                                )
                        break
                # Write the updated YAML back
                src_yaml_path_obj.parent.mkdir(parents=True, exist_ok=True)
                _write_yaml(
                    context.yaml_handler,
                    context.yaml_handler_lock,
                    src_yaml_path_obj,
                    existing_doc,
                    context.settings.dry_run,
                    context.register_mutations,
                )
                # Clear cache for the updated file
                with _YAML_BUFFER_CACHE_LOCK:
                    if src_yaml_path_obj in _YAML_BUFFER_CACHE:
                        del _YAML_BUFFER_CACHE[src_yaml_path_obj]
                did_side_effect = True
            else:
                logger.debug(
                    ":white_check_mark: No new tables found for source => %s",
                    source,
                )
        else:
            # Source doesn't exist - create new YAML file with all tables
            tables = list(db_tables.values())
            source_dict = {"name": source, "database": database, "schema": schema, "tables": tables}

            src_yaml_path_obj.parent.mkdir(parents=True, exist_ok=True)
            with src_yaml_path_obj.open("w") as f:
                logger.info(
                    ":books: Injecting new source => %s => %s with %d tables",
                    source_dict["name"],
                    src_yaml_path_obj,
                    len(tables),
                )
                context.yaml_handler.dump({"version": 2, "sources": [source_dict]}, f)
                context.register_mutations(1)

            did_side_effect = True

    if did_side_effect:
        logger.info(
            ":arrows_counterclockwise: Sources were updated, reloading the project.",
        )
        _reload_manifest(context.project)
