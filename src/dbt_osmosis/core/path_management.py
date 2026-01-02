from __future__ import annotations

import os
import typing as t
from dataclasses import dataclass, field
from pathlib import Path

from dbt.artifacts.resources.types import NodeType
from dbt.contracts.graph.nodes import ResultNode

if t.TYPE_CHECKING:
    from dbt_osmosis.core.dbt_protocols import YamlRefactorContextProtocol

import dbt_osmosis.core.logger as logger

__all__ = [
    "SchemaFileLocation",
    "SchemaFileMigration",
    "MissingOsmosisConfig",
    "_get_yaml_path_template",
    "get_current_yaml_path",
    "get_target_yaml_path",
    "build_yaml_file_mapping",
    "create_missing_source_yamls",
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
        default_factory=lambda: {"version": 2, "models": [], "sources": []}
    )
    supersede: dict[Path, list[ResultNode]] = field(default_factory=dict)


class MissingOsmosisConfig(Exception):
    """Raised when an osmosis configuration is missing."""


def _get_yaml_path_template(context: YamlRefactorContextProtocol, node: ResultNode) -> str | None:
    """Get the yaml path template for a dbt model or source node."""
    from dbt_osmosis.core.introspection import _find_first

    if node.resource_type == NodeType.Source:
        def_or_path = context.source_definitions.get(node.source_name)
        if isinstance(def_or_path, dict):
            return def_or_path.get("path")
        return def_or_path

    conf = [
        c.get(k)
        for k in ("dbt-osmosis", "dbt_osmosis")
        for c in (node.config.extra, node.config.meta, node.unrendered_config)
    ]
    path_template = _find_first(t.cast("list[str | None]", conf), lambda v: v is not None)
    if not path_template:
        raise MissingOsmosisConfig(
            f"Config key `dbt-osmosis: <path>` not set for model {node.name}"
        )
    logger.debug(":gear: Resolved YAML path template => %s", path_template)
    return path_template


def get_current_yaml_path(
    context: YamlRefactorContextProtocol, node: ResultNode
) -> t.Union[Path, None]:
    """Get the current yaml path for a dbt model or source node."""
    if node.resource_type in (NodeType.Model, NodeType.Seed) and getattr(node, "patch_path", None):
        path = Path(context.project.runtime_cfg.project_root).joinpath(
            t.cast(str, node.patch_path).partition("://")[-1]
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
    format_dict = {
        "model": node.name,
        "parent": path.parent.name,
        "schema": node.schema,
        "node": type(
            "obj",
            (object,),
            {
                "name": node.name,
                "schema": node.schema,
                "database": node.database,
                "package": node.package_name,
            },
        )(),
    }

    rendered = tpl.format(**format_dict)

    segments: list[t.Union[Path, str]] = []

    if node.resource_type == NodeType.Source:
        segments.append(context.project.runtime_cfg.model_paths[0])
    elif rendered.startswith("/"):
        segments.append(context.project.runtime_cfg.model_paths[0])
        # SECURITY: Remove only the first leading slash, not all slashes (prevents path traversal)
        rendered = rendered[1:] if rendered.startswith("/") else rendered
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
        raise ValueError(
            f"Security violation: Target YAML path '{resolved_path}' is outside project root '{project_root}'"
        )
    logger.debug(":star2: Target YAML path => %s", path)
    return path


def build_yaml_file_mapping(
    context: t.Any, create_missing_sources: bool = False
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
    """
    from dbt_osmosis.core.config import _reload_manifest
    from dbt_osmosis.core.introspection import _find_first, get_columns

    if context.project.config.disable_introspection:
        logger.warning(":warning: Introspection is disabled, cannot create missing source YAMLs.")
        return
    logger.info(":factory: Creating missing source YAMLs (if any).")
    database: str = context.project.runtime_cfg.credentials.database
    lowercase: bool = context.settings.output_to_lower

    did_side_effect: bool = False
    for source, spec in context.source_definitions.items():
        if isinstance(spec, str):
            schema = source
            src_yaml_path = spec
        elif isinstance(spec, dict):
            database = t.cast(str, spec.get("database", database))
            schema = t.cast(str, spec.get("schema", source))
            src_yaml_path = t.cast(str, spec["path"])
        else:
            continue

        if _find_first(
            context.project.manifest.sources.values(), lambda s: s.source_name == source
        ):
            logger.debug(
                ":white_check_mark: Source => %s already exists in the manifest, skipping creation.",
                source,
            )
            continue

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
            raise ValueError(
                f"Security violation: Source YAML path '{resolved_path}' is outside project root '{project_root}'"
            )

        def _describe(relation: t.Any) -> dict[str, t.Any]:
            assert relation.identifier, "No identifier found for relation."
            s = {
                "name": relation.identifier.lower() if lowercase else relation.identifier,
                "description": "",
                "columns": [
                    {
                        "name": name.lower() if lowercase else name,
                        "description": meta.comment or "",
                        "data_type": meta.type.lower() if lowercase else meta.type,
                    }
                    for name, meta in get_columns(context, relation).items()
                ],
            }
            if context.settings.skip_add_data_types:
                for col in t.cast(list[dict[str, t.Any]], s["columns"]):
                    _ = col.pop("data_type", None)
            return s

        tables = [
            _describe(relation)
            for relation in context.project.adapter.list_relations(database=database, schema=schema)
        ]
        source_dict = {"name": source, "database": database, "schema": schema, "tables": tables}

        src_yaml_path_obj.parent.mkdir(parents=True, exist_ok=True)
        with src_yaml_path_obj.open("w") as f:
            logger.info(
                ":books: Injecting new source => %s => %s", source_dict["name"], src_yaml_path_obj
            )
            context.yaml_handler.dump({"version": 2, "sources": [source_dict]}, f)
            context.register_mutations(1)

        did_side_effect = True

    if did_side_effect:
        logger.info(
            ":arrows_counterclockwise: Some new sources were created, reloading the project."
        )
        _reload_manifest(context.project)
