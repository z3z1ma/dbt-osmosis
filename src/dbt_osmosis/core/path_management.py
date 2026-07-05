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
    "_resolve_vars_routing",
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


def _project_vars_for_routing(context: YamlRefactorContextProtocol) -> dict[str, t.Any] | None:
    try:
        project_vars = context.project.runtime_cfg.vars.to_dict()
    except (AttributeError, TypeError, RuntimeError):
        return None
    return project_vars if isinstance(project_vars, dict) else None


def _dbt_osmosis_vars(project_vars: dict[str, t.Any]) -> dict[str, t.Any] | None:
    osmosis_vars = project_vars.get("dbt-osmosis", project_vars.get("dbt_osmosis", {}))
    return osmosis_vars if isinstance(osmosis_vars, dict) else None


def _routing_section(
    osmosis_vars: dict[str, t.Any],
    node: ResultNode,
) -> dict[str, str] | str | None:
    if node.resource_type == NodeType.Seed:
        routing = osmosis_vars.get("seeds")
        return routing if isinstance(routing, (dict, str)) else None

    routing = osmosis_vars.get("models")
    return routing if isinstance(routing, dict) else None


def _folder_routing_keys(node: ResultNode) -> list[str]:
    fqn_folders = node.fqn[1:-1] if len(node.fqn) > 2 else []
    return [".".join(fqn_folders[:depth]) for depth in range(len(fqn_folders), 0, -1)]


def _resolve_folder_routing(routing: dict[str, t.Any], node: ResultNode) -> str | None:
    for key in _folder_routing_keys(node):
        value = routing.get(key)
        if isinstance(value, str):
            logger.debug(
                ":rocket: Resolved YAML path from vars routing [%s]: %s",
                key,
                value,
            )
            return value
    return None


def _resolve_vars_routing(
    context: YamlRefactorContextProtocol,
    node: ResultNode,
) -> str | None:
    """Resolve YAML path template from vars.dbt-osmosis.models (or .seeds).

    This is the fusion-compatible alternative to +dbt-osmosis config keys in
    dbt_project.yml. Since dbt-fusion rejects unknown + prefixed keys but accepts
    vars, users can move their routing config here:

        vars:
          dbt-osmosis:
            models:
              staging: "_stg_{parent}__models.yml"
              intermediate: "_int_{parent}__models.yml"
              marts: "_marts_{parent}__models.yml"
            seeds: "_seeds__models.yml"

    Matching uses the node's FQN (fully qualified name). For a model with
    fqn = ["project", "staging", "oem_raw", "stg_oem_raw__mme"], we check
    the routing dict for the most specific match first:
      1. "staging.oem_raw" (deepest folder path)
      2. "staging" (parent folder)

    For seeds, the value can be a string (applies to all seeds) or a dict
    with per-folder routing like models.
    """
    project_vars = _project_vars_for_routing(context)
    osmosis_vars = _dbt_osmosis_vars(project_vars) if project_vars is not None else None
    routing = _routing_section(osmosis_vars, node) if osmosis_vars is not None else None
    if isinstance(routing, str):
        return routing
    return _resolve_folder_routing(routing, node) if routing is not None else None


def _source_yaml_path_template(
    context: YamlRefactorContextProtocol,
    node: ResultNode,
) -> str | None:
    source_name = getattr(node, "source_name", None)
    if not isinstance(source_name, str):
        return None

    def_or_path = context.source_definitions.get(source_name)
    if isinstance(def_or_path, dict):
        path_value = def_or_path.get("path")
        return path_value if isinstance(path_value, str) else None
    return def_or_path if isinstance(def_or_path, str) else None


def _resolver_yaml_path_template(
    context: YamlRefactorContextProtocol,
    node: ResultNode,
) -> str | None:
    from dbt_osmosis.core.introspection import SettingsResolver

    raw_path_template = SettingsResolver(context=context).get_yaml_path_template(node)
    return raw_path_template if isinstance(raw_path_template, str) else None


def _default_yaml_path_template(context: YamlRefactorContextProtocol) -> str | None:
    try:
        project_vars = context.project.runtime_cfg.vars.to_dict()
        path_template = project_vars.get("dbt_osmosis_default_path")
    except Exception as e:  # noqa: BLE001
        logger.debug(":warning: Failed to read global var: %s", e)
        return None

    if isinstance(path_template, str) and path_template:
        logger.debug(
            ":earth_americas: Using global var 'dbt_osmosis_default_path': %s",
            path_template,
        )
        return path_template
    return None


def _configured_yaml_path_template(
    context: YamlRefactorContextProtocol,
    node: ResultNode,
) -> str | None:
    return (
        _resolver_yaml_path_template(context, node)
        or _resolve_vars_routing(context, node)
        or _default_yaml_path_template(context)
    )


def _missing_yaml_template_error(node: ResultNode) -> MissingOsmosisConfig:
    return MissingOsmosisConfig(
        f"Config key `+dbt-osmosis:`, var `dbt_osmosis_default_path`,"
        f" or vars.dbt-osmosis.models not set for model {node.name}",
    )


def _get_yaml_path_template(context: YamlRefactorContextProtocol, node: ResultNode) -> str | None:
    """Get the yaml path template for a dbt model or source node.

    Resolution order:
    1. Source definitions (vars.dbt-osmosis.sources) for source nodes
    2. SettingsResolver: node.config.extra, config.meta, node.meta, unrendered_config
    3. vars.dbt-osmosis.models / .seeds per-folder routing (fusion-compatible)
    4. vars.dbt_osmosis_default_path global fallback
    """
    if node.resource_type == NodeType.Source:
        return _source_yaml_path_template(context, node)

    path_template = _configured_yaml_path_template(context, node)
    if not path_template:
        raise _missing_yaml_template_error(node)
    logger.debug(":gear: Resolved YAML path template => %s", path_template)
    return path_template


def get_current_yaml_path(
    context: YamlRefactorContextProtocol,
    node: ResultNode,
) -> Path | None:
    """Get the current yaml path for a dbt model or source node."""
    project_root = context.project.runtime_cfg.project_root
    if not project_root:
        return None
    if node.resource_type in (NodeType.Model, NodeType.Seed) and getattr(node, "patch_path", None):
        path = Path(project_root).joinpath(
            t.cast("str", node.patch_path).partition("://")[-1],
        )
        logger.debug(":page_facing_up: Current YAML path => %s", path)
        return path
    if node.resource_type == NodeType.Source:
        path = Path(project_root, node.path)
        logger.debug(":page_facing_up: Current YAML path => %s", path)
        return path
    return None


def get_target_yaml_path(context: YamlRefactorContextProtocol, node: ResultNode) -> Path:
    """Get the target yaml path for a dbt model or source node."""
    project_root = t.cast("str", context.project.runtime_cfg.project_root)
    if not project_root:
        raise PathResolutionError("Project root is not set in runtime config.")
    model_paths = t.cast("list[str]", context.project.runtime_cfg.model_paths or ["models"])

    tpl = _get_yaml_path_template(context, node)
    if not tpl:
        logger.warning(":warning: No path template found for => %s", node.unique_id)
        return Path(project_root, node.original_file_path)

    path = Path(project_root, node.original_file_path)

    format_dict = {
        "model": node.name,
        "parent": path.parent.name,
        "schema": node.schema,
        "node": node,
    }

    try:
        rendered = tpl.format(**format_dict)
    except KeyError as exc:
        missing_key = exc.args[0] if exc.args else "<unknown>"
        raise PathResolutionError(
            f"Unable to render YAML path template for node '{node.unique_id}' using template "
            f"'{tpl}': missing template key '{missing_key}'."
        ) from exc
    except AttributeError as exc:
        raise PathResolutionError(
            f"Unable to render YAML path template for node '{node.unique_id}' using template "
            f"'{tpl}': missing or invalid template attribute ({exc})."
        ) from exc

    segments: list[Path | str] = []

    if node.resource_type == NodeType.Source:
        segments.append(model_paths[0])
    elif rendered.startswith("/"):
        segments.append(model_paths[0])
        # SECURITY: Remove only the first leading slash, not all slashes (prevents path traversal)
        rendered = rendered.removeprefix("/")
    else:
        segments.append(path.parent)

    if not (rendered.endswith((".yml", ".yaml"))):
        rendered += ".yml"
    segments.append(rendered)

    path = Path(project_root, *segments)
    # SECURITY: Validate path is within project root to prevent directory traversal
    resolved_path = path.resolve()
    project_root_path = Path(project_root).resolve()
    if not resolved_path.is_relative_to(project_root_path):
        raise PathResolutionError(
            f"Security violation: Target YAML path '{resolved_path}' is outside project root '{project_root_path}'",
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


def _resolve_source_definition_spec(
    source: str,
    spec: t.Any,
    default_database: str,
) -> tuple[str, str, str] | None:
    if isinstance(spec, str):
        return default_database, source, spec
    if isinstance(spec, dict):
        return (
            t.cast("str", spec.get("database", default_database)),
            t.cast("str", spec.get("schema", source)),
            t.cast("str", spec["path"]),
        )
    return None


def _resolve_source_yaml_path(
    project_root: str,
    model_paths: list[str],
    src_yaml_path: str,
) -> Path:
    # SECURITY: Remove only the first leading separator, not all (prevents path traversal).
    cleaned_path = src_yaml_path[1:] if src_yaml_path.startswith(os.sep) else src_yaml_path
    src_yaml_path_obj = Path(project_root, model_paths[0], cleaned_path)

    # SECURITY: Validate path is within project root to prevent directory traversal.
    resolved_path = src_yaml_path_obj.resolve()
    project_root_path = Path(project_root).resolve()
    if not resolved_path.is_relative_to(project_root_path):
        raise PathResolutionError(
            f"Security violation: Source YAML path '{resolved_path}' is outside project root '{project_root_path}'",
        )
    return src_yaml_path_obj


def _format_source_value(value: str, *, uppercase: bool, lowercase: bool) -> str:
    if uppercase:
        return value.upper()
    if lowercase:
        return value.lower()
    return value


def _describe_source_relation(
    context: t.Any,
    relation: t.Any,
    *,
    uppercase: bool,
    lowercase: bool,
    get_columns: t.Callable[[t.Any, t.Any], dict[str, t.Any]],
) -> dict[str, t.Any]:
    assert relation.identifier, "No identifier found for relation."
    identifier_name = _format_source_value(
        relation.identifier,
        uppercase=uppercase,
        lowercase=lowercase,
    )

    columns = [
        {
            "name": _format_source_value(name, uppercase=uppercase, lowercase=lowercase),
            "description": meta.comment or "",
            "data_type": _format_source_value(meta.type, uppercase=uppercase, lowercase=lowercase),
        }
        for name, meta in get_columns(context, relation).items()
    ]
    if context.settings.skip_add_data_types:
        for col in columns:
            _ = col.pop("data_type", None)

    return {"name": identifier_name, "description": "", "columns": columns}


def _source_manifest_tables(context: t.Any, source: str) -> set[str] | None:
    source_nodes = [
        node for node in context.project.manifest.sources.values() if node.source_name == source
    ]
    if not source_nodes:
        return None

    manifest_tables = {node.name for node in source_nodes}
    logger.debug(
        ":white_check_mark: Source => %s already exists in the manifest with %d tables.",
        source,
        len(manifest_tables),
    )
    return manifest_tables


def _write_existing_source_tables(
    context: t.Any,
    *,
    src_yaml_path_obj: Path,
    source: str,
    db_tables: dict[str, dict[str, t.Any]],
    new_table_names: set[str],
    written_file_tracker: t.Callable[[Path], None] | None,
) -> bool:
    from dbt_osmosis.core.schema.reader import _read_yaml
    from dbt_osmosis.core.schema.writer import _write_yaml

    if not new_table_names:
        logger.debug(":white_check_mark: No new tables found for source => %s", source)
        return False

    logger.info(
        ":sparkles: Found %d new tables for source => %s: %s",
        len(new_table_names),
        source,
        sorted(new_table_names),
    )
    existing_doc = _read_yaml(
        context.yaml_handler,
        context.yaml_handler_lock,
        src_yaml_path_obj,
    )
    for src_entry in existing_doc.get("sources", []):
        if src_entry.get("name") != source:
            continue

        existing_tables_set = {tbl["name"] for tbl in src_entry.get("tables", [])}
        for table_name in sorted(new_table_names):
            if table_name in existing_tables_set:
                continue
            src_entry.setdefault("tables", []).append(db_tables[table_name])
            logger.info(":plus: Adding new table => %s to source => %s", table_name, source)
        break

    src_yaml_path_obj.parent.mkdir(parents=True, exist_ok=True)
    _write_yaml(
        context.yaml_handler,
        context.yaml_handler_lock,
        src_yaml_path_obj,
        existing_doc,
        context.settings.dry_run,
        context.register_mutations,
        context.settings.strip_eof_blank_lines,
        written_file_tracker=written_file_tracker,
    )
    return True


def _write_missing_source_yaml(
    context: t.Any,
    *,
    src_yaml_path_obj: Path,
    source: str,
    database: str,
    schema: str,
    db_tables: dict[str, dict[str, t.Any]],
    written_file_tracker: t.Callable[[Path], None] | None,
) -> bool:
    from dbt_osmosis.core.schema.writer import _write_yaml

    tables = list(db_tables.values())
    source_dict = {"name": source, "database": database, "schema": schema, "tables": tables}
    logger.info(
        ":books: Injecting new source => %s => %s with %d tables",
        source_dict["name"],
        src_yaml_path_obj,
        len(tables),
    )
    _write_yaml(
        context.yaml_handler,
        context.yaml_handler_lock,
        src_yaml_path_obj,
        {"version": 2, "sources": [source_dict]},
        context.settings.dry_run,
        context.register_mutations,
        context.settings.strip_eof_blank_lines,
        written_file_tracker=written_file_tracker,
    )
    return True


def _process_source_definition(
    context: t.Any,
    *,
    source: str,
    spec: t.Any,
    default_database: str,
    project_root: str,
    model_paths: list[str],
    uppercase: bool,
    lowercase: bool,
    written_file_tracker: t.Callable[[Path], None] | None,
    get_columns: t.Callable[[t.Any, t.Any], dict[str, t.Any]],
) -> bool:
    source_spec = _resolve_source_definition_spec(source, spec, default_database)
    if source_spec is None:
        return False

    database, schema, src_yaml_path = source_spec
    manifest_tables = _source_manifest_tables(context, source)
    src_yaml_path_obj = _resolve_source_yaml_path(project_root, model_paths, src_yaml_path)
    db_relations = list(context.project.adapter.list_relations(database=database, schema=schema))
    db_tables = {
        rel.identifier: _describe_source_relation(
            context,
            rel,
            uppercase=uppercase,
            lowercase=lowercase,
            get_columns=get_columns,
        )
        for rel in db_relations
    }

    if manifest_tables is None:
        return _write_missing_source_yaml(
            context,
            src_yaml_path_obj=src_yaml_path_obj,
            source=source,
            database=database,
            schema=schema,
            db_tables=db_tables,
            written_file_tracker=written_file_tracker,
        )

    return _write_existing_source_tables(
        context,
        src_yaml_path_obj=src_yaml_path_obj,
        source=source,
        db_tables=db_tables,
        new_table_names=set(db_tables) - manifest_tables,
        written_file_tracker=written_file_tracker,
    )


def create_missing_source_yamls(context: t.Any) -> None:
    """Create source files for sources defined in the dbt_project.yml dbt-osmosis var which don't exist as nodes.

    This is a useful preprocessing step to ensure that all sources are represented in the dbt project manifest. We
    do not have rich node information for non-existent sources, hence the alternative codepath here to bootstrap them.

    For existing sources, this function will also discover and add new tables from the database that are not yet
    defined in the source YAML file (fixes #217).
    """
    from dbt_osmosis.core.config import _reload_manifest
    from dbt_osmosis.core.introspection import get_columns

    if context.project.config.disable_introspection:
        logger.warning(":warning: Introspection is disabled, cannot create missing source YAMLs.")
        return
    logger.info(":factory: Creating missing source YAMLs and updating existing sources (if any).")
    default_database: str = context.project.runtime_cfg.credentials.database
    lowercase: bool = context.settings.output_to_lower
    uppercase: bool = context.settings.output_to_upper
    project_root = t.cast("str", context.project.runtime_cfg.project_root)
    if not project_root:
        raise PathResolutionError("Project root is not set in runtime config.")
    model_paths = t.cast("list[str]", context.project.runtime_cfg.model_paths or ["models"])

    starting_disk_mutations = getattr(context, "disk_mutation_count", 0)
    source_changes_detected = False
    written_file_tracker = getattr(context, "register_written_file", None)
    for source, spec in context.source_definitions.items():
        source_changes_detected = (
            _process_source_definition(
                context,
                source=source,
                spec=spec,
                default_database=default_database,
                project_root=project_root,
                model_paths=model_paths,
                uppercase=uppercase,
                lowercase=lowercase,
                written_file_tracker=written_file_tracker,
                get_columns=get_columns,
            )
            or source_changes_detected
        )

    if getattr(
        context, "disk_mutation_count", starting_disk_mutations
    ) > starting_disk_mutations or (source_changes_detected and not context.settings.dry_run):
        logger.info(
            ":arrows_counterclockwise: Sources were updated, reloading the project.",
        )
        _reload_manifest(context.project)
