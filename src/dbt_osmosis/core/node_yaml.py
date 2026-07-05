from __future__ import annotations

import typing as t
from pathlib import Path
from types import MappingProxyType

from dbt.contracts.graph.nodes import ModelNode, ResultNode, SeedNode, SourceDefinition

from dbt_osmosis.core.model_versions import _versioned_model_yaml_view
from dbt_osmosis.core.schema.reader import _read_yaml

if t.TYPE_CHECKING:
    from dbt_osmosis.core.dbt_protocols import YamlRefactorContextProtocol

__all__ = ["_get_node_yaml"]


def _find_first(
    coll: t.Iterable[t.Any],
    predicate: t.Callable[[t.Any], bool],
    default: t.Any = None,
) -> t.Any:
    for item in coll:
        if predicate(item):
            return item
    return default


def _get_node_yaml(
    context: YamlRefactorContextProtocol,
    member: ResultNode,
) -> MappingProxyType[str, t.Any] | None:
    """Get a read-only view of the parsed YAML for a dbt model or source node."""
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
            if isinstance(member, ModelNode):
                versioned_doc = _versioned_model_yaml_view(maybe_doc, member)
                if versioned_doc is not None:
                    return MappingProxyType(versioned_doc)
            return MappingProxyType(maybe_doc)

    return None
