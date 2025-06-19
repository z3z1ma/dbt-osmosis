import threading
import typing as t
from concurrent.futures import FIRST_EXCEPTION, Future, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

from dbt.contracts.graph.nodes import ResultNode, ModelNode, SeedNode, SourceDefinition
from dbt.artifacts.resources.types import NodeType

import dbt_osmosis.core.logger as logger

__all__ = [
    "RestructureOperation",
    "RestructureDeltaPlan",
    "_generate_minimal_model_yaml",
    "_generate_minimal_source_yaml",
    "_create_operations_for_node",
    "draft_restructure_delta_plan",
    "pretty_print_plan",
    "_remove_models",
    "_remove_seeds",
    "_remove_sources",
    "apply_restructure_plan",
]


@dataclass
class RestructureOperation:
    """Represents a single operation to perform on a YAML file.

    This might be CREATE, UPDATE, SUPERSEDE, etc. In a more advanced approach,
    we might unify multiple steps under a single operation with sub-operations.
    """

    file_path: Path
    content: dict[str, t.Any]
    superseded_paths: dict[Path, list[ResultNode]] = field(default_factory=dict)


@dataclass
class RestructureDeltaPlan:
    """Stores all the operations needed to restructure the project."""

    operations: list[RestructureOperation] = field(default_factory=list)


def _generate_minimal_model_yaml(node: Union[ModelNode, SeedNode]) -> dict[str, t.Any]:
    """Generate a minimal model yaml for a dbt model node."""
    logger.debug(":baby: Generating minimal yaml for Model/Seed => %s", node.name)
    return {"name": node.name, "columns": []}


def _generate_minimal_source_yaml(node: SourceDefinition) -> dict[str, t.Any]:
    """Generate a minimal source yaml for a dbt source node."""
    logger.debug(":baby: Generating minimal yaml for Source => %s", node.name)
    return {"name": node.source_name, "tables": [{"name": node.name, "columns": []}]}


def _create_operations_for_node(
    context: t.Any,
    uid: str,
    loc: t.Any,  # SchemaFileLocation
) -> list[RestructureOperation]:
    """Create restructure operations for a dbt model or source node."""
    logger.debug(":bricks: Creating restructure operations for => %s", uid)
    node = context.project.manifest.nodes.get(uid) or context.project.manifest.sources.get(uid)
    if not node:
        logger.warning(":warning: Node => %s not found in manifest.", uid)
        return []

    # If loc.current is None => we are generating a brand new file
    # If loc.current => we unify it with the new location
    ops: list[RestructureOperation] = []

    if loc.current is None:
        logger.info(":sparkles: No current YAML file, building minimal doc => %s", uid)
        if isinstance(node, (ModelNode, SeedNode)):
            minimal = _generate_minimal_model_yaml(node)
            ops.append(
                RestructureOperation(
                    file_path=loc.target,
                    content={"version": 2, f"{node.resource_type}s": [minimal]},
                )
            )
        else:
            minimal = _generate_minimal_source_yaml(t.cast(SourceDefinition, node))
            ops.append(
                RestructureOperation(
                    file_path=loc.target,
                    content={"version": 2, "sources": [minimal]},
                )
            )
    else:
        from dbt_osmosis.core.schema.reader import _read_yaml

        existing = _read_yaml(context.yaml_handler, context.yaml_handler_lock, loc.current)
        injectable: dict[str, t.Any] = {"version": 2}
        injectable.setdefault("models", [])
        injectable.setdefault("sources", [])
        injectable.setdefault("seeds", [])
        if loc.node_type == NodeType.Model:
            assert isinstance(node, ModelNode)
            for obj in existing.get("models", []):
                if obj["name"] == node.name:
                    injectable["models"].append(obj)
                    break
        elif loc.node_type == NodeType.Source:
            assert isinstance(node, SourceDefinition)
            for src in existing.get("sources", []):
                if src["name"] == node.source_name:
                    injectable["sources"].append(src)
                    break
        elif loc.node_type == NodeType.Seed:
            assert isinstance(node, SeedNode)
            for seed in existing.get("seeds", []):
                if seed["name"] == node.name:
                    injectable["seeds"].append(seed)
        ops.append(
            RestructureOperation(
                file_path=loc.target,
                content=injectable,
                superseded_paths={loc.current: [node]},
            )
        )
    return ops


def draft_restructure_delta_plan(context: t.Any) -> RestructureDeltaPlan:
    """Draft a restructure plan for the dbt project."""
    logger.info(":bulb: Drafting restructure delta plan for the project.")
    plan = RestructureDeltaPlan()
    lock = threading.Lock()

    def _job(uid: str, loc: t.Any) -> None:
        ops = _create_operations_for_node(context, uid, loc)
        with lock:
            plan.operations.extend(ops)

    futs: list[Future[None]] = []
    from dbt_osmosis.core.path_management import build_yaml_file_mapping

    for uid, loc in build_yaml_file_mapping(context).items():
        if not loc.is_valid:
            futs.append(context.pool.submit(_job, uid, loc))
    done, _ = wait(futs, return_when=FIRST_EXCEPTION)
    for fut in done:
        exc = fut.exception()
        if exc:
            logger.error(":bomb: Error encountered while drafting plan => %s", exc)
            raise exc

    # Deduplicate operations by target file path
    deduplicated_ops: dict[Path, RestructureOperation] = {}
    for op in plan.operations:
        if op.file_path in deduplicated_ops:
            # merge content rather than replacing
            existing_op = deduplicated_ops[op.file_path]
            for resource_type, resources in op.content.items():
                if resource_type not in existing_op.content:
                    existing_op.content[resource_type] = resources
                elif isinstance(resources, list) and isinstance(
                    existing_op.content[resource_type], list
                ):
                    # for model lists, deduplicate by model name
                    if resource_type in ("models", "seeds"):
                        existing_models = {
                            m.get("name"): m for m in existing_op.content[resource_type]
                        }
                        for model in resources:
                            if model.get("name") in existing_models:
                                # skip duplicate models - they're already included
                                continue
                            existing_op.content[resource_type].append(model)
                    else:
                        # for other types (like sources), just append
                        existing_op.content[resource_type].extend(resources)

            # merge superseded paths
            for path, nodes in op.superseded_paths.items():
                if path in existing_op.superseded_paths:
                    existing_op.superseded_paths[path].extend(nodes)
                else:
                    existing_op.superseded_paths[path] = nodes
        else:
            deduplicated_ops[op.file_path] = op

    plan.operations = list(deduplicated_ops.values())

    logger.info(":star2: Draft plan creation complete => %s operations", len(plan.operations))
    return plan


def pretty_print_plan(plan: RestructureDeltaPlan) -> None:
    """Pretty print the restructure plan for the dbt project."""
    logger.info(":mega: Restructure plan includes => %s operations.", len(plan.operations))
    for op in plan.operations:
        str_content = str(op.content)[:80] + "..."
        logger.info(":sparkles: Processing => %s", str_content)
        if not op.superseded_paths:
            logger.info(":blue_book: CREATE or MERGE => %s", op.file_path)
        else:
            old_paths = [p.name for p in op.superseded_paths.keys()] or ["UNKNOWN"]
            logger.info(":blue_book: %s -> %s", old_paths, op.file_path)


def _remove_models(existing_doc: dict[str, t.Any], nodes: list[ResultNode]) -> None:
    """Clean up the existing yaml doc by removing models superseded by the restructure plan."""
    logger.debug(":scissors: Removing superseded models => %s", [n.name for n in nodes])
    to_remove = {n.name for n in nodes if n.resource_type == NodeType.Model}
    keep = []
    for section in existing_doc.get("models", []):
        if section.get("name") not in to_remove:
            keep.append(section)
    existing_doc["models"] = keep


def _remove_seeds(existing_doc: dict[str, t.Any], nodes: list[ResultNode]) -> None:
    """Clean up the existing yaml doc by removing models superseded by the restructure plan."""
    logger.debug(":scissors: Removing superseded seeds => %s", [n.name for n in nodes])
    to_remove = {n.name for n in nodes if n.resource_type == NodeType.Seed}
    keep = []
    for section in existing_doc.get("seeds", []):
        if section.get("name") not in to_remove:
            keep.append(section)
    existing_doc["seeds"] = keep


def _remove_sources(existing_doc: dict[str, t.Any], nodes: list[ResultNode]) -> None:
    """Clean up the existing yaml doc by removing sources superseded by the restructure plan."""
    to_remove_sources = {
        (n.source_name, n.name) for n in nodes if n.resource_type == NodeType.Source
    }
    logger.debug(":scissors: Removing superseded sources => %s", sorted(to_remove_sources))
    keep_sources = []
    for section in existing_doc.get("sources", []):
        keep_tables = []
        for tbl in section.get("tables", []):
            if (section["name"], tbl["name"]) not in to_remove_sources:
                keep_tables.append(tbl)
        if keep_tables:
            section["tables"] = keep_tables
            keep_sources.append(section)
    existing_doc["sources"] = keep_sources


def apply_restructure_plan(
    context: t.Any, plan: RestructureDeltaPlan, *, confirm: bool = False
) -> None:
    """Apply the restructure plan for the dbt project."""
    if not plan.operations:
        logger.info(":white_check_mark: No changes needed in the restructure plan.")
        return

    if confirm:
        logger.info(":warning: Confirm option set => printing plan and waiting for user input.")
        pretty_print_plan(plan)

    while confirm:
        response = input("Apply the restructure plan? [y/N]: ")
        if response.lower() in ("y", "yes"):
            break
        elif response.lower() in ("n", "no", ""):
            logger.info("Skipping restructure plan.")
            return
        logger.warning(":loudspeaker: Please respond with 'y' or 'n'.")

    from dbt_osmosis.core.schema.reader import _read_yaml
    from dbt_osmosis.core.schema.writer import _write_yaml
    from dbt_osmosis.core.schema.reader import _YAML_BUFFER_CACHE
    from dbt_osmosis.core.config import _reload_manifest

    for op in plan.operations:
        logger.debug(":arrow_right: Applying restructure operation => %s", op)
        output_doc: dict[str, t.Any] = {"version": 2}
        if op.file_path.exists():
            existing_data = _read_yaml(
                context.yaml_handler, context.yaml_handler_lock, op.file_path
            )
            output_doc.update(existing_data)

        for key, val in op.content.items():
            if isinstance(val, list):
                output_doc.setdefault(key, []).extend(val)
            elif isinstance(val, dict):
                output_doc.setdefault(key, {}).update(val)
            else:
                output_doc[key] = val

        _write_yaml(
            context.yaml_handler,
            context.yaml_handler_lock,
            op.file_path,
            output_doc,
            context.settings.dry_run,
            context.register_mutations,
        )

        for path, nodes in op.superseded_paths.items():
            if path.is_file():
                existing_data = _read_yaml(context.yaml_handler, context.yaml_handler_lock, path)

                if "models" in existing_data:
                    _remove_models(existing_data, nodes)
                if "sources" in existing_data:
                    _remove_sources(existing_data, nodes)
                if "seeds" in existing_data:
                    _remove_seeds(existing_data, nodes)

                keys = set(existing_data.keys()) - {"version"}
                if all(len(existing_data.get(k, [])) == 0 for k in keys):
                    if not context.settings.dry_run:
                        path.unlink(missing_ok=True)
                        if path.parent.exists() and not any(path.parent.iterdir()):
                            path.parent.rmdir()
                        if path in _YAML_BUFFER_CACHE:
                            del _YAML_BUFFER_CACHE[path]
                    context.register_mutations(1)
                    logger.info(":heavy_minus_sign: Superseded entire file => %s", path)
                else:
                    _write_yaml(
                        context.yaml_handler,
                        context.yaml_handler_lock,
                        path,
                        existing_data,
                        context.settings.dry_run,
                        context.register_mutations,
                    )
                    logger.info(
                        ":arrow_forward: Migrated doc from => %s to => %s", path, op.file_path
                    )

    logger.info(
        ":arrows_counterclockwise: Committing all restructure changes and reloading manifest."
    )
    from dbt_osmosis.core.schema.writer import commit_yamls

    commit_yamls(
        context.yaml_handler,
        context.yaml_handler_lock,
        context.settings.dry_run,
        context.register_mutations,
    )
    _reload_manifest(context.project)
