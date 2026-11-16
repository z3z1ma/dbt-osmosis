from __future__ import annotations

import atexit
import time
import typing as t
from collections import ChainMap
from dataclasses import dataclass, field
from functools import partial
from types import MappingProxyType, NotImplementedType

from dbt.artifacts.resources.types import NodeType
from dbt.contracts.graph.nodes import (  # pyright: ignore[reportPrivateImportUsage]
    ColumnInfo,
    ResultNode,
)

if t.TYPE_CHECKING:
    from dbt_osmosis.core.dbt_protocols import (
        YamlRefactorContextProtocol,
    )

from dbt_osmosis.core import logger

__all__ = [
    "TransformOperation",
    "TransformPipeline",
    "_transform_op",
    "apply_semantic_analysis",
    "inherit_upstream_column_knowledge",
    "inject_missing_columns",
    "remove_columns_not_in_database",
    "sort_columns_alphabetically",
    "sort_columns_as_configured",
    "sort_columns_as_in_database",
    "suggest_improved_documentation",
    "synchronize_data_types",
    "synthesize_missing_documentation_with_openai",
]


def _order_preserving_union(primary: t.Iterable[str], secondary: t.Iterable[str]) -> list[str]:
    """Return primary items followed by unseen secondary items in their original order."""
    merged: list[str] = []
    seen: set[str] = set()
    for item in (*primary, *secondary):
        if item in seen:
            continue
        seen.add(item)
        merged.append(item)
    return merged


@dataclass
class TransformOperation:
    """An operation to be run on a dbt manifest node."""

    func: t.Callable[..., t.Any]
    name: str

    _result: t.Any | None = field(init=False, default=None)
    _context: t.Any | None = field(init=False, default=None)  # YamlRefactorContext
    _node: ResultNode | None = field(init=False, default=None)
    _metadata: dict[str, t.Any] = field(init=False, default_factory=dict)

    @property
    def result(self) -> t.Any:
        """The result of the operation or None."""
        return self._result

    @property
    def metadata(self) -> MappingProxyType[str, t.Any]:
        """Metadata about the operation."""
        return MappingProxyType(self._metadata)

    def __call__(
        self,
        context: YamlRefactorContextProtocol,
        node: ResultNode | None = None,  # YamlRefactorContextProtocol
    ) -> TransformOperation:
        """Run the operation and store the result."""
        self._context = context
        self._node = node
        self._metadata["started"] = True
        try:
            self.func(context, node)
            self._metadata["success"] = True
        except Exception as e:
            self._metadata["error"] = str(e)
            raise
        return self

    def __rshift__(self, next_op: TransformOperation) -> TransformPipeline:
        """Chain operations together."""
        return TransformPipeline([self]) >> next_op

    def __repr__(self) -> str:
        return f"<Operation: {self.name} (success={self.metadata.get('success', False)})>"


@dataclass
class TransformPipeline:
    """A pipeline of transform operations to be run on a dbt manifest node."""

    operations: list[TransformOperation] = field(default_factory=list)
    commit_mode: t.Literal["none", "batch", "atomic", "defer"] = "batch"

    _metadata: dict[str, t.Any] = field(init=False, default_factory=dict)

    @property
    def metadata(self) -> MappingProxyType[str, t.Any]:
        """Metadata about the pipeline."""
        return MappingProxyType(self._metadata)

    @t.overload
    def __rshift__(
        self,
        next_op: TransformOperation | t.Callable[..., t.Any],
    ) -> TransformPipeline:
        pass

    @t.overload
    def __rshift__(self, next_op: object) -> TransformPipeline | NotImplementedType:
        pass

    def __rshift__(self, next_op: object) -> TransformPipeline | NotImplementedType:
        """Chain operations together."""
        if isinstance(next_op, TransformOperation):
            self.operations.append(next_op)
        elif callable(next_op):
            operation_name = getattr(next_op, "__name__", next_op.__class__.__name__)
            self.operations.append(TransformOperation(next_op, operation_name))
        else:
            return NotImplemented
        return self

    def __call__(
        self,
        context: YamlRefactorContextProtocol,
        node: ResultNode | None = None,  # YamlRefactorContextProtocol
    ) -> TransformPipeline:
        """Run all operations in the pipeline."""
        logger.info(
            "\n:gear: [b]Running pipeline[/b] with => %s operations %s \n",
            len(self.operations),
            [op.name for op in self.operations],
        )

        self._metadata["started_at"] = (pipeline_start := time.time())
        for op in self.operations:
            logger.info(
                ":gear:  [b]Starting to[/b] [yellow]%s[/yellow]",
                op.name,
            )
            step_start = time.time()
            _ = op(context, node)
            step_end = time.time()
            logger.info(
                ":sparkles: [b]Done with[/b] [green]%s[/green] in %.2fs \n",
                op.name,
                step_end - step_start,
            )
            self._metadata.setdefault("steps", []).append({
                **op.metadata,
                "duration": step_end - step_start,
            })
            if self.commit_mode == "atomic":
                logger.info(
                    ":hourglass: [b]Committing[/b] Operation => [green]%s[/green]",
                    op.name,
                )
                from dbt_osmosis.core.sync_operations import sync_node_to_yaml

                sync_node_to_yaml(context, node, commit=True)
                logger.info(":checkered_flag: [b]Committed[/b] \n")
        self._metadata["completed_at"] = (pipeline_end := time.time())

        logger.info(
            ":checkered_flag: [b]Manifest transformation pipeline [green]completed[/green] in => %.2fs[/b]",
            pipeline_end - pipeline_start,
        )

        def _commit() -> None:
            """Commit changes to YAML files. Designed for use as an atexit handler."""
            logger.info(":hourglass: Committing all changes to YAML files in batch.")
            _commit_start = time.time()
            try:
                from dbt_osmosis.core.sync_operations import sync_node_to_yaml

                sync_node_to_yaml(context, node, commit=True)
                _commit_end = time.time()
                logger.info(
                    ":checkered_flag: YAML commits completed in => %.2fs",
                    _commit_end - _commit_start,
                )
            except Exception as e:  # noqa: BLE001
                # Log error but don't raise during atexit (prevents shutdown issues)
                logger.error(":boom: Failed to commit YAML changes during shutdown: %s", e)

        if self.commit_mode == "batch":
            _commit()
        elif self.commit_mode == "defer":
            logger.warning(
                ":warning: Using 'defer' commit mode with atexit.register. "
                "This may cause issues if locks are held during shutdown. "
                "Consider using 'batch' or 'atomic' mode instead.",
            )
            _ = atexit.register(_commit)

        return self

    def __repr__(self) -> str:
        steps = [op.name for op in self.operations]
        return f"<OperationPipeline: {len(self.operations)} operations, steps={steps!r}>"


def _transform_op(
    name: str | None = None,
) -> t.Callable[[t.Callable[[t.Any, ResultNode | None], None]], TransformOperation]:
    """Decorator to create a TransformOperation from a function."""

    def decorator(
        func: t.Callable[[t.Any, ResultNode | None], None],  # YamlRefactorContext
    ) -> TransformOperation:
        operation_name = t.cast("str", name or getattr(func, "__name__", func.__class__.__name__))
        return TransformOperation(func, name=operation_name)

    return decorator


@_transform_op("Inherit Upstream Column Knowledge")
def inherit_upstream_column_knowledge(
    context: YamlRefactorContextProtocol,
    node: ResultNode | None = None,  # YamlRefactorContext
) -> None:
    """Inherit column level knowledge from the ancestors of a dbt model or source node."""
    if node is None:
        logger.info(":wave: Inheriting column knowledge across all matched nodes.")
        from dbt_osmosis.core.node_filters import _iter_candidate_nodes

        for _ in context.pool.map(
            partial(inherit_upstream_column_knowledge, context),
            (n for _, n in _iter_candidate_nodes(context, exclude_external_patches=True)),
        ):
            pass
        return

    logger.info(":dna: Inheriting column knowledge for => %s", node.unique_id)

    from dbt_osmosis.core.inheritance import _build_column_knowledge_graph

    column_knowledge_graph = _build_column_knowledge_graph(context, node)
    for name, node_column in node.columns.items():
        kwargs = column_knowledge_graph.get(name)
        if kwargs is None:
            continue
        inheritable = _inheritable_metadata_keys(context, node, name, node_column, kwargs)
        updated_metadata = _metadata_to_inherit(kwargs, inheritable)
        logger.debug(
            ":star2: Inheriting updated metadata => %s for column => %s",
            updated_metadata,
            name,
        )
        node.columns[name] = node_column.replace(**updated_metadata)


def _inheritable_metadata_keys(
    context: YamlRefactorContextProtocol,
    node: ResultNode,
    column_name: str,
    node_column: ColumnInfo,
    upstream_metadata: dict[str, t.Any],
) -> list[str]:
    inheritable = _base_inheritable_metadata_keys(context, node, column_name)
    _add_progenitor_meta_key(context, node, column_name, upstream_metadata, inheritable)
    _remove_local_description_key(context, node, column_name, node_column, inheritable)
    return inheritable


def _base_inheritable_metadata_keys(
    context: YamlRefactorContextProtocol,
    node: ResultNode,
    column_name: str,
) -> list[str]:
    from dbt_osmosis.core.introspection import resolve_setting

    inheritable: list[str] = []
    if not resolve_setting(
        context,
        "skip-inherit-descriptions",
        node,
        column_name,
        fallback=context.settings.skip_inherit_descriptions,
    ):
        inheritable.append("description")
    if not resolve_setting(
        context,
        "skip-add-tags",
        node,
        column_name,
        fallback=context.settings.skip_add_tags,
    ):
        inheritable.append("tags")
    if not resolve_setting(
        context,
        "skip-merge-meta",
        node,
        column_name,
        fallback=context.settings.skip_merge_meta,
    ):
        inheritable.append("meta")
    for extra in resolve_setting(
        context,
        "add-inheritance-for-specified-keys",
        node,
        column_name,
        fallback=context.settings.add_inheritance_for_specified_keys,
    ):
        if extra not in inheritable:
            inheritable.append(extra)
    return inheritable


def _add_progenitor_meta_key(
    context: YamlRefactorContextProtocol,
    node: ResultNode,
    column_name: str,
    upstream_metadata: dict[str, t.Any],
    inheritable: list[str],
) -> None:
    from dbt_osmosis.core.introspection import resolve_setting

    if not resolve_setting(
        context,
        "add-progenitor-to-meta",
        node,
        column_name,
        fallback=context.settings.add_progenitor_to_meta,
    ):
        return

    meta_progenitor = upstream_metadata.get("meta", {}).get("osmosis_progenitor")
    if not meta_progenitor:
        meta_progenitor = (
            upstream_metadata.get("config", {}).get("meta", {}).get("osmosis_progenitor")
        )
    if meta_progenitor and "meta" not in inheritable:
        inheritable.append("meta")


def _remove_local_description_key(
    context: YamlRefactorContextProtocol,
    node: ResultNode,
    column_name: str,
    node_column: ColumnInfo,
    inheritable: list[str],
) -> None:
    from dbt_osmosis.core.introspection import resolve_setting

    if "description" not in inheritable or not node_column.description:
        return
    if not resolve_setting(
        context,
        "force-inherit-descriptions",
        node,
        column_name,
        fallback=context.settings.force_inherit_descriptions,
    ):
        inheritable.remove("description")


def _metadata_to_inherit(
    upstream_metadata: dict[str, t.Any],
    inheritable: list[str],
) -> dict[str, t.Any]:
    return {k: v for k, v in upstream_metadata.items() if v is not None and k in inheritable}


@_transform_op("Inject Missing Columns")
def inject_missing_columns(
    context: YamlRefactorContextProtocol,
    node: ResultNode | None = None,
) -> None:
    """Add missing columns to a dbt node and it's corresponding yaml section. Changes are implicitly buffered until commit_yamls is called."""
    from dbt_osmosis.core.introspection import get_columns
    from dbt_osmosis.core.node_filters import _iter_candidate_nodes

    if node is None:
        logger.info(":wave: Injecting missing columns for all matched nodes.")
        for _ in context.pool.map(
            partial(inject_missing_columns, context),
            (n for _, n in _iter_candidate_nodes(context, exclude_external_patches=True)),
        ):
            pass
        return
    if _skip_column_injection(context, node):
        logger.debug(":no_entry_sign: Skipping column injection (skip_add_columns=True).")
        return
    if _skip_source_column_injection(context, node):
        logger.debug(":no_entry_sign: Skipping column injection (skip_add_source_columns=True).")
        return

    incoming_columns = get_columns(context, node)
    output_to_upper, output_to_lower = _output_case_settings(context, node)
    current_columns = _current_column_compare_names(
        context, node, case_insensitive=bool(output_to_upper or output_to_lower)
    )

    for incoming_name, incoming_meta in incoming_columns.items():
        compare_name = (
            incoming_name.lower() if output_to_upper or output_to_lower else incoming_name
        )
        if compare_name not in current_columns:
            logger.info(
                ":heavy_plus_sign: Reconciling missing column => %s in node => %s",
                incoming_name,
                node.unique_id,
            )
            final_name = _output_column_name(incoming_name, output_to_upper, output_to_lower)
            gen_col = _generated_column_dict(
                context,
                node,
                final_name,
                incoming_meta.comment,
                incoming_meta.type,
                output_to_upper,
                output_to_lower,
            )
            node.columns[final_name] = ColumnInfo.from_dict(gen_col)


def _skip_column_injection(context: YamlRefactorContextProtocol, node: ResultNode) -> bool:
    from dbt_osmosis.core.introspection import resolve_setting

    return bool(
        resolve_setting(
            context, "skip-add-columns", node, fallback=context.settings.skip_add_columns
        )
    )


def _skip_source_column_injection(context: YamlRefactorContextProtocol, node: ResultNode) -> bool:
    from dbt_osmosis.core.introspection import resolve_setting

    return bool(
        resolve_setting(
            context,
            "skip-add-source-columns",
            node,
            fallback=context.settings.skip_add_source_columns,
        )
        and node.resource_type == NodeType.Source
    )


def _output_case_settings(
    context: YamlRefactorContextProtocol, node: ResultNode, column_name: str | None = None
) -> tuple[t.Any, t.Any]:
    from dbt_osmosis.core.introspection import resolve_setting

    output_to_upper = resolve_setting(
        context,
        "output-to-upper",
        node,
        column_name,
        fallback=context.settings.output_to_upper,
    )
    output_to_lower = resolve_setting(
        context,
        "output-to-lower",
        node,
        column_name,
        fallback=context.settings.output_to_lower,
    )
    return output_to_upper, output_to_lower


def _current_column_compare_names(
    context: YamlRefactorContextProtocol,
    node: ResultNode,
    *,
    case_insensitive: bool,
) -> set[str]:
    from dbt_osmosis.core.introspection import normalize_column_name

    credentials_type = context.project.runtime_cfg.credentials.type
    return {
        normalize_column_name(c.name, credentials_type).lower()
        if case_insensitive
        else normalize_column_name(c.name, credentials_type)
        for c in node.columns.values()
    }


def _output_column_name(column_name: str, output_to_upper: t.Any, output_to_lower: t.Any) -> str:
    if output_to_upper:
        return column_name.upper()
    if output_to_lower:
        return column_name.lower()
    return column_name


def _generated_column_dict(
    context: YamlRefactorContextProtocol,
    node: ResultNode,
    final_name: str,
    comment: str | None,
    data_type: str | None,
    output_to_upper: t.Any,
    output_to_lower: t.Any,
) -> dict[str, t.Any]:
    from dbt_osmosis.core.introspection import resolve_setting

    gen_col: dict[str, t.Any] = {"name": final_name, "description": comment or ""}
    if data_type and not resolve_setting(
        context,
        "skip-add-data-types",
        node,
        fallback=context.settings.skip_add_data_types,
    ):
        gen_col["data_type"] = _output_data_type(data_type, output_to_upper, output_to_lower)
    return gen_col


def _output_data_type(data_type: str, output_to_upper: t.Any, output_to_lower: t.Any) -> str:
    if output_to_upper:
        return data_type.upper()
    if output_to_lower:
        return data_type.lower()
    return data_type


@_transform_op("Remove Extra Columns")
def remove_columns_not_in_database(
    context: YamlRefactorContextProtocol,
    node: ResultNode | None = None,
) -> None:
    """Remove columns from a dbt node and it's corresponding yaml section that are not present in the database. Changes are implicitly buffered until commit_yamls is called."""
    from dbt_osmosis.core.introspection import (
        get_columns,
        normalize_column_name,
        resolve_setting,
    )
    from dbt_osmosis.core.node_filters import _iter_candidate_nodes

    if node is None:
        logger.info(":wave: Removing columns not in DB across all matched nodes.")
        for _ in context.pool.map(
            partial(remove_columns_not_in_database, context),
            (n for _, n in _iter_candidate_nodes(context, exclude_external_patches=True)),
        ):
            pass
        return
    output_to_upper = resolve_setting(
        context, "output-to-upper", node, fallback=context.settings.output_to_upper
    )
    output_to_lower = resolve_setting(
        context, "output-to-lower", node, fallback=context.settings.output_to_lower
    )
    case_insensitive = output_to_upper or output_to_lower
    current_columns = {
        (
            normalize_column_name(c.name, context.project.runtime_cfg.credentials.type).lower()
            if case_insensitive
            else normalize_column_name(c.name, context.project.runtime_cfg.credentials.type)
        ): key
        for key, c in node.columns.items()
    }
    incoming_columns = get_columns(context, node)
    if not incoming_columns:
        logger.info(
            ":no_entry_sign: No columns discovered for node => %s, skipping cleanup.",
            node.unique_id,
        )
        return
    incoming_keys = (
        {k.lower() for k in incoming_columns} if case_insensitive else set(incoming_columns.keys())
    )
    extra_columns = set(current_columns.keys()) - incoming_keys
    for extra_column in extra_columns:
        logger.info(
            ":heavy_minus_sign: Removing extra column => %s in node => %s",
            extra_column,
            node.unique_id,
        )
        _ = node.columns.pop(current_columns[extra_column], None)


@_transform_op("Sort Columns in DB Order")
def sort_columns_as_in_database(
    context: YamlRefactorContextProtocol,
    node: ResultNode | None = None,
) -> None:
    """Sort columns in a dbt node and it's corresponding yaml section as they appear in the database. Changes are implicitly buffered until commit_yamls is called."""
    from dbt_osmosis.core.introspection import get_columns, normalize_column_name, resolve_setting
    from dbt_osmosis.core.node_filters import _iter_candidate_nodes

    if node is None:
        logger.info(":wave: Sorting columns as they appear in DB across all matched nodes.")
        for _ in context.pool.map(
            partial(sort_columns_as_in_database, context),
            (n for _, n in _iter_candidate_nodes(context, exclude_external_patches=True)),
        ):
            pass
        return
    logger.info(":1234: Sorting columns by warehouse order => %s", node.unique_id)
    incoming_columns = get_columns(context, node)
    if not incoming_columns:
        logger.info(
            ":no_entry_sign: No columns discovered for node => %s, skipping db order sorting.",
            node.unique_id,
        )
        return

    credentials_type = context.project.runtime_cfg.credentials.type
    output_to_upper = resolve_setting(
        context, "output-to-upper", node, fallback=context.settings.output_to_upper
    )
    output_to_lower = resolve_setting(
        context, "output-to-lower", node, fallback=context.settings.output_to_lower
    )
    case_insensitive = output_to_upper or output_to_lower
    incoming_by_compare_name = {
        normalize_column_name(name, credentials_type).lower(): column
        for name, column in incoming_columns.items()
    }

    def _position(column: str) -> int:
        normalized_column = normalize_column_name(column, credentials_type)
        inc = (
            incoming_by_compare_name.get(normalized_column.lower())
            if case_insensitive
            else incoming_columns.get(normalized_column)
        )
        if inc is None or inc.index is None:
            return 99_999
        return inc.index

    node.columns = {k: v for k, v in sorted(node.columns.items(), key=lambda i: _position(i[0]))}


@_transform_op("Sort Columns Alphabetically")
def sort_columns_alphabetically(
    context: YamlRefactorContextProtocol,
    node: ResultNode | None = None,
) -> None:
    """Sort columns in a dbt node and it's corresponding yaml section alphabetically. Changes are implicitly buffered until commit_yamls is called."""
    from dbt_osmosis.core.introspection import resolve_setting
    from dbt_osmosis.core.node_filters import _iter_candidate_nodes

    if node is None:
        logger.info(":wave: Sorting columns alphabetically across all matched nodes.")
        for _ in context.pool.map(
            partial(sort_columns_alphabetically, context),
            (n for _, n in _iter_candidate_nodes(context, exclude_external_patches=True)),
        ):
            pass
        return
    logger.info(":abcd: Sorting columns alphabetically => %s", node.unique_id)

    # Determine the case conversion setting for sorting
    # We need to sort based on the FINAL case of the column names, not the original case
    output_to_lower = resolve_setting(
        context,
        "output-to-lower",
        node,
        fallback=context.settings.output_to_lower,
    )
    output_to_upper = resolve_setting(
        context,
        "output-to-upper",
        node,
        fallback=context.settings.output_to_upper,
    )

    def sort_key(item: tuple[str, t.Any]) -> str:
        """Generate a sort key based on the final case of the column name."""
        column_name = item[0]
        if output_to_upper:
            return column_name.upper()
        elif output_to_lower:
            return column_name.lower()
        else:
            return column_name

    node.columns = {k: v for k, v in sorted(node.columns.items(), key=sort_key)}


@_transform_op("Sort Columns")
def sort_columns_as_configured(
    context: YamlRefactorContextProtocol,
    node: ResultNode | None = None,
) -> None:
    from dbt_osmosis.core.introspection import resolve_setting
    from dbt_osmosis.core.node_filters import _iter_candidate_nodes

    if node is None:
        logger.info(":wave: Sorting columns as configured across all matched nodes.")
        for _ in context.pool.map(
            partial(sort_columns_as_configured, context),
            (n for _, n in _iter_candidate_nodes(context, exclude_external_patches=True)),
        ):
            pass
        return
    sort_by = resolve_setting(context, "sort-by", node, fallback="database")
    if sort_by == "database":
        _ = sort_columns_as_in_database(context, node)
    elif sort_by == "alphabetical":
        _ = sort_columns_alphabetically(context, node)
    else:
        raise ValueError(f"Invalid sort-by value: {sort_by} for node: {node.unique_id}")


@_transform_op("Synchronize Data Types")
def synchronize_data_types(
    context: YamlRefactorContextProtocol,
    node: ResultNode | None = None,
) -> None:
    """Populate data types for columns in a dbt node and it's corresponding yaml section. Changes are implicitly buffered until commit_yamls is called."""
    from dbt_osmosis.core.introspection import (
        get_columns,
        resolve_setting,
    )
    from dbt_osmosis.core.node_filters import _iter_candidate_nodes

    if node is None:
        logger.info(":wave: Populating data types across all matched nodes.")
        for _ in context.pool.map(
            partial(synchronize_data_types, context),
            (n for _, n in _iter_candidate_nodes(context, exclude_external_patches=True)),
        ):
            pass
        return
    logger.info(":1234: Synchronizing data types => %s", node.unique_id)
    incoming_columns = get_columns(context, node)
    incoming_columns_lower = {k.lower(): v for k, v in incoming_columns.items()}
    if resolve_setting(context, "skip-add-data-types", node, fallback=False):
        return
    for name, column in node.columns.items():
        if _skip_column_data_type_sync(context, node, name):
            continue
        uppercase, lowercase = _output_case_settings(context, node, name)
        incoming_column = _incoming_column_for_sync(
            context, name, incoming_columns, incoming_columns_lower, uppercase or lowercase
        )
        if incoming_column:
            _sync_column_data_type(column, incoming_column.type, uppercase, lowercase)


def _skip_column_data_type_sync(
    context: YamlRefactorContextProtocol, node: ResultNode, column_name: str
) -> bool:
    from dbt_osmosis.core.introspection import resolve_setting

    return bool(
        resolve_setting(
            context,
            "skip-add-data-types",
            node,
            column_name,
            fallback=context.settings.skip_add_data_types,
        )
    )


def _incoming_column_for_sync(
    context: YamlRefactorContextProtocol,
    column_name: str,
    incoming_columns: dict[str, t.Any],
    incoming_columns_lower: dict[str, t.Any],
    case_insensitive: bool,
) -> t.Any:
    from dbt_osmosis.core.introspection import normalize_column_name

    normalized = normalize_column_name(column_name, context.project.runtime_cfg.credentials.type)
    incoming_column = incoming_columns.get(normalized)
    if incoming_column is None and case_insensitive:
        return incoming_columns_lower.get(normalized.lower())
    return incoming_column


def _sync_column_data_type(
    column: ColumnInfo,
    incoming_type: str | None,
    output_to_upper: t.Any,
    output_to_lower: t.Any,
) -> None:
    if not incoming_type:
        return

    column.data_type = _synced_data_type(
        incoming_type,
        output_to_upper,
        output_to_lower,
        bool(column.data_type and column.data_type.islower()),
    )


def _synced_data_type(
    incoming_type: str,
    output_to_upper: t.Any,
    output_to_lower: t.Any,
    preserve_lowercase: bool,
) -> str:
    if output_to_upper:
        return incoming_type.upper()
    if output_to_lower or preserve_lowercase:
        return incoming_type.lower()
    return incoming_type


def _collect_upstream_documents(
    node: ResultNode,
    context: YamlRefactorContextProtocol,
) -> list[str]:
    """Collect upstream documentation from dependency nodes.

    Args:
        node: The dbt node to collect upstream docs for
        context: The YamlRefactorContext instance

    Returns:
        List of strings containing upstream documentation

    """
    import textwrap

    node_map = ChainMap(
        t.cast("dict[str, ResultNode]", context.project.manifest.nodes),
        t.cast("dict[str, ResultNode]", context.project.manifest.sources),
    )
    upstream_docs: list[str] = ["# The following is not exhaustive, but provides some context."]
    depends_on_nodes = t.cast("list[str]", node.depends_on_nodes)

    for i, uid in enumerate(depends_on_nodes):
        dep = node_map.get(uid)
        if dep is not None:
            _append_dependency_documents(upstream_docs, uid, dep, context, textwrap)
        # ensure our context window is bounded, semi-arbitrary
        if len(upstream_docs) > 100 and i < len(depends_on_nodes) - 1:
            upstream_docs.append(_remaining_dependencies_message(depends_on_nodes, i))
            break

    if len(upstream_docs) == 1:
        upstream_docs[0] = "(no upstream documentation found)"

    return upstream_docs


def _append_dependency_documents(
    upstream_docs: list[str],
    uid: str,
    dep: ResultNode,
    context: YamlRefactorContextProtocol,
    textwrap_module: t.Any,
) -> None:
    oneline_desc = dep.description.replace("\n", " ")
    upstream_docs.append(f"{uid}: # {oneline_desc}")
    for j, (name, meta) in enumerate(dep.columns.items()):
        if meta.description and meta.description not in context.placeholders:
            upstream_docs.append(f"- {name}: |\n{textwrap_module.indent(meta.description, '  ')}")
        if j > 20:
            # just a small amount of this supplementary context is sufficient
            upstream_docs.append("- (omitting additional columns for brevity)")
            break


def _remaining_dependencies_message(depends_on_nodes: list[str], index: int) -> str:
    return f"# remaining nodes are: {', '.join(depends_on_nodes[index:])}"


def _synthesize_bulk_documentation(
    node: ResultNode,
    upstream_docs: list[str],
    context: YamlRefactorContextProtocol,
) -> None:
    """Synthesize documentation in bulk for multiple columns.

    Args:
        node: The dbt node to synthesize documentation for
        upstream_docs: List of upstream documentation strings
        context: The YamlRefactorContext instance

    """
    from dbt_osmosis.core.llm import generate_model_spec_as_json

    logger.info(
        ":robot: Synthesizing bulk documentation for => %s columns in node => %s",
        len(node.columns)
        - len([
            c
            for c in node.columns.values()
            if c.description and c.description not in context.placeholders
        ]),
        node.unique_id,
    )

    spec = generate_model_spec_as_json(
        getattr(
            node,
            "compiled_sql",
            f"SELECT {', '.join(node.columns)} FROM {node.schema}.{node.name}",
        ),
        upstream_docs=upstream_docs,
        existing_context=f"NodeId={node.unique_id}\nTableDescription={node.description}",
        temperature=0.4,
    )

    if not node.description or node.description in context.placeholders:
        node.description = spec.get("description", node.description)

    for synth_col in spec.get("columns", []):
        usr_col = node.columns.get(synth_col["name"])
        if usr_col and (not usr_col.description or usr_col.description in context.placeholders):
            usr_col.description = synth_col.get("description", usr_col.description)


def _synthesize_node_documentation(
    node: ResultNode,
    upstream_docs: list[str],
    context: YamlRefactorContextProtocol,
) -> None:
    """Synthesize documentation for the node itself.

    Args:
        node: The dbt node to synthesize documentation for
        upstream_docs: List of upstream documentation strings
        context: The YamlRefactorContext instance

    """
    from dbt_osmosis.core.llm import generate_table_doc

    if not node.description or node.description in context.placeholders:
        logger.info(
            ":robot: Synthesizing documentation for node => %s",
            node.unique_id,
        )
        node.description = generate_table_doc(
            getattr(
                node,
                "compiled_sql",
                f"SELECT {', '.join(node.columns)} FROM {node.schema}.{node.name}",
            ),
            table_name=node.relation_name or node.name,
            upstream_docs=upstream_docs,
        )


def _synthesize_individual_column_documentation(
    node: ResultNode,
    upstream_docs: list[str],
    context: YamlRefactorContextProtocol,
) -> None:
    """Synthesize documentation for individual columns.

    Args:
        node: The dbt node to synthesize documentation for
        upstream_docs: List of upstream documentation strings
        context: The YamlRefactorContext instance

    """
    from dbt_osmosis.core.llm import generate_column_doc

    for column_name, column in node.columns.items():
        if not column.description or column.description in context.placeholders:
            logger.info(
                ":robot: Synthesizing documentation for column => %s in node => %s",
                column_name,
                node.unique_id,
            )
            column.description = generate_column_doc(
                column_name,
                existing_context=f"DataType={column.data_type or 'unknown'}>\nColumnParent={node.unique_id}\nTableDescription={node.description}",
                table_name=node.relation_name or node.name,
                upstream_docs=upstream_docs,
                temperature=0.7,
            )


def synthesize_missing_documentation_with_openai(
    context: YamlRefactorContextProtocol,
    node: ResultNode | None = None,
) -> None:
    """Synthesize missing documentation for a dbt node using OpenAI's GPT-4o API."""
    from dbt_osmosis.core.node_filters import _iter_candidate_nodes

    try:
        import importlib.util

        importlib.util.find_spec("dbt_osmosis.core.llm")
    except ImportError:
        raise ImportError(
            "Please install the 'dbt-osmosis[openai]' extra to use this feature.",
        ) from None
    if node is None:
        logger.info(":wave: Synthesizing missing documentation across all matched nodes.")
        for _ in context.pool.map(
            partial(synthesize_missing_documentation_with_openai, context),
            (n for _, n in _iter_candidate_nodes(context, exclude_external_patches=True)),
        ):
            pass
        return

    # since we are topologically sorted, we continually pass down synthesized knowledge leveraging our inheritance system
    # which minimizes synthesis requests -- in some cases by an order of magnitude while increasing accuracy
    _ = inherit_upstream_column_knowledge(context, node)
    total = len(node.columns)
    if total == 0:
        logger.info(
            ":no_entry_sign: No columns to synthesize documentation for => %s",
            node.unique_id,
        )
        return

    documented = len([
        column
        for column in node.columns.values()
        if column.description and column.description not in context.placeholders
    ])

    # Collect upstream documentation
    upstream_docs = _collect_upstream_documents(node, context)

    # Choose synthesis strategy based on number of missing columns
    if total - documented > 10:  # Use bulk synthesis for many missing columns
        _synthesize_bulk_documentation(node, upstream_docs, context)
    else:  # Use individual synthesis for few missing columns
        _synthesize_node_documentation(node, upstream_docs, context)
        _synthesize_individual_column_documentation(node, upstream_docs, context)


@_transform_op("Apply Semantic Analysis")
def apply_semantic_analysis(
    context: YamlRefactorContextProtocol, node: ResultNode | None = None
) -> None:
    """Apply AI semantic analysis to infer business meaning and relationships for columns.

    Uses LLM to analyze column names, data types, and context to:
    - Infer semantic types (primary_key, foreign_key, metric, dimension, etc.)
    - Detect relationships between columns (e.g., foreign keys)
    - Generate contextual descriptions based on semantic understanding
    - Suggest tags and metadata based on business meaning

    This transform enhances documentation by providing deeper business context
    beyond what traditional inheritance can provide.

    Args:
        context: The YAML refactor context
        node: The node to analyze. If None, analyzes all matched nodes.
    """
    from dbt_osmosis.core.node_filters import _iter_candidate_nodes

    if node is None:
        logger.info(":wave: Applying semantic analysis across all matched nodes.")
        for _ in context.pool.map(
            partial(apply_semantic_analysis, context),
            (n for _, n in _iter_candidate_nodes(context, exclude_external_patches=True)),
        ):
            pass
        return

    logger.info(":robot: Analyzing semantics for => %s", node.unique_id)

    # Check if LLM is configured
    llm_functions = _semantic_llm_functions()
    if llm_functions is None:
        return
    analyze_column_semantics, generate_semantic_description = llm_functions

    upstream_columns = _semantic_upstream_columns(context, node)
    model_context = _semantic_model_context(node)

    # Apply semantic analysis to each column
    for column_name, column_info in node.columns.items():
        _apply_column_semantic_analysis(
            node,
            column_name,
            column_info,
            upstream_columns,
            model_context,
            analyze_column_semantics,
            generate_semantic_description,
        )


def _semantic_llm_functions() -> tuple[t.Callable[..., t.Any], t.Callable[..., str]] | None:
    try:
        from dbt_osmosis.core.llm import analyze_column_semantics, generate_semantic_description

        # Verify LLM client can be created (will raise if not configured)
        _ = analyze_column_semantics.__globals__["get_llm_client"]()
        return analyze_column_semantics, generate_semantic_description
    except Exception as e:  # noqa: BLE001
        logger.warning(
            ":warning: LLM not configured or accessible. Skipping semantic analysis: %s",
            e,
        )
        return None


def _semantic_upstream_columns(
    context: YamlRefactorContextProtocol, node: ResultNode
) -> list[dict[str, str]]:
    from dbt_osmosis.core.inheritance import _build_column_knowledge_graph

    column_knowledge_graph = _build_column_knowledge_graph(context, node)
    return [
        {"name": name, "description": meta["description"]}
        for name, meta in column_knowledge_graph.items()
        if "description" in meta
    ]


def _semantic_model_context(node: ResultNode) -> str:
    model_context = node.description or ""
    raw_sql = getattr(node, "raw_sql", None)
    if isinstance(raw_sql, str) and raw_sql:
        # Include a snippet of SQL for context
        return f"{model_context}\n\nSQL: {raw_sql[:500]}..."
    return model_context


def _apply_column_semantic_analysis(
    node: ResultNode,
    column_name: str,
    column_info: ColumnInfo,
    upstream_columns: list[dict[str, str]],
    model_context: str,
    analyze_column_semantics: t.Callable[..., dict[str, t.Any]],
    generate_semantic_description: t.Callable[..., str],
) -> None:
    if _has_comprehensive_description(column_info):
        logger.debug(
            ":page_with_curl: Skipping semantic analysis for column => %s (already documented)",
            column_name,
        )
        return

    try:
        logger.info(":mag: Analyzing semantics for column => %s", column_name)
        semantic_result = analyze_column_semantics(
            column_name=column_name,
            data_type=column_info.data_type,
            table_name=node.name,
            model_context=model_context,
            upstream_columns=upstream_columns[:20],  # Limit for context
            temperature=0.3,
        )
        updated_column = _semantic_updated_column(
            node,
            column_name,
            column_info,
            semantic_result,
            generate_semantic_description,
        )
        node.columns[column_name] = updated_column
        logger.info(
            ":sparkles: Applied semantic analysis to column => %s: %s",
            column_name,
            semantic_result.get("semantic_type", "unknown"),
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(
            ":warning: Failed to analyze semantics for column %s: %s",
            column_name,
            e,
        )


def _has_comprehensive_description(column_info: ColumnInfo) -> bool:
    return bool(column_info.description and len(column_info.description) > 50)


def _semantic_updated_column(
    node: ResultNode,
    column_name: str,
    column_info: ColumnInfo,
    semantic_result: dict[str, t.Any],
    generate_semantic_description: t.Callable[..., str],
) -> ColumnInfo:
    new_description = generate_semantic_description(
        column_name=column_name,
        semantic_analysis=semantic_result,
        table_name=node.name,
        upstream_description=column_info.description,
        temperature=0.5,
    )
    updated_column = column_info.replace(description=new_description)
    updated_column = _apply_semantic_tags(column_name, updated_column, semantic_result)
    return _apply_semantic_meta(column_name, updated_column, semantic_result)


def _apply_semantic_tags(
    column_name: str, updated_column: ColumnInfo, semantic_result: dict[str, t.Any]
) -> ColumnInfo:
    if not semantic_result.get("tags"):
        return updated_column

    existing_tags = list(updated_column.tags) if updated_column.tags else []
    new_tags = t.cast(t.Iterable[str], semantic_result["tags"])
    merged_tags = _order_preserving_union(existing_tags, new_tags)
    if merged_tags == existing_tags:
        return updated_column

    logger.debug(":label: Added tags to column %s: %s", column_name, new_tags)
    return updated_column.replace(tags=merged_tags)


def _apply_semantic_meta(
    column_name: str, updated_column: ColumnInfo, semantic_result: dict[str, t.Any]
) -> ColumnInfo:
    if not semantic_result.get("meta"):
        return updated_column

    existing_meta = dict(updated_column.meta) if updated_column.meta else {}
    merged_meta = {**semantic_result["meta"], **existing_meta}
    if merged_meta == existing_meta:
        return updated_column

    logger.debug(":wrench: Added meta to column %s: %s", column_name, semantic_result["meta"])
    return updated_column.replace(meta=merged_meta)


@_transform_op("Suggest Improved Documentation")
def suggest_improved_documentation(
    context: YamlRefactorContextProtocol,
    node: ResultNode | None = None,
    threshold: float = 0.7,
    learning_mode: bool = True,
) -> None:
    """Suggest improved documentation using AI co-pilot with voice learning.

    This transform analyzes the project's documentation style and suggests
    improvements for model and column descriptions. It learns from existing
    documentation to match the team's voice and terminology.

    Args:
        context: The YamlRefactorContext instance
        node: The dbt node to suggest improvements for (None = all nodes)
        threshold: Confidence threshold for applying suggestions (0.0-1.0)
        learning_mode: Whether to analyze project style for voice learning

    Behavior:
        - For models with no documentation: generates new descriptions
        - For models with poor documentation: suggests improvements
        - Uses project style analysis to match team's voice
        - Only applies suggestions above confidence threshold
    """
    from dbt_osmosis.core.introspection import resolve_setting
    from dbt_osmosis.core.node_filters import _iter_candidate_nodes
    from dbt_osmosis.core.voice_learning import (
        analyze_project_documentation_style,
        extract_style_examples,
    )

    try:
        import importlib.util

        importlib.util.find_spec("dbt_osmosis.core.llm")
    except ImportError:
        raise ImportError(
            "Please install the 'dbt-osmosis[openai]' extra to use this feature."
        ) from None

    if node is None:
        logger.info(":wave: Suggesting improved documentation across all matched nodes.")
        operation = suggest_improved_documentation
        for _ in context.pool.map(
            partial(
                operation.func,
                context,
                threshold=threshold,
                learning_mode=learning_mode,
            ),
            (n for _, n in _iter_candidate_nodes(context, exclude_external_patches=True)),
        ):
            pass
        return

    # Check if AI co-pilot is disabled for this node
    if resolve_setting(context, "skip-ai-suggestions", node, fallback=False):
        logger.debug(":no_entry_sign: Skipping AI suggestions (skip_ai_suggestions=True).")
        return

    logger.info(":robot: Generating AI documentation suggestions for => %s", node.unique_id)

    style_profile, style_examples = _documentation_style_context(
        context,
        node,
        learning_mode,
        analyze_project_documentation_style,
        extract_style_examples,
    )
    upstream_docs = _collect_upstream_documents(node, context)

    # Track statistics
    suggestions_made = 0
    suggestions_applied = 0

    # Suggest model description
    made, applied = _suggest_model_documentation(
        node,
        context,
        threshold,
        upstream_docs,
        style_profile,
        style_examples,
    )
    suggestions_made += made
    suggestions_applied += applied

    # Suggest column descriptions
    for column_name, column in node.columns.items():
        made, applied = _suggest_column_documentation(
            node,
            column_name,
            column,
            context,
            threshold,
            upstream_docs,
            style_profile,
            style_examples,
        )
        suggestions_made += made
        suggestions_applied += applied

    logger.info(
        ":bar_chart: Generated %d suggestions, applied %d for node => %s",
        suggestions_made,
        suggestions_applied,
        node.unique_id,
    )


def _documentation_style_context(
    context: YamlRefactorContextProtocol,
    node: ResultNode,
    learning_mode: bool,
    analyze_project_documentation_style: t.Callable[..., t.Any],
    extract_style_examples: t.Callable[..., dict[str, list[str]]],
) -> tuple[t.Any | None, list[str] | None]:
    if learning_mode:
        logger.debug(":books: Analyzing project documentation style...")
        style_profile = analyze_project_documentation_style(
            context,
            max_nodes=50,
            max_columns_per_node=10,
        )
        logger.debug(
            ":mag: Found %d model examples, %d column examples",
            len(style_profile.model_description_samples),
            len(style_profile.column_description_samples),
        )
        return style_profile, None

    examples = extract_style_examples(context, node, max_examples=3)
    style_examples: list[str] = []
    style_examples.extend(examples.get("model_descriptions", []))
    style_examples.extend(examples.get("column_descriptions", []))
    return None, style_examples


def _suggest_model_documentation(
    node: ResultNode,
    context: YamlRefactorContextProtocol,
    threshold: float,
    upstream_docs: list[str],
    style_profile: t.Any | None,
    style_examples: list[str] | None,
) -> tuple[int, int]:
    needs_model_doc = not node.description or node.description in context.placeholders
    has_poor_model_doc = bool(node.description and len(node.description.split()) < 5)
    if not needs_model_doc and not has_poor_model_doc:
        return 0, 0

    from dbt_osmosis.core.llm import suggest_documentation_improvements

    suggestion = suggest_documentation_improvements(
        target="table",
        current_description=node.description if not needs_model_doc else None,
        table_name=node.relation_name or node.name,
        sql_content=getattr(node, "compiled_sql", f"SELECT * FROM {node.name}"),
        upstream_docs=upstream_docs,
        style_profile=style_profile,
        style_examples=style_examples,
        temperature=0.5,
    )
    if suggestion.confidence < threshold:
        logger.debug(
            ":heavy_check_mark: Model suggestion below threshold (confidence: %.2f): %s",
            suggestion.confidence,
            suggestion.reason,
        )
        return 1, 0

    node.description = suggestion.text
    logger.info(
        ":sparkles: Applied model description suggestion (confidence: %.2f): %s",
        suggestion.confidence,
        suggestion.reason,
    )
    return 1, 1


def _suggest_column_documentation(
    node: ResultNode,
    column_name: str,
    column: ColumnInfo,
    context: YamlRefactorContextProtocol,
    threshold: float,
    upstream_docs: list[str],
    style_profile: t.Any | None,
    style_examples: list[str] | None,
) -> tuple[int, int]:
    needs_col_doc = not column.description or column.description in context.placeholders
    has_poor_col_doc = bool(column.description and len(column.description.split()) < 3)
    if not needs_col_doc and not has_poor_col_doc:
        return 0, 0

    from dbt_osmosis.core.llm import suggest_documentation_improvements

    suggestion = suggest_documentation_improvements(
        target="column",
        current_description=column.description if not needs_col_doc else None,
        column_name=column_name,
        table_name=node.relation_name or node.name,
        existing_context=f"DataType={column.data_type or 'unknown'}",
        upstream_docs=upstream_docs,
        style_profile=style_profile,
        style_examples=style_examples,
        temperature=0.5,
    )
    if suggestion.confidence < threshold:
        logger.debug(
            ":heavy_check_mark: Column '%s' suggestion below threshold (confidence: %.2f)",
            column_name,
            suggestion.confidence,
        )
        return 1, 0

    column.description = suggestion.text
    logger.info(
        ":sparkles: Applied column suggestion for '%s' (confidence: %.2f)",
        column_name,
        suggestion.confidence,
    )
    return 1, 1
