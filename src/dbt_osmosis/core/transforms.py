import atexit
import time
import typing as t
from collections import ChainMap
from dataclasses import dataclass, field
from functools import partial
from types import MappingProxyType

from dbt.contracts.graph.nodes import ResultNode, ColumnInfo
from dbt.artifacts.resources.types import NodeType

import dbt_osmosis.core.logger as logger

__all__ = [
    "TransformOperation",
    "TransformPipeline",
    "_transform_op",
    "inherit_upstream_column_knowledge",
    "inject_missing_columns",
    "remove_columns_not_in_database",
    "sort_columns_as_in_database",
    "sort_columns_alphabetically",
    "sort_columns_as_configured",
    "synchronize_data_types",
    "synthesize_missing_documentation_with_openai",
]


@dataclass
class TransformOperation:
    """An operation to be run on a dbt manifest node."""

    func: t.Callable[..., t.Any]
    name: str

    _result: t.Optional[t.Any] = field(init=False, default=None)
    _context: t.Optional[t.Any] = field(init=False, default=None)  # YamlRefactorContext
    _node: t.Union[ResultNode, None] = field(init=False, default=None)
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
        context: t.Any,
        node: t.Optional[ResultNode] = None,  # YamlRefactorContext
    ) -> "TransformOperation":
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

    def __rshift__(self, next_op: "TransformOperation") -> "TransformPipeline":
        """Chain operations together."""
        return TransformPipeline([self]) >> next_op

    def __repr__(self) -> str:  # pyright: ignore[reportImplicitOverride]
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

    def __rshift__(
        self, next_op: t.Union["TransformOperation", t.Callable[..., t.Any]]
    ) -> "TransformPipeline":
        """Chain operations together."""
        if isinstance(next_op, TransformOperation):
            self.operations.append(next_op)
        elif callable(next_op):
            self.operations.append(TransformOperation(next_op, next_op.__name__))
        else:
            raise ValueError(f"Cannot chain non-callable: {next_op}")  # pyright: ignore[reportUnreachable]
        return self

    def __call__(
        self,
        context: t.Any,
        node: t.Optional[ResultNode] = None,  # YamlRefactorContext
    ) -> "TransformPipeline":
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
            logger.info(":hourglass: Committing all changes to YAML files in batch.")
            _commit_start = time.time()
            from dbt_osmosis.core.sync_operations import sync_node_to_yaml

            sync_node_to_yaml(context, node, commit=True)
            _commit_end = time.time()
            logger.info(
                ":checkered_flag: YAML commits completed in => %.2fs", _commit_end - _commit_start
            )

        if self.commit_mode == "batch":
            _commit()
        elif self.commit_mode == "defer":
            _ = atexit.register(_commit)

        return self

    def __repr__(self) -> str:  # pyright: ignore[reportImplicitOverride]
        steps = [op.name for op in self.operations]
        return f"<OperationPipeline: {len(self.operations)} operations, steps={steps!r}>"


def _transform_op(
    name: t.Optional[str] = None,
) -> t.Callable[[t.Callable[[t.Any, t.Optional[ResultNode]], None]], TransformOperation]:
    """Decorator to create a TransformOperation from a function."""

    def decorator(
        func: t.Callable[[t.Any, t.Union[ResultNode, None]], None],  # YamlRefactorContext
    ) -> TransformOperation:
        return TransformOperation(func, name=name or func.__name__)

    return decorator


@_transform_op("Inherit Upstream Column Knowledge")
def inherit_upstream_column_knowledge(
    context: t.Any,
    node: t.Union[ResultNode, None] = None,  # YamlRefactorContext
) -> None:
    """Inherit column level knowledge from the ancestors of a dbt model or source node."""
    if node is None:
        logger.info(":wave: Inheriting column knowledge across all matched nodes.")
        from dbt_osmosis.core.node_filters import _iter_candidate_nodes

        for _ in context.pool.map(
            partial(inherit_upstream_column_knowledge, context),
            (n for _, n in _iter_candidate_nodes(context, include_external=True)),
        ):
            ...
        return

    logger.info(":dna: Inheriting column knowledge for => %s", node.unique_id)

    from dbt_osmosis.core.inheritance import _build_column_knowledge_graph
    from dbt_osmosis.core.introspection import _get_setting_for_node

    column_knowledge_graph = _build_column_knowledge_graph(context, node)
    kwargs = None
    for name, node_column in node.columns.items():
        kwargs = column_knowledge_graph.get(name)
        if kwargs is None:
            continue
        inheritable = ["description"]
        if not _get_setting_for_node(
            "skip-add-tags", node, name, fallback=context.settings.skip_add_tags
        ):
            inheritable.append("tags")
        if not _get_setting_for_node(
            "skip-merge-meta", node, name, fallback=context.settings.skip_merge_meta
        ):
            inheritable.append("meta")
        for extra in _get_setting_for_node(
            "add-inheritance-for-specified-keys",
            node,
            name,
            fallback=context.settings.add_inheritance_for_specified_keys,
        ):
            if extra not in inheritable:
                inheritable.append(extra)

        updated_metadata = {k: v for k, v in kwargs.items() if v is not None and k in inheritable}
        logger.debug(
            ":star2: Inheriting updated metadata => %s for column => %s", updated_metadata, name
        )
        node.columns[name] = node_column.replace(**updated_metadata)


@_transform_op("Inject Missing Columns")
def inject_missing_columns(context: t.Any, node: t.Union[ResultNode, None] = None) -> None:
    """Add missing columns to a dbt node and it's corresponding yaml section. Changes are implicitly buffered until commit_yamls is called."""
    from dbt_osmosis.core.introspection import _get_setting_for_node, get_columns
    from dbt_osmosis.core.node_filters import _iter_candidate_nodes

    if _get_setting_for_node("skip-add-columns", node, fallback=context.settings.skip_add_columns):
        logger.debug(":no_entry_sign: Skipping column injection (skip_add_columns=True).")
        return
    if node is None:
        logger.info(":wave: Injecting missing columns for all matched nodes.")
        for _ in context.pool.map(
            partial(inject_missing_columns, context),
            (n for _, n in _iter_candidate_nodes(context)),
        ):
            ...
        return
    if (
        _get_setting_for_node(
            "skip-add-source-columns", node, fallback=context.settings.skip_add_source_columns
        )
        and node.resource_type == NodeType.Source
    ):
        logger.debug(":no_entry_sign: Skipping column injection (skip_add_source_columns=True).")
        return

    from dbt_osmosis.core.introspection import normalize_column_name

    current_columns = {
        normalize_column_name(c.name, context.project.runtime_cfg.credentials.type)
        for c in node.columns.values()
    }
    incoming_columns = get_columns(context, node)
    for incoming_name, incoming_meta in incoming_columns.items():
        if incoming_name not in current_columns:
            logger.info(
                ":heavy_plus_sign: Reconciling missing column => %s in node => %s",
                incoming_name,
                node.unique_id,
            )
            gen_col = {"name": incoming_name, "description": incoming_meta.comment or ""}
            if (dtype := incoming_meta.type) and not _get_setting_for_node(
                "skip-add-data-types", node, fallback=context.settings.skip_add_data_types
            ):
                gen_col["data_type"] = dtype.lower() if context.settings.output_to_lower else dtype
            node.columns[incoming_name] = ColumnInfo.from_dict(gen_col)


@_transform_op("Remove Extra Columns")
def remove_columns_not_in_database(context: t.Any, node: t.Union[ResultNode, None] = None) -> None:
    """Remove columns from a dbt node and it's corresponding yaml section that are not present in the database. Changes are implicitly buffered until commit_yamls is called."""
    from dbt_osmosis.core.introspection import normalize_column_name, get_columns
    from dbt_osmosis.core.node_filters import _iter_candidate_nodes

    if node is None:
        logger.info(":wave: Removing columns not in DB across all matched nodes.")
        for _ in context.pool.map(
            partial(remove_columns_not_in_database, context),
            (n for _, n in _iter_candidate_nodes(context)),
        ):
            ...
        return
    current_columns = {
        normalize_column_name(c.name, context.project.runtime_cfg.credentials.type): key
        for key, c in node.columns.items()
    }
    incoming_columns = get_columns(context, node)
    if not incoming_columns:
        logger.info(
            ":no_entry_sign: No columns discovered for node => %s, skipping cleanup.",
            node.unique_id,
        )
        return
    extra_columns = set(current_columns.keys()) - set(incoming_columns.keys())
    for extra_column in extra_columns:
        logger.info(
            ":heavy_minus_sign: Removing extra column => %s in node => %s",
            extra_column,
            node.unique_id,
        )
        _ = node.columns.pop(current_columns[extra_column], None)


@_transform_op("Sort Columns in DB Order")
def sort_columns_as_in_database(context: t.Any, node: t.Union[ResultNode, None] = None) -> None:
    """Sort columns in a dbt node and it's corresponding yaml section as they appear in the database. Changes are implicitly buffered until commit_yamls is called."""
    from dbt_osmosis.core.introspection import normalize_column_name, get_columns
    from dbt_osmosis.core.node_filters import _iter_candidate_nodes

    if node is None:
        logger.info(":wave: Sorting columns as they appear in DB across all matched nodes.")
        for _ in context.pool.map(
            partial(sort_columns_as_in_database, context),
            (n for _, n in _iter_candidate_nodes(context)),
        ):
            ...
        return
    logger.info(":1234: Sorting columns by warehouse order => %s", node.unique_id)
    incoming_columns = get_columns(context, node)
    if not incoming_columns:
        logger.info(
            ":no_entry_sign: No columns discovered for node => %s, skipping db order sorting.",
            node.unique_id,
        )
        return

    def _position(column: str) -> int:
        inc = incoming_columns.get(
            normalize_column_name(column, context.project.runtime_cfg.credentials.type)
        )
        if inc is None or inc.index is None:  # pyright: ignore[reportUnnecessaryComparison]
            return 99_999
        return inc.index

    node.columns = {k: v for k, v in sorted(node.columns.items(), key=lambda i: _position(i[0]))}


@_transform_op("Sort Columns Alphabetically")
def sort_columns_alphabetically(context: t.Any, node: t.Union[ResultNode, None] = None) -> None:
    """Sort columns in a dbt node and it's corresponding yaml section alphabetically. Changes are implicitly buffered until commit_yamls is called."""
    from dbt_osmosis.core.node_filters import _iter_candidate_nodes

    if node is None:
        logger.info(":wave: Sorting columns alphabetically across all matched nodes.")
        for _ in context.pool.map(
            partial(sort_columns_alphabetically, context),
            (n for _, n in _iter_candidate_nodes(context)),
        ):
            ...
        return
    logger.info(":abcd: Sorting columns alphabetically => %s", node.unique_id)
    node.columns = {k: v for k, v in sorted(node.columns.items(), key=lambda i: i[0])}


@_transform_op("Sort Columns")
def sort_columns_as_configured(context: t.Any, node: t.Union[ResultNode, None] = None) -> None:
    from dbt_osmosis.core.introspection import _get_setting_for_node
    from dbt_osmosis.core.node_filters import _iter_candidate_nodes

    if node is None:
        logger.info(":wave: Sorting columns as configured across all matched nodes.")
        for _ in context.pool.map(
            partial(sort_columns_as_configured, context),
            (n for _, n in _iter_candidate_nodes(context)),
        ):
            ...
        return
    sort_by = _get_setting_for_node("sort-by", node, fallback="database")
    if sort_by == "database":
        _ = sort_columns_as_in_database(context, node)
    elif sort_by == "alphabetical":
        _ = sort_columns_alphabetically(context, node)
    else:
        raise ValueError(f"Invalid sort-by value: {sort_by} for node: {node.unique_id}")


@_transform_op("Synchronize Data Types")
def synchronize_data_types(context: t.Any, node: t.Union[ResultNode, None] = None) -> None:
    """Populate data types for columns in a dbt node and it's corresponding yaml section. Changes are implicitly buffered until commit_yamls is called."""
    from dbt_osmosis.core.introspection import (
        _get_setting_for_node,
        normalize_column_name,
        get_columns,
    )
    from dbt_osmosis.core.node_filters import _iter_candidate_nodes

    if node is None:
        logger.info(":wave: Populating data types across all matched nodes.")
        for _ in context.pool.map(
            partial(synchronize_data_types, context), (n for _, n in _iter_candidate_nodes(context))
        ):
            ...
        return
    logger.info(":1234: Synchronizing data types => %s", node.unique_id)
    incoming_columns = get_columns(context, node)
    if _get_setting_for_node("skip-add-data-types", node, fallback=False):
        return
    for name, column in node.columns.items():
        if _get_setting_for_node(
            "skip-add-data-types", node, name, fallback=context.settings.skip_add_data_types
        ):
            continue
        lowercase = _get_setting_for_node(
            "output-to-lower", node, name, fallback=context.settings.output_to_lower
        )
        if inc_c := incoming_columns.get(
            normalize_column_name(name, context.project.runtime_cfg.credentials.type)
        ):
            is_lower = column.data_type and column.data_type.islower()
            if inc_c.type:
                column.data_type = inc_c.type.lower() if lowercase or is_lower else inc_c.type


@_transform_op("Synthesize Missing Documentation")
def synthesize_missing_documentation_with_openai(
    context: t.Any, node: t.Union[ResultNode, None] = None
) -> None:
    """Synthesize missing documentation for a dbt node using OpenAI's GPT-4o API."""
    import textwrap
    from dbt_osmosis.core.node_filters import _iter_candidate_nodes

    try:
        from dbt_osmosis.core.llm import (
            generate_column_doc,
            generate_model_spec_as_json,
            generate_table_doc,
        )
    except ImportError:
        raise ImportError(
            "Please install the 'dbt-osmosis[openai]' extra to use this feature."
        ) from None
    if node is None:
        logger.info(":wave: Synthesizing missing documentation across all matched nodes.")
        for _ in context.pool.map(
            partial(synthesize_missing_documentation_with_openai, context),
            (n for _, n in _iter_candidate_nodes(context)),
        ):
            ...
        return
    # since we are topologically sorted, we continually pass down synthesized knowledge leveraging our inheritance system
    # which minimizes synthesis requests -- in some cases by an order of magnitude while increasing accuracy
    _ = inherit_upstream_column_knowledge(context, node)
    total = len(node.columns)
    if total == 0:
        logger.info(
            ":no_entry_sign: No columns to synthesize documentation for => %s", node.unique_id
        )
        return
    documented = len([
        column
        for column in node.columns.values()
        if column.description and column.description not in context.placeholders
    ])
    node_map = ChainMap(
        t.cast(dict[str, ResultNode], context.project.manifest.nodes),
        t.cast(dict[str, ResultNode], context.project.manifest.sources),
    )
    upstream_docs: list[str] = ["# The following is not exhaustive, but provides some context."]
    depends_on_nodes = t.cast(list[str], node.depends_on_nodes)
    for i, uid in enumerate(depends_on_nodes):
        dep = node_map.get(uid)
        if dep is not None:
            oneline_desc = dep.description.replace("\n", " ")
            upstream_docs.append(f"{uid}: # {oneline_desc}")
            for j, (name, meta) in enumerate(dep.columns.items()):
                if meta.description and meta.description not in context.placeholders:
                    upstream_docs.append(f"- {name}: |\n{textwrap.indent(meta.description, '  ')}")
                if j > 20:
                    # just a small amount of this supplementary context is sufficient
                    upstream_docs.append("- (omitting additional columns for brevity)")
                    break
        # ensure our context window is bounded, semi-arbitrary
        if len(upstream_docs) > 100 and i < len(depends_on_nodes) - 1:
            upstream_docs.append(f"# remaining nodes are: {', '.join(depends_on_nodes[i:])}")
            break
    if len(upstream_docs) == 1:
        upstream_docs[0] = "(no upstream documentation found)"
    if (
        total - documented
        > 10  # a semi-arbitrary limit by which its probably better to one shot the table versus many smaller requests
    ):
        logger.info(
            ":robot: Synthesizing bulk documentation for => %s columns in node => %s",
            total - documented,
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
    else:
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
