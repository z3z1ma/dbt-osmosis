from __future__ import annotations

import json
import re
import typing as t
from collections import OrderedDict
from datetime import datetime, timezone
from itertools import chain
from pathlib import Path

from dbt.adapters.base.column import Column as BaseColumn
from dbt.adapters.base.relation import BaseRelation
from dbt.contracts.graph.nodes import ResultNode
from dbt.contracts.results import CatalogArtifact, CatalogResults, ColumnMetadata
from dbt.task.docs.generate import Catalog

import dbt_osmosis.core.logger as logger

__all__ = [
    "_find_first",
    "normalize_column_name",
    "_maybe_use_precise_dtype",
    "get_columns",
    "_load_catalog",
    "_generate_catalog",
    "_COLUMN_LIST_CACHE",
]

T = t.TypeVar("T")

_COLUMN_LIST_CACHE: dict[str, OrderedDict[str, ColumnMetadata]] = {}
"""Cache for column lists to avoid redundant introspection."""


@t.overload
def _find_first(coll: t.Iterable[T], predicate: t.Callable[[T], bool], default: T) -> T: ...


@t.overload
def _find_first(
    coll: t.Iterable[T], predicate: t.Callable[[T], bool], default: None = ...
) -> t.Union[T, None]: ...


def _find_first(
    coll: t.Iterable[T], predicate: t.Callable[[T], bool], default: t.Union[T, None] = None
) -> t.Union[T, None]:
    """Find the first item in a container that satisfies a predicate."""
    for item in coll:
        if predicate(item):
            return item
    return default


def normalize_column_name(column: str, credentials_type: str) -> str:
    """Apply case normalization to a column name based on the credentials type."""
    if credentials_type == "snowflake" and column.startswith('"') and column.endswith('"'):
        logger.debug(":snowflake: Column name found with double-quotes => %s", column)
        pass
    elif credentials_type == "snowflake":
        return column.upper()
    return column.strip('"').strip("`").strip("[]")


def _maybe_use_precise_dtype(
    col: BaseColumn, settings: t.Any, node: t.Union[ResultNode, None] = None
) -> str:
    """Use the precise data type if enabled in the settings."""
    use_num_prec = _get_setting_for_node(
        "numeric-precision-and-scale", node, col.name, fallback=settings.numeric_precision_and_scale
    )
    use_chr_prec = _get_setting_for_node(
        "string-length", node, col.name, fallback=settings.string_length
    )
    if (col.is_numeric() and use_num_prec) or (col.is_string() and use_chr_prec):
        logger.debug(":ruler: Using precise data type => %s", col.data_type)
        return col.data_type
    if hasattr(col, "mode"):
        return col.data_type
    return col.dtype


def _get_setting_for_node(
    opt: str,
    /,
    node: t.Union[ResultNode, None] = None,
    col: t.Union[str, None] = None,
    *,
    fallback: t.Union[t.Any, None] = None,
) -> t.Any:
    """Get a configuration value for a dbt node from the node's meta and config.

    models: # dbt_project
      project:
        staging:
          +dbt-osmosis: path/spec.yml
          +dbt-osmosis-options:
            string-length: true
            numeric-precision-and-scale: true
            skip-add-columns: true
          +dbt-osmosis-skip-add-tags: true

    models: # schema
      - name: foo
        meta:
          string-length: false
          prefix: user_ # we strip this prefix to inherit from columns upstream, useful in staging models that prefix everything
        columns:
          - bar:
            meta:
              dbt-osmosis-skip-meta-merge: true # per-column options
              dbt-osmosis-options:
                output-to-lower: true

    {{ config(..., dbt_osmosis_options={"prefix": "account_"}) }} -- sql

    We check for
    From node column meta
    - <key>
    - dbt-osmosis-<key>
    - dbt-osmosis-options.<key>
    From node meta
    - <key>
    - dbt-osmosis-<key>
    - dbt-osmosis-options.<key>
    From node config
    - dbt-osmosis-<key>
    - dbt-osmosis-options.<key>
    - dbt_osmosis_<key> # allows use in {{ config(...) }} by being a valid python identifier
    - dbt_osmosis_options.<key> # allows use in {{ config(...) }} by being a valid python identifier
    """
    if node is None:
        return fallback
    k, identifier = opt.replace("_", "-"), opt.replace("-", "_")
    sources = [
        node.meta,
        node.meta.get("dbt-osmosis-options", {}),
        node.meta.get("dbt_osmosis_options", {}),
        node.config.extra,
        node.config.extra.get("dbt-osmosis-options", {}),
        node.config.extra.get("dbt_osmosis_options", {}),
    ]
    if col and (column := node.columns.get(col)):
        sources = [
            column.meta,
            column.meta.get("dbt-osmosis-options", {}),
            column.meta.get("dbt_osmosis_options", {}),
            *sources,
        ]
    for source in sources:
        for variation in (f"dbt-osmosis-{k}", f"dbt_osmosis_{identifier}"):
            if variation in source:
                return source[variation]
        if source is not node.config.extra:
            if k in source:
                return source[k]
            if identifier in source:
                return source[identifier]
    return fallback


def get_columns(
    context: t.Any, relation: t.Union[BaseRelation, ResultNode, None]
) -> dict[str, ColumnMetadata]:
    """Equivalent to get_columns_meta in old code but directly referencing a key, not a node."""
    normalized_columns: OrderedDict[str, ColumnMetadata] = OrderedDict()

    if relation is None:
        logger.debug(":blue_book: Relation is empty, skipping column collection.")
        return normalized_columns

    result_node: ResultNode | None = None
    if not isinstance(relation, BaseRelation):
        # NOTE: Technically, we should use `isinstance(relation, ResultNode)` to verify it's a ResultNode,
        #       but since ResultNode is defined as a Union[...], Python 3.9 raises
        #       > TypeError: Subscripted generics cannot be used with class and instance checks
        #       To avoid that, we're skipping the isinstance check.
        result_node = relation  # may be a ResultNode
        relation = context.project.adapter.Relation.create_from(
            context.project.adapter.config,  # pyright: ignore[reportUnknownArgumentType]
            relation,  # pyright: ignore[reportArgumentType]
        )

    rendered_relation = relation.render()
    if rendered_relation in _COLUMN_LIST_CACHE:
        logger.debug(":blue_book: Column list cache HIT => %s", rendered_relation)
        return _COLUMN_LIST_CACHE[rendered_relation]

    logger.info(":mag_right: Collecting columns for table => %s", rendered_relation)
    index = 0

    def process_column(c: t.Union[BaseColumn, ColumnMetadata], /) -> None:
        nonlocal index

        columns = [c]
        if hasattr(c, "flatten"):
            columns.extend(c.flatten())  # pyright: ignore[reportUnknownMemberType]

        for column in columns:
            if any(re.match(b, column.name) for b in context.ignore_patterns):
                logger.debug(
                    ":no_entry_sign: Skipping column => %s due to skip pattern match.", column.name
                )
                continue
            normalized = normalize_column_name(
                column.name, context.project.runtime_cfg.credentials.type
            )
            if not isinstance(column, ColumnMetadata):
                dtype = _maybe_use_precise_dtype(column, context.settings, result_node)
                column = ColumnMetadata(
                    name=normalized,
                    type=dtype,
                    index=index,
                    comment=getattr(column, "comment", None),
                )
            normalized_columns[normalized] = column
            index += 1

    if catalog := context.read_catalog():
        logger.debug(":blue_book: Catalog found => Checking for ref => %s", rendered_relation)
        catalog_entry = _find_first(
            chain(catalog.nodes.values(), catalog.sources.values()),
            lambda c: relation.matches(*c.key()),
        )
        if catalog_entry:
            logger.info(
                ":books: Found catalog entry for => %s. Using it to process columns.",
                rendered_relation,
            )
            for column in catalog_entry.columns.values():
                process_column(column)
            return normalized_columns

    if context.project.config.disable_introspection:
        logger.warning(
            ":warning: Introspection is disabled, cannot introspect columns and no catalog entry."
        )
        return normalized_columns

    try:
        logger.info(":mag: Introspecting columns in warehouse for => %s", rendered_relation)
        for column in t.cast(
            t.Iterable[BaseColumn], context.project.adapter.get_columns_in_relation(relation)
        ):
            process_column(column)
    except Exception as ex:
        logger.warning(":warning: Could not introspect columns for %s: %s", rendered_relation, ex)

    _COLUMN_LIST_CACHE[rendered_relation] = normalized_columns
    return normalized_columns


def _load_catalog(settings: t.Any) -> CatalogResults | None:
    """Load the catalog file if it exists and return a CatalogResults instance."""
    logger.debug(":mag: Attempting to load catalog from => %s", settings.catalog_path)
    if not settings.catalog_path:
        return None
    fp = Path(settings.catalog_path)
    if not fp.exists():
        logger.warning(":warning: Catalog path => %s does not exist.", fp)
        return None
    logger.info(":books: Loading existing catalog => %s", fp)
    return t.cast(CatalogResults, CatalogArtifact.from_dict(json.loads(fp.read_text())))


# NOTE: this is mostly adapted from dbt-core with some cruft removed, strict pyright is not a fan of dbt's shenanigans
def _generate_catalog(context: t.Any) -> CatalogResults | None:
    """Generate the dbt catalog file for the project."""
    import dbt.utils as dbt_utils

    if context.config.disable_introspection:
        logger.warning(":warning: Introspection is disabled, cannot generate catalog.")
        return None
    logger.info(
        ":books: Generating a new catalog for the project => %s", context.runtime_cfg.project_name
    )
    catalogable_nodes = chain(
        [
            t.cast(t.Any, node)  # pyright: ignore[reportInvalidCast]
            for node in context.manifest.nodes.values()
            if node.is_relational and not node.is_ephemeral_model
        ],
        [t.cast(t.Any, node) for node in context.manifest.sources.values()],  # pyright: ignore[reportInvalidCast]
    )
    table, exceptions = context.adapter.get_filtered_catalog(
        catalogable_nodes,
        context.manifest.get_used_schemas(),  # pyright: ignore[reportArgumentType]
    )

    logger.debug(":mag_right: Building catalog from returned table => %s", table)
    catalog = Catalog(
        [dict(zip(table.column_names, map(dbt_utils._coerce_decimal, row))) for row in table]  # pyright: ignore[reportUnknownArgumentType,reportPrivateUsage]
    )

    errors: list[str] | None = None
    if exceptions:
        errors = [str(e) for e in exceptions]
        logger.warning(":warning: Exceptions encountered in get_filtered_catalog => %s", errors)

    nodes, sources = catalog.make_unique_id_map(context.manifest)
    artifact = CatalogArtifact.from_results(
        nodes=nodes,
        sources=sources,
        generated_at=datetime.now(timezone.utc),
        compile_results=None,
        errors=errors,
    )
    artifact_path = Path(context.runtime_cfg.project_target_path, "catalog.json")
    logger.info(":bookmark_tabs: Writing fresh catalog => %s", artifact_path)
    artifact.write(str(artifact_path.resolve()))  # Cache it, same as dbt
    return t.cast(CatalogResults, artifact)
