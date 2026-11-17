# pyright: reportPrivateImportUsage=false
from __future__ import annotations

import json
import typing as t
from datetime import datetime, timezone
from itertools import chain
from pathlib import Path

from dbt.artifacts.schemas.catalog import (
    CatalogArtifact,
    CatalogResults,
)
from dbt.task.docs.generate import Catalog
from mashumaro.exceptions import InvalidFieldValue, MissingField

from dbt_osmosis.core import logger
from dbt_osmosis.core.exceptions import CatalogValidationError

__all__ = ["_generate_catalog", "_load_catalog"]


class _CatalogArtifactProtocol(t.Protocol):
    nodes: t.Mapping[str, object]
    sources: t.Mapping[str, object]

    def write(self, path: str) -> None:
        raise NotImplementedError


class _CatalogArtifactFactoryProtocol(t.Protocol):
    @staticmethod
    def from_dict(data: object) -> _CatalogArtifactProtocol:
        raise NotImplementedError

    @staticmethod
    def from_results(
        *,
        nodes: object,
        sources: object,
        generated_at: datetime,
        compile_results: object,
        errors: list[str] | None,
    ) -> _CatalogArtifactProtocol:
        del generated_at, compile_results
        raise NotImplementedError


def _catalog_artifact_factory() -> _CatalogArtifactFactoryProtocol:
    """Return a typed view over dbt's versioned catalog artifact factory."""
    return t.cast("_CatalogArtifactFactoryProtocol", t.cast("object", CatalogArtifact))


def _as_catalog_results(artifact: _CatalogArtifactProtocol) -> CatalogResults:
    """Normalize a versioned catalog artifact to the concrete CatalogResults alias."""
    return t.cast("CatalogResults", t.cast("object", artifact))


_CATALOG_SCHEMA = "https://schemas.getdbt.com/dbt/catalog/v1.json"


def _validate_catalog_input(data: object) -> None:
    """Check the catalog envelope and fields used by column introspection."""
    if not isinstance(data, dict):
        raise TypeError("expected a JSON object")
    metadata = data.get("metadata")
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be an object")
    if "dbt_schema_version" in metadata and metadata["dbt_schema_version"] != _CATALOG_SCHEMA:
        raise ValueError(f"expected catalog v1 schema {_CATALOG_SCHEMA!r}")
    if any(key in data for key in ("macros", "parent_map", "child_map")):
        raise ValueError("received manifest fields; supply catalog.json, not manifest.json")
    for section in ("nodes", "sources"):
        entries = data.get(section)
        if not isinstance(entries, dict):
            raise TypeError(f"{section} must be an object")
        for unique_id, table in entries.items():
            location = f"{section}.{unique_id}"
            if not isinstance(table, dict):
                raise TypeError(f"{location} must be an object")
            table_metadata = table.get("metadata")
            if not isinstance(table_metadata, dict):
                raise TypeError(f"{location}.metadata must be an object")
            for key in ("type", "schema", "name"):
                if not isinstance(table_metadata.get(key), str):
                    raise TypeError(f"{location}.metadata.{key} must be a string")
            columns = table.get("columns")
            if not isinstance(columns, dict):
                raise TypeError(f"{location}.columns must be an object")
            for column_key, column in columns.items():
                column_location = f"{location}.columns.{column_key}"
                if not isinstance(column, dict):
                    raise TypeError(f"{column_location} must be an object")
                for key in ("name", "type"):
                    if not isinstance(column.get(key), str):
                        raise TypeError(f"{column_location}.{key} must be a string")
                if type(column.get("index")) is not int:
                    raise TypeError(f"{column_location}.index must be an integer")
                if column.get("comment") is not None and not isinstance(column["comment"], str):
                    raise TypeError(f"{column_location}.comment must be a string or null")


def _load_catalog(settings: t.Any) -> CatalogResults | None:
    """Load a compatible catalog, retaining the existing missing-path fallback."""
    logger.debug(":mag: Attempting to load catalog from => %s", settings.catalog_path)
    if not settings.catalog_path:
        return None
    fp = Path(settings.catalog_path)
    if not fp.exists():
        logger.warning(":warning: Catalog path => %s does not exist.", fp)
        return None
    logger.info(":books: Loading existing catalog => %s", fp)
    try:
        data = json.loads(fp.read_text(encoding="utf-8"))
        _validate_catalog_input(data)
        return _as_catalog_results(_catalog_artifact_factory().from_dict(data))
    except (OSError, UnicodeError, ValueError, TypeError, InvalidFieldValue, MissingField) as exc:
        raise CatalogValidationError(
            f"Cannot load catalog {str(fp)!r}: {exc}. "
            "Supply a valid dbt catalog v1 JSON artifact (not manifest.json)."
        ) from exc


# NOTE: this is mostly adapted from dbt-core with some cruft removed, strict pyright is not a fan of dbt's shenanigans
def _generate_catalog(context: t.Any) -> CatalogResults | None:
    """Generate dbt catalog file for the project."""
    import dbt.utils as dbt_utils  # pyright: ignore[reportPrivateImportUsage]

    if context.config.disable_introspection:
        logger.warning(":warning: Introspection is disabled, cannot generate catalog.")
        return None
    logger.info(
        ":books: Generating a new catalog for the project => %s",
        context.runtime_cfg.project_name,
    )
    catalogable_nodes = chain(
        [
            node
            for node in context.manifest.nodes.values()
            if node.is_relational and not node.is_ephemeral_model
        ],
        context.manifest.sources.values(),
    )
    table, exceptions = context.adapter.get_filtered_catalog(
        catalogable_nodes,
        context.manifest.get_used_schemas(),  # pyright: ignore[reportArgumentType]
    )

    logger.debug(":mag_right: Building catalog from returned table => %s", table)
    catalog = Catalog(
        [dict(zip(table.column_names, map(dbt_utils._coerce_decimal, row))) for row in table],  # pyright: ignore[reportUnknownArgumentType,reportPrivateUsage]
    )

    errors: list[str] | None = None
    if exceptions:
        errors = [str(e) for e in exceptions]
        logger.warning(":warning: Exceptions encountered in get_filtered_catalog => %s", errors)

    nodes, sources = catalog.make_unique_id_map(context.manifest)
    artifact = _catalog_artifact_factory().from_results(
        nodes=nodes,
        sources=sources,
        generated_at=datetime.now(timezone.utc),
        compile_results=None,
        errors=errors,
    )
    artifact_path = Path(context.runtime_cfg.project_target_path, "catalog.json")
    logger.info(":bookmark_tabs: Writing fresh catalog => %s", artifact_path)
    artifact.write(str(artifact_path.resolve()))  # Cache it, same as dbt
    return _as_catalog_results(artifact)
