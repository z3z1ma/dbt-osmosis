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

from dbt_osmosis.core import logger

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
        raise NotImplementedError


def _catalog_artifact_factory() -> _CatalogArtifactFactoryProtocol:
    """Return a typed view over dbt's versioned catalog artifact factory."""
    return t.cast("_CatalogArtifactFactoryProtocol", t.cast("object", CatalogArtifact))


def _as_catalog_results(artifact: _CatalogArtifactProtocol) -> CatalogResults:
    """Normalize a versioned catalog artifact to the concrete CatalogResults alias."""
    return t.cast("CatalogResults", t.cast("object", artifact))


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
    return _as_catalog_results(_catalog_artifact_factory().from_dict(json.loads(fp.read_text())))


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
            t.cast("t.Any", node)  # pyright: ignore[reportInvalidCast]
            for node in context.manifest.nodes.values()
            if node.is_relational and not node.is_ephemeral_model
        ],
        [t.cast("t.Any", node) for node in context.manifest.sources.values()],  # pyright: ignore[reportInvalidCast]
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
