from __future__ import annotations

import os
import threading
import typing as t
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import ruamel.yaml
from dbt.artifacts.schemas.catalog import CatalogResults

from dbt_osmosis.core import logger

if t.TYPE_CHECKING:
    from dbt_osmosis.core.config import DbtProjectContext


def _create_yaml_instance() -> ruamel.yaml.YAML:
    from dbt_osmosis.core.schema.parser import create_yaml_instance

    return create_yaml_instance()


__all__ = [
    "EMPTY_STRING",
    "YamlRefactorContext",
    "YamlRefactorSettings",
]

EMPTY_STRING = ""
"""A null string constant for use in placeholder lists, this is always considered undocumented"""


@dataclass
class YamlRefactorSettings:
    """Settings for yaml based refactoring operations."""

    fqn: list[str] = field(default_factory=list)
    """Filter models to action via a fully qualified name match such as returned by `dbt ls`."""
    models: list[Path | str] = field(default_factory=list)
    """Filter models to action via a file path match."""
    dry_run: bool = False
    """Do not write changes to disk."""
    skip_merge_meta: bool = False
    """Skip merging upstream meta fields in the yaml files."""
    skip_add_columns: bool = False
    """Skip adding missing columns in the yaml files."""
    skip_add_tags: bool = False
    """Skip appending upstream tags in the yaml files."""
    skip_add_data_types: bool = False
    """Skip adding data types in the yaml files."""
    skip_add_source_columns: bool = False
    """Skip adding columns in the source yaml files specifically."""
    add_progenitor_to_meta: bool = False
    """Add a custom progenitor field to the meta section indicating a column's origin."""
    numeric_precision_and_scale: bool = False
    """Include numeric precision in the data type."""
    string_length: bool = False
    """Include character length in the data type."""
    force_inherit_descriptions: bool = False
    """Force inheritance of descriptions from upstream models, even if node has a valid description."""
    use_unrendered_descriptions: bool = False
    """Use unrendered descriptions preserving things like {{ doc(...) }} which are otherwise pre-rendered in the manifest object"""
    prefer_yaml_values: bool = False
    """Prefer YAML values as-is for all fields, preserving unrendered jinja templates like {{ var(...) }} and {{ env_var(...) }}"""
    add_inheritance_for_specified_keys: list[str] = field(default_factory=list)
    """Include additional keys in the inheritance process."""
    output_to_lower: bool = False
    """Force column name and data type output to lowercase in the yaml files."""
    output_to_upper: bool = False
    """Force column name and data type output to uppercase in the yaml files."""
    catalog_path: str | None = None
    """Path to the dbt catalog.json file to use preferentially instead of live warehouse introspection"""
    create_catalog_if_not_exists: bool = False
    """Generate the catalog.json for the project if it doesn't exist and use it for introspective queries."""
    scaffold_empty_configs: bool = False
    """When True, include empty/placeholder fields (e.g., empty descriptions) in YAML. When False, skip writing them."""
    strip_eof_blank_lines: bool = False
    include_external: bool = False
    """Include models and sources from external dbt packages in the processing."""
    formatter: str | None = None
    """External command to format written YAML files (e.g. 'prettier --write'). File paths appended as args."""
    fusion_compat: bool | None = None
    """When True, output Fusion-compatible YAML with meta/tags nested inside config blocks.
    When False, output classic format with meta/tags at top level.
    When None (default), auto-detect from Fusion manifest evidence or installed dbt version:
    True if the project has a Fusion manifest or dbt >= 1.9.6."""


@dataclass
class YamlRefactorContext:
    """A data object that includes references to:

    - The dbt project context
    - The yaml refactor settings
    - A thread pool executor
    - A ruamel.yaml instance
    - A tuple of placeholder strings
    - The mutation count incremented during refactoring operations

    Resource Management:
        The context manager should be used to ensure resources are properly cleaned up:
            with YamlRefactorContext(project, settings) as context:
                ...  # use context
        Or explicitly call close() when done:
            context = YamlRefactorContext(project, settings)
            try:
                ...  # use context
            finally:
                context.close()
    """

    project: DbtProjectContext  # Forward reference to avoid circular import
    settings: YamlRefactorSettings = field(default_factory=YamlRefactorSettings)
    pool: ThreadPoolExecutor = field(
        default_factory=lambda: ThreadPoolExecutor(max_workers=min(32, (os.cpu_count() or 1) + 4)),
    )
    yaml_handler: ruamel.yaml.YAML = field(
        init=False,
        default_factory=_create_yaml_instance,
    )
    yaml_handler_lock: threading.Lock = field(default_factory=threading.Lock)
    current_node: t.Any | None = field(default=None, repr=False)

    placeholders: tuple[str, ...] = (
        EMPTY_STRING,
        "Pending further documentation",
        "No description for this column",
        "Not documented",
        "Undefined",
    )

    _mutation_count: int = field(default=0, init=False)
    _written_files: set[Path] = field(default_factory=set, init=False, repr=False)
    """Tracks which YAML files were actually written to disk (for external formatter integration)."""
    _disk_mutation_count: int = field(default=0, init=False, repr=False)
    """Counts actual on-disk mutations so callers can distinguish dry-run from real writes."""
    _catalog: CatalogResults | None = field(default=None, init=False)
    _closed: bool = field(default=False, init=False, repr=False)
    """Track whether the context has been closed to prevent double-cleanup."""

    def __enter__(self) -> YamlRefactorContext:
        """Enter the context manager.

        Returns:
            self for use in with statements

        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the context manager, ensuring resources are cleaned up.

        Args:
            exc_type: Exception type if an exception was raised
            exc_val: Exception value if an exception was raised
            exc_tb: Exception traceback if an exception was raised

        """
        self.close()

    def close(self) -> None:
        """Close the thread pool and cleanup resources.

        This method is idempotent - calling it multiple times is safe.
        It shuts down the thread pool and closes the project context
        to prevent resource leaks.
        """
        if self._closed:
            return

        try:
            # Shutdown the thread pool
            if hasattr(self, "pool") and self.pool is not None:
                logger.debug(":lock: Shutting down thread pool")
                self.pool.shutdown(wait=True)
        except Exception as e:
            logger.warning(":warning: Error shutting down thread pool: %s", e)

        try:
            # Close the project context
            if hasattr(self, "project") and self.project is not None:
                self.project.close()
        except Exception as e:
            logger.warning(":warning: Error closing project context: %s", e)

        self._closed = True

    def register_mutations(self, count: int) -> None:
        """Increment the mutation count by a specified amount."""
        logger.debug(
            ":sparkles: Registering %s new mutations. Current count => %s",
            count,
            self._mutation_count,
        )
        self._mutation_count += count

    def register_disk_mutation(self, count: int = 1) -> None:
        """Record actual on-disk changes so callers can react truthfully after dry-run flows."""
        self._disk_mutation_count += count

    @property
    def mutation_count(self) -> int:
        """Read only property to access the mutation count."""
        return self._mutation_count

    @property
    def mutated(self) -> bool:
        """Check if the context has performed any mutations."""
        has_mutated = self._mutation_count > 0
        logger.debug(":white_check_mark: Has the context mutated anything? => %s", has_mutated)
        return has_mutated

    @property
    def disk_mutation_count(self) -> int:
        """Read-only count of actual on-disk mutations in this session."""
        return self._disk_mutation_count

    def register_written_file(self, path: Path) -> None:
        """Register a file path that was successfully written to disk.

        Paths are resolved to absolute to ensure the external formatter
        can find them regardless of its working directory.
        """
        self._written_files.add(path.resolve())
        self.register_disk_mutation()

    @property
    def written_files(self) -> frozenset[Path]:
        """Read-only view of all files written to disk during this session."""
        return frozenset(self._written_files)

    @property
    def resolved_formatter(self) -> str | None:
        """Resolve the formatter command: CLI setting > dbt-osmosis.yml > None."""
        # 1. CLI flag (highest priority)
        if self.settings.formatter:
            return self.settings.formatter
        # 2. dbt-osmosis.yml supplementary file
        supp_file = self.project_root / "dbt-osmosis.yml"
        if supp_file.is_file():
            try:
                import yaml

                with supp_file.open("r") as f:
                    data = yaml.safe_load(f) or {}
                formatter = data.get("formatter")
                if isinstance(formatter, str) and formatter.strip():
                    return formatter.strip()
            except Exception as e:
                logger.warning(
                    ":warning: Failed to read formatter from %s: %s",
                    supp_file,
                    e,
                )
        return None

    # Convenience properties for commonly accessed nested attributes
    # These reduce repetition and improve readability throughout the codebase

    @property
    def manifest(self):
        """Shortcut to context.project.manifest for brevity."""
        return self.project.manifest

    @property
    def runtime_cfg(self):
        """Shortcut to context.project.runtime_cfg for brevity."""
        return self.project.runtime_cfg

    @property
    def adapter(self):
        """Shortcut to context.project.adapter for brevity."""
        return self.project.adapter

    @property
    def project_root(self) -> Path:
        """Shortcut to context.project.runtime_cfg.project_root for brevity."""
        return Path(self.project.runtime_cfg.project_root)

    @property
    def credentials(self):
        """Shortcut to context.project.runtime_cfg.credentials for brevity."""
        return self.project.runtime_cfg.credentials

    @property
    def database_type(self) -> str:
        """Shortcut to context.project.runtime_cfg.credentials.type for brevity."""
        return self.project.runtime_cfg.credentials.type

    @property
    def fusion_compat(self) -> bool:
        """Whether to output Fusion-compatible YAML (meta/tags inside config block).

        Resolution order:
        1. Explicit setting from YamlRefactorSettings.fusion_compat (True/False)
        2. Fusion manifest detected in target directory (schema version > v12)
        3. Auto-detect from dbt version: True if dbt >= 1.9.6

        The 1.9.6 threshold is used because column-level config support was
        introduced in that version. Below 1.9.6, nesting meta/tags under config
        on columns causes silent data loss.

        Fusion manifest detection (step 2) handles teams running both dbt Fusion
        and dbt-core side by side: even if the installed dbt-core is older than
        1.9.6, a Fusion manifest signals that Fusion-compatible output is needed.
        """
        if self.settings.fusion_compat is not None:
            return self.settings.fusion_compat
        if getattr(self.project, "is_fusion_manifest", False):
            return True
        return getattr(self.project, "is_dbt_v1_9_6_or_greater", False)

    @property
    def source_definitions(self) -> dict[str, t.Any]:
        """The source definitions from the dbt project config."""
        c = self.project.runtime_cfg.vars.to_dict()
        toplevel_conf = self._find_first(
            [c.get(k, {}) for k in ["dbt-osmosis", "dbt_osmosis"]],
            lambda v: bool(v),
            {},
        )
        return toplevel_conf.get("sources", {})

    @property
    def ignore_patterns(self) -> list[str]:
        """The column name ignore patterns from the dbt project config."""
        c = self.project.runtime_cfg.vars.to_dict()
        toplevel_conf = self._find_first(
            [c.get(k, {}) for k in ["dbt-osmosis", "dbt_osmosis"]],
            lambda v: bool(v),
            {},
        )
        return toplevel_conf.get("column_ignore_patterns", [])

    @property
    def yaml_settings(self) -> dict[str, t.Any]:
        """The column name ignore patterns from the dbt project config."""
        c = self.project.runtime_cfg.vars.to_dict()
        toplevel_conf = self._find_first(
            [c.get(k, {}) for k in ["dbt-osmosis", "dbt_osmosis"]],
            lambda v: bool(v),
            {},
        )
        return toplevel_conf.get("yaml_settings", {})

    def read_catalog(self) -> CatalogResults | None:
        """Read the catalog file if it exists."""
        logger.debug(":mag: Checking if catalog is already loaded => %s", bool(self._catalog))
        if not self._catalog:
            from dbt_osmosis.core.introspection import _generate_catalog, _load_catalog

            catalog = _load_catalog(self.settings)
            if not catalog and self.settings.create_catalog_if_not_exists:
                logger.info(
                    ":bookmark_tabs: No existing catalog found, generating new catalog.json.",
                )
                catalog = _generate_catalog(self.project)
            self._catalog = catalog
        return self._catalog

    def _find_first(
        self,
        coll: t.Iterable[t.Any],
        predicate: t.Callable[[t.Any], bool],
        default: t.Any = None,
    ) -> t.Any:
        """Find the first item in a container that satisfies a predicate."""
        for item in coll:
            if predicate(item):
                return item
        return default

    def __post_init__(self) -> None:
        logger.debug(":green_book: Running post-init for YamlRefactorContext.")
        if EMPTY_STRING not in self.placeholders:
            self.placeholders = (EMPTY_STRING, *self.placeholders)
        for setting, val in self.yaml_settings.items():
            setattr(self.yaml_handler, setting, val)
        # Override max_workers with dbt's thread count when available.
        if hasattr(self.project.runtime_cfg, "threads") and self.project.runtime_cfg.threads:
            self.pool._max_workers = self.project.runtime_cfg.threads
            logger.info(
                ":notebook: Osmosis ThreadPoolExecutor max_workers using dbt threads => %s",
                self.pool._max_workers,
            )
        else:
            logger.info(
                ":notebook: Osmosis ThreadPoolExecutor max_workers using default => %s",
                self.pool._max_workers,
            )
