from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import typing as t

import ruamel.yaml
from dbt.contracts.results import CatalogResults

import dbt_osmosis.core.logger as logger

if t.TYPE_CHECKING:
    from dbt_osmosis.core.config import DbtProjectContext

__all__ = [
    "EMPTY_STRING",
    "YamlRefactorSettings",
    "YamlRefactorContext",
]

EMPTY_STRING = ""
"""A null string constant for use in placeholder lists, this is always considered undocumented"""


@dataclass
class YamlRefactorSettings:
    """Settings for yaml based refactoring operations."""

    fqn: list[str] = field(default_factory=list)
    """Filter models to action via a fully qualified name match such as returned by `dbt ls`."""
    models: list[t.Union[Path, str]] = field(default_factory=list)
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
    add_inheritance_for_specified_keys: list[str] = field(default_factory=list)
    """Include additional keys in the inheritance process."""
    output_to_lower: bool = False
    """Force column name and data type output to lowercase in the yaml files."""
    catalog_path: t.Optional[str] = None
    """Path to the dbt catalog.json file to use preferentially instead of live warehouse introspection"""
    create_catalog_if_not_exists: bool = False
    """Generate the catalog.json for the project if it doesn't exist and use it for introspective queries."""


@dataclass
class YamlRefactorContext:
    """A data object that includes references to:

    - The dbt project context
    - The yaml refactor settings
    - A thread pool executor
    - A ruamel.yaml instance
    - A tuple of placeholder strings
    - The mutation count incremented during refactoring operations
    """

    project: DbtProjectContext  # Forward reference to avoid circular import
    settings: YamlRefactorSettings = field(default_factory=YamlRefactorSettings)
    pool: ThreadPoolExecutor = field(default_factory=ThreadPoolExecutor)
    yaml_handler: ruamel.yaml.YAML = field(
        default_factory=lambda: None
    )  # Will be set in __post_init__
    yaml_handler_lock: threading.Lock = field(default_factory=threading.Lock)

    placeholders: tuple[str, ...] = (
        EMPTY_STRING,
        "Pending further documentation",
        "No description for this column",
        "Not documented",
        "Undefined",
    )

    _mutation_count: int = field(default=0, init=False)
    _catalog: t.Optional[CatalogResults] = field(default=None, init=False)

    def register_mutations(self, count: int) -> None:
        """Increment the mutation count by a specified amount."""
        logger.debug(
            ":sparkles: Registering %s new mutations. Current count => %s",
            count,
            self._mutation_count,
        )
        self._mutation_count += count

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
    def source_definitions(self) -> dict[str, t.Any]:
        """The source definitions from the dbt project config."""
        c = self.project.runtime_cfg.vars.to_dict()
        toplevel_conf = self._find_first(
            [c.get(k, {}) for k in ["dbt-osmosis", "dbt_osmosis"]], lambda v: bool(v), {}
        )
        return toplevel_conf.get("sources", {})

    @property
    def ignore_patterns(self) -> list[str]:
        """The column name ignore patterns from the dbt project config."""
        c = self.project.runtime_cfg.vars.to_dict()
        toplevel_conf = self._find_first(
            [c.get(k, {}) for k in ["dbt-osmosis", "dbt_osmosis"]], lambda v: bool(v), {}
        )
        return toplevel_conf.get("column_ignore_patterns", [])

    @property
    def yaml_settings(self) -> dict[str, t.Any]:
        """The column name ignore patterns from the dbt project config."""
        c = self.project.runtime_cfg.vars.to_dict()
        toplevel_conf = self._find_first(
            [c.get(k, {}) for k in ["dbt-osmosis", "dbt_osmosis"]], lambda v: bool(v), {}
        )
        return toplevel_conf.get("yaml_settings", {})

    def read_catalog(self) -> CatalogResults | None:
        """Read the catalog file if it exists."""
        logger.debug(":mag: Checking if catalog is already loaded => %s", bool(self._catalog))
        if not self._catalog:
            from dbt_osmosis.core.introspection import _load_catalog, _generate_catalog

            catalog = _load_catalog(self.settings)
            if not catalog and self.settings.create_catalog_if_not_exists:
                logger.info(
                    ":bookmark_tabs: No existing catalog found, generating new catalog.json."
                )
                catalog = _generate_catalog(self.project)
            self._catalog = catalog
        return self._catalog

    def _find_first(
        self, coll: t.Iterable[t.Any], predicate: t.Callable[[t.Any], bool], default: t.Any = None
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
        # Initialize yaml_handler here
        from dbt_osmosis.core.schema.parser import create_yaml_instance

        self.yaml_handler = create_yaml_instance()
        for setting, val in self.yaml_settings.items():
            setattr(self.yaml_handler, setting, val)
        self.pool._max_workers = self.project.runtime_cfg.threads
        logger.info(
            ":notebook: Osmosis ThreadPoolExecutor max_workers synced with dbt => %s",
            self.pool._max_workers,
        )
