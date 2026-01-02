"""Dbt project configuration and context management.

This module provides a high-level interface to dbt-core for dbt-osmosis.
It wraps dbt-core-interface to provide a cleaner API and reduce coupling to
internal dbt-core modules.
"""

from __future__ import annotations

import argparse
import importlib
import os
import threading
import time
import typing as t
from dataclasses import dataclass, field
from pathlib import Path
from threading import get_ident
from types import ModuleType

# Import from dbt-core-interface instead of internal dbt modules
from dbt_core_interface import DbtConfiguration as InterfaceDbtConfiguration
from dbt_core_interface import DbtProject as InterfaceDbtProject

# Type imports for compatibility
from dbt.adapters.base.impl import BaseAdapter
from dbt.contracts.graph.manifest import Manifest
from dbt.parser.models import ModelParser

import dbt_osmosis.core.logger as logger

__all__ = [
    "discover_project_dir",
    "discover_profiles_dir",
    "DbtConfiguration",
    "DbtProjectContext",
    "create_dbt_project_context",
    "_reload_manifest",
    "MAX_PREVIEW_LENGTH",
    "config_to_namespace",
]


def discover_project_dir() -> str:
    """Return the directory containing a dbt_project.yml if found, else the current dir. Checks DBT_PROJECT_DIR first if set."""
    if "DBT_PROJECT_DIR" in os.environ:
        project_dir = Path(os.environ["DBT_PROJECT_DIR"])
        if project_dir.is_dir():
            logger.info(":mag: DBT_PROJECT_DIR detected => %s", project_dir)
            return str(project_dir.resolve())
        logger.warning(":warning: DBT_PROJECT_DIR %s is not a valid directory.", project_dir)
    cwd = Path.cwd()
    for p in [cwd] + list(cwd.parents):
        if (p / "dbt_project.yml").exists():
            logger.info(":mag: Found dbt_project.yml at => %s", p)
            return str(p.resolve())
    logger.info(":mag: Defaulting to current directory => %s", cwd)
    return str(cwd.resolve())


def discover_profiles_dir() -> str:
    """Return the directory containing a profiles.yml if found, else ~/.dbt. Checks DBT_PROFILES_DIR first if set."""
    if "DBT_PROFILES_DIR" in os.environ:
        profiles_dir = Path(os.environ["DBT_PROFILES_DIR"])
        if profiles_dir.is_dir():
            logger.info(":mag: DBT_PROFILES_DIR detected => %s", profiles_dir)
            return str(profiles_dir.resolve())
        logger.warning(":warning: DBT_PROFILES_DIR %s is not a valid directory.", profiles_dir)
    if (Path.cwd() / "profiles.yml").exists():
        logger.info(":mag: Found profiles.yml in current directory.")
        return str(Path.cwd().resolve())
    home_profiles = str(Path.home() / ".dbt")
    logger.info(":mag: Defaulting to => %s", home_profiles)
    return home_profiles


@dataclass
class DbtConfiguration:
    """Configuration for a dbt project.

    This is a simplified configuration class that maps to dbt-core-interface's
    DbtConfiguration while maintaining dbt-osmosis-specific settings.
    """

    project_dir: str = field(default_factory=discover_project_dir)
    profiles_dir: str = field(default_factory=discover_profiles_dir)
    target: str | None = None
    profile: str | None = None
    threads: int | None = None
    single_threaded: bool | None = None
    vars: dict[str, t.Any] = field(default_factory=dict)
    quiet: bool = True
    disable_introspection: bool = False  # Internal

    def __post_init__(self) -> None:
        if self.threads and self.threads > 1:
            self.single_threaded = False

    def to_interface_config(self) -> InterfaceDbtConfiguration:
        """Convert to dbt-core-interface DbtConfiguration."""
        return InterfaceDbtConfiguration(
            project_dir=self.project_dir,
            profiles_dir=self.profiles_dir,
            target=self.target,
            profile=self.profile,
            threads=self.threads or 1,
            vars=self.vars,
            quiet=self.quiet,
            use_experimental_parser=True,
            static_parser=True,
            partial_parse=True,
            defer=True,
            favor_state=False,
        )


@dataclass
class DbtProjectContext:
    """A data object that includes references to the dbt project.

    This class wraps dbt-core-interface's DbtProject to provide a cleaner
    API for dbt-osmosis while maintaining backwards compatibility.

    Thread-safety: The underlying DbtProject handles thread safety for
    adapter and manifest access. Additional mutexes protect manifest reload
    operations to prevent concurrent modifications.
    """

    config: DbtConfiguration
    """The configuration for the dbt project"""

    _project: InterfaceDbtProject = field(default=None, init=False, repr=False)
    """The underlying dbt-core-interface DbtProject instance"""

    connection_ttl: float = 3600.0
    """Max time in seconds to keep a db connection alive before recycling it"""

    _manifest_mutex: threading.RLock = field(default_factory=threading.RLock, init=False)
    """Lock protecting manifest reload operations.

    Thread-safety: Use this lock to synchronize manifest reload operations.
    The underlying DbtProject has its own thread-safety mechanisms; this
    additional lock provides extra protection for manifest reloads.

    Uses RLock (reentrant lock) to allow nested acquisitions within the same thread,
    which is necessary because the adapter property may be accessed while holding
    this lock (e.g., in compile_sql_code).
    """

    _connection_created_at: dict[int, float] = field(default_factory=dict, init=False)
    """Per-thread connection creation timestamps for TTL tracking.

    Thread-safety: Protected by DbtProject's internal locks. Keys are thread IDs.
    """

    @classmethod
    def from_project(
        cls, config: DbtConfiguration, project: InterfaceDbtProject
    ) -> "DbtProjectContext":
        """Create a DbtProjectContext from an existing DbtProject instance.

        This is used internally to avoid creating the DbtProject twice.

        Args:
            config: The dbt project configuration
            project: The already-created DbtProject instance

        Returns:
            A DbtProjectContext wrapping the DbtProject
        """
        instance = cls(config=config)
        instance._project = project
        return instance

    @property
    def runtime_cfg(self):
        """Get the dbt runtime config.

        Delegates to the underlying DbtProject's runtime_config.
        """
        return self._project.runtime_config

    @property
    def manifest(self) -> Manifest:
        """Get the dbt project manifest.

        Delegates to the underlying DbtProject's manifest.
        """
        return self._project.manifest

    @manifest.setter
    def manifest(self, value: Manifest) -> None:
        """Set the dbt project manifest.

        This allows manifest reloading while maintaining the same DbtProject instance.
        """
        # We need to update the internal manifest in DbtProject
        # DbtProject doesn't expose a setter, so we set it on the object
        object.__setattr__(self._project, "_DbtProject__manifest", value)

    @property
    def sql_parser(self):
        """Get the SQL parser.

        Delegates to the underlying DbtProject's sql_parser property.
        """
        return self._project.sql_parser

    @property
    def macro_parser(self):
        """Get the macro parser.

        Delegates to the underlying DbtProject's macro_parser property.
        """
        return self._project.macro_parser

    @property
    def adapter(self) -> BaseAdapter:
        """Get the dbt adapter instance.

        Delegates to the underlying DbtProject's adapter property.
        Implements TTL-based connection recycling.
        """
        with self._manifest_mutex:
            if not hasattr(self, "_connection_created_at"):
                self._connection_created_at = {}

            ident = get_ident()
            current_time = time.time()
            last_created = self._connection_created_at.get(ident, 0.0)

            if current_time - last_created > self.connection_ttl:
                logger.info(
                    ":hourglass_flowing_sand: Connection expired for thread => %s, refreshing",
                    ident,
                )
                # Force adapter refresh by clearing and reacquiring
                adapter = self._project.adapter
                # Trigger connection refresh by accessing connections
                if hasattr(adapter, "connections"):
                    try:
                        adapter.connections.release()
                        adapter.connections.clear_thread_connection()
                        adapter.acquire_connection()
                    except Exception:
                        # If refresh fails, continue with existing adapter
                        pass
                self._connection_created_at[ident] = current_time

            return self._project.adapter

    @property
    def manifest_mutex(self) -> threading.Lock:
        """Return the manifest mutex for thread safety.

        Thread-safety: Use this lock to synchronize manifest reload operations.
        """
        return self._manifest_mutex


def _add_cross_project_references(
    manifest: Manifest, dbt_loom: ModuleType, project_name: str
) -> Manifest:
    """Add cross-project references to the dbt manifest from dbt-loom defined manifests.

    Wraps dbt_loom API calls with error handling to prevent failures from breaking
    the manifest loading process.
    """
    loomnodes: list[t.Any] = []
    try:
        loom = dbt_loom.dbtLoom(project_name)
        loom_manifests = loom.manifests
    except (AttributeError, KeyError, TypeError) as e:
        logger.warning(":warning: Failed to load dbt loom manifests: %s", e)
        return manifest
    except Exception as e:
        logger.warning(":warning: Unexpected error loading dbt loom manifests: %s", e)
        return manifest

    logger.info(":arrows_counterclockwise: Loaded dbt loom manifests!")
    for name, loom_manifest in loom_manifests.items():
        if loom_manifest.get("nodes"):
            loom_manifest_nodes = loom_manifest.get("nodes")
            for _, node in loom_manifest_nodes.items():
                if node.get("access"):
                    node_access = node.get("access")
                    if node_access != "protected":
                        if node.get("resource_type") == "model":
                            try:
                                loomnodes.append(ModelParser.parse_from_dict(None, node))  # pyright: ignore[reportArgumentType]
                            except Exception as e:
                                logger.warning(
                                    ":warning: Failed to parse node %s from dbt loom: %s",
                                    node.get("unique_id", "unknown"),
                                    e,
                                )
            for node in loomnodes:
                manifest.nodes[node.unique_id] = node
            logger.info(
                f":arrows_counterclockwise: added {len(loomnodes)} exposed nodes from {name} to the dbt manifest!"
            )
    return manifest


def create_dbt_project_context(config: DbtConfiguration) -> DbtProjectContext:
    """Build a DbtProjectContext from a DbtConfiguration.

    This creates a DbtProject instance using dbt-core-interface and wraps it
    in a DbtProjectContext for backwards compatibility with existing dbt-osmosis code.

    Args:
        config: The dbt project configuration

    Returns:
        A DbtProjectContext with initialized manifest, adapter, and parsers
    """
    logger.info(":wave: Creating DBT project context using config => %s", config)

    # Create the interface config
    interface_config = config.to_interface_config()

    # Create DbtProject instance (this loads the manifest)
    project = InterfaceDbtProject.from_config(interface_config)

    # Handle dbt-loom cross-project references if available
    try:
        dbt_loom = importlib.import_module("dbt_loom")
    except ImportError:
        logger.debug(
            ":information_source: dbt_loom not available, skipping cross-project references"
        )
    else:
        try:
            manifest = _add_cross_project_references(
                project.manifest, dbt_loom, project.project_name
            )
            # Update the manifest in the project
            project.manifest = manifest
        except Exception as e:
            logger.warning(":warning: Failed to add cross-project references from dbt_loom: %s", e)

    logger.info(":sparkles: DbtProjectContext successfully created!")

    # Create the context wrapper with the existing project
    return DbtProjectContext.from_project(config=config, project=project)


def _reload_manifest(context: DbtProjectContext) -> None:
    """Reload the dbt project manifest. Useful for picking up mutations.

    This uses DbtProject's parse_project method to reload the manifest
    from disk, ensuring all changes are picked up.

    Args:
        context: The DbtProjectContext to reload the manifest for
    """
    logger.info(":arrows_counterclockwise: Reloading the dbt project manifest!")
    with context.manifest_mutex:
        # Use DbtProject's parse_project to reload
        context._project.parse_project(write_manifest=False)
        logger.info(":white_check_mark: Manifest reloaded => %s", context.manifest.metadata)


# Constants
MAX_PREVIEW_LENGTH = 100


def config_to_namespace(config: DbtConfiguration) -> argparse.Namespace:
    """Convert a DbtConfiguration to an argparse.Namespace for backwards compatibility.

    This function is maintained for backwards compatibility with code that expects
    an argparse.Namespace object. The dbt-core-interface uses DbtConfiguration directly.

    Args:
        config: The DbtConfiguration to convert

    Returns:
        An argparse.Namespace with the same attributes as the config
    """
    return argparse.Namespace(
        project_dir=config.project_dir,
        profiles_dir=config.profiles_dir,
        target=config.target,
        profile=config.profile,
        threads=config.threads,
        single_threaded=config.single_threaded,
        vars=config.vars,
        quiet=config.quiet,
        disable_introspection=config.disable_introspection,
    )
