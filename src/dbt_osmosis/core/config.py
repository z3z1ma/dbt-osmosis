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

# Type imports for compatibility
from dbt.adapters.base.impl import BaseAdapter
from dbt.contracts.graph.manifest import Manifest
from dbt.mp_context import get_mp_context
from dbt.parser.models import ModelParser

# Import from dbt-core-interface instead of internal dbt modules
from dbt_core_interface import DbtConfiguration as InterfaceDbtConfiguration
from dbt_core_interface import DbtProject as InterfaceDbtProject
from packaging.version import parse as parse_version

from dbt_osmosis.core import logger

# Import dbt version for compatibility checking
# Use a try/except in case the version module structure changes
try:
    from dbt.version import get_installed_version

    _raw_version = str(get_installed_version())
    # Version may be returned as "=1.10.17" or similar, strip the prefix
    _dbt_version = _raw_version.lstrip("=")
except (ImportError, AttributeError):
    try:
        from dbt import version as _dbt_version_module

        _dbt_version = getattr(_dbt_version_module, "__version__", "1.8.0")
    except (ImportError, AttributeError):
        _dbt_version = "1.8.0"

__all__ = [
    "DEFAULT_CONNECTION_TTL",
    "MAX_PREVIEW_LENGTH",
    "DbtConfiguration",
    "DbtProjectContext",
    "_reload_manifest",
    "config_to_namespace",
    "create_dbt_project_context",
    "discover_profiles_dir",
    "discover_project_dir",
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


# Constants
MAX_PREVIEW_LENGTH = 100
DEFAULT_CONNECTION_TTL = 3600.0


@dataclass
class DbtProjectContext:
    """A data object that includes references to the dbt project.

    This class wraps dbt-core-interface's DbtProject to provide a cleaner
    API for dbt-osmosis while maintaining backwards compatibility.

    Lazy Adapter Initialization:
        The DbtProject uses lazy initialization for the adapter property to defer
        connection overhead until actually needed. This pattern avoids establishing
        database connections during context creation when they may not be used,
        improving startup performance and reducing resource consumption. The
        connection_ttl field implements TTL-based connection recycling on top of
        this lazy initialization to refresh stale connections in long-running
        processes.

    Thread-safety: The underlying DbtProject handles thread safety for
    adapter and manifest access. Additional mutexes protect manifest reload
    operations to prevent concurrent modifications.

    Resource Management:
        The context manager should be used to ensure connections are properly closed:
            with create_dbt_project_context(config) as context:
                ...  # use context
        Or explicitly call close() when done:
            context = create_dbt_project_context(config)
            try:
                ...  # use context
            finally:
                context.close()
    """

    config: DbtConfiguration
    """The configuration for the dbt project"""

    _project: InterfaceDbtProject = field(default=None, init=False, repr=False)
    """The underlying dbt-core-interface DbtProject instance"""

    connection_ttl: float = DEFAULT_CONNECTION_TTL
    """Max time in seconds to keep a db connection alive before recycling it.

    Defaults to DEFAULT_CONNECTION_TTL. Connection recycling ensures stale connections
    are refreshed, preventing timeouts and connection leaks in long-running processes.
    """

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

    _closed: bool = field(default=False, init=False, repr=False)
    """Track whether the context has been closed to prevent double-cleanup.

    Thread-safety: Protected by _manifest_mutex when checking/setting.
    """

    dbt_version: str = field(init=False, repr=True)
    """The dbt-core version being used (e.g., "1.10.0")."""

    is_dbt_v1_10_or_greater: bool = field(init=False, repr=False)
    """Whether the dbt version is 1.10.0 or higher.

    This is used for compatibility handling of the meta/tags namespace change
    in dbt 1.10, where these fields moved from top-level to the config block.
    """

    def __enter__(self) -> DbtProjectContext:
        """Enter the context manager.

        Returns:
            self for use in with statements

        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the context manager, ensuring connections are closed.

        Args:
            exc_type: Exception type if an exception was raised
            exc_val: Exception value if an exception was raised
            exc_tb: Exception traceback if an exception was raised

        """
        self.close()

    def close(self) -> None:
        """Close the adapter connections and cleanup resources.

        This method is idempotent - calling it multiple times is safe.
        It should be called when the DbtProjectContext is no longer needed
        to prevent connection leaks.

        Thread-safety: This method is thread-safe and can be called from any thread.
        """
        with self._manifest_mutex:
            if self._closed:
                return

            try:
                if self._project is not None and hasattr(self._project, "adapter"):
                    adapter = self._project.adapter
                    if hasattr(adapter, "connections"):
                        logger.debug(":lock: Closing adapter connections")
                        # Use cleanup_all() instead of close() because close() requires
                        # a connection parameter. cleanup_all() closes all open connections.
                        if hasattr(adapter.connections, "cleanup_all"):
                            adapter.connections.cleanup_all()
                        elif hasattr(adapter.connections, "close_all_connections"):
                            # Fallback for adapters that have close_all_connections
                            adapter.connections.close_all_connections()
            except Exception as e:
                logger.warning(":warning: Error closing adapter connections: %s", e)
            finally:
                self._closed = True

    @classmethod
    def from_project(
        cls,
        config: DbtConfiguration,
        project: InterfaceDbtProject,
    ) -> DbtProjectContext:
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
        instance.dbt_version = _dbt_version
        instance.is_dbt_v1_10_or_greater = parse_version(_dbt_version) >= parse_version("1.10.0")
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
    manifest: Manifest,
    dbt_loom: ModuleType,
    project_name: str,
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
                f":arrows_counterclockwise: added {len(loomnodes)} exposed nodes from {name} to the dbt manifest!",
            )
    return manifest


def _patch_adapter_factory_registration() -> None:
    """Monkey-patch dbt-core-interface to register adapters in FACTORY.adapters.

    The dbt-core-interface's create_adapter() method sets self.runtime_config.adapter
    but doesn't call FACTORY.register_adapter(). This causes KeyError when dbt parsers
    try to get the adapter via get_adapter(config) because FACTORY.lookup_adapter()
    looks in FACTORY.adapters, not runtime_config.adapter.

    This monkey-patch wraps DbtProject.create_adapter to register the adapter in FACTORY
    immediately after creation.
    """
    from dbt.adapters.factory import FACTORY
    from dbt_core_interface import DbtProject

    original_create_adapter = DbtProject.create_adapter

    def patched_create_adapter(self, replace: bool = False, verify_connectivity: bool = True):
        """Wrapper that ensures the adapter is registered in FACTORY.adapters."""
        # Call the original create_adapter
        adapter = original_create_adapter(
            self,
            replace=replace,
            verify_connectivity=verify_connectivity,
        )

        # Register the adapter in FACTORY.adapters if not already registered
        adapter_type = self.runtime_config.credentials.type
        if adapter_type not in FACTORY.adapters:
            FACTORY.adapters[adapter_type] = adapter
            logger.debug(
                f":wrench: Registered adapter '{adapter_type}' in FACTORY.adapters (monkey-patch)",
            )

        return adapter

    # Apply the monkey-patch
    DbtProject.create_adapter = patched_create_adapter


def _ensure_adapter_loaded(config: DbtConfiguration) -> None:
    """Ensure the dbt adapter plugin is loaded before creating the DbtProject.

    This is necessary because dbt-core-interface's DbtProject initialization calls
    RuntimeConfig.from_args() which needs the adapter to be registered in the factory.
    Without loading the plugin first, a KeyError will be raised for the adapter type.

    Args:
        config: The dbt project configuration

    """
    try:
        # Apply the monkey-patch to ensure adapters are registered in FACTORY.adapters
        _patch_adapter_factory_registration()

        # Try to read the profiles.yml to determine the adapter type
        from dbt.adapters.factory import FACTORY
        from dbt.config.project import read_profile_from_disk

        # Read the profile to get the adapter type
        raw_profile = read_profile_from_disk(
            config.profiles_dir,
            config.project_dir,
            config.profile,
            config.target,
        )

        # Get the adapter type from the profile's credentials
        adapter_type = raw_profile.credentials.type

        # Load the plugin if not already loaded
        if adapter_type not in FACTORY.plugins:
            logger.info(f":wrench: Loading dbt adapter plugin for '{adapter_type}'...")
            FACTORY.load_plugin(adapter_type)
            logger.info(f":white_check_mark: Successfully loaded adapter plugin '{adapter_type}'")
    except Exception as e:
        logger.debug(
            f":information_source: Could not pre-load adapter plugin: {e}. "
            "This is OK - the adapter will be loaded on-demand.",
        )


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

    # Ensure the adapter plugin is loaded before creating the DbtProject
    # This is necessary because RuntimeConfig.from_args() needs the adapter
    # to be registered in the factory, and load_plugin needs to be called
    # before that happens.
    _ensure_adapter_loaded(config)

    # Create the interface config
    interface_config = config.to_interface_config()

    # Create DbtProject instance (this loads the manifest)
    project = InterfaceDbtProject.from_config(interface_config)

    # Workaround: Ensure the adapter is registered in FACTORY.adapters
    # The dbt-core-interface's create_adapter() sets self.runtime_config.adapter
    # but doesn't call FACTORY.register_adapter(). This causes issues when
    # dbt parsers try to get the adapter via get_adapter(config) because
    # FACTORY.lookup_adapter() looks in FACTORY.adapters, not runtime_config.adapter.
    # We register the adapter here to ensure parsers can find it.
    try:
        from dbt.adapters.factory import FACTORY

        adapter_type = project.runtime_config.credentials.type
        if adapter_type not in FACTORY.adapters:
            FACTORY.register_adapter(
                project.runtime_config,  # pyright: ignore[reportArgumentType]
                get_mp_context("spawn"),  # pyright: ignore[reportArgumentType]
            )
            logger.info(f":white_check_mark: Registered adapter '{adapter_type}' in factory")
    except Exception as e:
        logger.debug(f":information_source: Could not register adapter in factory: {e}")

    # Handle dbt-loom cross-project references if available
    try:
        dbt_loom = importlib.import_module("dbt_loom")
    except ImportError:
        logger.debug(
            ":information_source: dbt_loom not available, skipping cross-project references",
        )
    else:
        try:
            manifest = _add_cross_project_references(
                project.manifest,
                dbt_loom,
                project.project_name,
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
