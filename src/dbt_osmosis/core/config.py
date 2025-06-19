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

import dbt.flags as dbt_flags
from dbt.adapters.base.impl import BaseAdapter
from dbt.adapters.factory import get_adapter, register_adapter
from dbt.config.runtime import RuntimeConfig
from dbt.context.providers import generate_runtime_macro_context
from dbt.contracts.graph.manifest import Manifest
from dbt.mp_context import get_mp_context
from dbt.parser.manifest import ManifestLoader
from dbt.parser.models import ModelParser
from dbt.parser.sql import SqlBlockParser, SqlMacroParser
from dbt.tracking import disable_tracking
from dbt_common.clients.system import get_env
from dbt_common.context import set_invocation_context

import dbt_osmosis.core.logger as logger

__all__ = [
    "discover_project_dir",
    "discover_profiles_dir",
    "DbtConfiguration",
    "DbtProjectContext",
    "create_dbt_project_context",
    "_reload_manifest",
]

disable_tracking()


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
    """Configuration for a dbt project."""

    project_dir: str = field(default_factory=discover_project_dir)
    profiles_dir: str = field(default_factory=discover_profiles_dir)
    target: t.Optional[str] = None
    profile: t.Optional[str] = None
    threads: t.Optional[int] = None
    single_threaded: t.Optional[bool] = None
    vars: dict[str, t.Any] = field(default_factory=dict)
    quiet: bool = True
    disable_introspection: bool = False  # Internal

    def __post_init__(self) -> None:
        logger.debug(":bookmark_tabs: Setting invocation context with environment variables.")
        set_invocation_context(get_env())
        if self.threads and self.threads > 1:
            self.single_threaded = False


def config_to_namespace(cfg: DbtConfiguration) -> argparse.Namespace:
    """Convert a DbtConfiguration into a dbt-friendly argparse.Namespace."""
    logger.debug(":blue_book: Converting DbtConfiguration to argparse.Namespace => %s", cfg)
    ns = argparse.Namespace(
        project_dir=cfg.project_dir,
        profiles_dir=cfg.profiles_dir,
        target=cfg.target or os.getenv("DBT_TARGET"),
        profile=cfg.profile or os.getenv("DBT_PROFILE"),
        threads=cfg.threads,
        single_threaded=cfg.single_threaded,
        vars=cfg.vars,
        which="parse",
        quiet=cfg.quiet,
        DEBUG=False,
        REQUIRE_RESOURCE_NAMES_WITHOUT_SPACES=False,
    )
    return ns


@dataclass
class DbtProjectContext:
    """A data object that includes references to:

    - The loaded dbt config
    - The manifest
    - The sql/macro parsers

    With mutexes for thread safety. The adapter is lazily instantiated and has a TTL which allows
    for re-use across multiple operations in long-running processes. (is the idea)
    """

    config: DbtConfiguration
    """The configuration for the dbt project used to initialize the runtime cfg and manifest"""
    runtime_cfg: RuntimeConfig
    """The dbt project runtime config associated with the context"""
    manifest: Manifest
    """The dbt project manifest"""
    sql_parser: SqlBlockParser
    """Parser for dbt Jinja SQL blocks"""
    macro_parser: SqlMacroParser
    """Parser for dbt Jinja macros"""
    connection_ttl: float = 3600.0
    """Max time in seconds to keep a db connection alive before recycling it, mostly useful for very long runs"""

    _adapter_mutex: threading.Lock = field(default_factory=threading.Lock, init=False)
    _manifest_mutex: threading.Lock = field(default_factory=threading.Lock, init=False)
    _adapter: t.Optional[BaseAdapter] = field(default=None, init=False)
    _connection_created_at: dict[int, float] = field(default_factory=dict, init=False)

    @property
    def is_connection_expired(self) -> bool:
        """Check if the adapter has expired based on the adapter TTL."""
        expired = (
            time.time() - self._connection_created_at.setdefault(get_ident(), 0.0)
            > self.connection_ttl
        )
        logger.debug(":hourglass_flowing_sand: Checking if connection is expired => %s", expired)
        return expired

    @property
    def adapter(self) -> BaseAdapter:
        """Get the adapter instance, creating a new one if the current one has expired."""
        with self._adapter_mutex:
            if not self._adapter:
                logger.info(":wrench: Instantiating new adapter because none is currently set.")
                adapter = _instantiate_adapter(self.runtime_cfg)
                adapter.set_macro_resolver(self.manifest)
                _ = adapter.acquire_connection()
                self._adapter = adapter
                self._connection_created_at[get_ident()] = time.time()
                logger.info(
                    ":wrench: Successfully acquired new adapter connection for thread => %s",
                    get_ident(),
                )
            elif self.is_connection_expired:
                logger.info(
                    ":wrench: Refreshing db connection for thread => %s",
                    get_ident(),
                )
                self._adapter.connections.release()
                self._adapter.connections.clear_thread_connection()
                _ = self._adapter.acquire_connection()
                self._connection_created_at[get_ident()] = time.time()
        return self._adapter

    @property
    def manifest_mutex(self) -> threading.Lock:
        """Return the manifest mutex for thread safety."""
        return self._manifest_mutex


def _add_cross_project_references(
    manifest: Manifest, dbt_loom: ModuleType, project_name: str
) -> Manifest:
    """Add cross-project references to the dbt manifest from dbt-loom defined manifests."""
    loomnodes: list[t.Any] = []
    loom = dbt_loom.dbtLoom(project_name)
    loom_manifests = loom.manifests
    logger.info(":arrows_counterclockwise: Loaded dbt loom manifests!")
    for name, loom_manifest in loom_manifests.items():
        if loom_manifest.get("nodes"):
            loom_manifest_nodes = loom_manifest.get("nodes")
            for _, node in loom_manifest_nodes.items():
                if node.get("access"):
                    node_access = node.get("access")
                    if node_access != "protected":
                        if node.get("resource_type") == "model":
                            loomnodes.append(ModelParser.parse_from_dict(None, node))  # pyright: ignore[reportArgumentType]
        for node in loomnodes:
            manifest.nodes[node.unique_id] = node
        logger.info(
            f":arrows_counterclockwise: added {len(loomnodes)} exposed nodes from {name} to the dbt manifest!"
        )
    return manifest


def _instantiate_adapter(runtime_config: RuntimeConfig) -> BaseAdapter:
    """Instantiate a dbt adapter based on the runtime configuration."""
    logger.debug(":mag: Registering adapter for runtime config => %s", runtime_config)
    adapter = get_adapter(runtime_config)
    adapter.set_macro_context_generator(t.cast(t.Any, generate_runtime_macro_context))
    adapter.connections.set_connection_name("dbt-osmosis")
    logger.debug(":hammer_and_wrench: Adapter instantiated => %s", adapter)
    return t.cast(BaseAdapter, t.cast(t.Any, adapter))


def create_dbt_project_context(config: DbtConfiguration) -> DbtProjectContext:
    """Build a DbtProjectContext from a DbtConfiguration."""
    logger.info(":wave: Creating DBT project context using config => %s", config)
    args = config_to_namespace(config)
    dbt_flags.set_from_args(args, args)
    runtime_cfg = RuntimeConfig.from_args(args)

    logger.info(":bookmark_tabs: Registering adapter as part of project context creation.")
    register_adapter(runtime_cfg, get_mp_context())

    loader = ManifestLoader(
        runtime_cfg,
        runtime_cfg.load_dependencies(),
    )
    manifest = loader.load()

    try:
        dbt_loom = importlib.import_module("dbt_loom")
    except ImportError:
        pass
    else:
        manifest = _add_cross_project_references(manifest, dbt_loom, runtime_cfg.project_name)

    manifest.build_flat_graph()
    logger.info(":arrows_counterclockwise: Loaded the dbt project manifest!")

    if not config.disable_introspection:
        adapter = _instantiate_adapter(runtime_cfg)
        runtime_cfg.adapter = adapter  # pyright: ignore[reportAttributeAccessIssue]
        adapter.set_macro_resolver(manifest)

    sql_parser = SqlBlockParser(runtime_cfg, manifest, runtime_cfg)
    macro_parser = SqlMacroParser(runtime_cfg, manifest)

    logger.info(":sparkles: DbtProjectContext successfully created!")
    return DbtProjectContext(
        config=config,
        runtime_cfg=runtime_cfg,
        manifest=manifest,
        sql_parser=sql_parser,
        macro_parser=macro_parser,
    )


def _reload_manifest(context: DbtProjectContext) -> None:
    """Reload the dbt project manifest. Useful for picking up mutations."""
    logger.info(":arrows_counterclockwise: Reloading the dbt project manifest!")
    loader = ManifestLoader(context.runtime_cfg, context.runtime_cfg.load_dependencies())
    manifest = loader.load()
    manifest.build_flat_graph()
    if not context.config.disable_introspection:
        context.adapter.set_macro_resolver(manifest)
    context.manifest = manifest
    logger.info(":white_check_mark: Manifest reloaded => %s", context.manifest.metadata)
