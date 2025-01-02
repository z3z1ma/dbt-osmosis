# pyright: reportUnknownVariableType=false, reportPrivateImportUsage=false, reportAny=false, reportUnknownMemberType=false

from __future__ import annotations

import argparse
import io
import json
import os
import re
import threading
import time
import typing as t
import uuid
from collections import OrderedDict
from collections.abc import Iterable, Iterator
from concurrent.futures import FIRST_EXCEPTION, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache, partial
from itertools import chain
from pathlib import Path
from threading import get_ident
from types import MappingProxyType

import dbt.flags as dbt_flags
import dbt.utils as dbt_utils
import pluggy
import ruamel.yaml
from agate.table import Table  # pyright: ignore[reportMissingTypeStubs]
from dbt.adapters.base.column import Column as BaseColumn
from dbt.adapters.base.impl import BaseAdapter
from dbt.adapters.base.relation import BaseRelation
from dbt.adapters.contracts.connection import AdapterResponse
from dbt.adapters.contracts.relation import RelationConfig
from dbt.adapters.factory import get_adapter, register_adapter
from dbt.config.runtime import RuntimeConfig
from dbt.context.providers import generate_runtime_macro_context
from dbt.contracts.graph.manifest import Manifest
from dbt.contracts.graph.nodes import (
    ColumnInfo,
    ManifestSQLNode,
    ModelNode,
    ResultNode,
    SeedNode,
    SourceDefinition,
)
from dbt.contracts.results import CatalogArtifact, CatalogResults, ColumnMetadata
from dbt.contracts.results import (
    CatalogKey as TableRef,
)
from dbt.mp_context import get_mp_context
from dbt.node_types import NodeType
from dbt.parser.manifest import ManifestLoader, process_node
from dbt.parser.sql import SqlBlockParser, SqlMacroParser
from dbt.task.docs.generate import Catalog
from dbt.task.sql import SqlCompileRunner
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
    "create_yaml_instance",
    "YamlRefactorSettings",
    "YamlRefactorContext",
    "compile_sql_code",
    "execute_sql_code",
    "normalize_column_name",
    "get_table_ref",
    "get_columns",
    "create_missing_source_yamls",
    "get_current_yaml_path",
    "get_target_yaml_path",
    "build_yaml_file_mapping",
    "commit_yamls",
    "draft_restructure_delta_plan",
    "pretty_print_plan",
    "sync_node_to_yaml",
    "apply_restructure_plan",
    "inherit_upstream_column_knowledge",
    "inject_missing_columns",
    "remove_columns_not_in_database",
    "sort_columns_as_in_database",
    "sort_columns_alphabetically",
    "synchronize_data_types",
]

disable_tracking()

T = t.TypeVar("T")

EMPTY_STRING = ""
"""A null string constant for use in placeholder lists, this is always considered undocumented"""


# Basic DBT Setup
# ===============


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
    target: str | None = None
    profile: str | None = None
    threads: int = 1
    single_threaded: bool = True
    vars: dict[str, t.Any] = field(default_factory=dict)
    quiet: bool = True

    def __post_init__(self) -> None:
        logger.debug(":bookmark_tabs: Setting invocation context with environment variables.")
        set_invocation_context(get_env())
        if self.threads > 1:
            self.single_threaded = False
        elif self.threads < 1:
            raise ValueError("DbtConfiguration.threads must be >= 1")


def config_to_namespace(cfg: DbtConfiguration) -> argparse.Namespace:
    """Convert a DbtConfiguration into a dbt-friendly argparse.Namespace."""
    logger.debug(":blue_book: Converting DbtConfiguration to argparse.Namespace => %s", cfg)
    return argparse.Namespace(
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


@dataclass
class DbtProjectContext:
    """A data object that includes references to:

    - The loaded dbt config
    - The manifest
    - The sql/macro parsers

    With mutexes for thread safety. The adapter is lazily instantiated and has a TTL which allows
    for re-use across multiple operations in long-running processes. (is the idea)
    """

    args: argparse.Namespace
    config: RuntimeConfig
    manifest: Manifest
    sql_parser: SqlBlockParser
    macro_parser: SqlMacroParser
    connection_ttl: float = 3600.0

    _adapter_mutex: threading.Lock = field(default_factory=threading.Lock, init=False)
    _manifest_mutex: threading.Lock = field(default_factory=threading.Lock, init=False)
    _adapter: BaseAdapter | None = field(default=None, init=False)
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
                adapter = _instantiate_adapter(self.config)
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


def _instantiate_adapter(runtime_config: RuntimeConfig) -> BaseAdapter:
    """Instantiate a dbt adapter based on the runtime configuration."""
    logger.debug(":mag: Registering adapter for runtime config => %s", runtime_config)
    register_adapter(runtime_config, get_mp_context())
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

    logger.info(":bookmark_tabs: Instantiating adapter as part of project context creation.")
    adapter = _instantiate_adapter(runtime_cfg)
    setattr(runtime_cfg, "adapter", adapter)
    loader = ManifestLoader(
        runtime_cfg,
        runtime_cfg.load_dependencies(),
    )
    manifest = loader.load()
    manifest.build_flat_graph()
    logger.info(":arrows_counterclockwise: Loaded the dbt project manifest!")

    adapter.set_macro_resolver(manifest)

    sql_parser = SqlBlockParser(runtime_cfg, manifest, runtime_cfg)
    macro_parser = SqlMacroParser(runtime_cfg, manifest)

    logger.info(":sparkles: DbtProjectContext successfully created!")
    return DbtProjectContext(
        args=args,
        config=runtime_cfg,
        manifest=manifest,
        sql_parser=sql_parser,
        macro_parser=macro_parser,
    )


def _reload_manifest(context: DbtProjectContext) -> None:
    """Reload the dbt project manifest. Useful for picking up mutations."""
    logger.info(":arrows_counterclockwise: Reloading the dbt project manifest!")
    loader = ManifestLoader(context.config, context.config.load_dependencies())
    manifest = loader.load()
    manifest.build_flat_graph()
    context.adapter.set_macro_resolver(manifest)
    context.manifest = manifest
    logger.info(":white_check_mark: Manifest reloaded => %s", context.manifest.metadata)


# YAML + File Data
# ================


def create_yaml_instance(
    indent_mapping: int = 2,
    indent_sequence: int = 4,
    indent_offset: int = 2,
    width: int = 800,
    preserve_quotes: bool = True,
    default_flow_style: bool = False,
    encoding: str = "utf-8",
) -> ruamel.yaml.YAML:
    """Returns a ruamel.yaml.YAML instance configured with the provided settings."""
    logger.debug(":notebook: Creating ruamel.yaml.YAML instance with custom formatting.")
    y = ruamel.yaml.YAML()
    y.indent(mapping=indent_mapping, sequence=indent_sequence, offset=indent_offset)
    y.width = width
    y.preserve_quotes = preserve_quotes
    y.default_flow_style = default_flow_style
    y.encoding = encoding
    logger.debug(":notebook: YAML instance created => %s", y)
    return y


@dataclass
class SchemaFileLocation:
    """Describes the current and target location of a schema file."""

    target: Path
    current: Path | None = None
    node_type: NodeType = NodeType.Model

    @property
    def is_valid(self) -> bool:
        """Check if the current and target locations are valid."""
        valid = self.current == self.target
        logger.debug(":white_check_mark: Checking if schema file location is valid => %s", valid)
        return valid


@dataclass
class SchemaFileMigration:
    """Describes a schema file migration operation."""

    output: dict[str, t.Any] = field(
        default_factory=lambda: {"version": 2, "models": [], "sources": []}
    )
    supersede: dict[Path, list[ResultNode]] = field(default_factory=dict)


@dataclass
class RestructureOperation:
    """Represents a single operation to perform on a YAML file.

    This might be CREATE, UPDATE, SUPERSEDE, etc. In a more advanced approach,
    we might unify multiple steps under a single operation with sub-operations.
    """

    file_path: Path
    content: dict[str, t.Any]
    superseded_paths: dict[Path, list[ResultNode]] = field(default_factory=dict)


@dataclass
class RestructureDeltaPlan:
    """Stores all the operations needed to restructure the project."""

    operations: list[RestructureOperation] = field(default_factory=list)


@dataclass
class YamlRefactorSettings:
    """Settings for yaml based refactoring operations."""

    fqn: list[str] = field(default_factory=list)
    """Filter models to action via a fully qualified name match such as returned by `dbt ls`."""
    models: list[str] = field(default_factory=list)
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
    catalog_path: str | None = None
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

    project: DbtProjectContext
    settings: YamlRefactorSettings = field(default_factory=YamlRefactorSettings)
    pool: ThreadPoolExecutor = field(default_factory=ThreadPoolExecutor)
    yaml_handler: ruamel.yaml.YAML = field(default_factory=create_yaml_instance)
    yaml_handler_lock: threading.Lock = field(default_factory=threading.Lock)

    placeholders: tuple[str, ...] = (
        EMPTY_STRING,
        "Pending further documentation",
        "No description for this column",
        "Not documented",
        "Undefined",
    )

    _mutation_count: int = field(default=0, init=False)
    _catalog: CatalogResults | None = field(default=None, init=False)

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
        c = self.project.config.vars.to_dict()
        toplevel_conf = _find_first(
            [c.get(k, {}) for k in ["dbt-osmosis", "dbt_osmosis"]], lambda v: bool(v), {}
        )
        return toplevel_conf.get("sources", {})

    @property
    def ignore_patterns(self) -> list[str]:
        """The column name ignore patterns from the dbt project config."""
        c = self.project.config.vars.to_dict()
        toplevel_conf = _find_first(
            [c.get(k, {}) for k in ["dbt-osmosis", "dbt_osmosis"]], lambda v: bool(v), {}
        )
        return toplevel_conf.get("column_ignore_patterns", [])

    @property
    def yaml_settings(self) -> dict[str, t.Any]:
        """The column name ignore patterns from the dbt project config."""
        c = self.project.config.vars.to_dict()
        toplevel_conf = _find_first(
            [c.get(k, {}) for k in ["dbt-osmosis", "dbt_osmosis"]], lambda v: bool(v), {}
        )
        return toplevel_conf.get("yaml_settings", {})

    def read_catalog(self) -> CatalogResults | None:
        """Read the catalog file if it exists."""
        logger.debug(":mag: Checking if catalog is already loaded => %s", bool(self._catalog))
        if not self._catalog:
            catalog = _load_catalog(self.settings)
            if not catalog and self.settings.create_catalog_if_not_exists:
                logger.info(
                    ":bookmark_tabs: No existing catalog found, generating new catalog.json."
                )
                catalog = _generate_catalog(self.project)
            self._catalog = catalog
        return self._catalog

    def __post_init__(self) -> None:
        logger.debug(":green_book: Running post-init for YamlRefactorContext.")
        if EMPTY_STRING not in self.placeholders:
            self.placeholders = (EMPTY_STRING, *self.placeholders)
        for setting, val in self.yaml_settings.items():
            setattr(self.yaml_handler, setting, val)


def _load_catalog(settings: YamlRefactorSettings) -> CatalogResults | None:
    """Load the catalog file if it exists and return a CatalogResults instance."""
    logger.debug(":mag: Attempting to load catalog from => %s", settings.catalog_path)
    if not settings.catalog_path:
        return None
    fp = Path(settings.catalog_path)
    if not fp.exists():
        logger.warning(":warning: Catalog path => %s does not exist.", fp)
        return None
    logger.info(":books: Loading existing catalog => %s", fp)
    return t.cast(CatalogResults, CatalogArtifact.from_dict(json.loads(fp.read_text())))  # pyright: ignore[reportInvalidCast]


# NOTE: this is mostly adapted from dbt-core with some cruft removed, strict pyright is not a fan of dbt's shenanigans
def _generate_catalog(context: DbtProjectContext) -> CatalogResults | None:
    """Generate the dbt catalog file for the project."""
    logger.info(
        ":books: Generating a new catalog for the project => %s", context.config.project_name
    )
    catalogable_nodes = chain(
        [
            t.cast(RelationConfig, node)  # pyright: ignore[reportInvalidCast]
            for node in context.manifest.nodes.values()
            if node.is_relational and not node.is_ephemeral_model
        ],
        [t.cast(RelationConfig, node) for node in context.manifest.sources.values()],  # pyright: ignore[reportInvalidCast]
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
    artifact = CatalogArtifact.from_results(  # pyright: ignore[reportAttributeAccessIssue]
        nodes=nodes,
        sources=sources,
        generated_at=datetime.now(timezone.utc),
        compile_results=None,
        errors=errors,
    )
    artifact_path = Path(context.config.project_target_path, "catalog.json")
    logger.info(":bookmark_tabs: Writing fresh catalog => %s", artifact_path)
    artifact.write(str(artifact_path.resolve()))  # Cache it, same as dbt
    return t.cast(CatalogResults, artifact)


# Basic compile & execute
# =======================


def _has_jinja(code: str) -> bool:
    """Check if a code string contains jinja tokens."""
    logger.debug(":crystal_ball: Checking if code snippet has Jinja => %s", code[:50] + "...")
    return any(token in code for token in ("{{", "}}", "{%", "%}", "{#", "#}"))


def compile_sql_code(context: DbtProjectContext, raw_sql: str) -> ManifestSQLNode:
    """Compile jinja SQL using the context's manifest and adapter."""
    logger.info(":zap: Compiling SQL code. Possibly with jinja => %s", raw_sql[:75] + "...")
    tmp_id = str(uuid.uuid4())
    with context.manifest_mutex:
        key = f"{NodeType.SqlOperation}.{context.config.project_name}.{tmp_id}"
        _ = context.manifest.nodes.pop(key, None)

        node = context.sql_parser.parse_remote(raw_sql, tmp_id)
        if not _has_jinja(raw_sql):
            logger.debug(":scroll: No jinja found in the raw SQL, skipping compile steps.")
            return node
        process_node(context.config, context.manifest, node)
        compiled_node = SqlCompileRunner(
            context.config,
            context.adapter,
            node=node,
            node_index=1,
            num_nodes=1,
        ).compile(context.manifest)

        _ = context.manifest.nodes.pop(key, None)

    logger.info(":sparkles: Compilation complete.")
    return compiled_node


def execute_sql_code(context: DbtProjectContext, raw_sql: str) -> tuple[AdapterResponse, Table]:
    """Execute jinja SQL using the context's manifest and adapter."""
    logger.info(":running: Attempting to execute SQL => %s", raw_sql[:75] + "...")
    if _has_jinja(raw_sql):
        comp = compile_sql_code(context, raw_sql)
        sql_to_exec = comp.compiled_code or comp.raw_code
    else:
        sql_to_exec = raw_sql

    resp, table = context.adapter.execute(sql_to_exec, auto_begin=False, fetch=True)
    logger.info(":white_check_mark: SQL execution complete => %s rows returned.", len(table.rows))  # pyright: ignore[reportUnknownArgumentType]
    return resp, table


# Node filtering
# ==============


def _is_fqn_match(node: ResultNode, fqns: list[str]) -> bool:
    """Filter models based on the provided fully qualified name matching on partial segments."""
    logger.debug(":mag_right: Checking if node => %s matches any FQNs => %s", node.unique_id, fqns)
    for fqn_str in fqns:
        parts = fqn_str.split(".")
        segment_match = len(node.fqn[1:]) >= len(parts) and all(
            left == right for left, right in zip(parts, node.fqn[1:])
        )
        if segment_match:
            logger.debug(":white_check_mark: FQN matched => %s", fqn_str)
            return True
    return False


def _is_file_match(node: ResultNode, paths: list[str]) -> bool:
    """Check if a node's file path matches any of the provided file paths or names."""
    node_path = _get_node_path(node)
    for model in paths:
        if node.name == model:
            logger.debug(":white_check_mark: Name match => %s", model)
            return True
        try_path = Path(model).resolve()
        if try_path.is_dir():
            if node_path and try_path in node_path.parents:
                logger.debug(":white_check_mark: Directory path match => %s", model)
                return True
        elif try_path.is_file():
            if node_path and try_path == node_path:
                logger.debug(":white_check_mark: File path match => %s", model)
                return True
    return False


def _get_node_path(node: ResultNode) -> Path | None:
    """Return the path to the node's original file if available."""
    if node.original_file_path and hasattr(node, "root_path"):
        path = Path(getattr(node, "root_path"), node.original_file_path).resolve()
        logger.debug(":file_folder: Resolved node path => %s", path)
        return path
    return None


def _iter_candidate_nodes(
    context: YamlRefactorContext,
) -> Iterator[tuple[str, ResultNode]]:
    """Iterate over the models in the dbt project manifest applying the filter settings."""
    logger.debug(
        ":mag: Filtering nodes (models/sources/seeds) with user-specified settings => %s",
        context.settings,
    )

    def f(node: ResultNode) -> bool:
        """Closure to filter models based on the context settings."""
        if node.resource_type not in (NodeType.Model, NodeType.Source, NodeType.Seed):
            return False
        if node.package_name != context.project.config.project_name:
            return False
        if node.resource_type == NodeType.Model and node.config.materialized == "ephemeral":
            return False
        if context.settings.models:
            if not _is_file_match(node, context.settings.models):
                return False
        if context.settings.fqn:
            if not _is_fqn_match(node, context.settings.fqn):
                return False
        logger.debug(":white_check_mark: Node => %s passed filtering logic.", node.unique_id)
        return True

    items = chain(context.project.manifest.nodes.items(), context.project.manifest.sources.items())
    for uid, dbt_node in items:
        if f(dbt_node):
            yield uid, dbt_node


# Introspection
# =============


@t.overload
def _find_first(coll: Iterable[T], predicate: t.Callable[[T], bool], default: T) -> T: ...


@t.overload
def _find_first(
    coll: Iterable[T], predicate: t.Callable[[T], bool], default: None = ...
) -> T | None: ...


def _find_first(
    coll: Iterable[T], predicate: t.Callable[[T], bool], default: T | None = None
) -> T | None:
    """Find the first item in a container that satisfies a predicate."""
    for item in coll:
        if predicate(item):
            return item
    return default


def normalize_column_name(column: str, credentials_type: str) -> str:
    """Apply case normalization to a column name based on the credentials type."""
    if credentials_type == "snowflake" and column.startswith('"') and column.endswith('"'):
        logger.debug(":snowflake: Column name found with double-quotes => %s", column)
        return column.strip('"')
    if credentials_type == "snowflake":
        return column.upper()
    return column


def _maybe_use_precise_dtype(
    col: BaseColumn, settings: YamlRefactorSettings, node: ResultNode | None = None
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
    return col.dtype


def get_table_ref(node: ResultNode | BaseRelation) -> TableRef:
    """Make an appropriate table ref for a dbt node or relation."""
    if isinstance(node, BaseRelation):
        assert node.schema, "Schema must be set for a BaseRelation to generate a TableRef"
        assert node.identifier, "Identifier must be set for a BaseRelation to generate a TableRef"
        return TableRef(node.database, node.schema, node.identifier)
    elif node.resource_type == NodeType.Source:
        return TableRef(node.database, node.schema, node.identifier or node.name)
    else:
        return TableRef(node.database, node.schema, node.name)


_COLUMN_LIST_CACHE = {}
"""Cache for column lists to avoid redundant introspection."""


def get_columns(context: YamlRefactorContext, ref: TableRef) -> dict[str, ColumnMetadata]:
    """Equivalent to get_columns_meta in old code but directly referencing a key, not a node."""
    if ref in _COLUMN_LIST_CACHE:
        logger.debug(":blue_book: Column list cache HIT => %s", ref)
        return _COLUMN_LIST_CACHE[ref]

    logger.info(":mag_right: Collecting columns for table => %s", ref)
    normalized_cols = OrderedDict()
    offset = 0

    def process_column(col: BaseColumn | ColumnMetadata):
        nonlocal offset
        if any(re.match(b, col.name) for b in context.ignore_patterns):
            logger.debug(
                ":no_entry_sign: Skipping column => %s due to skip pattern match.", col.name
            )
            return
        normalized = normalize_column_name(col.name, context.project.config.credentials.type)
        if not isinstance(col, ColumnMetadata):
            dtype = _maybe_use_precise_dtype(col, context.settings)
            col = ColumnMetadata(
                name=normalized, type=dtype, index=offset, comment=getattr(col, "comment", None)
            )
        normalized_cols[normalized] = col
        offset += 1
        if hasattr(col, "flatten"):
            for struct_field in t.cast(Iterable[BaseColumn], getattr(col, "flatten")()):
                process_column(struct_field)

    if catalog := context.read_catalog():
        logger.debug(":blue_book: Catalog found => Checking for ref => %s", ref)
        catalog_entry = _find_first(
            chain(catalog.nodes.values(), catalog.sources.values()), lambda c: c.key() == ref
        )
        if catalog_entry:
            logger.info(":books: Found catalog entry for => %s. Using it to process columns.", ref)
            for column in catalog_entry.columns.values():
                process_column(column)
            return normalized_cols

    relation: BaseRelation | None = context.project.adapter.get_relation(*ref)
    if relation is None:
        logger.warning(":warning: No relation found => %s", ref)
        return normalized_cols

    try:
        logger.info(":mag: Introspecting columns in warehouse for => %s", relation)
        for column in t.cast(
            Iterable[BaseColumn], context.project.adapter.get_columns_in_relation(relation)
        ):
            process_column(column)
    except Exception as ex:
        logger.warning(":warning: Could not introspect columns for %s: %s", ref, ex)

    _COLUMN_LIST_CACHE[ref] = normalized_cols
    return normalized_cols


def _get_setting_for_node(
    opt: str,
    /,
    node: ResultNode | None = None,
    col: str | None = None,
    *,
    fallback: t.Any | None = None,
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


# Restructuring Logic
# ===================


def create_missing_source_yamls(context: YamlRefactorContext) -> None:
    """Create source files for sources defined in the dbt_project.yml dbt-osmosis var which don't exist as nodes.

    This is a useful preprocessing step to ensure that all sources are represented in the dbt project manifest. We
    do not have rich node information for non-existent sources, hence the alternative codepath here to bootstrap them.
    """
    logger.info(":factory: Creating missing source YAMLs (if any).")
    database: str = context.project.config.credentials.database
    lowercase: bool = context.settings.output_to_lower

    did_side_effect: bool = False
    for source, spec in context.source_definitions.items():
        if isinstance(spec, str):
            schema = source
            src_yaml_path = spec
        elif isinstance(spec, dict):
            database = t.cast(str, spec.get("database", database))
            schema = t.cast(str, spec.get("schema", source))
            src_yaml_path = t.cast(str, spec["path"])
        else:
            continue

        if _find_first(
            context.project.manifest.sources.values(), lambda s: s.source_name == source
        ):
            logger.debug(
                ":white_check_mark: Source => %s already exists in the manifest, skipping creation.",
                source,
            )
            continue

        src_yaml_path = Path(
            context.project.config.project_root,
            context.project.config.model_paths[0],
            src_yaml_path.lstrip(os.sep),
        )

        def _describe(rel: BaseRelation) -> dict[str, t.Any]:
            s = {
                "name": rel.identifier,
                "description": "",
                "columns": [
                    {
                        "name": name.lower() if lowercase else name,
                        "description": meta.comment or "",
                        "data_type": meta.type.lower() if lowercase else meta.type,
                    }
                    for name, meta in get_columns(context, get_table_ref(rel)).items()
                ],
            }
            if context.settings.skip_add_data_types:
                for col in t.cast(list[dict[str, t.Any]], s["columns"]):
                    _ = col.pop("data_type", None)
            return s

        tables = [
            schema
            for schema in context.pool.map(
                _describe,
                context.project.adapter.list_relations(database=database, schema=schema),
            )
        ]
        source = {"name": source, "database": database, "schema": schema, "tables": tables}

        src_yaml_path.parent.mkdir(parents=True, exist_ok=True)
        with src_yaml_path.open("w") as f:
            logger.info(":books: Injecting new source => %s => %s", source["name"], src_yaml_path)
            context.yaml_handler.dump({"version": 2, "sources": [source]}, f)
            context.register_mutations(1)

        did_side_effect = True

    if did_side_effect:
        logger.info(
            ":arrows_counterclockwise: Some new sources were created, reloading the project."
        )
        _reload_manifest(context.project)


class MissingOsmosisConfig(Exception):
    """Raised when an osmosis configuration is missing."""


def _get_yaml_path_template(context: YamlRefactorContext, node: ResultNode) -> str | None:
    """Get the yaml path template for a dbt model or source node."""
    if node.resource_type == NodeType.Source:
        def_or_path = context.source_definitions.get(node.source_name)
        if isinstance(def_or_path, dict):
            return def_or_path.get("path")
        return def_or_path
    conf = [
        c.get(k)
        for k in ("dbt-osmosis", "dbt_osmosis")
        for c in (node.config.extra, node.unrendered_config)
    ]
    path_template = _find_first(t.cast(list[str | None], conf), lambda v: v is not None)
    if not path_template:
        raise MissingOsmosisConfig(
            f"Config key `dbt-osmosis: <path>` not set for model {node.name}"
        )
    logger.debug(":gear: Resolved YAML path template => %s", path_template)
    return path_template


def get_current_yaml_path(context: YamlRefactorContext, node: ResultNode) -> Path | None:
    """Get the current yaml path for a dbt model or source node."""
    if node.resource_type in (NodeType.Model, NodeType.Seed) and getattr(node, "patch_path", None):
        path = Path(context.project.config.project_root).joinpath(
            t.cast(str, node.patch_path).partition("://")[-1]
        )
        logger.debug(":page_facing_up: Current YAML path => %s", path)
        return path
    if node.resource_type == NodeType.Source:
        path = Path(context.project.config.project_root, node.path)
        logger.debug(":page_facing_up: Current YAML path => %s", path)
        return path
    return None


def get_target_yaml_path(context: YamlRefactorContext, node: ResultNode) -> Path:
    """Get the target yaml path for a dbt model or source node."""
    tpl = _get_yaml_path_template(context, node)
    if not tpl:
        logger.warning(":warning: No path template found for => %s", node.unique_id)
        return Path(context.project.config.project_root, node.original_file_path)

    rendered = tpl.format(node=node, model=node.name, parent=node.fqn[-2])
    segments: list[Path | str] = []

    if node.resource_type == NodeType.Source:
        segments.append(context.project.config.model_paths[0])
    else:
        segments.append(Path(node.original_file_path).parent)

    if not (rendered.endswith(".yml") or rendered.endswith(".yaml")):
        rendered += ".yml"
    segments.append(rendered)

    path = Path(context.project.config.project_root, *segments)
    logger.debug(":star2: Target YAML path => %s", path)
    return path


def build_yaml_file_mapping(
    context: YamlRefactorContext, create_missing_sources: bool = False
) -> dict[str, SchemaFileLocation]:
    """Build a mapping of dbt model and source nodes to their current and target yaml paths."""
    logger.info(
        ":globe_with_meridians: Building YAML file mapping. create_missing_sources => %s",
        create_missing_sources,
    )

    if create_missing_sources:
        create_missing_source_yamls(context)

    out_map: dict[str, SchemaFileLocation] = {}
    for uid, node in _iter_candidate_nodes(context):
        current_path = get_current_yaml_path(context, node)
        out_map[uid] = SchemaFileLocation(
            target=get_target_yaml_path(context, node).resolve(),
            current=current_path.resolve() if current_path else None,
            node_type=node.resource_type,
        )

    logger.debug(":card_index_dividers: Built YAML file mapping => %s", out_map)
    return out_map


_YAML_BUFFER_CACHE: dict[Path, t.Any] = {}
"""Cache for yaml file buffers to avoid redundant disk reads/writes and simplify edits."""


def _read_yaml(context: YamlRefactorContext, path: Path) -> dict[str, t.Any]:
    """Read a yaml file from disk. Adds an entry to the buffer cache so all operations on a path are consistent."""
    if path not in _YAML_BUFFER_CACHE:
        if not path.is_file():
            logger.debug(":warning: Path => %s is not a file. Returning empty doc.", path)
            return {}
        logger.debug(":open_file_folder: Reading YAML doc => %s", path)
        with context.yaml_handler_lock:
            _YAML_BUFFER_CACHE[path] = t.cast(dict[str, t.Any], context.yaml_handler.load(path))
    return _YAML_BUFFER_CACHE[path]


def _write_yaml(context: YamlRefactorContext, path: Path, data: dict[str, t.Any]) -> None:
    """Write a yaml file to disk and register a mutation with the context. Clears the path from the buffer cache."""
    logger.debug(":page_with_curl: Attempting to write YAML to => %s", path)
    if not context.settings.dry_run:
        with context.yaml_handler_lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            original = path.read_bytes() if path.is_file() else b""
            context.yaml_handler.dump(data, staging := io.BytesIO())
            modified = staging.getvalue()
            if modified != original:
                logger.info(":writing_hand: Writing changes to => %s", path)
                with path.open("wb") as f:
                    _ = f.write(modified)
                    context.register_mutations(1)
            else:
                logger.debug(":white_check_mark: Skipping write => %s (no changes)", path)
            del staging
        if path in _YAML_BUFFER_CACHE:
            del _YAML_BUFFER_CACHE[path]


def commit_yamls(context: YamlRefactorContext) -> None:
    """Commit all files in the yaml buffer cache to disk. Clears the buffer cache and registers mutations."""
    logger.info(":inbox_tray: Committing all YAMLs from buffer cache to disk.")
    if not context.settings.dry_run:
        with context.yaml_handler_lock:
            for path in list(_YAML_BUFFER_CACHE.keys()):
                original = path.read_bytes() if path.is_file() else b""
                context.yaml_handler.dump(_YAML_BUFFER_CACHE[path], staging := io.BytesIO())
                modified = staging.getvalue()
                if modified != original:
                    logger.info(":writing_hand: Writing => %s", path)
                    with path.open("wb") as f:
                        logger.info(f"Writing {path}")
                        _ = f.write(modified)
                        context.register_mutations(1)
                else:
                    logger.debug(":white_check_mark: Skipping => %s (no changes)", path)
                del _YAML_BUFFER_CACHE[path]


def _generate_minimal_model_yaml(node: ModelNode | SeedNode) -> dict[str, t.Any]:
    """Generate a minimal model yaml for a dbt model node."""
    logger.debug(":baby: Generating minimal yaml for Model/Seed => %s", node.name)
    return {"name": node.name, "columns": []}


def _generate_minimal_source_yaml(node: SourceDefinition) -> dict[str, t.Any]:
    """Generate a minimal source yaml for a dbt source node."""
    logger.debug(":baby: Generating minimal yaml for Source => %s", node.name)
    return {"name": node.source_name, "tables": [{"name": node.name, "columns": []}]}


def _create_operations_for_node(
    context: YamlRefactorContext, uid: str, loc: SchemaFileLocation
) -> list[RestructureOperation]:
    """Create restructure operations for a dbt model or source node."""
    logger.debug(":bricks: Creating restructure operations for => %s", uid)
    node = context.project.manifest.nodes.get(uid) or context.project.manifest.sources.get(uid)
    if not node:
        logger.warning(":warning: Node => %s not found in manifest.", uid)
        return []

    # If loc.current is None => we are generating a brand new file
    # If loc.current => we unify it with the new location
    ops: list[RestructureOperation] = []

    if loc.current is None:
        logger.info(":sparkles: No current YAML file, building minimal doc => %s", uid)
        if isinstance(node, (ModelNode, SeedNode)):
            minimal = _generate_minimal_model_yaml(node)
            ops.append(
                RestructureOperation(
                    file_path=loc.target,
                    content={"version": 2, f"{node.resource_type}s": [minimal]},
                )
            )
        else:
            minimal = _generate_minimal_source_yaml(t.cast(SourceDefinition, node))
            ops.append(
                RestructureOperation(
                    file_path=loc.target,
                    content={"version": 2, "sources": [minimal]},
                )
            )
    else:
        existing = _read_yaml(context, loc.current)
        injectable: dict[str, t.Any] = {"version": 2}
        injectable.setdefault("models", [])
        injectable.setdefault("sources", [])
        injectable.setdefault("seeds", [])
        if loc.node_type == NodeType.Model:
            assert isinstance(node, ModelNode)
            for obj in existing.get("models", []):
                if obj["name"] == node.name:
                    injectable["models"].append(obj)
                    break
        elif loc.node_type == NodeType.Source:
            assert isinstance(node, SourceDefinition)
            for src in existing.get("sources", []):
                if src["name"] == node.source_name:
                    injectable["sources"].append(src)
                    break
        elif loc.node_type == NodeType.Seed:
            assert isinstance(node, SeedNode)
            for seed in existing.get("seeds", []):
                if seed["name"] == node.name:
                    injectable["seeds"].append(seed)
        ops.append(
            RestructureOperation(
                file_path=loc.target,
                content=injectable,
                superseded_paths={loc.current: [node]},
            )
        )
    return ops


def draft_restructure_delta_plan(context: YamlRefactorContext) -> RestructureDeltaPlan:
    """Draft a restructure plan for the dbt project."""
    logger.info(":bulb: Drafting restructure delta plan for the project.")
    plan = RestructureDeltaPlan()
    lock = threading.Lock()

    def _job(uid: str, loc: SchemaFileLocation) -> None:
        ops = _create_operations_for_node(context, uid, loc)
        with lock:
            plan.operations.extend(ops)

    futs: list[Future[None]] = []
    for uid, loc in build_yaml_file_mapping(context).items():
        if not loc.is_valid:
            futs.append(context.pool.submit(_job, uid, loc))
    done, _ = wait(futs, return_when=FIRST_EXCEPTION)
    for fut in done:
        exc = fut.exception()
        if exc:
            logger.error(":bomb: Error encountered while drafting plan => %s", exc)
            raise exc
    logger.info(":star2: Draft plan creation complete => %s operations", len(plan.operations))
    return plan


def pretty_print_plan(plan: RestructureDeltaPlan) -> None:
    """Pretty print the restructure plan for the dbt project."""
    logger.info(":mega: Restructure plan includes => %s operations.", len(plan.operations))
    for op in plan.operations:
        str_content = str(op.content)[:80] + "..."
        logger.info(":sparkles: Processing => %s", str_content)
        if not op.superseded_paths:
            logger.info(":blue_book: CREATE or MERGE => %s", op.file_path)
        else:
            old_paths = [p.name for p in op.superseded_paths.keys()] or ["UNKNOWN"]
            logger.info(":blue_book: %s -> %s", old_paths, op.file_path)


def _remove_models(existing_doc: dict[str, t.Any], nodes: list[ResultNode]) -> None:
    """Clean up the existing yaml doc by removing models superseded by the restructure plan."""
    logger.debug(":scissors: Removing superseded models => %s", [n.name for n in nodes])
    to_remove = {n.name for n in nodes if n.resource_type == NodeType.Model}
    keep = []
    for section in existing_doc.get("models", []):
        if section.get("name") not in to_remove:
            keep.append(section)
    existing_doc["models"] = keep


def _remove_seeds(existing_doc: dict[str, t.Any], nodes: list[ResultNode]) -> None:
    """Clean up the existing yaml doc by removing models superseded by the restructure plan."""
    logger.debug(":scissors: Removing superseded seeds => %s", [n.name for n in nodes])
    to_remove = {n.name for n in nodes if n.resource_type == NodeType.Seed}
    keep = []
    for section in existing_doc.get("seeds", []):
        if section.get("name") not in to_remove:
            keep.append(section)
    existing_doc["seeds"] = keep


def _remove_sources(existing_doc: dict[str, t.Any], nodes: list[ResultNode]) -> None:
    """Clean up the existing yaml doc by removing sources superseded by the restructure plan."""
    to_remove_sources = {
        (n.source_name, n.name) for n in nodes if n.resource_type == NodeType.Source
    }
    logger.debug(":scissors: Removing superseded sources => %s", sorted(to_remove_sources))
    keep_sources = []
    for section in existing_doc.get("sources", []):
        keep_tables = []
        for tbl in section.get("tables", []):
            if (section["name"], tbl["name"]) not in to_remove_sources:
                keep_tables.append(tbl)
        if keep_tables:
            section["tables"] = keep_tables
            keep_sources.append(section)
    existing_doc["sources"] = keep_sources


def _sync_doc_section(
    context: YamlRefactorContext, node: ResultNode, doc_section: dict[str, t.Any]
) -> None:
    """Helper function that overwrites 'doc_section' with data from 'node'.

    This includes columns, description, meta, tags, etc.
    We assume node is the single source of truth, so doc_section is replaced.
    """
    logger.debug(":arrows_counterclockwise: Syncing doc_section with node => %s", node.unique_id)
    if node.description:
        doc_section["description"] = node.description
    else:
        doc_section.pop("description", None)

    current_columns: list[dict[str, t.Any]] = doc_section.setdefault("columns", [])
    incoming_columns: list[dict[str, t.Any]] = []

    current_map = {}
    for c in current_columns:
        norm_name = normalize_column_name(c["name"], context.project.config.credentials.type)
        current_map[norm_name] = c

    for name, meta in node.columns.items():
        cdict = meta.to_dict()
        cdict["name"] = name
        norm_name = normalize_column_name(name, context.project.config.credentials.type)

        current_yaml = t.cast(dict[str, t.Any], current_map.get(norm_name, {}))
        merged = dict(current_yaml)

        skip_add_types = _get_setting_for_node(
            "skip-add-data-types", node, name, fallback=context.settings.skip_add_data_types
        )
        for k, v in cdict.items():
            if k == "description" and not v:
                merged.pop("description", None)
            elif k == "data_type" and skip_add_types and merged.get("data_type") is None:
                pass
            else:
                merged[k] = v

        if not merged.get("description"):
            merged.pop("description", None)
        if merged.get("tags") == []:
            merged.pop("tags", None)
        if merged.get("meta") == {}:
            merged.pop("meta", None)

        for k in list(merged.keys()):
            if not merged[k]:
                merged.pop(k)

        if _get_setting_for_node(
            "output-to-lower", node, name, fallback=context.settings.output_to_lower
        ):
            merged["name"] = merged["name"].lower()

        incoming_columns.append(merged)

    doc_section["columns"] = incoming_columns


def sync_node_to_yaml(
    context: YamlRefactorContext, node: ResultNode | None = None, *, commit: bool = True
) -> None:
    """Synchronize a single node's columns, description, tags, meta, etc. from the manifest into its corresponding YAML file.

    We assume the manifest node is the single source of truth, so the YAML file is overwritten to match.

    - If the YAML file doesn't exist yet, we create it with minimal structure.
    - If the YAML file exists, we read it from the file/ cache, locate the node's section,
      and then overwrite that section to match the node's current columns, meta, etc.

    This is a one-way sync:
        Manifest Node => YAML

    All changes to the Node (columns, metadata, etc.) should happen before calling this function.
    """
    if node is None:
        logger.info(":wave: No single node specified; synchronizing all matched nodes.")
        for _, node in _iter_candidate_nodes(context):
            sync_node_to_yaml(context, node, commit=commit)
        return

    current_path = get_current_yaml_path(context, node)
    if not current_path or not current_path.exists():
        logger.debug(
            ":warning: Current path does not exist => %s. Using target path instead.", current_path
        )
        current_path = get_target_yaml_path(context, node)

    doc: dict[str, t.Any] = _read_yaml(context, current_path)
    if not doc:
        doc = {"version": 2}

    if node.resource_type == NodeType.Source:
        resource_k = "sources"
    elif node.resource_type == NodeType.Seed:
        resource_k = "seeds"
    else:
        resource_k = "models"

    if node.resource_type == NodeType.Source:
        # The doc structure => sources: [ { "name": <source_name>, "tables": [...]}, ... ]
        # Step A: find or create the source
        doc_source: dict[str, t.Any] | None = None
        for s in doc.setdefault(resource_k, []):
            if s.get("name") == node.source_name:
                doc_source = s
                break
        if not doc_source:
            doc_source = {
                "name": node.source_name,
                "tables": [],
            }
            doc["sources"].append(doc_source)

        # Step B: find or create the table
        doc_table: dict[str, t.Any] | None = None
        for t_ in doc_source["tables"]:
            if t_.get("name") == node.name:
                doc_table = t_
                break
        if not doc_table:
            doc_table = {
                "name": node.name,
                "columns": [],
            }
            doc_source["tables"].append(doc_table)

        # We'll store the columns & description on "doc_table"
        # For source, "description" is stored at table-level in the Node
        _sync_doc_section(context, node, doc_table)

    else:
        # Models or Seeds => doc[ "models" ] or doc[ "seeds" ] is a list of { "name", "description", "columns", ... }
        doc_list = doc.setdefault(resource_k, [])
        doc_obj: dict[str, t.Any] | None = None
        for item in doc_list:
            if item.get("name") == node.name:
                doc_obj = item
                break
        if not doc_obj:
            doc_obj = {
                "name": node.name,
                "columns": [],
            }
            doc_list.append(doc_obj)

        _sync_doc_section(context, node, doc_obj)

    for k in ("models", "sources", "seeds"):
        if len(doc.get(k, [])) == 0:
            _ = doc.pop(k, None)

    if commit:
        logger.info(":inbox_tray: Committing YAML doc changes for => %s", node.unique_id)
        _write_yaml(context, current_path, doc)


def apply_restructure_plan(
    context: YamlRefactorContext, plan: RestructureDeltaPlan, *, confirm: bool = False
) -> None:
    """Apply the restructure plan for the dbt project."""
    if not plan.operations:
        logger.info(":white_check_mark: No changes needed in the restructure plan.")
        return

    if confirm:
        logger.info(":warning: Confirm option set => printing plan and waiting for user input.")
        pretty_print_plan(plan)

    while confirm:
        response = input("Apply the restructure plan? [y/N]: ")
        if response.lower() in ("y", "yes"):
            break
        elif response.lower() in ("n", "no", ""):
            logger.info("Skipping restructure plan.")
            return
        logger.warning(":loudspeaker: Please respond with 'y' or 'n'.")

    for op in plan.operations:
        logger.debug(":arrow_right: Applying restructure operation => %s", op)
        output_doc: dict[str, t.Any] = {"version": 2}
        if op.file_path.exists():
            existing_data = _read_yaml(context, op.file_path)
            output_doc.update(existing_data)

        for key, val in op.content.items():
            if isinstance(val, list):
                output_doc.setdefault(key, []).extend(val)
            elif isinstance(val, dict):
                output_doc.setdefault(key, {}).update(val)
            else:
                output_doc[key] = val

        _write_yaml(context, op.file_path, output_doc)

        for path, nodes in op.superseded_paths.items():
            if path.is_file():
                existing_data = _read_yaml(context, path)

                if "models" in existing_data:
                    _remove_models(existing_data, nodes)
                if "sources" in existing_data:
                    _remove_sources(existing_data, nodes)
                if "seeds" in existing_data:
                    _remove_seeds(existing_data, nodes)

                keys = set(existing_data.keys()) - {"version"}
                if all(len(existing_data.get(k, [])) == 0 for k in keys):
                    if not context.settings.dry_run:
                        path.unlink(missing_ok=True)
                        if path.parent.exists() and not any(path.parent.iterdir()):
                            path.parent.rmdir()
                        if path in _YAML_BUFFER_CACHE:
                            del _YAML_BUFFER_CACHE[path]
                    context.register_mutations(1)
                    logger.info(":heavy_minus_sign: Superseded entire file => %s", path)
                else:
                    _write_yaml(context, path, existing_data)
                    logger.info(
                        ":arrow_forward: Migrated doc from => %s to => %s", path, op.file_path
                    )

    logger.info(
        ":arrows_counterclockwise: Committing all restructure changes and reloading manifest."
    )
    _ = commit_yamls(context), _reload_manifest(context.project)


# Inheritance Logic
# =================


def _build_node_ancestor_tree(
    manifest: Manifest,
    node: ResultNode,
    tree: dict[str, list[str]] | None = None,
    visited: set[str] | None = None,
    depth: int = 1,
) -> dict[str, list[str]]:
    """Build a flat graph of a node and it's ancestors."""
    logger.debug(":seedling: Building ancestor tree/branch for => %s", node.unique_id)
    if tree is None or visited is None:
        visited = set(node.unique_id)
        tree = {"generation_0": [node.unique_id]}
        depth = 1

    if not hasattr(node, "depends_on"):
        return tree

    for dep in getattr(node.depends_on, "nodes", []):
        if not dep.startswith(("model.", "seed.", "source.")):
            continue
        if dep not in visited:
            visited.add(dep)
            member = manifest.nodes.get(dep, manifest.sources.get(dep))
            if member:
                tree.setdefault(f"generation_{depth}", []).append(dep)
                _ = _build_node_ancestor_tree(manifest, member, tree, visited, depth + 1)

    for generation in tree.values():
        generation.sort()  # For deterministic ordering

    return tree


def _get_node_yaml(
    context: YamlRefactorContext, member: ResultNode
) -> MappingProxyType[str, t.Any] | None:
    """Get a read-only view of the parsed YAML for a dbt model or source node."""
    project_dir = Path(context.project.config.project_root)

    if isinstance(member, SourceDefinition):
        if not member.original_file_path:
            return None
        path = project_dir.joinpath(member.original_file_path)
        sources = t.cast(list[dict[str, t.Any]], _read_yaml(context, path).get("sources", []))
        source = _find_first(sources, lambda s: s["name"] == member.source_name, {})
        tables = source.get("tables", [])
        maybe_doc = _find_first(tables, lambda tbl: tbl["name"] == member.name)
        if maybe_doc is not None:
            return MappingProxyType(maybe_doc)

    elif isinstance(member, (ModelNode, SeedNode)):
        if not member.patch_path:
            return None
        path = project_dir.joinpath(member.patch_path.split("://")[-1])
        section = f"{member.resource_type}s"
        models = t.cast(list[dict[str, t.Any]], _read_yaml(context, path).get(section, []))
        maybe_doc = _find_first(models, lambda model: model["name"] == member.name)
        if maybe_doc is not None:
            return MappingProxyType(maybe_doc)

    return None


def _build_column_knowledge_graph(
    context: YamlRefactorContext, node: ResultNode
) -> dict[str, dict[str, t.Any]]:
    """Generate a column knowledge graph for a dbt model or source node."""
    tree = _build_node_ancestor_tree(context.project.manifest, node)
    logger.debug(":family_tree: Node ancestor tree => %s", tree)

    pm = get_plugin_manager()
    node_column_variants: dict[str, list[str]] = {}
    for column_name, _ in node.columns.items():
        variants = node_column_variants.setdefault(column_name, [column_name])
        for v in pm.hook.get_candidates(name=column_name, node=node, context=context.project):
            variants.extend(t.cast(list[str], v))

    column_knowledge_graph: dict[str, dict[str, t.Any]] = {}
    for generation in reversed(sorted(tree.keys())):
        ancestors = tree[generation]
        for ancestor_uid in ancestors:
            ancestor = context.project.manifest.nodes.get(
                ancestor_uid, context.project.manifest.sources.get(ancestor_uid)
            )
            if not isinstance(ancestor, (SourceDefinition, SeedNode, ModelNode)):
                continue

            for name, _ in node.columns.items():
                graph_node = column_knowledge_graph.setdefault(name, {})
                for variant in node_column_variants[name]:
                    incoming = ancestor.columns.get(variant)
                    if incoming is not None:
                        break
                else:
                    continue
                graph_edge = incoming.to_dict()

                if _get_setting_for_node(
                    "add-progenitor-to-meta",
                    node,
                    name,
                    fallback=context.settings.add_progenitor_to_meta,
                ):
                    graph_node.setdefault("meta", {}).setdefault(
                        "osmosis_progenitor", ancestor.unique_id
                    )

                if _get_setting_for_node(
                    "use-unrendered-descriptions",
                    node,
                    name,
                    fallback=context.settings.use_unrendered_descriptions,
                ):
                    raw_yaml = _get_node_yaml(context, ancestor) or {}
                    raw_columns = t.cast(list[dict[str, t.Any]], raw_yaml.get("columns", []))
                    raw_column_metadata = _find_first(
                        raw_columns,
                        lambda c: normalize_column_name(
                            c["name"], context.project.config.credentials.type
                        )
                        in node_column_variants[name],
                        {},
                    )
                    if unrendered_description := raw_column_metadata.get("description"):
                        graph_edge["description"] = unrendered_description

                current_tags = graph_node.get("tags", [])
                if merged_tags := (set(graph_edge.pop("tags", [])) | set(current_tags)):
                    graph_edge["tags"] = list(merged_tags)

                current_meta = graph_node.get("meta", {})
                if merged_meta := {**current_meta, **graph_edge.pop("meta", {})}:
                    graph_edge["meta"] = merged_meta

                for inheritable in _get_setting_for_node(
                    "add-inheritance-for-specified-keys",
                    node,
                    name,
                    fallback=context.settings.add_inheritance_for_specified_keys,
                ):
                    current_val = graph_node.get(inheritable)
                    if incoming_val := graph_edge.pop(inheritable, current_val):
                        graph_edge[inheritable] = incoming_val

                if graph_edge.get("description", EMPTY_STRING) in context.placeholders or (
                    generation == "generation_0"
                    and _get_setting_for_node(
                        "force_inherit_descriptions",
                        node,
                        name,
                        fallback=context.settings.force_inherit_descriptions,
                    )
                ):
                    _ = graph_edge.pop("description", None)
                if graph_edge.get("tags") == []:
                    del graph_edge["tags"]
                if graph_edge.get("meta") == {}:
                    del graph_edge["meta"]
                for k in list(graph_edge.keys()):
                    if graph_edge[k] is None:
                        graph_edge.pop(k)

                graph_node.update(graph_edge)

    return column_knowledge_graph


def inherit_upstream_column_knowledge(
    context: YamlRefactorContext, node: ResultNode | None = None
) -> None:
    """Inherit column level knowledge from the ancestors of a dbt model or source node."""
    if node is None:
        logger.info(":wave: Inheriting column knowledge across all matched nodes.")
        for _ in context.pool.map(
            partial(inherit_upstream_column_knowledge, context),
            (n for _, n in _iter_candidate_nodes(context)),
        ):
            ...
        return

    logger.info(":dna: Inheriting column knowledge for => %s", node.unique_id)

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


def inject_missing_columns(context: YamlRefactorContext, node: ResultNode | None = None) -> None:
    """Add missing columns to a dbt node and it's corresponding yaml section. Changes are implicitly buffered until commit_yamls is called."""
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
    current_columns = {
        normalize_column_name(c.name, context.project.config.credentials.type)
        for c in node.columns.values()
    }
    incoming_columns = get_columns(context, get_table_ref(node))
    for incoming_name, incoming_meta in incoming_columns.items():
        if incoming_name not in node.columns and incoming_name not in current_columns:
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


def remove_columns_not_in_database(
    context: YamlRefactorContext, node: ResultNode | None = None
) -> None:
    """Remove columns from a dbt node and it's corresponding yaml section that are not present in the database. Changes are implicitly buffered until commit_yamls is called."""
    if node is None:
        logger.info(":wave: Removing columns not in DB across all matched nodes.")
        for _ in context.pool.map(
            partial(remove_columns_not_in_database, context),
            (n for _, n in _iter_candidate_nodes(context)),
        ):
            ...
        return
    current_columns = {
        normalize_column_name(c.name, context.project.config.credentials.type)
        for c in node.columns.values()
    }
    incoming_columns = get_columns(context, get_table_ref(node))
    extra_columns = current_columns - set(incoming_columns.keys())
    for extra_column in extra_columns:
        logger.info(
            ":heavy_minus_sign: Removing extra column => %s in node => %s",
            extra_column,
            node.unique_id,
        )
        _ = node.columns.pop(extra_column, None)


def sort_columns_as_in_database(
    context: YamlRefactorContext, node: ResultNode | None = None
) -> None:
    """Sort columns in a dbt node and it's corresponding yaml section as they appear in the database. Changes are implicitly buffered until commit_yamls is called."""
    if node is None:
        logger.info(":wave: Sorting columns as they appear in DB across all matched nodes.")
        for _ in context.pool.map(
            partial(sort_columns_as_in_database, context),
            (n for _, n in _iter_candidate_nodes(context)),
        ):
            ...
        return
    logger.info(":1234: Sorting columns by warehouse order => %s", node.unique_id)
    incoming_columns = get_columns(context, get_table_ref(node))

    def _position(column: dict[str, t.Any]):
        db_info = incoming_columns.get(column["name"])
        if db_info is None:
            return 99999
        return db_info.index

    node.columns = {
        k: v for k, v in sorted(node.columns.items(), key=lambda i: _position(i[1].to_dict()))
    }


def sort_columns_alphabetically(
    context: YamlRefactorContext, node: ResultNode | None = None
) -> None:
    """Sort columns in a dbt node and it's corresponding yaml section alphabetically. Changes are implicitly buffered until commit_yamls is called."""
    if node is None:
        logger.info(":wave: Sorting columns alphabetically across all matched nodes.")
        for _ in context.pool.map(
            partial(sort_columns_alphabetically, context),
            (n for _, n in _iter_candidate_nodes(context)),
        ):
            ...
        return
    logger.info(":alphabet_white: Sorting columns alphabetically => %s", node.unique_id)
    node.columns = {k: v for k, v in sorted(node.columns.items(), key=lambda i: i[0])}


def synchronize_data_types(context: YamlRefactorContext, node: ResultNode | None = None) -> None:
    """Populate data types for columns in a dbt node and it's corresponding yaml section. Changes are implicitly buffered until commit_yamls is called."""
    if node is None:
        logger.info(":wave: Populating data types across all matched nodes.")
        for _ in context.pool.map(
            partial(synchronize_data_types, context), (n for _, n in _iter_candidate_nodes(context))
        ):
            ...
        return
    logger.info(":1234: Synchronizing data types => %s", node.unique_id)
    incoming_columns = get_columns(context, get_table_ref(node))
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
        if inc_c := incoming_columns.get(name):
            is_lower = column.data_type and column.data_type.islower()
            if inc_c.type:
                column.data_type = inc_c.type.lower() if lowercase or is_lower else inc_c.type


# Fuzzy Plugins
# =============

_hookspec = pluggy.HookspecMarker("dbt-osmosis")
hookimpl = pluggy.HookimplMarker("dbt-osmosis")


@_hookspec
def get_candidates(name: str, node: ResultNode, context: DbtProjectContext) -> list[str]:  # pyright: ignore[reportUnusedParameter]
    """Get a list of candidate names for a column."""
    raise NotImplementedError


class FuzzyCaseMatching:
    @hookimpl
    def get_candidates(self, name: str, node: ResultNode, context: DbtProjectContext) -> list[str]:
        """Get a list of candidate names for a column based on case variants."""
        _ = node, context
        variants = [
            name.lower(),  # lowercase
            name.upper(),  # UPPERCASE
            cc := re.sub("_(.)", lambda m: m.group(1).upper(), name),  # camelCase
            cc[0].upper() + cc[1:],  # PascalCase
        ]
        logger.debug(":lower_upper_case: FuzzyCaseMatching variants => %s", variants)
        return variants


class FuzzyPrefixMatching:
    @hookimpl
    def get_candidates(self, name: str, node: ResultNode, context: DbtProjectContext) -> list[str]:
        """Get a list of candidate names for a column excluding a prefix."""
        _ = context
        variants = []
        p = _get_setting_for_node("prefix", node, name)
        if p:
            mut_name = name.removeprefix(p)
            logger.debug(
                ":scissors: FuzzyPrefixMatching => removing prefix '%s' => %s", p, mut_name
            )
            variants.append(mut_name)
        return variants


@lru_cache(maxsize=None)
def get_plugin_manager():
    """Get the pluggy plugin manager for dbt-osmosis."""
    manager = pluggy.PluginManager("dbt-osmosis")
    _ = manager.register(FuzzyCaseMatching())
    _ = manager.register(FuzzyPrefixMatching())
    _ = manager.load_setuptools_entrypoints("dbt-osmosis")
    return manager


# NOTE: usage example of the more FP style module below


def run_example_compilation_flow(c: DbtConfiguration) -> None:
    c.vars["foo"] = "bar"

    context = create_dbt_project_context(c)

    node = compile_sql_code(context, "select '{{ 1+1 }}' as col_{{ var('foo') }}")
    print("Compiled =>", node.compiled_code)

    resp, t_ = execute_sql_code(context, "select '{{ 1+2 }}' as col_{{ var('foo') }}")
    print("Resp =>", resp)

    t_.print_csv()


if __name__ == "__main__":
    # Kitchen sink
    c = DbtConfiguration(
        project_dir="demo_duckdb", profiles_dir="demo_duckdb", vars={"dbt-osmosis": {}}
    )

    run_example_compilation_flow(c)

    project = create_dbt_project_context(c)
    _ = _generate_catalog(project)

    yaml_context = YamlRefactorContext(
        project, settings=YamlRefactorSettings(use_unrendered_descriptions=True)
    )

    plan = draft_restructure_delta_plan(yaml_context)
    steps = (
        (create_missing_source_yamls, (yaml_context,), {}),
        (apply_restructure_plan, (yaml_context, plan), {"confirm": True}),
        (inject_missing_columns, (yaml_context,), {}),
        (remove_columns_not_in_database, (yaml_context,), {}),
        (inherit_upstream_column_knowledge, (yaml_context,), {}),
        (sort_columns_as_in_database, (yaml_context,), {}),
        (synchronize_data_types, (yaml_context,), {}),
        (sync_node_to_yaml, (yaml_context,), {}),
        (commit_yamls, (yaml_context,), {}),
    )
    steps = iter(t.cast(t.Any, steps))

    DONE = object()
    nr = 1
    while (step := next(steps, DONE)) is not DONE:
        step, args, kwargs = step  # pyright: ignore[reportGeneralTypeIssues]
        step(*args, **kwargs)
        logger.info("Completed step %d (%s).", nr, getattr(t.cast(object, step), "__name__"))
        nr += 1
