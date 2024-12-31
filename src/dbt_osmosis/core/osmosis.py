# pyright: reportUnknownVariableType=false, reportPrivateImportUsage=false, reportAny=false, reportUnknownMemberType=false

from __future__ import annotations

import argparse
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
from functools import lru_cache
from itertools import chain
from pathlib import Path

import dbt.flags as dbt_flags
import pluggy
import ruamel.yaml
from dbt.adapters.base.column import Column as BaseColumn
from dbt.adapters.base.impl import BaseAdapter
from dbt.adapters.base.relation import BaseRelation
from dbt.adapters.contracts.connection import AdapterResponse
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
from dbt.contracts.results import CatalogArtifact, ColumnMetadata
from dbt.contracts.results import (
    CatalogKey as TableRef,
)
from dbt.mp_context import get_mp_context
from dbt.node_types import NodeType
from dbt.parser.manifest import ManifestLoader, process_node
from dbt.parser.sql import SqlBlockParser, SqlMacroParser
from dbt.task.sql import SqlCompileRunner
from dbt.tracking import disable_tracking
from dbt_common.clients.system import get_env
from dbt_common.context import set_invocation_context

import dbt_osmosis.core.logger as logger

disable_tracking()

T = t.TypeVar("T")

EMPTY_STRING = ""
"""A null string constant for use in placeholder lists, this is always considered undocumented"""

SKIP_PATTERNS = "_column_ignore_patterns"
"""This key is used to skip certain column name patterns in dbt-osmosis"""


# Basic DBT Setup
# ===============


def discover_project_dir() -> str:
    """Return the directory containing a dbt_project.yml if found, else the current dir."""
    cwd = Path.cwd()
    for p in [cwd] + list(cwd.parents):
        if (p / "dbt_project.yml").exists():
            return str(p.resolve())
    return str(cwd.resolve())


def discover_profiles_dir() -> str:
    """Return the directory containing a profiles.yml if found, else ~/.dbt."""
    if (Path.cwd() / "profiles.yml").exists():
        return str(Path.cwd().resolve())
    return str(Path.home() / ".dbt")


@dataclass
class DbtConfiguration:
    """Configuration for a dbt project."""

    project_dir: str = field(default_factory=discover_project_dir)
    profiles_dir: str = field(default_factory=discover_profiles_dir)
    target: str | None = None
    profile: str | None = None
    threads: int = 1
    single_threaded: bool = True

    _vars: str | dict[str, t.Any] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        set_invocation_context(get_env())
        if self.threads != 1:
            self.single_threaded = False

    @property
    def vars(self) -> dict[str, t.Any]:
        if isinstance(self._vars, str):
            return json.loads(self._vars)
        return self._vars

    @vars.setter
    def vars(self, value: t.Any) -> None:
        if not isinstance(value, (str, dict)):
            raise ValueError("DbtConfiguration.vars must be a string or dict")
        self._vars = value


def config_to_namespace(cfg: DbtConfiguration) -> argparse.Namespace:
    """Convert a DbtConfiguration into a dbt-friendly argparse.Namespace."""
    return argparse.Namespace(
        project_dir=cfg.project_dir,
        profiles_dir=cfg.profiles_dir,
        target=cfg.target,
        profile=cfg.profile,
        threads=cfg.threads,
        single_threaded=cfg.single_threaded,
        vars=cfg.vars,
        which="parse",
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
    adapter_ttl: float = 3600.0

    _adapter_mutex: threading.Lock = field(default_factory=threading.Lock, init=False)
    _manifest_mutex: threading.Lock = field(default_factory=threading.Lock, init=False)
    _adapter: BaseAdapter | None = None
    _adapter_created_at: float = 0.0

    @property
    def is_adapter_expired(self) -> bool:
        """Check if the adapter has expired based on the adapter TTL."""
        return time.time() - self._adapter_created_at > self.adapter_ttl

    @property
    def adapter(self) -> BaseAdapter:
        """Get the adapter instance, creating a new one if the current one has expired."""
        with self._adapter_mutex:
            if not self._adapter or self.is_adapter_expired:
                self._adapter = instantiate_adapter(self.config)
                self._adapter.set_macro_resolver(self.manifest)
                self._adapter_created_at = time.time()
        return self._adapter

    @property
    def manifest_mutex(self) -> threading.Lock:
        """Return the manifest mutex for thread safety."""
        return self._manifest_mutex


def instantiate_adapter(runtime_config: RuntimeConfig) -> BaseAdapter:
    """Instantiate a dbt adapter based on the runtime configuration."""
    register_adapter(runtime_config, get_mp_context())
    adapter = get_adapter(runtime_config)
    adapter.set_macro_context_generator(t.cast(t.Any, generate_runtime_macro_context))
    adapter.connections.set_connection_name("dbt-osmosis")
    return t.cast(BaseAdapter, t.cast(t.Any, adapter))


def create_dbt_project_context(config: DbtConfiguration) -> DbtProjectContext:
    """Build a DbtProjectContext from a DbtConfiguration."""
    args = config_to_namespace(config)
    dbt_flags.set_from_args(args, args)
    runtime_cfg = RuntimeConfig.from_args(args)

    adapter = instantiate_adapter(runtime_cfg)
    setattr(runtime_cfg, "adapter", adapter)
    loader = ManifestLoader(
        runtime_cfg,
        runtime_cfg.load_dependencies(),
    )
    manifest = loader.load()
    manifest.build_flat_graph()

    adapter.set_macro_resolver(manifest)

    sql_parser = SqlBlockParser(runtime_cfg, manifest, runtime_cfg)
    macro_parser = SqlMacroParser(runtime_cfg, manifest)

    return DbtProjectContext(
        args=args,
        config=runtime_cfg,
        manifest=manifest,
        sql_parser=sql_parser,
        macro_parser=macro_parser,
    )


def reload_manifest(context: DbtProjectContext) -> None:
    """Reload the dbt project manifest. Useful for picking up mutations."""
    loader = ManifestLoader(context.config, context.config.load_dependencies())
    manifest = loader.load()
    manifest.build_flat_graph()
    context.adapter.set_macro_resolver(manifest)
    context.manifest = manifest


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
    y = ruamel.yaml.YAML()
    y.indent(mapping=indent_mapping, sequence=indent_sequence, offset=indent_offset)
    y.width = width
    y.preserve_quotes = preserve_quotes
    y.default_flow_style = default_flow_style
    y.encoding = encoding
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
        return self.current == self.target


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


class MissingOsmosisConfig(Exception):
    """Raised when an osmosis configuration is missing."""

    pass


class InvalidOsmosisConfig(Exception):
    """Raised when an osmosis configuration is invalid."""

    pass


@dataclass
class YamlRefactorSettings:
    """Settings for yaml based refactoring operations."""

    fqn: str | None = None
    """Filter models to action via a fully qualified name match."""
    models: list[str] = field(default_factory=list)
    """Filter models to action via a file path match."""
    dry_run: bool = False
    """Do not write changes to disk."""
    catalog_file: str | None = None
    """Path to the dbt catalog.json file to use preferentially instead of live warehouse introspection"""
    skip_add_columns: bool = False
    """Skip adding missing columns in the yaml files."""
    skip_add_tags: bool = False
    """Skip appending upstream tags in the yaml files."""
    skip_add_data_types: bool = False
    """Skip adding data types in the yaml files."""
    numeric_precision: bool = False
    """Include numeric precision in the data type."""
    char_length: bool = False
    """Include character length in the data type."""
    skip_merge_meta: bool = False
    """Skip merging upstream meta fields in the yaml files."""
    add_progenitor_to_meta: bool = False
    """Add a custom progenitor field to the meta section indicating a column's origin."""
    use_unrendered_descriptions: bool = False
    """Use unrendered descriptions preserving things like {{ doc(...) }} which are otherwise pre-rendered in the manifest object"""
    add_inheritance_for_specified_keys: list[str] = field(default_factory=list)
    """Include additional keys in the inheritance process."""
    output_to_lower: bool = False
    """Force column name and data type output to lowercase in the yaml files."""
    force_inherit_descriptions: bool = False
    """Force inheritance of descriptions from upstream models, even if node has a valid description."""


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
    _catalog: CatalogArtifact | None = field(default=None, init=False)

    def register_mutations(self, count: int) -> None:
        """Increment the mutation count by a specified amount."""
        self._mutation_count += count

    @property
    def mutation_count(self) -> int:
        """Read only property to access the mutation count."""
        return self._mutation_count

    @property
    def mutated(self) -> bool:
        """Check if the context has performed any mutations."""
        return self._mutation_count > 0

    @property
    def source_definitions(self) -> dict[str, t.Any]:
        """The source definitions from the dbt project config."""
        defs = self.project.config.vars.to_dict().get("dbt-osmosis", {}).copy()
        defs.pop(SKIP_PATTERNS, None)
        return defs

    @property
    def skip_patterns(self) -> list[str]:
        """The column name skip patterns from the dbt project config."""
        defs = self.project.config.vars.to_dict().get("dbt-osmosis", {}).copy()
        return defs.pop(SKIP_PATTERNS, [])

    def read_catalog(self) -> CatalogArtifact | None:
        """Read the catalog file if it exists."""
        if self._catalog:
            return self._catalog
        if not self.settings.catalog_file:
            return None
        fp = Path(self.settings.catalog_file)
        if not fp.exists():
            return None
        self._catalog = CatalogArtifact.from_dict(json.loads(fp.read_text()))
        return self._catalog

    def __post_init__(self) -> None:
        if EMPTY_STRING not in self.placeholders:
            self.placeholders = (EMPTY_STRING, *self.placeholders)


def load_catalog(settings: YamlRefactorSettings) -> CatalogArtifact | None:
    """Load the catalog file if it exists and return a CatalogArtifact instance."""
    if not settings.catalog_file:
        return None
    fp = Path(settings.catalog_file)
    if not fp.exists():
        return None
    return CatalogArtifact.from_dict(json.loads(fp.read_text()))


# Basic compile & execute
# =======================


def _has_jinja(code: str) -> bool:
    """Check if a code string contains jinja tokens."""
    return any(token in code for token in ("{{", "}}", "{%", "%}", "{#", "#}"))


def compile_sql_code(context: DbtProjectContext, raw_sql: str) -> ManifestSQLNode:
    """Compile jinja SQL using the context's manifest and adapter."""
    tmp_id = str(uuid.uuid4())
    with context.manifest_mutex:
        key = f"{NodeType.SqlOperation}.{context.config.project_name}.{tmp_id}"
        _ = context.manifest.nodes.pop(key, None)

        node = context.sql_parser.parse_remote(raw_sql, tmp_id)
        if not _has_jinja(raw_sql):
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

    return compiled_node


def execute_sql_code(context: DbtProjectContext, raw_sql: str) -> AdapterResponse:
    """Execute jinja SQL using the context's manifest and adapter."""
    if _has_jinja(raw_sql):
        comp = compile_sql_code(context, raw_sql)
        sql_to_exec = comp.compiled_code or comp.raw_code
    else:
        sql_to_exec = raw_sql

    resp, _ = context.adapter.execute(sql_to_exec, auto_begin=False, fetch=True)
    return resp


# Node filtering
# ==============


def _is_fqn_match(node: ResultNode, fqn_str: str) -> bool:
    """Filter models based on the provided fully qualified name matching on partial segments."""
    if not fqn_str:
        return True
    parts = fqn_str.split(".")
    return len(node.fqn[1:]) >= len(parts) and all(
        left == right for left, right in zip(parts, node.fqn[1:])
    )


def _is_file_match(node: ResultNode, paths: list[str]) -> bool:
    """Check if a node's file path matches any of the provided file paths or names."""
    node_path = _get_node_path(node)
    for model in paths:
        if node.name == model:
            return True
        try_path = Path(model).resolve()
        if try_path.is_dir():
            if node_path and try_path in node_path.parents:
                return True
        elif try_path.is_file():
            if node_path and try_path == node_path:
                return True
    return False


def _get_node_path(node: ResultNode) -> Path | None:
    """Return the path to the node's original file if available."""
    if node.original_file_path and hasattr(node, "root_path"):
        return Path(getattr(node, "root_path"), node.original_file_path).resolve()
    return None


def filter_models(
    context: YamlRefactorContext,
) -> Iterator[tuple[str, ResultNode]]:
    """Iterate over the models in the dbt project manifest applying the filter settings."""

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
        elif context.settings.fqn:
            if not _is_fqn_match(node, context.settings.fqn):
                return False
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
        return column
    if credentials_type == "snowflake":
        return column.upper()
    return column


def _maybe_use_precise_dtype(col: BaseColumn, settings: YamlRefactorSettings) -> str:
    """Use the precise data type if enabled in the settings."""
    if (col.is_numeric() and settings.numeric_precision) or (
        col.is_string() and settings.char_length
    ):
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
        return _COLUMN_LIST_CACHE[ref]

    normalized_cols = OrderedDict()
    offset = 0

    def process_column(col: BaseColumn | ColumnMetadata):
        nonlocal offset
        if any(re.match(b, col.name) for b in context.skip_patterns):
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
        catalog_entry = _find_first(
            chain(catalog.nodes.values(), catalog.sources.values()), lambda c: c.key() == ref
        )
        if catalog_entry:
            for column in catalog_entry.columns.values():
                process_column(column)
            return normalized_cols

    relation: BaseRelation | None = context.project.adapter.get_relation(*ref)
    if relation is None:
        return normalized_cols

    try:
        for column in t.cast(
            Iterable[BaseColumn], context.project.adapter.get_columns_in_relation(relation)
        ):
            process_column(column)
    except Exception as ex:
        logger.warning(f"Could not introspect columns for {ref}: {ex}")

    _COLUMN_LIST_CACHE[ref] = normalized_cols
    return normalized_cols


# Restructuring Logic
# ===================


def create_missing_source_yamls(context: YamlRefactorContext) -> None:
    """Create source files for sources defined in the dbt_project.yml dbt-osmosis var which don't exist as nodes.

    This is a useful preprocessing step to ensure that all sources are represented in the dbt project manifest. We
    do not have rich node information for non-existent sources, hence the alternative codepath here to bootstrap them.
    """
    database: str = context.project.config.credentials.database

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
            continue

        src_yaml_path = Path(
            context.project.config.project_root,
            context.project.config.model_paths[0],
            src_yaml_path.lstrip(os.sep),
        )

        def _describe(rel: BaseRelation) -> dict[str, t.Any]:
            return {
                "name": rel.identifier,
                "description": "",
                "columns": [
                    {
                        "name": name,
                        "description": meta.comment or "",
                        "data_type": meta.type.lower()
                        if context.settings.output_to_lower
                        else meta.type,
                    }
                    for name, meta in get_columns(context, get_table_ref(rel)).items()
                ],
            }

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
            logger.info(f"Injecting source {source} => {src_yaml_path}")
            context.yaml_handler.dump({"version": 2, "sources": [source]}, f)

        did_side_effect = True
        context.register_mutations(1)

    if did_side_effect:
        logger.info("Reloading project to pick up new sources.")
        reload_manifest(context.project)


def _get_yaml_path_template(context: YamlRefactorContext, node: ResultNode) -> str | None:
    """Get the yaml path template for a dbt model or source node."""
    if node.resource_type == NodeType.Source:
        def_or_path = context.source_definitions.get(node.source_name)
        if isinstance(def_or_path, dict):
            return def_or_path.get("path")
        return def_or_path
    path_template = node.config.extra.get("dbt-osmosis", node.unrendered_config.get("dbt-osmosis"))
    if not path_template:
        raise MissingOsmosisConfig(
            f"Config key `dbt-osmosis: <path>` not set for model {node.name}"
        )
    return path_template


def get_current_yaml_path(context: YamlRefactorContext, node: ResultNode) -> Path | None:
    """Get the current yaml path for a dbt model or source node."""
    if node.resource_type in (NodeType.Model, NodeType.Seed) and getattr(node, "patch_path", None):
        return Path(context.project.config.project_root).joinpath(
            t.cast(str, node.patch_path).partition("://")[-1]
        )
    if node.resource_type == NodeType.Source:
        return Path(context.project.config.project_root, node.path)
    return None


def get_target_yaml_path(context: YamlRefactorContext, node: ResultNode) -> Path:
    """Get the target yaml path for a dbt model or source node."""
    tpl = _get_yaml_path_template(context, node)
    if not tpl:
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

    return Path(context.project.config.project_root, *segments)


def build_yaml_file_mapping(
    context: YamlRefactorContext, create_missing_sources: bool = False
) -> dict[str, SchemaFileLocation]:
    """Build a mapping of dbt model and source nodes to their current and target yaml paths."""

    if create_missing_sources:
        create_missing_source_yamls(context)

    out_map: dict[str, SchemaFileLocation] = {}
    for uid, node in filter_models(context):
        current_path = get_current_yaml_path(context, node)
        out_map[uid] = SchemaFileLocation(
            target=get_target_yaml_path(context, node).resolve(),
            current=current_path.resolve() if current_path else None,
            node_type=node.resource_type,
        )
    return out_map


# TODO: detect if something is dirty to minimize disk writes on commits
_YAML_BUFFER_CACHE: dict[Path, t.Any] = {}
"""Cache for yaml file buffers to avoid redundant disk reads/writes and simplify edits."""


def _read_yaml(context: YamlRefactorContext, path: Path) -> dict[str, t.Any]:
    """Read a yaml file from disk. Adds an entry to the buffer cache so all operations on a path are consistent."""
    if path not in _YAML_BUFFER_CACHE:
        if not path.is_file():
            return {}
        with context.yaml_handler_lock:
            _YAML_BUFFER_CACHE[path] = t.cast(dict[str, t.Any], context.yaml_handler.load(path))
    return _YAML_BUFFER_CACHE[path]


def _write_yaml(context: YamlRefactorContext, path: Path, data: dict[str, t.Any]) -> None:
    """Write a yaml file to disk and register a mutation with the context. Clears the path from the buffer cache."""
    if not context.settings.dry_run:
        with context.yaml_handler_lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            context.yaml_handler.dump(data, path)
        if path in _YAML_BUFFER_CACHE:
            del _YAML_BUFFER_CACHE[path]
    context.register_mutations(1)


def commit_yamls(context: YamlRefactorContext) -> None:
    """Commit all files in the yaml buffer cache to disk. Clears the buffer cache and registers mutations."""
    if not context.settings.dry_run:
        with context.yaml_handler_lock:
            for path in list(_YAML_BUFFER_CACHE.keys()):
                with path.open("w") as f:
                    context.yaml_handler.dump(_YAML_BUFFER_CACHE[path], f)
                del _YAML_BUFFER_CACHE[path]
                context.register_mutations(1)


def _generate_minimal_model_yaml(node: ModelNode | SeedNode) -> dict[str, t.Any]:
    """Generate a minimal model yaml for a dbt model node."""
    return {"name": node.name, "columns": []}


def _generate_minimal_source_yaml(node: SourceDefinition) -> dict[str, t.Any]:
    """Generate a minimal source yaml for a dbt source node."""
    return {"name": node.source_name, "tables": [{"name": node.name, "columns": []}]}


def _create_operations_for_node(
    context: YamlRefactorContext, uid: str, loc: SchemaFileLocation
) -> list[RestructureOperation]:
    """Create restructure operations for a dbt model or source node."""
    node = context.project.manifest.nodes.get(uid) or context.project.manifest.sources.get(uid)
    if not node:
        logger.warning(f"Node {uid} not found in manifest.")
        return []

    # If loc.current is None => we are generating a brand new file
    # If loc.current => we unify it with the new location
    ops: list[RestructureOperation] = []

    if loc.current is None:
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
            raise exc
    return plan


def pretty_print_plan(plan: RestructureDeltaPlan) -> None:
    """Pretty print the restructure plan for the dbt project."""
    for op in plan.operations:
        str_content = str(op.content)[:80] + "..."
        logger.info(f"Processing {str_content}")
        if not op.superseded_paths:
            logger.info(f"CREATE or MERGE => {op.file_path}")
        else:
            old_paths = [p.name for p in op.superseded_paths.keys()] or ["UNKNOWN"]
            logger.info(f"{old_paths} -> {op.file_path}")


def _remove_models(existing_doc: dict[str, t.Any], nodes: list[ResultNode]) -> None:
    """Clean up the existing yaml doc by removing models superseded by the restructure plan."""
    to_remove = {n.name for n in nodes if n.resource_type == NodeType.Model}
    keep = []
    for section in existing_doc.get("models", []):
        if section.get("name") not in to_remove:
            keep.append(section)
    existing_doc["models"] = keep


def _remove_seeds(existing_doc: dict[str, t.Any], nodes: list[ResultNode]) -> None:
    """Clean up the existing yaml doc by removing models superseded by the restructure plan."""
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


def apply_restructure_plan(
    context: YamlRefactorContext, plan: RestructureDeltaPlan, *, confirm: bool = False
) -> None:
    """Apply the restructure plan for the dbt project."""
    if not plan.operations:
        logger.info("No changes needed.")
        return

    if confirm:
        pretty_print_plan(plan)

    while confirm:
        response = input("Apply the restructure plan? [y/N]: ")
        if response.lower() in ("y", "yes"):
            break
        elif response.lower() in ("n", "no", ""):
            logger.info("Skipping restructure plan.")
            return
        logger.info("Please respond with 'y' or 'n'.")

    for op in plan.operations:
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
                    logger.info(f"Superseded entire file {path}")
                else:
                    _write_yaml(context, path, existing_data)
                    logger.info(f"Migrated doc from {path} -> {op.file_path}")

    _ = commit_yamls(context), reload_manifest(context.project)


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


def _get_member_yaml(context: YamlRefactorContext, member: ResultNode) -> dict[str, t.Any] | None:
    """Get the parsed YAML for a dbt model or source node."""
    project_dir = Path(context.project.config.project_root)

    if isinstance(member, SourceDefinition):
        if not member.original_file_path:
            return None
        path = project_dir.joinpath(member.original_file_path)
        sources = t.cast(list[dict[str, t.Any]], _read_yaml(context, path).get("sources", []))
        source = _find_first(sources, lambda s: s["name"] == member.source_name, {})
        tables = source.get("tables", [])
        return _find_first(tables, lambda tbl: tbl["name"] == member.name)

    elif isinstance(member, (ModelNode, SeedNode)):
        if not member.patch_path:
            return None
        path = project_dir.joinpath(member.patch_path.split("://")[-1])
        section = f"{member.resource_type}s"
        models = t.cast(list[dict[str, t.Any]], _read_yaml(context, path).get(section, []))
        return _find_first(models, lambda model: model["name"] == member.name)

    return None


def _build_column_knowledge_graph(
    context: YamlRefactorContext, node: ResultNode
) -> dict[str, dict[str, t.Any]]:
    """Generate a column knowledge graph for a dbt model or source node."""
    tree = _build_node_ancestor_tree(context.project.manifest, node)

    column_knowledge_graph: dict[str, dict[str, t.Any]] = {}
    for generation in reversed(sorted(tree.keys())):
        ancestors = tree[generation]
        for ancestor_uid in ancestors:
            ancestor = context.project.manifest.nodes.get(
                ancestor_uid, context.project.manifest.sources.get(ancestor_uid)
            )
            if not isinstance(ancestor, (SourceDefinition, SeedNode, ModelNode)):
                continue

            for name, metadata in ancestor.columns.items():
                graph_node = column_knowledge_graph.setdefault(name, {})
                if context.settings.add_progenitor_to_meta:
                    graph_node.setdefault("meta", {}).setdefault(
                        "osmosis_progenitor", ancestor.unique_id
                    )

                graph_edge = metadata.to_dict()

                if context.settings.use_unrendered_descriptions:
                    raw_yaml = _get_member_yaml(context, ancestor) or {}
                    raw_columns = t.cast(list[dict[str, t.Any]], raw_yaml.get("columns", []))
                    raw_column_metadata = _find_first(
                        raw_columns,
                        lambda c: normalize_column_name(
                            c["name"], context.project.config.credentials.type
                        )
                        == name,
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

                for inheritable in context.settings.add_inheritance_for_specified_keys:
                    current_val = graph_node.get(inheritable)
                    if incoming_val := graph_edge.pop(inheritable, current_val):
                        graph_edge[inheritable] = incoming_val

                if graph_edge.get("description", EMPTY_STRING) in context.placeholders or (
                    generation == "generation_0" and context.settings.force_inherit_descriptions
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
        for _, node in filter_models(context):
            inherit_upstream_column_knowledge(context, node)
        return None

    inheritable = ["description"]
    if not context.settings.skip_add_tags:
        inheritable.append("tags")
    if not context.settings.skip_merge_meta:
        inheritable.append("meta")
    for extra in context.settings.add_inheritance_for_specified_keys:
        if extra not in inheritable:
            inheritable.append(extra)

    yaml_section = _get_member_yaml(context, node)
    column_knowledge_graph = _build_column_knowledge_graph(context, node)
    kwargs = None
    for name, node_column in node.columns.items():
        variants: list[str] = [name]
        pm = get_plugin_manager()
        for v in pm.hook.get_candidates(name=name, node=node, context=context.project):
            variants.extend(t.cast(list[str], v))
        for variant in variants:
            kwargs = column_knowledge_graph.get(variant)
            if kwargs is not None:
                break
        else:
            continue

        updated_metadata = {k: v for k, v in kwargs.items() if v is not None and k in inheritable}
        node.columns[name] = node_column.replace(**updated_metadata)

        if not yaml_section:
            continue
        for column in yaml_section.get("columns", []):
            yaml_name = normalize_column_name(
                column["name"], context.project.config.credentials.type
            )
            if yaml_name == name:
                column.update(**updated_metadata)


def inject_missing_columns(context: YamlRefactorContext, node: ResultNode | None = None) -> None:
    """Add missing columns to a dbt node and it's corresponding yaml section. Changes are implicitly buffered until commit_yamls is called."""
    if context.settings.skip_add_columns:
        return
    if node is None:
        for _, node in filter_models(context):
            inject_missing_columns(context, node)
        return
    yaml_section = _get_member_yaml(context, node)
    if yaml_section is None:
        return
    current_columns = {
        normalize_column_name(c["name"], context.project.config.credentials.type)
        for c in yaml_section.get("columns", [])
    }
    incoming_columns = get_columns(context, get_table_ref(node))
    for incoming_name, incoming_meta in incoming_columns.items():
        if incoming_name not in node.columns and incoming_name not in current_columns:
            logger.info(
                f"Detected and reconciling missing column {incoming_name} in node {node.unique_id}"
            )
            gen_col = {"name": incoming_name, "description": incoming_meta.comment or ""}
            if dtype := incoming_meta.type:
                gen_col["data_type"] = dtype.lower() if context.settings.output_to_lower else dtype
            node.columns[incoming_name] = ColumnInfo.from_dict(gen_col)
            yaml_section.setdefault("columns", []).append(gen_col)


def remove_columns_not_in_database(
    context: YamlRefactorContext, node: ResultNode | None = None
) -> None:
    """Remove columns from a dbt node and it's corresponding yaml section that are not present in the database. Changes are implicitly buffered until commit_yamls is called."""
    if node is None:
        for _, node in filter_models(context):
            remove_columns_not_in_database(context, node)
        return
    yaml_section = _get_member_yaml(context, node)
    if yaml_section is None:
        return
    current_columns = {
        normalize_column_name(c["name"], context.project.config.credentials.type)
        for c in yaml_section.get("columns", [])
    }
    incoming_columns = get_columns(context, get_table_ref(node))
    extra_columns = current_columns - set(incoming_columns.keys())
    for extra_column in extra_columns:
        logger.info(f"Detected and removing extra column {extra_column} in node {node.unique_id}")
        _ = node.columns.pop(extra_column, None)
        yaml_section["columns"] = [
            c for c in yaml_section.get("columns", []) if c["name"] != extra_column
        ]


def sort_columns_as_in_database(
    context: YamlRefactorContext, node: ResultNode | None = None
) -> None:
    """Sort columns in a dbt node and it's corresponding yaml section as they appear in the database. Changes are implicitly buffered until commit_yamls is called."""
    if node is None:
        for _, node in filter_models(context):
            sort_columns_as_in_database(context, node)
        return
    yaml_section = _get_member_yaml(context, node)
    if yaml_section is None:
        return
    incoming_columns = get_columns(context, get_table_ref(node))

    def _position(column: dict[str, t.Any]):
        db_info = incoming_columns.get(column["name"])
        if db_info is None:
            return 99999
        return db_info.index

    t.cast(list[dict[str, t.Any]], yaml_section["columns"]).sort(key=_position)
    node.columns = {
        k: v for k, v in sorted(node.columns.items(), key=lambda i: _position(i[1].to_dict()))
    }
    context.register_mutations(1)


def sort_columns_alphabetically(
    context: YamlRefactorContext, node: ResultNode | None = None
) -> None:
    """Sort columns in a dbt node and it's corresponding yaml section alphabetically. Changes are implicitly buffered until commit_yamls is called."""
    if node is None:
        for _, node in filter_models(context):
            sort_columns_alphabetically(context, node)
        return
    yaml_section = _get_member_yaml(context, node)
    if yaml_section is None:
        return
    t.cast(list[dict[str, t.Any]], yaml_section["columns"]).sort(key=lambda c: c["name"])
    node.columns = {k: v for k, v in sorted(node.columns.items(), key=lambda i: i[0])}
    context.register_mutations(1)


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
        return variants


class FuzzyPrefixMatching:
    @hookimpl
    def get_candidates(self, name: str, node: ResultNode, context: DbtProjectContext) -> list[str]:
        """Get a list of candidate names for a column excluding a prefix."""
        _ = context
        variants = []
        prefix = t.cast(
            str,
            node.config.extra.get(
                "dbt-osmosis-prefix", node.unrendered_config.get("dbt-osmosis-prefix")
            ),
        )
        if prefix and name.startswith(prefix):
            variants.append(name[len(prefix) :])
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


def run_example_compilation_flow() -> None:
    config = DbtConfiguration(target="some_target", threads=2)
    config.vars = {"foo": "bar"}
    proj_ctx = create_dbt_project_context(config)

    node = compile_sql_code(proj_ctx, "select '{{ 1+1 }}' as col")
    print("Compiled =>", node.compiled_code)

    resp = execute_sql_code(proj_ctx, "select '{{ 1+2 }}' as col")
    print("Resp =>", resp)


if __name__ == "__main__":
    c = DbtConfiguration(project_dir="demo_duckdb", profiles_dir="demo_duckdb")
    c.vars = {"dbt-osmosis": {}}

    project = create_dbt_project_context(c)
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
