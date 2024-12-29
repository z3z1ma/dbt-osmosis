# pyright: reportUnknownVariableType=false, reportPrivateImportUsage=false, reportAny=false, reportUnknownMemberType=false

import argparse
import json
import logging
import re
import threading
import time
import typing as t
import uuid
from collections import OrderedDict
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path

import dbt.flags as dbt_flags
import ruamel.yaml
from dbt.adapters.contracts.connection import AdapterResponse
from dbt.adapters.factory import Adapter, get_adapter_class_by_name
from dbt.config.runtime import RuntimeConfig
from dbt.contracts.graph.manifest import Manifest
from dbt.contracts.graph.nodes import ColumnInfo, ManifestNode, ManifestSQLNode, SourceDefinition
from dbt.contracts.results import CatalogArtifact, CatalogKey, CatalogTable, ColumnMetadata
from dbt.node_types import NodeType
from dbt.parser.manifest import ManifestLoader, process_node
from dbt.parser.sql import SqlBlockParser, SqlMacroParser
from dbt.task.sql import SqlCompileRunner
from dbt.tracking import disable_tracking

disable_tracking()

EMPTY_STRING = ""

logger = logging.getLogger("dbt-osmosis")


def has_jinja(code: str) -> bool:
    """Check if a code string contains jinja tokens."""
    return any(token in code for token in ("{{", "}}", "{%", "%}", "{#", "#}"))


def column_casing(column: str, credentials_type: str, to_lower: bool) -> str:
    """Apply case normalization to a column name based on the credentials type."""
    if credentials_type == "snowflake" and column.startswith('"') and column.endswith('"'):
        return column
    if to_lower:
        return column.lower()
    if credentials_type == "snowflake":
        return column.upper()
    return column


@dataclass
class DbtConfiguration:
    """Configuration for a dbt project."""

    project_dir: str
    profiles_dir: str
    target: str | None = None
    profile: str | None = None
    threads: int = 1
    single_threaded: bool = True
    which: str = ""

    debug: bool = False
    _vars: str | dict[str, t.Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.threads != 1:
            self.single_threaded = False

    @property
    def vars(self) -> str:
        if isinstance(self._vars, dict):
            return json.dumps(self._vars)
        return self._vars

    @vars.setter
    def vars(self, value: t.Any) -> None:
        if not isinstance(value, (str, dict)):
            raise ValueError("vars must be a string or dict")
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
        which=cfg.which,
        vars=cfg.vars,
        DEBUG=cfg.debug,
    )


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
    supersede: dict[Path, list[str]] = field(default_factory=dict)


# FIXME: fold this in from the other file
@dataclass
class ColumnLevelKnowledgePropagator:
    """Example usage for doc-propagation logic. placeholders is a tuple to avoid accidental mutation."""

    placeholders: tuple[str, ...] = (
        EMPTY_STRING,
        "Pending further documentation",
        "Pending further documentation.",
        "No description for this column",
        "No description for this column.",
        "Not documented",
        "Not documented.",
        "Undefined",
        "Undefined.",
    )

    @staticmethod
    def get_node_columns_with_inherited_knowledge(
        manifest: Manifest,
        node: ManifestNode,
        placeholders: list[str],
        use_unrendered_descriptions: bool,
    ) -> dict[str, dict[str, t.Any]]:
        _ = manifest, node, placeholders, use_unrendered_descriptions
        return {}

    @staticmethod
    def update_undocumented_columns_with_prior_knowledge(
        columns_to_update: Iterable[str],
        node: ManifestNode,
        yaml_section: dict[str, t.Any],
        known_knowledge: dict[str, dict[str, t.Any]],
    ) -> int:
        changed_count = 0
        for col in columns_to_update:
            if col not in node.columns:
                continue
            cinfo = node.columns[col]
            old_desc = getattr(cinfo, "description", "")
            new_desc = old_desc
            if col in known_knowledge and not old_desc:
                new_desc = known_knowledge[col].get("description", "")
            if new_desc and new_desc != old_desc:
                setattr(cinfo, "description", new_desc)
                for c in yaml_section.get("columns", []):
                    if c["name"].lower() == col.lower():
                        c["description"] = new_desc
                changed_count += 1
        return changed_count


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
    models: list[str] = field(default_factory=list)
    dry_run: bool = False
    catalog_file: str | None = None
    skip_add_columns: bool = False
    skip_add_tags: bool = False
    skip_add_data_types: bool = False
    numeric_precision: bool = False
    char_length: bool = False
    skip_merge_meta: bool = False
    add_progenitor_to_meta: bool = False
    use_unrendered_descriptions: bool = False
    add_inheritance_for_specified_keys: list[str] = field(default_factory=list)
    output_to_lower: bool = False


@dataclass
class DbtProjectContext:
    """A data object that includes references to:

    - The loaded dbt config
    - The manifest
    - The sql/macro parsers

    With mutexes for thread safety. The adapter is lazily instantiated and has a TTL which allows
    for re-use across multiple operations in long-running processes. (is the idea)
    """

    config: RuntimeConfig
    manifest: Manifest
    sql_parser: SqlBlockParser
    macro_parser: SqlMacroParser
    adapter_ttl: float = 3600.0

    _adapter_mutex: threading.Lock = field(default_factory=threading.Lock)
    _manifest_mutex: threading.Lock = field(default_factory=threading.Lock)
    _adapter: Adapter | None = None
    _adapter_created_at: float = 0.0

    @property
    def is_adapter_expired(self) -> bool:
        """Check if the adapter has expired based on the adapter TTL."""
        return time.time() - self._adapter_created_at > self.adapter_ttl

    # NOTE: the way we use the adapter, the generics are irrelevant
    @property
    def adapter(self) -> Adapter[t.Any, t.Any, t.Any, t.Any]:
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


def instantiate_adapter(runtime_config: RuntimeConfig) -> Adapter[t.Any, t.Any, t.Any, t.Any]:
    """Instantiate a dbt adapter based on the runtime configuration."""
    adapter_cls = get_adapter_class_by_name(runtime_config.credentials.type)
    if not adapter_cls:
        raise RuntimeError(
            f"No valid adapter class found for credentials type: {runtime_config.credentials.type}"
        )

    # NOTE: this exists to patch over an API change in dbt core at some point I don't remember
    try:
        adapter = adapter_cls(runtime_config)
    except TypeError:
        from dbt.mp_context import get_mp_context

        adapter = adapter_cls(runtime_config, get_mp_context())  # pyright: ignore[reportCallIssue]

    adapter.connections.set_connection_name("dbt-osmosis")
    return adapter


def create_dbt_project_context(config: DbtConfiguration) -> DbtProjectContext:
    """Build a DbtProjectContext from a DbtConfiguration."""
    args = config_to_namespace(config)
    dbt_flags.set_from_args(args, args)
    runtime_cfg = RuntimeConfig.from_args(args)

    loader = ManifestLoader(runtime_cfg, runtime_cfg.load_dependencies())
    manifest = loader.load()
    manifest.build_flat_graph()

    adapter = instantiate_adapter(runtime_cfg)
    adapter.set_macro_resolver(manifest)

    sql_parser = SqlBlockParser(runtime_cfg, manifest, runtime_cfg)
    macro_parser = SqlMacroParser(runtime_cfg, manifest)

    return DbtProjectContext(
        config=runtime_cfg,
        manifest=manifest,
        sql_parser=sql_parser,
        macro_parser=macro_parser,
    )


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
    settings: YamlRefactorSettings

    pool: ThreadPoolExecutor = field(default_factory=ThreadPoolExecutor)

    yaml_handler: ruamel.yaml.YAML = field(default_factory=create_yaml_instance)

    placeholders: tuple[str, ...] = (
        EMPTY_STRING,
        "Pending further documentation",
        "No description for this column",
        "Not documented",
        "Undefined",
    )

    mutation_count: int = 0

    def register_mutations(self, count: int) -> None:
        """Increment the mutation count by a specified amount."""
        self.mutation_count += count

    def __post_init__(self) -> None:
        if EMPTY_STRING not in self.placeholders:
            self.placeholders = (EMPTY_STRING, *self.placeholders)


def compile_sql_code(context: DbtProjectContext, raw_sql: str) -> ManifestSQLNode:
    """Compile jinja SQL using the context's manifest and adapter."""
    tmp_id = str(uuid.uuid4())
    with context.manifest_mutex:
        key = f"{NodeType.SqlOperation}.{context.config.project_name}.{tmp_id}"
        _ = context.manifest.nodes.pop(key, None)

        node = context.sql_parser.parse_remote(raw_sql, tmp_id)
        if not has_jinja(raw_sql):
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
    if has_jinja(raw_sql):
        comp = compile_sql_code(context, raw_sql)
        sql_to_exec = comp.compiled_code or comp.raw_code
    else:
        sql_to_exec = raw_sql

    resp, _ = context.adapter.execute(sql_to_exec, auto_begin=False, fetch=True)
    return resp


def filter_models(
    context: YamlRefactorContext,
) -> Iterator[tuple[str, ManifestNode | SourceDefinition]]:
    """Iterate over the models in the dbt project manifest applying the filter settings."""

    def f(node: ManifestNode | SourceDefinition) -> bool:
        """Closure to filter models based on the context settings."""
        if node.resource_type not in (NodeType.Model, NodeType.Source):
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


def _is_fqn_match(node: ManifestNode | SourceDefinition, fqn_str: str) -> bool:
    """Filter models based on the provided fully qualified name matching on partial segments."""
    if not fqn_str:
        return True
    parts = fqn_str.split(".")
    return len(node.fqn[1:]) >= len(parts) and all(
        left == right for left, right in zip(parts, node.fqn[1:])
    )


def _is_file_match(node: ManifestNode | SourceDefinition, paths: list[str]) -> bool:
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


def _get_node_path(node: ManifestNode | SourceDefinition) -> Path | None:
    """Return the path to the node's original file if available."""
    if node.original_file_path and hasattr(node, "root_path"):
        return Path(getattr(node, "root_path"), node.original_file_path).resolve()
    return None


def load_catalog(settings: YamlRefactorSettings) -> CatalogArtifact | None:
    """Load the catalog file if it exists and return a CatalogArtifact instance."""
    if not settings.catalog_file:
        return None
    fp = Path(settings.catalog_file)
    if not fp.exists():
        return None
    return CatalogArtifact.from_dict(json.loads(fp.read_text()))


# TODO: more work to do below the fold here


# NOTE: in multithreaded operations, we need to use the thread connection for the adapter
def get_columns_meta(
    context: YamlRefactorContext,
    node: ManifestNode,
    catalog: CatalogArtifact | None,
) -> dict[str, ColumnMetadata]:
    """Get the column metadata for a node from the catalog or the adapter."""
    cased_cols = OrderedDict()
    blacklist = context.project.config.vars.get("dbt-osmosis", {}).get("_blacklist", [])

    key = _catalog_key_for_node(node)
    if catalog:
        cat_objs = {**catalog.nodes, **catalog.sources}
        matched = [v for k, v in cat_objs.items() if k.split(".")[-1] == key.name]
        if matched:
            for col in matched[0].columns.values():
                if any(re.match(p, col.name) for p in blacklist):
                    continue
                cased = column_casing(
                    col.name,
                    context.project.config.credentials.type,
                    context.settings.output_to_lower,
                )
                cased_cols[cased] = ColumnMetadata(
                    name=cased,
                    type=col.type,
                    index=col.index,
                    comment=col.comment,
                )
            return cased_cols

    rel = context.project.adapter.get_relation(key.database, key.schema, key.name)
    if not rel:
        return cased_cols
    try:
        col_objs = context.project.adapter.get_columns_in_relation(rel)
        for col_ in col_objs:
            if any(re.match(b, col_.name) for b in blacklist):
                continue
            cased = column_casing(
                col_.name,
                context.project.config.credentials.type,
                context.settings.output_to_lower,
            )
            dtype = _maybe_use_precise_dtype(col_, context.settings)
            cased_cols[cased] = ColumnMetadata(
                name=cased,
                type=dtype,
                index=None,
                comment=getattr(col_, "comment", None),
            )
            if hasattr(col_, "flatten"):
                for exp in col_.flatten():
                    if any(re.match(b, exp.name) for b in blacklist):
                        continue
                    cased2 = column_casing(
                        exp.name,
                        context.project.config.credentials.type,
                        context.settings.output_to_lower,
                    )
                    dtype2 = _maybe_use_precise_dtype(exp, context.settings)
                    cased_cols[cased2] = ColumnMetadata(
                        name=cased2,
                        type=dtype2,
                        index=None,
                        comment=getattr(exp, "comment", None),
                    )
    except Exception as exc:
        logger.warning(f"Could not introspect columns for {key}: {exc}")

    return cased_cols


def _maybe_use_precise_dtype(col: t.Any, settings: YamlRefactorSettings) -> str:
    """Use the precise data type if enabled in the settings."""
    if (col.is_numeric() and settings.numeric_precision) or (
        col.is_string() and settings.char_length
    ):
        return col.data_type
    return col.dtype


def _catalog_key_for_node(node: ManifestNode) -> CatalogKey:
    """Make an appropriate catalog key for a dbt node."""
    # TODO: pyright seems to think something is wrong below
    if node.resource_type == NodeType.Source:
        return CatalogKey(node.database, node.schema, getattr(node, "identifier", node.name))
    return CatalogKey(node.database, node.schema, getattr(node, "alias", node.name))


# NOTE: usage examples of the more FP style module below


def build_dbt_project_context(cfg: DbtConfiguration) -> DbtProjectContext:
    if not cfg.project_dir:
        cfg.project_dir = discover_project_dir()
    if not cfg.profiles_dir:
        cfg.profiles_dir = discover_profiles_dir()
    return create_dbt_project_context(cfg)


def run_example_compilation_flow() -> None:
    cfg = DbtConfiguration(
        project_dir="", profiles_dir="", target="some_target", threads=2, _vars={"foo": "bar"}
    )
    proj_ctx = build_dbt_project_context(cfg)

    cr = compile_sql_code(proj_ctx, "select '{{ 1+1 }}' as col")
    print("Compiled =>", cr.compiled_code)

    ex = execute_sql_code(proj_ctx, "select '{{ 1+2 }}' as col")
    print("Rows =>", ex.table)
