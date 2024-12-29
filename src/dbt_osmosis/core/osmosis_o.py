# pyright: reportUnknownVariableType=false, reportPrivateImportUsage=false, reportAny=false, reportUnknownMemberType=false
import json
import logging
import os
import re
import sys
import threading
import time
import typing as t
import uuid
from argparse import Namespace
from collections import OrderedDict, UserDict
from collections.abc import Iterable, Iterator, MutableMapping
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import contextmanager
from copy import copy
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import chain
from pathlib import Path

import ruamel.yaml
from dbt.adapters.factory import get_adapter_class_by_name
from dbt.config.runtime import RuntimeConfig
from dbt.contracts.graph.manifest import Manifest
from dbt.contracts.graph.nodes import ColumnInfo, ManifestNode
from dbt.contracts.results import CatalogArtifact, CatalogKey, CatalogTable, ColumnMetadata
from dbt.flags import set_from_args
from dbt.node_types import NodeType
from dbt.parser.manifest import ManifestLoader, process_node
from dbt.parser.sql import SqlBlockParser, SqlMacroParser
from dbt.task.sql import SqlCompileRunner
from dbt.tracking import disable_tracking

# Disabling dbt tracking for non-standard usage
disable_tracking()


def logger() -> logging.Logger:
    """Get the log handle for dbt-osmosis"""
    return logging.getLogger("dbt-osmosis")


def has_jinja(code: str) -> bool:
    """Check if code contains Jinja tokens"""
    return any(token in code for token in ("{{", "}}", "{%", "%}", "{#", "#}"))


def column_casing(column: str, credentials_type: str, to_lower: bool) -> str:
    """Utility to handle column name casing based on dbt adapter & user flag."""
    # If quoted in snowflake, pass verbatim
    if credentials_type == "snowflake" and column.startswith('"') and column.endswith('"'):
        return column
    # Otherwise apply user-specified transformations
    if to_lower:
        return column.lower()
    if credentials_type == "snowflake":
        return column.upper()
    return column


class YamlHandler(ruamel.yaml.YAML):
    """A ruamel.yaml wrapper to handle dbt YAML files with sane defaults."""

    def __init__(self, **kwargs: t.Any) -> None:
        super().__init__(**kwargs)
        self.indent(mapping=2, sequence=4, offset=2)
        self.width: int = 800
        self.preserve_quotes: bool = True
        self.default_flow_style: bool = False
        self.encoding: str = os.getenv("DBT_OSMOSIS_ENCODING", "utf-8")


@dataclass
class SchemaFileLocation:
    """Dataclass to store schema file location details."""

    target: Path
    current: Path | None = None
    node_type: NodeType = NodeType.Model

    @property
    def is_valid(self) -> bool:
        return self.current == self.target


@dataclass
class SchemaFileMigration:
    """Dataclass to store schema file migration details."""

    output: dict[str, t.Any] = field(
        default_factory=lambda: {"version": 2, "models": [], "sources": []}
    )
    supersede: dict[Path, list[str]] = field(default_factory=dict)


@dataclass
class DbtConfiguration:
    """Stores dbt project configuration in a namespace"""

    project_dir: str
    profiles_dir: str
    threads: int = 1
    single_threaded: bool = True
    which: str = ""
    target: str | None = None
    profile: str | None = None

    DEBUG: bool = False

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
    def vars(self, v: t.Any) -> None:
        if not isinstance(v, (str, dict)):
            raise ValueError("vars must be a string or dict")
        self._vars = v


class DbtManifestProxy(UserDict[str, t.Any]):
    """Proxy for the manifest's flat_graph, read-only by design."""

    def _readonly(self, *args: t.Any, **kwargs: t.Any) -> t.Never:
        _ = args, kwargs
        raise RuntimeError("Cannot modify DbtManifestProxy")

    __setitem__: t.Callable[..., None] = _readonly
    __delitem__: t.Callable[..., None] = _readonly
    pop: t.Callable[..., None] = _readonly
    popitem: t.Callable[..., t.Any] = _readonly
    clear: t.Callable[..., None] = _readonly
    update: t.Callable[..., None] = _readonly
    setdefault: t.Callable[..., None] = _readonly


@dataclass
class DbtAdapterExecutionResult:
    adapter_response: t.Any
    table: t.Any
    raw_code: str
    compiled_code: str


@dataclass
class DbtAdapterCompilationResult:
    raw_code: str
    compiled_code: str
    node: ManifestNode
    injected_code: str | None = None


def find_default_project_dir() -> str:
    cwd = Path.cwd()
    # Walk up if needed
    for p in [cwd] + list(cwd.parents):
        if (p / "dbt_project.yml").exists():
            return str(p.resolve())
    return str(cwd.resolve())


def find_default_profiles_dir() -> str:
    # Common fallback for DBT_PROFILES_DIR
    if (Path.cwd() / "profiles.yml").exists():
        return str(Path.cwd().resolve())
    return str(Path.home() / ".dbt")


class DbtProject:
    """Wraps dbt's in-memory project & adapter, enabling queries, compilation, etc."""

    ADAPTER_TTL: float = 3600.0

    def __init__(
        self,
        target: str | None = None,
        profiles_dir: str | None = None,
        project_dir: str | None = None,
        threads: int = 1,
        vars: str | dict[str, t.Any] | None = None,
        profile: str | None = None,
    ):
        if not profiles_dir:
            profiles_dir = find_default_profiles_dir()
        if not project_dir:
            project_dir = find_default_project_dir()

        self.base_config: DbtConfiguration = DbtConfiguration(
            project_dir=project_dir,
            profiles_dir=profiles_dir,
            target=target,
            threads=threads,
            profile=profile,
        )
        if vars:
            self.base_config.vars = vars

        self.adapter_mutex: threading.Lock = threading.Lock()
        self.parsing_mutex: threading.Lock = threading.Lock()
        self.manifest_mutation_mutex: threading.Lock = threading.Lock()

        self._config: RuntimeConfig | None = None
        self._manifest: Manifest | None = None
        self.parse_project(init=True)

        self._sql_parser: SqlBlockParser | None = None
        self._macro_parser: SqlMacroParser | None = None
        self._adapter_created_at: float = 0.0

    @property
    def config(self) -> RuntimeConfig:
        """Get the dbt project configuration."""
        if self._config is None:
            raise RuntimeError("DbtProject not initialized. parse_project() must be called first.")
        return self._config

    @property
    def manifest(self) -> Manifest:
        """Get the dbt project manifest."""
        if self._manifest is None:
            raise RuntimeError("DbtProject not initialized. parse_project() must be called first.")
        return self._manifest

    def parse_project(self, init: bool = False) -> None:
        """Parse the dbt project configuration and manifest."""
        with self.parsing_mutex:
            if init:
                ns = Namespace(
                    **self.base_config.__dict__
                )  # TODO: replace with method call to handle _vars prop
                set_from_args(ns, ns)
                self._config = RuntimeConfig.from_args(ns)
                self.initialize_adapter()
            loader = ManifestLoader(
                self.config,
                self.config.load_dependencies(),
                self.adapter.connections.set_query_header,
            )
            self._manifest = loader.load()
            self._manifest.build_flat_graph()
            loader.save_macros_to_adapter(self.adapter)
            self._sql_parser = None
            self._macro_parser = None

    def safe_parse_project(self, init: bool = False) -> None:
        """Safely re-parse the dbt project configuration and manifest preserving internal state on error."""
        old_config = copy(getattr(self, "config", None))
        try:
            self.parse_project(init=init)
        except Exception as exc:
            if old_config:
                self._config = old_config
            raise exc
        # Write manifest to disk here
        self.write_manifest_artifact()

    def initialize_adapter(self) -> None:
        """Initialize the dbt adapter."""
        if hasattr(self, "_adapter"):
            try:
                self.adapter.connections.cleanup_all()
            except Exception:
                pass
        try:
            adapter_cls = get_adapter_class_by_name(
                self.base_config.target or self.base_config.profile or ""
            )
        except Exception:
            # fallback if none found (dbt should raise if invalid type)
            raise RuntimeError("Could not find an adapter class by name.")
        if not adapter_cls:
            raise RuntimeError("No valid adapter class found.")

        # NOTE: this smooths over an API change upstream
        try:
            self.adapter = adapter_cls(self.config)
        except TypeError:
            from dbt.mp_context import get_mp_context

            self.adapter = adapter_cls(self.config, get_mp_context())  # pyright: ignore[reportCallIssue]

        self.adapter.connections.set_connection_name()
        self._adapter_created_at = time.time()
        setattr(self.config, "adapter", self.adapter)

    @property
    def adapter(self) -> t.Any:
        """Get the dbt adapter. Automatically refreshes if TTL exceeded."""
        if (time.time() - getattr(self, "_adapter_created_at", 0)) > self.ADAPTER_TTL:
            self.initialize_adapter()
        return self._adapter  # FIXME: add to init

    @adapter.setter
    def adapter(self, v: t.Any) -> None:
        """Set the dbt adapter. Thread-safe."""
        if self.adapter_mutex.acquire(blocking=False):
            try:
                setattr(self, "_adapter", v)
                v.debug_query()  # Verify connection
                self._adapter_created_at = time.time()
                setattr(self.config, "adapter", v)
            finally:
                self.adapter_mutex.release()

    @property
    def manifest_dict(self) -> DbtManifestProxy:
        """Get a read-only proxy for the manifest's flat_graph."""
        return DbtManifestProxy(self.manifest.flat_graph)

    def write_manifest_artifact(self) -> None:
        """Convenience method to write the manifest to disk."""
        artifact_path = Path(self.config.project_root) / self.config.target_path / "manifest.json"
        self.manifest.write(str(artifact_path))

    def clear_internal_caches(self) -> None:
        """Clear internal lru caches for the project instance."""
        self.compile_code.cache_clear()
        self.unsafe_compile_code.cache_clear()

    def get_relation(self, database: str, schema: str, name: str) -> t.Any:
        """Get a relation from the adapter."""
        return self.adapter.get_relation(database, schema, name)

    def adapter_execute(
        self, sql: str, auto_begin: bool = False, fetch: bool = False
    ) -> tuple[t.Any, t.Any]:
        """Convenience method to execute a query via the adapter."""
        return self.adapter.execute(sql, auto_begin, fetch)

    def execute_code(self, raw_code: str) -> DbtAdapterExecutionResult:
        """Execute SQL, compiling jinja if necessary and wrapping result in a consistent interface."""
        compiled = raw_code
        if has_jinja(raw_code):
            compiled = self.compile_code(raw_code).compiled_code
        resp, table = self.adapter_execute(compiled, fetch=True)
        return DbtAdapterExecutionResult(resp, table, raw_code, compiled)

    @contextmanager
    def generate_server_node(self, sql: str, node_name: str = "anonymous_node"):
        """Generate a server node, process it, and clear it after use. Mutates manifest during context."""
        with self.manifest_mutation_mutex:
            self._clear_node(node_name)
            sql_node = self.sql_parser.parse_remote(sql, node_name)
            process_node(self.config, self.manifest, sql_node)
            yield sql_node
            self._clear_node(node_name)

    def unsafe_generate_server_node(
        self, sql: str, node_name: str = "anonymous_node"
    ) -> ManifestNode:
        """Generate a server node without context, mutating manifest."""
        self._clear_node(node_name)
        sql_node = self.sql_parser.parse_remote(sql, node_name)
        process_node(self.config, self.manifest, sql_node)
        return sql_node

    def _clear_node(self, name: str) -> None:
        """Clear a node from the manifest."""
        _ = self.manifest.nodes.pop(
            f"{NodeType.SqlOperation}.{self.config.project_name}.{name}", None
        )

    @property
    def sql_parser(self) -> SqlBlockParser:
        """Lazy handle to the dbt SQL parser for the project."""
        if not self._sql_parser:
            self._sql_parser = SqlBlockParser(self.config, self.manifest, self._config)
        return self._sql_parser

    @property
    def macro_parser(self) -> SqlMacroParser:
        """Lazy handle to the dbt SQL macro parser for the project."""
        if not self._macro_parser:
            self._macro_parser = SqlMacroParser(self.config, self.manifest)
        return self._macro_parser

    def compile_from_node(self, node: ManifestNode) -> DbtAdapterCompilationResult:
        """Compile a node and wrap the result in a consistent interface."""
        compiled_node = SqlCompileRunner(
            self._config, self.adapter, node=node, node_index=1, num_nodes=1
        ).compile(self.manifest)
        return DbtAdapterCompilationResult(
            raw_code=getattr(compiled_node, "raw_code"),
            compiled_code=getattr(compiled_node, "compiled_code"),
            node=compiled_node,
        )

    @lru_cache(maxsize=100)
    def compile_code(self, raw_code: str) -> DbtAdapterCompilationResult:
        """Compile raw SQL and wrap the result in a consistent interface, leveraging lru cache."""
        tmp_id = str(uuid.uuid4())
        with self.generate_server_node(raw_code, tmp_id) as node:
            return self.compile_from_node(node)

    @lru_cache(maxsize=100)
    def unsafe_compile_code(self, raw_code: str, retry: int = 3) -> DbtAdapterCompilationResult:
        """Compile raw SQL and wrap the result in a consistent interface, leveraging lru cache. Technically less thread-safe than compile_code but faster in a high throughput server scenario"""
        tmp_id = str(uuid.uuid4())
        try:
            node = self.unsafe_generate_server_node(raw_code, tmp_id)
            return self.compile_from_node(node)
        except Exception as e:
            if retry > 0:
                return self.compile_code(raw_code)
            raise e
        finally:
            self._clear_node(tmp_id)


# TODO: we will collapse this from the file it is in currently
class ColumnLevelKnowledgePropagator:
    """Stub for doc-propagation logic. For brevity, only the relevant part is included."""

    @staticmethod
    def get_node_columns_with_inherited_knowledge(
        manifest: t.Any,
        node: ManifestNode,
        placeholders: list[str],
        project_dir: str,
        use_unrendered_descriptions: bool,
    ) -> dict[str, dict[str, t.Any]]:
        """
        Return known doc/metadata from related lineage.
        In real usage, you would gather from multiple upstream nodes.
        """
        # This is a stub.
        # For now, returning an empty dict or minimal placeholders
        _ = manifest, node, placeholders, project_dir, use_unrendered_descriptions
        return {}

    @staticmethod
    def update_undocumented_columns_with_prior_knowledge(
        columns_to_update: Iterable[str],
        node: ManifestNode,
        yaml_section: dict[str, t.Any],
        known_knowledge: dict[str, dict[str, t.Any]],
        skip_add_tags: bool,
        skip_merge_meta: bool,
        add_progenitor_to_meta: bool,
        add_inheritance_keys: list[str],
    ) -> int:
        """
        Propagate docs from known_knowledge onto columns in node + yaml_section.
        Return count of columns that changed.
        """
        _ = skip_add_tags, skip_merge_meta, add_progenitor_to_meta, add_inheritance_keys
        n = 0
        for col in columns_to_update:
            if col not in node.columns:
                continue
            cinfo = node.columns[col]
            old_desc = getattr(cinfo, "description", "")
            # If we have prior knowledge, do something
            # (for example, update cinfo.description if old_desc is blank).
            new_desc = old_desc
            if col in known_knowledge and not old_desc:
                new_desc = known_knowledge[col].get("description", "")
            if new_desc and new_desc != old_desc:
                setattr(cinfo, "description", new_desc)
                # Mirror in yaml
                for c in yaml_section.get("columns", []):
                    if c["name"].lower() == col.lower():
                        c["description"] = new_desc
                n += 1
        return n


class MissingOsmosisConfig(Exception):
    pass


class InvalidOsmosisConfig(Exception):
    pass


@dataclass
class DbtYamlManager(DbtProject):
    """Automates tasks around schema yml files, organization, coverage, etc.

    Inherits from DbtProject to access manifest and adapter.
    """

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

    _mutex: threading.Lock = threading.Lock()
    _pool: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=(os.cpu_count() or 1) * 2)
    _catalog: CatalogArtifact | None = field(default=None, init=False, repr=False)
    _mutations: int = 0

    placeholders: tuple[str, ...] = (
        "Pending further documentation",
        "Pending further documentation.",
        "No description for this column",
        "No description for this column.",
        "Not documented",
        "Not documented.",
        "Undefined",
        "Undefined.",
        "",
    )

    def __post_init__(self) -> None:
        super(DbtProject, self).__init__()  # FIXME: this is not right

        # Re-parse to ensure our newly added attributes (like skip_add_columns) are recognized
        if not list(self.filtered_models()):
            logger().warning("No models found to process given fqn/models arguments")
            logger().info("Check your filters or supply a valid model name/fqn.")
            sys.exit(0)

    @property
    def yaml_handler(self) -> YamlHandler:
        """Get a canonical YAML handler for dbt project files"""
        if not hasattr(self, "_yaml_handler"):
            self._yaml_handler = YamlHandler()  # FIXME: do like DbtProject
        return self._yaml_handler

    @property
    def catalog(self) -> CatalogArtifact | None:
        """Get the catalog artifact, loading from disk if needed."""
        if self._catalog:
            return self._catalog
        if not self.catalog_file:
            return None
        fp = Path(self.catalog_file)
        if not fp.exists():
            return None
        self._catalog = CatalogArtifact.from_dict(json.loads(fp.read_text()))
        return self._catalog

    def _filter_model_by_fqn(self, node: ManifestNode) -> bool:
        """Filter a model node by its fqn."""
        if not self.fqn:
            return True
        fqn_parts = self.fqn.split(".")
        return len(node.fqn[1:]) >= len(fqn_parts) and all(
            left == right for left, right in zip(fqn_parts, node.fqn[1:])
        )

    def _filter_model_by_models(self, node: ManifestNode) -> bool:
        """Filter a model node by its name."""
        for m in self.models:
            if node.name == m:
                return True
            node_path = self.get_node_path(node)
            inp_path = Path(m).resolve()
            if inp_path.is_dir():
                if node_path and inp_path in node_path.parents:
                    return True
            elif inp_path.is_file():
                if node_path and inp_path == node_path:
                    return True
        return False

    def _filter_model(self, node: ManifestNode) -> bool:
        """Filter a model node by fqn or models depending on input."""
        if self.models:
            filter_method = self._filter_model_by_models
        elif self.fqn:
            filter_method = self._filter_model_by_fqn
        else:
            # FIXME: make this more concise
            def _filter_method(_):
                return True

            filter_method = _filter_method

        return (
            node.resource_type in (NodeType.Model, NodeType.Source)
            and node.package_name == self.project_name
            and not (
                node.resource_type == NodeType.Model and node.config.materialized == "ephemeral"
            )
            and filter_method(node)
        )

    def filtered_models(
        self, subset: MutableMapping[str, ManifestNode] | None = None
    ) -> Iterator[tuple[str, ManifestNode]]:
        """Iterate over models in the manifest, applying filters."""
        items = (
            subset.items()
            if subset
            else chain(self.manifest.nodes.items(), self.manifest.sources.items())
        )
        for unique_id, dbt_node in items:
            if self._filter_model(dbt_node):
                yield unique_id, dbt_node

    @staticmethod
    def get_node_path(node: ManifestNode) -> Path | None:
        """Get the resolved path for a node."""
        if node.original_file_path:
            return Path(node.root_path, node.original_file_path).resolve()
        return None

    @staticmethod
    def get_patch_path(node: ManifestNode) -> Path | None:
        """Get the resolved path for a node's patch (YAML) file."""
        if node.patch_path:
            return Path(node.patch_path.split("://")[-1])
        return None

    def get_columns_meta(
        self, catalog_key: CatalogKey, output_to_lower: bool = False
    ) -> dict[str, ColumnMetadata]:
        """
        Resolve columns metadata (type, comment, etc.) either from an external CatalogArtifact
        or from a live introspection query with the adapter.
        """
        columns = OrderedDict()
        blacklist = self._config.vars.get("dbt-osmosis", {}).get("_blacklist", [])
        # if catalog is loaded:
        if self.catalog:
            # Attempt to match node in catalog
            cat_objs = {**self.catalog.nodes, **self.catalog.sources}
            matched = [
                obj for key, obj in cat_objs.items() if key.split(".")[-1] == catalog_key.name
            ]
            if matched:
                for col in matched[0].columns.values():
                    if any(re.match(pat, col.name) for pat in blacklist):
                        continue
                    columns[
                        column_casing(col.name, self._config.credentials.type, output_to_lower)
                    ] = ColumnMetadata(
                        name=column_casing(
                            col.name, self._config.credentials.type, output_to_lower
                        ),
                        type=col.type,
                        index=col.index,
                        comment=col.comment,
                    )
                return columns

        # fallback to adapter-based introspection
        with self.adapter.connection_named("dbt-osmosis"):
            table = self.adapter.get_relation(
                catalog_key.database, catalog_key.schema, catalog_key.name
            )
            if not table:
                return columns
            try:
                for c in self.adapter.get_columns_in_relation(table):
                    if any(re.match(p, c.name) for p in blacklist):
                        continue
                    col_cased = column_casing(
                        c.name, self._config.credentials.type, output_to_lower
                    )
                    columns[col_cased] = ColumnMetadata(
                        name=col_cased,
                        type=c.dtype
                        if not (
                            c.is_numeric()
                            and self.numeric_precision
                            or c.is_string()
                            and self.char_length
                        )
                        else c.data_type,
                        index=None,
                        comment=getattr(c, "comment", None),
                    )
                    if hasattr(c, "flatten"):
                        for exp in c.flatten():
                            if any(re.match(p, exp.name) for p in blacklist):
                                continue
                            col_exp_cased = column_casing(
                                exp.name, self._config.credentials.type, output_to_lower
                            )
                            columns[col_exp_cased] = ColumnMetadata(
                                name=col_exp_cased,
                                type=exp.dtype
                                if not (
                                    exp.is_numeric()
                                    and self.numeric_precision
                                    or exp.is_string()
                                    and self.char_length
                                )
                                else exp.data_type,
                                index=None,
                                comment=getattr(exp, "comment", None),
                            )
            except Exception as e:
                logger().info(f"Could not resolve columns for {catalog_key}: {e}")
        return columns

    def get_catalog_key(self, node: ManifestNode) -> CatalogKey:
        if node.resource_type == NodeType.Source:
            return CatalogKey(node.database, node.schema, getattr(node, "identifier", node.name))
        return CatalogKey(node.database, node.schema, getattr(node, "alias", node.name))

    def propagate_documentation_downstream(
        self, force_inheritance: bool = False, output_to_lower: bool = False
    ) -> None:
        schema_map = self.build_schema_folder_mapping(output_to_lower)
        futures = []
        with self.adapter.connection_named("dbt-osmosis"):
            for unique_id, node in self.filtered_models():
                futures.append(
                    self._pool.submit(
                        self._run, unique_id, node, schema_map, force_inheritance, output_to_lower
                    )
                )
            wait(futures)

    def build_schema_folder_mapping(self, output_to_lower: bool) -> dict[str, SchemaFileLocation]:
        """
        Build a mapping of model unique_id -> (target schema yml path, existing path)
        """
        self.bootstrap_sources(output_to_lower)
        out = {}
        for uid, node in self.filtered_models():
            sc_path = self.get_schema_path(node)
            target_sc_path = self.get_target_schema_path(node)
            out[uid] = SchemaFileLocation(
                target=target_sc_path.resolve(),
                current=sc_path.resolve() if sc_path else None,
                node_type=node.resource_type,
            )
        return out

    def bootstrap_sources(self, output_to_lower: bool = False) -> None:
        """
        Quick approach: if the user has declared sources in 'dbt-osmosis' vars,
        create or augment the schema files for them. For brevity, direct approach only.
        """
        performed_disk_mutation = False
        spec_dict = self._config.vars.get("dbt-osmosis", {})
        blacklist = spec_dict.get("_blacklist", [])

        for source, spec in spec_dict.items():
            if source == "_blacklist":
                continue
            if isinstance(spec, str):
                schema = source
                database = self._config.credentials.database
                path = spec
            elif isinstance(spec, dict):
                schema = spec.get("schema", source)
                database = spec.get("database", self._config.credentials.database)
                path = spec["path"]
            else:
                continue

            # Check if source in manifest
            dbt_node = next(
                (s for s in self.manifest.sources.values() if s.source_name == source), None
            )
            if not dbt_node:
                # create file with tables from introspection
                sc_file = (
                    Path(self._config.project_root)
                    / self._config.model_paths[0]
                    / path.lstrip(os.sep)
                )
                relations = self.adapter.list_relations(database=database, schema=schema)
                tables_data = []
                for rel in relations:
                    cols = []
                    for c in self.adapter.get_columns_in_relation(rel):
                        if any(re.match(p, c.name) for p in blacklist):
                            continue
                        col_cased = column_casing(
                            c.name, self._config.credentials.type, output_to_lower
                        )
                        dt = c.dtype.lower() if output_to_lower else c.dtype
                        cols.append({"name": col_cased, "description": "", "data_type": dt})
                    tables_data.append({"name": rel.identifier, "description": "", "columns": cols})

                sc_file.parent.mkdir(parents=True, exist_ok=True)
                with open(sc_file, "w") as f:
                    logger().info(f"Injecting source {source} => {sc_file}")
                    self.yaml_handler.dump(
                        {
                            "version": 2,
                            "sources": [
                                {
                                    "name": source,
                                    "database": database,
                                    "schema": schema,
                                    "tables": tables_data,
                                }
                            ],
                        },
                        f,
                    )
                self._mutations += 1
                performed_disk_mutation = True

        if performed_disk_mutation:
            logger().info("Reloading project to pick up new sources.")
            self.safe_parse_project(init=True)

    def get_schema_path(self, node: ManifestNode) -> Optional[Path]:
        if node.resource_type == NodeType.Model and node.patch_path:
            return Path(self._config.project_root).joinpath(node.patch_path.partition("://")[-1])
        if node.resource_type == NodeType.Source and hasattr(node, "source_name"):
            return Path(self._config.project_root).joinpath(node.path)
        return None

    def get_target_schema_path(self, node: ManifestNode) -> Path:
        path_spec = self.get_osmosis_path_spec(node)
        if not path_spec:
            return Path(self._config.project_root, node.original_file_path)
        sc = path_spec.format(node=node, model=node.name, parent=node.fqn[-2])
        parts = []
        if node.resource_type == NodeType.Source:
            parts.append(self._config.model_paths[0])
        else:
            parts.append(Path(node.original_file_path).parent)
        if not (sc.endswith(".yml") or sc.endswith(".yaml")):
            sc += ".yml"
        parts.append(sc)
        return Path(self._config.project_root, *parts)

    def get_osmosis_path_spec(self, node: ManifestNode) -> Optional[str]:
        if node.resource_type == NodeType.Source:
            source_specs = self._config.vars.get("dbt-osmosis", {})
            source_spec = source_specs.get(node.source_name)
            if isinstance(source_spec, dict):
                return source_spec.get("path")
            return source_spec
        osm_spec = node.unrendered_config.get("dbt-osmosis")
        if not osm_spec:
            raise MissingOsmosisConfig(f"Config not set for model {node.name}")
        return osm_spec

    def get_columns(self, key: CatalogKey, to_lower: bool) -> list[str]:
        return list(self.get_columns_meta(key, to_lower).keys())

    def get_base_model(self, node: ManifestNode, to_lower: bool) -> dict[str, t.Any]:
        cols = self.get_columns(self.get_catalog_key(node), to_lower)
        return {
            "name": node.name,
            "columns": [{"name": c, "description": ""} for c in cols],
        }

    def augment_existing_model(
        self, doc: dict[str, t.Any], node: ManifestNode, to_lower: bool
    ) -> dict[str, t.Any]:
        existing_cols = [c["name"] for c in doc.get("columns", [])]
        db_cols = self.get_columns(self.get_catalog_key(node), to_lower)
        new_cols = [c for c in db_cols if not any(c.lower() == e.lower() for e in existing_cols)]
        for col in new_cols:
            doc.setdefault("columns", []).append({"name": col, "description": ""})
            logger().info(f"Injecting column {col} into {node.unique_id}")
        return doc

    def draft_project_structure_update_plan(
        self, output_to_lower: bool = False
    ) -> dict[Path, SchemaFileMigration]:
        blueprint = {}
        logger().info("Building structure update plan.")
        futs = []
        with self.adapter.connection_named("dbt-osmosis"):
            for uid, sf_loc in self.build_schema_folder_mapping(output_to_lower).items():
                if not sf_loc.is_valid:
                    futs.append(
                        self._pool.submit(self._draft, sf_loc, uid, blueprint, output_to_lower)
                    )
            wait(futs)
        return blueprint

    def _draft(
        self,
        sf_loc: SchemaFileLocation,
        uid: str,
        blueprint: dict[Path, SchemaFileMigration],
        to_lower: bool,
    ):
        try:
            with self._mutex:
                if sf_loc.target not in blueprint:
                    blueprint[sf_loc.target] = SchemaFileMigration()
            if sf_loc.node_type == NodeType.Model:
                node = self.manifest.nodes[uid]
            else:
                node = self.manifest.sources[uid]

            if sf_loc.current is None:
                # model not documented yet
                with self._mutex:
                    if sf_loc.node_type == NodeType.Model:
                        blueprint[sf_loc.target].output["models"].append(
                            self.get_base_model(node, to_lower)
                        )
            else:
                # We have existing doc, but we want to unify it into the new location
                with self._mutex:
                    doc = self.yaml_handler.load(sf_loc.current)
                if sf_loc.node_type == NodeType.Model:
                    for m in doc.get("models", []):
                        if m["name"] == node.name:
                            newm = self.augment_existing_model(m, node, to_lower)
                            with self._mutex:
                                blueprint[sf_loc.target].output["models"].append(newm)
                                blueprint[sf_loc.target].supersede.setdefault(
                                    sf_loc.current, []
                                ).append(node)
                            break
                else:
                    for source in doc.get("sources", []):
                        if source["name"] == node.source_name:
                            for table in source["tables"]:
                                if table["name"] == node.name:
                                    newt = self.augment_existing_model(table, node, to_lower)
                                    with self._mutex:
                                        if not any(
                                            s["name"] == node.source_name
                                            for s in blueprint[sf_loc.target].output["sources"]
                                        ):
                                            blueprint[sf_loc.target].output["sources"].append(
                                                source
                                            )
                                        for s in blueprint[sf_loc.target].output["sources"]:
                                            if s["name"] == node.source_name:
                                                for t2 in s["tables"]:
                                                    if t2["name"] == node.name:
                                                        t2.update(newt)
                                                        break
                                        blueprint[sf_loc.target].supersede.setdefault(
                                            sf_loc.current, []
                                        ).append(node)
                                    break
        except Exception as e:
            logger().error(f"Drafting structure plan for {uid} failed: {e}")
            raise e

    def cleanup_blueprint(
        self, blueprint: dict[Path, SchemaFileMigration]
    ) -> dict[Path, SchemaFileMigration]:
        for k in list(blueprint.keys()):
            out = blueprint[k].output
            # remove empty models/sources
            if "models" in out and not out["models"]:
                del out["models"]
            if "sources" in out and not out["sources"]:
                del out["sources"]
            if not out.get("models") and not out.get("sources"):
                del blueprint[k]
        return blueprint

    def commit_project_restructure_to_disk(
        self,
        blueprint: Optional[dict[Path, SchemaFileMigration]] = None,
        output_to_lower: bool = False,
    ) -> bool:
        if not blueprint:
            blueprint = self.draft_project_structure_update_plan(output_to_lower)
        blueprint = self.cleanup_blueprint(blueprint)
        if not blueprint:
            logger().info("Project structure is already conformed.")
            return False
        self.pretty_print_restructure_plan(blueprint)

        for target, struct in blueprint.items():
            if not target.exists():
                logger().info(f"Creating schema file {target}")
                if not self.dry_run:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.touch()
                    self.yaml_handler.dump(struct.output, target)
                    self._mutations += 1
            else:
                logger().info(f"Updating schema file {target}")
                existing = self.yaml_handler.load(target)
                if not existing:
                    existing = {"version": 2}
                if "version" not in existing:
                    existing["version"] = 2

                if "models" in struct.output:
                    existing.setdefault("models", []).extend(struct.output["models"])
                if "sources" in struct.output:
                    existing.setdefault("sources", []).extend(struct.output["sources"])
                if not self.dry_run:
                    self.yaml_handler.dump(existing, target)
                    self._mutations += 1

            # handle superseded
            for sup_path, nodes in struct.supersede.items():
                raw_sc = self.yaml_handler.load(sup_path)
                # figure out which ones to remove
                to_remove_models = {n.name for n in nodes if n.resource_type == NodeType.Model}
                to_remove_sources = {
                    (n.source_name, n.name) for n in nodes if n.resource_type == NodeType.Source
                }

                keep_models = []
                for m in raw_sc.get("models", []):
                    if m["name"] not in to_remove_models:
                        keep_models.append(m)
                raw_sc["models"] = keep_models

                # remove relevant source tables
                keep_src = []
                for s in raw_sc.get("sources", []):
                    keep_tables = []
                    for t_ in s.get("tables", []):
                        if (s["name"], t_["name"]) not in to_remove_sources:
                            keep_tables.append(t_)
                    if keep_tables:
                        s["tables"] = keep_tables
                        keep_src.append(s)
                raw_sc["sources"] = keep_src

                # if file is empty => remove it
                if (not raw_sc.get("models")) and (not raw_sc.get("sources")):
                    logger().info(f"Superseding entire file {sup_path}")
                    if not self.dry_run:
                        sup_path.unlink(missing_ok=True)
                        if sup_path.parent.exists() and not any(sup_path.parent.iterdir()):
                            sup_path.parent.rmdir()
                else:
                    if not self.dry_run:
                        self.yaml_handler.dump(raw_sc, sup_path)
                        self._mutations += 1
                    logger().info(f"Migrated doc from {sup_path} -> {target}")
        return True

    @staticmethod
    def pretty_print_restructure_plan(blueprint: dict[Path, SchemaFileMigration]) -> None:
        summary = []
        for plan in blueprint.keys():
            files_superseded = [s.name for s in blueprint[plan].supersede] or ["CREATE"]
            summary.append((files_superseded, "->", plan.name))
        logger().info(summary)

    ############################################################################
    # Column Sync
    ############################################################################
    @staticmethod
    def get_column_sets(
        database_cols: Iterable[str],
        yaml_cols: Iterable[str],
        documented_cols: Iterable[str],
    ) -> t.tuple[list[str], list[str], list[str]]:
        """
        Return: (missing_in_yaml, undocumented_in_yaml, extra_in_yaml)
        """
        missing = [x for x in database_cols if x.lower() not in (y.lower() for y in yaml_cols)]
        undocumented = [
            x for x in database_cols if x.lower() not in (y.lower() for y in documented_cols)
        ]
        extra = [x for x in yaml_cols if x.lower() not in (y.lower() for y in database_cols)]
        return missing, undocumented, extra

    def _run(
        self,
        uid: str,
        node: ManifestNode,
        schema_map: dict[str, SchemaFileLocation],
        force_inheritance: bool,
        output_to_lower: bool,
    ):
        try:
            with self._mutex:
                logger().info(f"Processing model: {uid}")
            sf_loc = schema_map.get(uid)
            if not sf_loc or not sf_loc.current:
                with self._mutex:
                    logger().info(f"No schema file for {uid}, skipping.")
                return
            db_cols_list = self.get_columns(self.get_catalog_key(node), output_to_lower)
            if not db_cols_list:
                with self._mutex:
                    logger().info(
                        f"No database columns found for {uid}, falling back to yaml columns."
                    )
                db_cols_list = list(node.columns.keys())

            db_cols_set = set(db_cols_list)
            yaml_cols_list = list(node.columns.keys())
            documented_cols_set = {
                c
                for c, info in node.columns.items()
                if info.description and info.description not in self.placeholders
            }

            missing, undocumented, extra = self.get_column_sets(
                db_cols_list, yaml_cols_list, documented_cols_set
            )

            if force_inheritance:
                undocumented = list(db_cols_set)  # treat all as needing doc

            with self._mutex:
                sc_data = self.yaml_handler.load(sf_loc.current)
                section = self.maybe_get_section_from_schema_file(sc_data, node)
                if not section:
                    logger().info(f"No section in {sf_loc.current} for {uid}")
                    return
                # Perform updates
                n_added = n_doc_inh = n_removed = n_type_changed = n_desc_changed = 0
                if any([missing, undocumented, extra]):
                    (
                        n_added,
                        n_doc_inh,
                        n_removed,
                        n_type_changed,
                        n_desc_changed,
                    ) = self.update_schema_file_and_node(
                        missing,
                        undocumented,
                        extra,
                        node,
                        section,
                        self.get_columns_meta(self.get_catalog_key(node), output_to_lower),
                        output_to_lower,
                    )

                reorder = tuple(db_cols_list) != tuple(yaml_cols_list)
                if reorder:

                    def _sort(c: dict[str, t.Any]) -> int:
                        try:
                            return db_cols_list.index(
                                column_casing(
                                    c["name"], self._config.credentials.type, output_to_lower
                                )
                            )
                        except ValueError:
                            return 999999

                    section["columns"].sort(key=_sort)

                if (
                    n_added + n_doc_inh + n_removed + n_type_changed + n_desc_changed or reorder
                ) and not self.dry_run:
                    self.yaml_handler.dump(sc_data, sf_loc.current)
                    self._mutations += 1
                    logger().info(f"Updated {sf_loc.current}")
                else:
                    logger().info(f"{sf_loc.current} is up to date")

        except Exception as e:
            logger().error(f"Error while processing {uid}: {e}")
            raise e

    @staticmethod
    def maybe_get_section_from_schema_file(
        yaml_data: dict[str, t.Any], node: ManifestNode
    ) -> Optional[dict[str, t.Any]]:
        if node.resource_type == NodeType.Source:
            for s in yaml_data.get("sources", []):
                for t_ in s.get("tables", []):
                    if s["name"] == node.source_name and t_["name"] == node.name:
                        return t_
        else:
            for m in yaml_data.get("models", []):
                if m["name"] == node.name:
                    return m
        return None

    @staticmethod
    def remove_columns_not_in_database(
        extra_columns: Iterable[str],
        node: ManifestNode,
        yaml_section: dict[str, t.Any],
    ) -> int:
        c = 0
        for e in extra_columns:
            node.columns.pop(e, None)
            yaml_section["columns"] = [col for col in yaml_section["columns"] if col["name"] != e]
            c += 1
        return c

    def update_columns_attribute(
        self,
        node: ManifestNode,
        yaml_section: dict[str, t.Any],
        db_meta: dict[str, ColumnMetadata],
        attr: str,
        meta_key: str,
        skip_flag: bool,
        output_to_lower: bool,
    ) -> int:
        if skip_flag:
            return 0
        changed = 0
        for col_name, col_meta in db_meta.items():
            if col_name in node.columns:
                new_val = getattr(col_meta, meta_key, "") or ""
                old_val = getattr(node.columns[col_name], attr, "")
                if new_val and old_val != new_val:
                    setattr(node.columns[col_name], attr, new_val)
                    for c in yaml_section["columns"]:
                        if (
                            column_casing(c["name"], self._config.credentials.type, output_to_lower)
                            == col_name
                        ):
                            if output_to_lower and isinstance(new_val, str):
                                new_val = new_val.lower()
                            c[attr] = new_val
                    changed += 1
        return changed

    def add_missing_cols_to_node_and_model(
        self,
        missing_cols: Iterable[str],
        node: ManifestNode,
        yaml_section: dict[str, t.Any],
        db_meta: dict[str, ColumnMetadata],
        output_to_lower: bool,
    ) -> int:
        c = 0
        for col in missing_cols:
            if col not in db_meta:
                continue
            dtype = db_meta[col].type or ""
            desc = db_meta[col].comment or ""
            meta_name = col.lower() if output_to_lower else col
            meta_type = dtype.lower() if output_to_lower else dtype
            node.columns[col] = ColumnInfo.from_dict(
                {"name": meta_name, "description": desc, "data_type": meta_type}
            )
            yaml_section.setdefault("columns", []).append(
                {"name": meta_name, "description": desc, "data_type": meta_type}
            )
            c += 1
        return c

    def update_schema_file_and_node(
        self,
        missing_cols: Iterable[str],
        undocumented_cols: Iterable[str],
        extra_cols: Iterable[str],
        node: ManifestNode,
        yaml_section: dict[str, t.Any],
        db_meta: dict[str, ColumnMetadata],
        output_to_lower: bool,
    ) -> t.tuple[int, int, int, int, int]:
        n_added = 0
        n_doc_inherited = 0
        n_removed = 0
        n_type_updated = 0
        n_desc_updated = 0

        if not self.skip_add_columns:
            n_added = self.add_missing_cols_to_node_and_model(
                missing_cols, node, yaml_section, db_meta, output_to_lower
            )

        knowledge = ColumnLevelKnowledgePropagator.get_node_columns_with_inherited_knowledge(
            self.manifest,
            node,
            self.placeholders,
            self._config.project_root,
            self.use_unrendered_descriptions,
        )
        n_doc_inherited = (
            ColumnLevelKnowledgePropagator.update_undocumented_columns_with_prior_knowledge(
                undocumented_cols,
                node,
                yaml_section,
                knowledge,
                self.skip_add_tags,
                self.skip_merge_meta,
                self.add_progenitor_to_meta,
                self.add_inheritance_for_specified_keys,
            )
        )
        n_type_updated = self.update_columns_attribute(
            node,
            yaml_section,
            db_meta,
            attr="data_type",
            meta_key="type",
            skip_flag=self.skip_add_data_types,
            output_to_lower=output_to_lower,
        )
        # We piggyback the "catalog_file" presence as "update description?" flag in original code
        n_desc_updated = self.update_columns_attribute(
            node,
            yaml_section,
            db_meta,
            attr="description",
            meta_key="comment",
            skip_flag=(self.catalog_file is None),
            output_to_lower=output_to_lower,
        )
        n_removed = self.remove_columns_not_in_database(extra_cols, node, yaml_section)
        return n_added, n_doc_inherited, n_removed, n_type_updated, n_desc_updated
