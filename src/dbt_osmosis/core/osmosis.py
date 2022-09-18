import os
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache

from hashlib import md5
from itertools import chain
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Set,
    Tuple,
    Union,
    cast,
)

import agate
import sqlparse
from dbt.adapters.factory import Adapter, get_adapter, register_adapter, reset_adapters
from dbt.clients import jinja
from dbt.clients.jinja import extract_toplevel_blocks
from dbt.config.runtime import RuntimeConfig
from dbt.context.providers import generate_runtime_model_context
from dbt.contracts.connection import AdapterResponse
from dbt.contracts.graph.compiled import (
    COMPILED_TYPES,
    CompiledGenericTestNode,
    InjectedCTE,
    ManifestNode,
    NonSourceCompiledNode,
)
from dbt.contracts.graph.manifest import Manifest, ManifestNode, NodeType, SourceFile
from dbt.contracts.graph.parsed import ColumnInfo, ParsedMacro, ParsedModelNode, ParsedSqlNode
from dbt.contracts.graph.unparsed import UnparsedMacro
from dbt.exceptions import CompilationException, InternalException, RuntimeException
from dbt.flags import DEFAULT_PROFILES_DIR, env_set_truthy, set_from_args
from dbt.node_types import NodeType
from dbt.parser.base import SimpleSQLParser
from dbt.parser.macros import MacroParser
from dbt.parser.manifest import ManifestLoader, process_macro, process_node
from dbt.parser.search import FileBlock
from dbt.tracking import disable_tracking
from pydantic import BaseModel
from rich.progress import track
from ruamel.yaml import YAML

from dbt_osmosis.core.exceptions import (
    InvalidOsmosisConfig,
    MissingOsmosisConfig,
    SanitizationRequired,
)
from dbt_osmosis.core.log_controller import logger


class MemoContainer:
    _MEMO = {}


def memoize_get_rendered(function):
    def wrapper(
        string: str,
        ctx: Dict[str, Any],
        node=None,
        capture_macros: bool = False,
        native: bool = False,
    ):
        v = md5(string.encode("utf-8")).hexdigest()
        if capture_macros == True and node is not None:
            # Now the node is important, cache mutated node
            v += node.name
        rv = MemoContainer._MEMO.get(v)
        if rv is not None:
            return rv
        else:
            rv = function(string, ctx, node, capture_macros, native)
            MemoContainer._MEMO[v] = rv
            return rv

    return wrapper


disable_tracking()
jinja.get_rendered = memoize_get_rendered(jinja.get_rendered)

AUDIT_REPORT = """
:white_check_mark: [bold]Audit Report[/bold]
-------------------------------

Database: [bold green]{database}[/bold green]
Schema: [bold green]{schema}[/bold green]
Table: [bold green]{table}[/bold green]

Total Columns in Database: {total_columns}
Total Documentation Coverage: {coverage}%

Action Log:
Columns Added to dbt: {n_cols_added}
Column Knowledge Inherited: {n_cols_doc_inherited}
Extra Columns Removed: {n_cols_removed}
"""

# TODO: Let user supply a custom config file / csv of strings which we consider "not-documented placeholders", these are just my own
PLACEHOLDERS = [
    "Pending further documentation",
    "Pending further documentation.",
    "No description for this column",
    "No description for this column.",
    "Not documented",
    "Not documented.",
    "Undefined",
    "Undefined.",
    "",
]

FILE_ADAPTER_POSTFIX = "://"

SINGLE_THREADED_HANDLER = env_set_truthy("DBT_SINGLE_THREADED_HANDLER")


class PseudoArgs:
    def __init__(
        self,
        threads: Optional[int] = 1,
        target: Optional[str] = None,
        profiles_dir: Optional[str] = None,
        project_dir: Optional[str] = None,
        vars: Optional[str] = "{}",
    ):
        self.threads = threads
        if target:
            self.target = target  # We don't want target in args context if it is None
        self.profiles_dir = profiles_dir or DEFAULT_PROFILES_DIR
        self.project_dir = project_dir
        self.vars = vars  # json.dumps str
        self.dependencies = []
        self.single_threaded = threads == 1
        self.quiet = True


class OsmosisConfig(str, Enum):
    SchemaYaml = "schema.yml"
    FolderYaml = "folder.yml"
    ModelYaml = "model.yml"
    UnderscoreModelYaml = "_model.yml"
    SchemaModelYaml = "schema/model.yml"


class SchemaFile(BaseModel):
    target: Path
    current: Optional[Path] = None

    @property
    def is_valid(self) -> bool:
        return self.current == self.target


class RestructureQuantum(BaseModel):
    output: Dict[str, Any] = {}
    supersede: Dict[Path, List[str]] = {}


@dataclass
class SqlBlock(FileBlock):
    sql_name: str

    @property
    def name(self):
        return self.sql_name


class LocalCallParser(SimpleSQLParser[ParsedSqlNode]):
    def parse_from_dict(self, dct, validate=True) -> ParsedSqlNode:
        if validate:
            ParsedSqlNode.validate(dct)
        return ParsedSqlNode.from_dict(dct)

    @property
    def resource_type(self) -> NodeType:
        return NodeType.SqlOperation

    def get_compiled_path(cls, block: FileBlock):
        if not isinstance(block, SqlBlock):
            raise InternalException(
                "While parsing SQL calls, got an actual file block instead of "
                "an SQL block: {}".format(block)
            )

        return os.path.join("sql", block.name)

    def parse_remote(self, sql: str, name: str) -> ParsedSqlNode:
        """Add a node to the manifest parsing arbitrary SQL"""
        source_file = SourceFile.remote(sql, self.project.project_name)
        contents = SqlBlock(sql_name=name, file=source_file)
        return self.parse_node(contents)


class LocalMacroParser(MacroParser):
    @lru_cache(maxsize=None)
    def parse_remote(self, contents) -> Iterable[ParsedMacro]:
        base = UnparsedMacro(
            path="from local system",
            original_file_path="from local system",
            package_name=self.project.project_name,
            raw_sql=contents,
            root_path=self.project.project_root,
            resource_type=NodeType.Macro,
        )
        return [node for node in self.parse_unparsed_macros(base)]


class DbtOsmosis:
    def __init__(
        self,
        fqn: Optional[str] = None,
        target: Optional[str] = None,
        profiles_dir: Optional[str] = None,
        project_dir: Optional[str] = None,
        threads: Optional[int] = 1,
        dry_run: bool = False,
    ):
        # Build pseudo args
        args = PseudoArgs(
            threads=threads,
            target=target,
            profiles_dir=profiles_dir,
            project_dir=project_dir,
        )
        self.args = args

        # Load dbt + verify connection to data warehhouse
        set_from_args(args, args)
        self.project, self.profile = RuntimeConfig.collect_parts(args)
        self.config = RuntimeConfig.from_parts(self.project, self.profile, args)
        reset_adapters()
        register_adapter(self.config)
        self.adapter = self._verify_connection(get_adapter(self.config))

        # Parse project
        self.dbt = ManifestLoader.get_full_manifest(self.config)

        # Selector Passed in From CLI
        self.fqn = fqn

        # Utilities
        self.yaml = self._build_yaml_parser()
        self.dry_run = dry_run
        self.track_package_install = (
            lambda *args, **kwargs: None
        )  # Monkey patching to make self compatible with DepsTask
        self._cached_exec_nodes = set()

        # Dependency injection
        self._sql_parser = None
        self._macro_parser = None

    # PARSERS

    @property
    def sql_parser(self) -> LocalCallParser:
        if self._sql_parser is None:
            self._sql_parser = LocalCallParser(self.project, self.dbt, self.config)
        return self._sql_parser

    @property
    def macro_parser(self) -> LocalMacroParser:
        if self._macro_parser is None:
            self._macro_parser = LocalMacroParser(self.project, self.dbt)
        return self._macro_parser

    @staticmethod
    def _build_yaml_parser() -> YAML:
        yaml = YAML()
        yaml.indent(mapping=2, sequence=4, offset=2)
        yaml.width = 800
        yaml.preserve_quotes = True
        yaml.default_flow_style = False
        return yaml

    # RUNTIME VALIDATORS

    @staticmethod
    def _verify_connection(adapter: Adapter) -> Adapter:
        try:
            with adapter.connection_named("debug"):
                adapter.debug_query()
        except Exception as exc:
            raise Exception("Could not connect to Database") from exc
        else:
            return adapter

    # HELPERS

    @property
    def project_name(self) -> str:
        return self.project.project_name

    @property
    def project_root(self) -> str:
        return self.project.project_root

    @property
    def manifest(self) -> Dict[str, Any]:
        return self.dbt.flat_graph

    # REPARSE

    def rebuild_dbt_manifest(self, reset: bool = False) -> None:
        reset_adapters()
        if reset:
            # Make this as atomic as possible
            self.clear_caches()
            proj, prof, conf = self.project, self.profile, self.config
            try:
                self.rebuild_project_and_profile()
                self.rebuild_config()
            except Exception as parse_error:
                self.project, self.profile, self.config = proj, prof, conf
                register_adapter(self.config)
                raise parse_error
        register_adapter(self.config)
        self.dbt = ManifestLoader.get_full_manifest(self.config, reset=reset)
        self._sql_parser = LocalCallParser(self.project, self.dbt, self.config)
        self._macro_parser = LocalMacroParser(self.project, self.dbt)
        path = os.path.join(self.config.project_root, self.config.target_path, "manifest.json")
        logger().debug("Rewriting manifest to %s", path)
        self.dbt.write(path)

    def rebuild_project_and_profile(self):
        self.project, self.profile = RuntimeConfig.collect_parts(self.args)

    def rebuild_config(self):
        self.config = RuntimeConfig.from_parts(self.project, self.profile, self.args)

    def clear_caches(self):
        MemoContainer._MEMO.clear()
        self._extract_jinja_data.cache_clear()

    # FIND MODEL

    def get_ref_node(self, target_model_name: str):
        """
        target_model_name:
            ie, ref('stg_users') = name -> stg_users
        """
        return self.dbt.resolve_ref(
            target_model_name=target_model_name,
            target_model_package=None,
            current_project=self.config.project_name,
            node_package=self.config.project_name,
        )

    def get_source_node(self, target_source_name: str, target_table_name: str):
        """
        target_table_name:
            ie, source('jira', 'users') = source_name -> jira, table_name -> users
        """
        return self.dbt.resolve_source(
            target_source_name=target_source_name,
            target_table_name=target_table_name,
            current_project=self.config.project_name,
            node_package=self.config.project_name,
        )

    # DBT EXECUTION

    def execute_macro(
        self,
        macro: str,
        kwargs: Optional[Dict[str, Any]] = None,
        run_compiled_sql: bool = False,
        fetch: bool = False,
    ) -> Tuple[
        str, Optional[AdapterResponse], Optional[agate.Table]
    ]:  # returns Macro `return` value from Jinja be it string, SQL, or dict
        """Wraps adapter execute_macro"""
        with self.adapter.connection_named("dbt-osmosis"):
            compiled_macro = self.adapter.execute_macro(
                macro_name=macro, manifest=self.dbt, kwargs=kwargs
            )
            if run_compiled_sql:
                resp, table = self.adapter.execute(compiled_macro, fetch=fetch)
                return compiled_macro, resp, table
        return compiled_macro, None, None

    def execute_sql(
        self,
        sql: str,
        compile: bool = False,
        fetch: bool = False,
    ) -> Tuple[AdapterResponse, agate.Table]:
        """Wraps adapter execute"""
        if compile:
            sql = self.compile_sql(sql).compiled_sql
        with self.adapter.connection_named("dbt-osmosis"):
            resp, table = self.adapter.execute(sql, fetch=fetch)
        return resp, table

    # DBT COMPILATION

    def compile_sql(self, sql: str, name: str = "dbt_osmosis_node") -> ManifestNode:
        """Compile dbt SQL 🔥"""
        self.dbt.nodes.pop(f"{NodeType.SqlOperation}.{self.project_name}.{name}", None)
        with self.adapter.connection_named("dbt-osmosis"):
            return self.compile_node(self._get_exec_node(sql, name=name))

    def compile_node(
        self,
        node: ManifestNode,
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> NonSourceCompiledNode:

        if extra_context is None:
            extra_context = {}

        data = node.to_dict(omit_none=True)
        data.update(
            {
                "compiled": False,
                "compiled_sql": None,
                "extra_ctes_injected": False,
                "extra_ctes": [],
            }
        )
        compiled_node = _compiled_type_for(node).from_dict(data)
        context = self._create_node_context(compiled_node, extra_context)

        compiled_node.compiled_sql = jinja.get_rendered(
            node.raw_sql,
            context,
            node,
        )

        compiled_node.relation_name = self._get_relation_name(node)
        compiled_node.compiled = True
        compiled_node, _ = self._recursively_prepend_ctes(compiled_node, extra_context)
        return compiled_node

    def _add_new_refs(self, node: ParsedSqlNode, macros: Dict[str, Any]) -> None:

        # TODO: Vet multi-threaded execution as-is
        # if self.args.single_threaded or SINGLE_THREADED_HANDLER:
        #     manifest = self.dbt.deepcopy()
        # else:
        #     manifest = self.dbt

        self.dbt.macros.update(macros)

        for macro in macros.values():
            process_macro(self.config, self.dbt, macro)

        process_node(self.config, self.dbt, node)

    @lru_cache(maxsize=None)
    def _extract_jinja_data(self, sql: str):
        macro_blocks = []
        data_chunks = []
        for block in extract_toplevel_blocks(sql):
            if block.block_type_name == "macro":
                macro_blocks.append(block.full_block)
            else:
                data_chunks.append(block.full_block)
        macros = "\n".join(macro_blocks)
        sql = "".join(data_chunks)
        return sql, macros

    def _get_exec_node(self, sql: str, name: str = "dbt_osmosis_node"):
        macro_overrides = {}
        sql, macros = self._extract_jinja_data(sql)

        if macros:
            for node in self.macro_parser.parse_remote(macros):
                macro_overrides[node.unique_id] = node
            self.dbt.macros.update(macro_overrides)

        # TODO: Alternatively, we could supply a node and make it FAST when working with existing models
        # Adds node to manifest (dbt)
        sql_node = self.sql_parser.parse_remote(sql, name)

        # Populate depends_on
        # TODO: If we use an existing node, we should set node.depends_on to [] since its an append op to update
        sql_node.depends_on.nodes = []
        self._add_new_refs(node=sql_node, macros=macro_overrides)

        # TODO: the extra compile below is likely uneeded
        # self.graph = self.adapter.get_compiler().compile(self.dbt, write=False)
        return sql_node

    def _create_node_context(
        self,
        node: NonSourceCompiledNode,
        extra_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Creates a ModelContext which is converted to a dict for jinja rendering of SQL"""
        context = generate_runtime_model_context(node, self.config, self.dbt)
        context.update(extra_context)
        if isinstance(node, CompiledGenericTestNode):
            # for test nodes, add a special keyword args value to the context
            jinja.add_rendered_test_kwargs(context, node)

        return context

    def _recursively_prepend_ctes(
        self,
        model: NonSourceCompiledNode,
        extra_context: Optional[Dict[str, Any]],
    ) -> Tuple[NonSourceCompiledNode, List[InjectedCTE]]:
        if model.compiled_sql is None:
            raise RuntimeException("Cannot inject ctes into an unparsed node", model)
        if model.extra_ctes_injected:
            return (model, model.extra_ctes)

        if not model.extra_ctes:
            model.extra_ctes_injected = True
            self.dbt.update_node(model)
            return (model, model.extra_ctes)

        prepended_ctes: List[InjectedCTE] = []

        for cte in model.extra_ctes:
            if cte.id not in self.dbt.nodes:
                raise InternalException(
                    f"During compilation, found a cte reference that "
                    f"could not be resolved: {cte.id}"
                )
            cte_model = self.dbt.nodes[cte.id]

            if not cte_model.is_ephemeral_model:
                raise InternalException(f"{cte.id} is not ephemeral")

            if getattr(cte_model, "compiled", False):
                assert isinstance(cte_model, tuple(COMPILED_TYPES.values()))
                cte_model = cast(NonSourceCompiledNode, cte_model)
                new_prepended_ctes = cte_model.extra_ctes

            else:
                cte_model = self.compile_node(cte_model, extra_context)
                cte_model, new_prepended_ctes = self._recursively_prepend_ctes(
                    cte_model, extra_context
                )
                self.dbt.sync_update_node(cte_model)

            _extend_prepended_ctes(prepended_ctes, new_prepended_ctes)

            new_cte_name = self._add_ephemeral_prefix(cte_model.name)
            rendered_sql = cte_model._pre_injected_sql or cte_model.compiled_sql
            sql = f" {new_cte_name} as (\n{rendered_sql}\n)"

            _add_prepended_cte(prepended_ctes, InjectedCTE(id=cte.id, sql=sql))

        injected_sql = self._inject_ctes_into_sql(
            model.compiled_sql,
            prepended_ctes,
        )
        model._pre_injected_sql = model.compiled_sql
        model.compiled_sql = injected_sql
        model.extra_ctes_injected = True
        model.extra_ctes = prepended_ctes
        model.validate(model.to_dict(omit_none=True))

        self.dbt.update_node(model)

        return model, prepended_ctes

    def _add_ephemeral_prefix(self, name: str):
        relation_cls = self.adapter.Relation
        return relation_cls.add_ephemeral_prefix(name)

    def _inject_ctes_into_sql(self, sql: str, ctes: List[InjectedCTE]) -> str:
        if len(ctes) == 0:
            return sql

        parsed_stmts = sqlparse.parse(sql)
        parsed = parsed_stmts[0]

        with_stmt = None
        for token in parsed.tokens:
            if token.is_keyword and token.normalized == "WITH":
                with_stmt = token
                break

        if with_stmt is None:
            # no with stmt, add one, and inject CTEs right at the beginning
            first_token = parsed.token_first()
            with_stmt = sqlparse.sql.Token(sqlparse.tokens.Keyword, "with")
            parsed.insert_before(first_token, with_stmt)
        else:
            # stmt exists, add a comma (which will come after injected CTEs)
            trailing_comma = sqlparse.sql.Token(sqlparse.tokens.Punctuation, ",")
            parsed.insert_after(with_stmt, trailing_comma)

        token = sqlparse.sql.Token(sqlparse.tokens.Keyword, ", ".join(c.sql for c in ctes))
        parsed.insert_after(with_stmt, token)

        return str(parsed)

    def _safe_release_connection(self):
        try:
            self.adapter.release_connection()
        except Exception as exc:
            return str(exc)
        return None

    def _get_relation_name(self, node: ManifestNode):
        relation_name = None
        if node.is_relational and not node.is_ephemeral_model:
            relation_cls = self.adapter.Relation
            relation_name = str(relation_cls.create_from(self.config, node))
        return relation_name

    # DBT-OSMOSIS SPECIFIC METHODS

    def _filter_model(self, node: ManifestNode) -> bool:
        """Validates a node as being a targetable model. Validates both models and sources."""
        fqn = self.fqn or ".".join(node.fqn[1:])
        fqn_parts = fqn.split(".")
        logger().debug("%s: %s -> %s", node.resource_type, fqn, node.fqn[1:])
        return (
            # Verify Resource Type
            node.resource_type in (NodeType.Model, NodeType.Source)
            # Verify Package == Current Project
            and node.package_name == self.project_name
            # Verify Materialized is Not Ephemeral if NodeType is Model [via short-circuit]
            and (node.resource_type != NodeType.Model or node.config.materialized != "ephemeral")
            # Verify FQN Length [Always true if no fqn was supplied]
            and len(node.fqn[1:]) >= len(fqn_parts)
            # Verify FQN Matches Parts [Always true if no fqn was supplied]
            and all(left == right for left, right in zip(fqn_parts, node.fqn[1:]))
        )

    @staticmethod
    def get_patch_path(node: ManifestNode) -> Optional[Path]:
        if node is not None and node.patch_path:
            return Path(node.patch_path.split(FILE_ADAPTER_POSTFIX)[-1])

    def filtered_models(
        self, subset: Optional[MutableMapping[str, ManifestNode]] = None
    ) -> Iterator[Tuple[str, ManifestNode]]:
        """Generates an iterator of valid models"""
        for unique_id, dbt_node in (
            subset.items() if subset else chain(self.dbt.nodes.items(), self.dbt.sources.items())
        ):
            if self._filter_model(dbt_node):
                yield unique_id, dbt_node

    def get_osmosis_config(self, node: ManifestNode) -> Optional[OsmosisConfig]:
        """Validates a config string. If input is a source, we return the resource type str instead"""
        if node.resource_type == NodeType.Source:
            return None
        osmosis_config = node.config.get("dbt-osmosis")
        if not osmosis_config:
            raise MissingOsmosisConfig(
                f"Config not set for model {node.name}, we recommend setting the config at a directory level through the `dbt_project.yml`"
            )
        try:
            return OsmosisConfig(osmosis_config)
        except ValueError as exc:
            raise InvalidOsmosisConfig(
                f"Invalid config for model {node.name}: {osmosis_config}"
            ) from exc

    def get_schema_path(self, node: ManifestNode) -> Optional[Path]:
        """Resolve absolute schema file path for a manifest node"""
        schema_path = None
        if node.resource_type == NodeType.Model and node.patch_path:
            schema_path: str = node.patch_path.partition(FILE_ADAPTER_POSTFIX)[-1]
        elif node.resource_type == NodeType.Source:
            if hasattr(node, "source_name"):
                schema_path: str = node.path
        if schema_path:
            return Path(self.project_root).joinpath(schema_path)

    def get_target_schema_path(self, node: ManifestNode) -> Path:
        """Resolve the correct schema yml target based on the dbt-osmosis config for the model / directory"""
        osmosis_config = self.get_osmosis_config(node)
        if not osmosis_config:
            return Path(node.root_path, node.original_file_path)
        # Here we resolve file migration targets based on the config
        if osmosis_config == OsmosisConfig.SchemaYaml:
            schema = "schema"
        elif osmosis_config == OsmosisConfig.FolderYaml:
            schema = node.fqn[-2]
        elif osmosis_config == OsmosisConfig.ModelYaml:
            schema = node.name
        elif osmosis_config == OsmosisConfig.SchemaModelYaml:
            schema = "schema/" + node.name
        elif osmosis_config == OsmosisConfig.UnderscoreModelYaml:
            schema = "_" + node.name
        else:
            raise InvalidOsmosisConfig(f"Invalid dbt-osmosis config for model: {node.fqn}")
        return Path(node.root_path, node.original_file_path).parent / Path(f"{schema}.yml")

    @staticmethod
    def get_database_parts(node: ManifestNode) -> Tuple[str, str, str]:
        return node.database, node.schema, getattr(node, "alias", node.name)

    def get_base_model(self, node: ManifestNode) -> Dict[str, Any]:
        """Construct a base model object with model name, column names populated from database"""
        columns = self.get_columns(node)
        return {
            "name": node.alias or node.name,
            "columns": [{"name": column_name} for column_name in columns],
        }

    def bootstrap_existing_model(
        self, model_documentation: Dict[str, Any], node: ManifestNode
    ) -> Dict[str, Any]:
        """Injects columns from database into existing model if not found"""
        model_columns: List[str] = [
            c["name"].lower() for c in model_documentation.get("columns", [])
        ]
        database_columns = self.get_columns(node)
        for column in database_columns:
            if column.lower() not in model_columns:
                logger().info(":syringe: Injecting column %s into dbt schema", column)
                model_documentation.setdefault("columns", []).append({"name": column})
        return model_documentation

    def get_columns(self, node: ManifestNode) -> List[str]:
        """Get all columns in a list for a model"""
        parts = self.get_database_parts(node)
        table = self.adapter.get_relation(*parts)
        columns = []
        if not table:
            logger().info(
                ":cross_mark: Relation %s.%s.%s does not exist in target database, cannot resolve columns",
                *parts,
            )
            return columns
        try:
            columns = [c.name for c in self.adapter.get_columns_in_relation(table)]
        except CompilationException as error:
            logger().info(
                ":cross_mark: Could not resolve relation %s.%s.%s against database active tables during introspective query: %s",
                *parts,
                str(error),
            )
        return columns

    @staticmethod
    def assert_schema_has_no_sources(schema: Mapping) -> Mapping:
        """Inline assertion ensuring that a schema does not have a source key"""
        if schema.get("sources"):
            raise SanitizationRequired(
                "Found `sources:` block in a models schema file. We require you separate sources in order to organize your project."
            )
        return schema

    def build_schema_folder_mapping(
        self,
        target_node_type: Optional[Union[NodeType.Model, NodeType.Source]] = None,
    ) -> Dict[str, SchemaFile]:
        """Builds a mapping of models or sources to their existing and target schema file paths"""
        if target_node_type == NodeType.Source:
            # Source folder mapping is reserved for source importing
            target_nodes = self.dbt.sources
        elif target_node_type == NodeType.Model:
            target_nodes = self.dbt.nodes
        else:
            target_nodes = {**self.dbt.nodes, **self.dbt.sources}
        # Container for output
        schema_map = {}
        logger().info("...building project structure mapping in memory")
        # Iterate over models and resolve current path vs declarative target path
        for unique_id, dbt_node in self.filtered_models(target_nodes):
            schema_path = self.get_schema_path(dbt_node)
            osmosis_schema_path = self.get_target_schema_path(dbt_node)
            schema_map[unique_id] = SchemaFile(target=osmosis_schema_path, current=schema_path)
        return schema_map

    def draft_project_structure_update_plan(self) -> Dict[Path, RestructureQuantum]:
        """Build project structure update plan based on `dbt-osmosis:` configs set across dbt_project.yml and model files.
        The update plan includes injection of undocumented models. Unless this plan is constructed and executed by the `commit_project_restructure` function,
        dbt-osmosis will only operate on models it is aware of through the existing documentation.

        Returns:
            MutableMapping: Update plan where dict keys consist of targets and contents consist of outputs which match the contents of the `models` to be output in the
            target file and supersede lists of what files are superseded by a migration
        """

        # Container for output
        blueprint: Dict[Path, RestructureQuantum] = {}
        logger().info(
            ":chart_increasing: Searching project stucture for required updates and building action plan"
        )
        with self.adapter.connection_named("dbt-osmosis"):
            for unique_id, schema_file in self.build_schema_folder_mapping(
                target_node_type=NodeType.Model
            ).items():
                if not schema_file.is_valid:
                    blueprint.setdefault(
                        schema_file.target,
                        RestructureQuantum(output={"version": 2, "models": []}, supersede={}),
                    )
                    node = self.dbt.nodes[unique_id]
                    if schema_file.current is None:
                        # Bootstrapping Undocumented Model
                        blueprint[schema_file.target].output["models"].append(
                            self.get_base_model(node)
                        )
                    else:
                        # Model Is Documented but Must be Migrated
                        if not schema_file.current.exists():
                            continue
                        # TODO: We avoid sources for complexity reasons but if we are opinionated, we don't have to
                        schema = self.assert_schema_has_no_sources(
                            self.yaml.load(schema_file.current)
                        )
                        models_in_file: Iterable[Dict[str, Any]] = schema.get("models", [])
                        for documented_model in models_in_file:
                            if documented_model["name"] == node.name:
                                # Bootstrapping Documented Model
                                blueprint[schema_file.target].output["models"].append(
                                    self.bootstrap_existing_model(documented_model, node)
                                )
                                # Target to supersede current
                                blueprint[schema_file.target].supersede.setdefault(
                                    schema_file.current, []
                                ).append(documented_model["name"])
                                break
                        else:
                            ...  # Model not found at patch path -- We should pass on this for now
                else:
                    ...  # Valid schema file found for model -- We will update the columns in the `Document` task

        return blueprint

    def commit_project_restructure_to_disk(
        self, blueprint: Optional[Dict[Path, RestructureQuantum]] = None
    ) -> bool:
        """Given a project restrucure plan of pathlib Paths to a mapping of output and supersedes which is in itself a mapping of Paths to model names,
        commit changes to filesystem to conform project to defined structure as code fully or partially superseding existing models as needed.

        Args:
            blueprint (Dict[Path, RestructureQuantum]): Project restructure plan as typically created by `build_project_structure_update_plan`

        Returns:
            bool: True if the project was restructured, False if no action was required
        """

        # Build blueprint if not user supplied
        if not blueprint:
            blueprint = self.draft_project_structure_update_plan()

        # Verify we have actions in the plan
        if not blueprint:
            logger().info(":1st_place_medal: Project structure approved")
            return False

        # Print plan for user auditability
        self.pretty_print_restructure_plan(blueprint)

        logger().info(
            ":construction_worker: Executing action plan and conforming projecting schemas to defined structure"
        )
        for target, structure in blueprint.items():
            if not target.exists():
                # Build File
                logger().info(":construction: Building schema file %s", target.name)
                if not self.dry_run:
                    target.parent.mkdir(exist_ok=True, parents=True)
                    target.touch()
                    self.yaml.dump(structure.output, target)

            else:
                # Update File
                logger().info(":toolbox: Updating schema file %s", target.name)
                target_schema: Dict[str, Any] = self.yaml.load(target)
                if "version" not in target_schema:
                    target_schema["version"] = 2
                target_schema.setdefault("models", []).extend(structure.output["models"])
                if not self.dry_run:
                    self.yaml.dump(target_schema, target)

            # Clean superseded schema files
            for dir, models in structure.supersede.items():
                preserved_models = []
                raw_schema: Dict[str, Any] = self.yaml.load(dir)
                models_marked_for_superseding = set(models)
                models_in_schema = set(map(lambda mdl: mdl["name"], raw_schema.get("models", [])))
                non_superseded_models = models_in_schema - models_marked_for_superseding
                if len(non_superseded_models) == 0:
                    logger().info(":rocket: Superseded schema file %s", dir.name)
                    if not self.dry_run:
                        dir.unlink(missing_ok=True)
                else:
                    for model in raw_schema["models"]:
                        if model["name"] in non_superseded_models:
                            preserved_models.append(model)
                    raw_schema["models"] = preserved_models
                    if not self.dry_run:
                        self.yaml.dump(raw_schema, dir)
                    logger().info(
                        ":satellite: Model documentation migrated from %s to %s",
                        dir.name,
                        target.name,
                    )

        return True

    @staticmethod
    def pretty_print_restructure_plan(blueprint: Dict[Path, RestructureQuantum]) -> None:
        logger().info(
            list(
                map(
                    lambda plan: (blueprint[plan].supersede or "CREATE", "->", plan),
                    blueprint.keys(),
                )
            )
        )

    def build_node_ancestor_tree(
        self,
        node: ManifestNode,
        family_tree: Optional[Dict[str, List[str]]] = None,
        members_found: Optional[List[str]] = None,
        depth: int = 0,
    ) -> Dict[str, List[str]]:
        """Recursively build dictionary of parents in generational order"""
        if family_tree is None:
            family_tree = {}
        if members_found is None:
            members_found = []
        for parent in node.depends_on.nodes:
            member = self.dbt.nodes.get(parent, self.dbt.sources.get(parent))
            if member and parent not in members_found:
                family_tree.setdefault(f"generation_{depth}", []).append(parent)
                members_found.append(parent)
                # Recursion
                family_tree = self.build_node_ancestor_tree(
                    member, family_tree, members_found, depth + 1
                )
        return family_tree

    def inherit_column_level_knowledge(
        self,
        family_tree: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        """Inherit knowledge from ancestors in reverse insertion order to ensure that the most recent ancestor is always the one to inherit from"""
        knowledge: Dict[str, Dict[str, Any]] = {}
        for generation in reversed(family_tree):
            for ancestor in family_tree[generation]:
                member: ManifestNode = self.dbt.nodes.get(ancestor, self.dbt.sources.get(ancestor))
                if not member:
                    continue
                for name, info in member.columns.items():
                    knowledge.setdefault(name, {"progenitor": ancestor})
                    deserialized_info = info.to_dict()
                    # Handle Info:
                    # 1. tags are additive
                    # 2. descriptions are overriden
                    # 3. meta is merged
                    # 4. tests are ignored until I am convinced those shouldn't be hand curated with love
                    if deserialized_info["description"] in PLACEHOLDERS:
                        deserialized_info.pop("description", None)
                    deserialized_info["tags"] = list(
                        set(deserialized_info.pop("tags", []) + knowledge[name].get("tags", []))
                    )
                    if not deserialized_info["tags"]:
                        deserialized_info.pop("tags")  # poppin' tags like Macklemore
                    deserialized_info["meta"] = {
                        **knowledge[name].get("meta", {}),
                        **deserialized_info["meta"],
                    }
                    if not deserialized_info["meta"]:
                        deserialized_info.pop("meta")
                    knowledge[name].update(deserialized_info)
        return knowledge

    def get_node_columns_with_inherited_knowledge(
        self,
        node: ManifestNode,
    ) -> Dict[str, Dict[str, Any]]:
        """Build a knowledgebase for the model based on iterating through ancestors"""
        family_tree = self.build_node_ancestor_tree(node)
        knowledge = self.inherit_column_level_knowledge(family_tree)
        return knowledge

    @staticmethod
    def get_column_sets(
        database_columns: Iterable[str],
        yaml_columns: Iterable[str],
        documented_columns: Iterable[str],
    ) -> Tuple[List[str], List[str], List[str]]:
        """Returns:
        missing_columns: Columns in database not in dbt -- will be injected into schema file
        undocumented_columns: Columns missing documentation -- descriptions will be inherited and injected into schema file where prior knowledge exists
        extra_columns: Columns in schema file not in database -- will be removed from schema file
        """
        missing_columns = [
            x for x in database_columns if x.lower() not in (y.lower() for y in yaml_columns)
        ]
        undocumented_columns = [
            x for x in database_columns if x.lower() not in (y.lower() for y in documented_columns)
        ]
        extra_columns = [
            x for x in yaml_columns if x.lower() not in (y.lower() for y in database_columns)
        ]
        return missing_columns, undocumented_columns, extra_columns

    def propagate_documentation_downstream(self, force_inheritance: bool = False) -> None:
        schema_map = self.build_schema_folder_mapping()
        with self.adapter.connection_named("dbt-osmosis"):
            for unique_id, node in track(list(self.filtered_models())):
                logger().info("\n:point_right: Processing model: [bold]%s[/bold] \n", unique_id)
                # Get schema file path, must exist to propagate documentation
                schema_path: Optional[SchemaFile] = schema_map.get(unique_id)
                if schema_path is None or schema_path.current is None:
                    logger().info(
                        ":bow: No valid schema file found for model %s", unique_id
                    )  # We can't take action
                    continue

                # Build Sets
                database_columns: Set[str] = set(self.get_columns(node))
                yaml_columns: Set[str] = set(column for column in node.columns)

                if not database_columns:
                    logger().info(
                        ":safety_vest: Unable to resolve columns in database, falling back to using yaml columns as base column set\n"
                    )
                    database_columns = yaml_columns

                # Get documentated columns
                documented_columns: Set[str] = set(
                    column
                    for column, info in node.columns.items()
                    if info.description and info.description not in PLACEHOLDERS
                )

                # Queue
                missing_columns, undocumented_columns, extra_columns = self.get_column_sets(
                    database_columns, yaml_columns, documented_columns
                )

                if force_inheritance:
                    # Consider all columns "undocumented" so that inheritance is not selective
                    undocumented_columns = database_columns

                # Engage
                n_cols_added = 0
                n_cols_doc_inherited = 0
                n_cols_removed = 0
                if len(missing_columns) > 0 or len(undocumented_columns) or len(extra_columns) > 0:
                    schema_file = self.yaml.load(schema_path.current)
                    (
                        n_cols_added,
                        n_cols_doc_inherited,
                        n_cols_removed,
                    ) = self.update_schema_file_and_node(
                        missing_columns,
                        undocumented_columns,
                        extra_columns,
                        node,
                        schema_file,
                    )
                    if n_cols_added + n_cols_doc_inherited + n_cols_removed > 0:
                        # Dump the mutated schema file back to the disk
                        if not self.dry_run:
                            self.yaml.dump(schema_file, schema_path.current)
                        logger().info(":sparkles: Schema file updated")

                # Print Audit Report
                n_cols = float(len(database_columns))
                n_cols_documented = float(len(documented_columns)) + n_cols_doc_inherited
                perc_coverage = (
                    min(100.0 * round(n_cols_documented / n_cols, 3), 100.0)
                    if n_cols > 0
                    else "Unable to Determine"
                )
                logger().info(
                    AUDIT_REPORT.format(
                        database=node.database,
                        schema=node.schema,
                        table=node.name,
                        total_columns=n_cols,
                        n_cols_added=n_cols_added,
                        n_cols_doc_inherited=n_cols_doc_inherited,
                        n_cols_removed=n_cols_removed,
                        coverage=perc_coverage,
                    )
                )

    @staticmethod
    def remove_columns_not_in_database(
        extra_columns: Iterable[str],
        node: ManifestNode,
        yaml_file_model_section: Dict[str, Any],
    ) -> int:
        """Removes columns found in dbt model that do not exist in database from both node and model simultaneously
        THIS MUTATES THE NODE AND MODEL OBJECTS so that state is always accurate"""
        changes_committed = 0
        for column in extra_columns:
            node.columns.pop(column, None)
            yaml_file_model_section["columns"] = [
                c for c in yaml_file_model_section["columns"] if c["name"] != column
            ]
            changes_committed += 1
            logger().info(":wrench: Removing column %s from dbt schema", column)
        return changes_committed

    def update_undocumented_columns_with_prior_knowledge(
        self,
        undocumented_columns: Iterable[str],
        node: ManifestNode,
        yaml_file_model_section: Dict[str, Any],
    ) -> int:
        """Update undocumented columns with prior knowledge in node and model simultaneously
        THIS MUTATES THE NODE AND MODEL OBJECTS so that state is always accurate"""
        knowledge = self.get_node_columns_with_inherited_knowledge(node)
        inheritables = ("description", "tags", "meta")
        changes_committed = 0
        for column in undocumented_columns:
            prior_knowledge = knowledge.get(column, {})
            progenitor = prior_knowledge.pop("progenitor", "Unknown")
            prior_knowledge = {k: v for k, v in prior_knowledge.items() if k in inheritables}
            if not prior_knowledge:
                continue
            if column not in node.columns:
                node.columns[column] = ColumnInfo.from_dict({"name": column, **prior_knowledge})
            else:
                node.columns[column].replace(kwargs={"name": column, **prior_knowledge})
            for model_column in yaml_file_model_section["columns"]:
                if model_column["name"] == column:
                    model_column.update(prior_knowledge)
            changes_committed += 1
            logger().info(
                ":light_bulb: Column %s is inheriting knowledge from the lineage of progenitor (%s)",
                column,
                progenitor,
            )
            logger().info(prior_knowledge)
        return changes_committed

    @staticmethod
    def add_missing_cols_to_node_and_model(
        missing_columns: Iterable,
        node: ManifestNode,
        yaml_file_model_section: Dict[str, Any],
    ) -> int:
        """Add missing columns to node and model simultaneously
        THIS MUTATES THE NODE AND MODEL OBJECTS so that state is always accurate"""
        changes_committed = 0
        for column in missing_columns:
            node.columns[column] = ColumnInfo.from_dict({"name": column})
            yaml_file_model_section.setdefault("columns", []).append({"name": column})
            changes_committed += 1
            logger().info(":syringe: Injecting column %s into dbt schema", column)
        return changes_committed

    def update_schema_file_and_node(
        self,
        missing_columns: Iterable[str],
        undocumented_columns: Iterable[str],
        extra_columns: Iterable[str],
        node: ManifestNode,
        yaml_file: Dict[str, Any],
    ) -> Tuple[int, int, int]:
        """Take action on a schema file mirroring changes in the node."""
        # We can extrapolate this to a general func
        noop = 0, 0, 0
        if node.resource_type == NodeType.Source:
            KEY = "tables"
            yaml_file_models = None
            for src in yaml_file.get("sources", []):
                if src["name"] == node.source_name:
                    # Scope our pointer to a specific portion of the object
                    yaml_file_models = src
        else:
            KEY = "models"
            yaml_file_models = yaml_file
        if yaml_file_models is None:
            return noop
        for yaml_file_model_section in yaml_file_models[KEY]:
            if yaml_file_model_section["name"] == node.name:
                logger().info(":microscope: Looking for actions")
                n_cols_added = self.add_missing_cols_to_node_and_model(
                    missing_columns, node, yaml_file_model_section
                )
                n_cols_doc_inherited = self.update_undocumented_columns_with_prior_knowledge(
                    undocumented_columns, node, yaml_file_model_section
                )
                n_cols_removed = self.remove_columns_not_in_database(
                    extra_columns, node, yaml_file_model_section
                )
                return n_cols_added, n_cols_doc_inherited, n_cols_removed
        logger().info(":thumbs_up: No actions needed")
        return noop


def get_raw_profiles(profiles_dir: Optional[str] = None) -> Dict[str, Any]:
    import dbt.config.profile as dbt_profile

    return dbt_profile.read_profile(profiles_dir or DEFAULT_PROFILES_DIR)


def uncompile_node(node: ManifestNode) -> ManifestNode:
    """Uncompile a node by removing the compiled_resource_path and compiled_resource_hash"""
    return ParsedModelNode.from_dict(node.to_dict())


def _compiled_type_for(model: ManifestNode):
    if type(model) not in COMPILED_TYPES:
        raise InternalException(f"Asked to compile {type(model)} node, but it has no compiled form")
    return COMPILED_TYPES[type(model)]


def _add_prepended_cte(prepended_ctes, new_cte):
    for cte in prepended_ctes:
        if cte.id == new_cte.id:
            cte.sql = new_cte.sql
            return
    prepended_ctes.append(new_cte)


def _extend_prepended_ctes(prepended_ctes, new_prepended_ctes):
    for new_cte in new_prepended_ctes:
        _add_prepended_cte(prepended_ctes, new_cte)
