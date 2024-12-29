# pyright: reportUnknownVariableType=false, reportPrivateImportUsage=false, reportAny=false, reportUnknownMemberType=false

from __future__ import annotations

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
from dbt.adapters.base.impl import BaseAdapter
from dbt.adapters.contracts.connection import AdapterResponse
from dbt.adapters.factory import get_adapter_class_by_name
from dbt.config.runtime import RuntimeConfig
from dbt.contracts.graph.manifest import Manifest
from dbt.contracts.graph.nodes import (
    ColumnInfo,
    ManifestNode,
    ManifestSQLNode,
    ModelNode,
    ResultNode,
    SeedNode,
    SourceDefinition,
)
from dbt.contracts.results import CatalogArtifact, CatalogKey, CatalogTable, ColumnMetadata
from dbt.node_types import NodeType
from dbt.parser.manifest import ManifestLoader, process_node
from dbt.parser.sql import SqlBlockParser, SqlMacroParser
from dbt.task.sql import SqlCompileRunner
from dbt.tracking import disable_tracking

disable_tracking()

EMPTY_STRING = ""

logger = logging.getLogger("dbt-osmosis")


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
    which: str = ""

    debug: bool = False
    _vars: str | dict[str, t.Any] = field(default_factory=dict, init=False)

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

    _mutation_count: int = field(default=0, init=False)

    def register_mutations(self, count: int) -> None:
        """Increment the mutation count by a specified amount."""
        self._mutation_count += count

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


def normalize_column_name(column: str, credentials_type: str, to_lower: bool) -> str:
    """Apply case normalization to a column name based on the credentials type."""
    if credentials_type == "snowflake" and column.startswith('"') and column.endswith('"'):
        return column
    if to_lower:
        return column.lower()
    if credentials_type == "snowflake":
        return column.upper()
    return column


@dataclass
class ColumnData:
    """Simple data object for column information"""

    name: str
    description: str
    data_type: str


def _maybe_use_precise_dtype(col: t.Any, settings: YamlRefactorSettings) -> str:
    """Use the precise data type if enabled in the settings."""
    if (col.is_numeric() and settings.numeric_precision) or (
        col.is_string() and settings.char_length
    ):
        return col.data_type
    return col.dtype


def _catalog_key_for_node(node: ResultNode) -> CatalogKey:
    """Make an appropriate catalog key for a dbt node."""
    if node.resource_type == NodeType.Source:
        return CatalogKey(node.database, node.schema, getattr(node, "identifier", node.name))
    return CatalogKey(node.database, node.schema, getattr(node, "alias", node.name))


def get_columns_meta_for_key(
    context: YamlRefactorContext, key: CatalogKey, output_to_lower: bool
) -> dict[str, ColumnMetadata]:
    """Equivalent to get_columns_meta in old code but directly referencing a key, not a node."""
    cased_cols = OrderedDict()
    blacklist = context.project.config.vars.to_dict().get("dbt-osmosis", {}).get("_blacklist", [])
    catalog = None
    if context.settings.catalog_file:
        path = Path(context.settings.catalog_file)
        if path.is_file():
            catalog = CatalogArtifact.from_dict(json.loads(path.read_text()))

    # Catalog first
    if catalog:
        cat_objs = {**catalog.nodes, **catalog.sources}
        matched = [table for cat_k, table in cat_objs.items() if cat_k.split(".")[-1] == key.name]
        if matched:
            for col in matched[0].columns.values():
                if any(re.match(p, col.name) for p in blacklist):
                    continue
                cased = normalize_column_name(
                    col.name,
                    context.project.config.credentials.type,
                    output_to_lower,
                )
                cased_cols[cased] = ColumnMetadata(
                    name=cased,
                    type=col.type,
                    index=col.index,
                    comment=col.comment,
                )
            return cased_cols

    # Fallback to adapter-based
    adapter = context.project.adapter
    rel = adapter.get_relation(key.database, key.schema, key.name)
    if not rel:
        return cased_cols
    try:
        for col_ in adapter.get_columns_in_relation(rel):
            if any(re.match(b, col_.name) for b in blacklist):
                continue
            cased = normalize_column_name(
                col_.name,
                context.project.config.credentials.type,
                output_to_lower,
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
                    cased2 = normalize_column_name(
                        exp.name,
                        context.project.config.credentials.type,
                        output_to_lower,
                    )
                    dtype2 = _maybe_use_precise_dtype(exp, context.settings)
                    cased_cols[cased2] = ColumnMetadata(
                        name=cased2,
                        type=dtype2,
                        index=None,
                        comment=getattr(exp, "comment", None),
                    )
    except Exception as ex:
        logger.warning(f"Could not introspect columns for {key}: {ex}")
    return cased_cols


def get_columns_for_key(
    context: YamlRefactorContext, key: CatalogKey, output_to_lower: bool
) -> list[str]:
    """Equivalent to get_columns in old code; returns just the list of column names."""
    meta = get_columns_meta_for_key(context, key, output_to_lower)
    return list(meta.keys())


# TODO: more work to do below the fold here


_ColumnLevelKnowledge = dict[str, t.Any]
_KnowledgeBase = dict[str, _ColumnLevelKnowledge]


def _build_node_ancestor_tree(
    manifest: Manifest,
    node: ResultNode,
    family_tree: dict[str, list[str]] | None = None,
    members_found: list[str] | None = None,
    depth: int = 0,
) -> dict[str, list[str]]:
    """Recursively build dictionary of parents in generational order using a simple DFS algorithm"""
    # Set initial values
    if family_tree is None:
        family_tree = {}
    if members_found is None:
        members_found = []

    # If the node has no dependencies, return the family tree as it is
    if not hasattr(node, "depends_on"):
        return family_tree

    # Iterate over the parents of the node mutating family_tree
    for parent in getattr(node.depends_on, "nodes", []):
        member = manifest.nodes.get(parent, manifest.sources.get(parent))
        if member and parent not in members_found:
            family_tree.setdefault(f"generation_{depth}", []).append(parent)
            _ = _build_node_ancestor_tree(manifest, member, family_tree, members_found, depth + 1)
            members_found.append(parent)

    return family_tree


def _find_first(coll: Iterable[dict[str, t.Any]], predicate: t.Callable[[t.Any], bool]) -> t.Any:
    """Find the first item in a container that satisfies a predicate."""
    for item in coll:
        if predicate(item):
            return item


def get_member_yaml(context: YamlRefactorContext, member: ResultNode) -> dict[str, t.Any] | None:
    """Get the parsed YAML for a dbt model or source node."""
    project_dir = Path(context.project.config.project_root)
    yaml_handler = context.yaml_handler

    if isinstance(member, SourceDefinition):
        if not member.original_file_path:
            return None
        path = project_dir.joinpath(member.original_file_path)
        if not path.exists():
            return None
        with path.open("r") as f:
            parsed_yaml = yaml_handler.load(f)
        data: t.Any = parsed_yaml.get("sources", [])
        src = _find_first(data, lambda s: s["name"] == member.source_name)
        if not src:
            return None
        tables = src.get("tables", [])
        return _find_first(tables, lambda tbl: tbl["name"] == member.name)

    elif isinstance(member, (ModelNode, SeedNode)):
        if not member.patch_path:
            return None
        patch_file = project_dir.joinpath(member.patch_path.split("://")[-1])
        if not patch_file.is_file():
            return None
        with patch_file.open("r") as f:
            parsed_yaml = yaml_handler.load(f)
        section_key = f"{member.resource_type}s"
        data = parsed_yaml.get(section_key, [])
        return _find_first(data, lambda model: model["name"] == member.name)

    return None


def inherit_column_level_knowledge(
    context: YamlRefactorContext, family_tree: dict[str, list[str]]
) -> _KnowledgeBase:
    """Generate a knowledge base by applying inheritance logic based on the family tree graph."""
    knowledge: _KnowledgeBase = {}
    placeholders = context.placeholders
    manifest = context.project.manifest

    # If the user wants to use unrendered descriptions
    use_unrendered = context.settings.use_unrendered_descriptions

    # We traverse from the last generation to the earliest
    # so that the "nearest" ancestor overwrites the older ones.
    for gen_name in reversed(family_tree.keys()):
        members_in_generation = family_tree[gen_name]
        for ancestor_id in members_in_generation:
            member = manifest.nodes.get(ancestor_id, manifest.sources.get(ancestor_id))
            if not member:
                continue

            member_yaml: dict[str, t.Any] | None = None
            if use_unrendered:
                member_yaml = get_member_yaml(context, member)

            # For each column in the ancestor
            for col_name, col_info in member.columns.items():
                # If we haven't seen this column name yet, seed it with minimal data
                _ = knowledge.setdefault(
                    col_name,
                    {"progenitor": ancestor_id, "generation": gen_name},
                )
                merged_info = col_info.to_dict()

                # If the description is in placeholders, discard it
                if merged_info.get("description", "") in placeholders:
                    merged_info["description"] = ""

                # If user wants unrendered, read from YAML file if present
                if member_yaml and "columns" in member_yaml:
                    col_in_yaml = _find_first(
                        member_yaml["columns"], lambda c: c["name"] == merged_info["name"]
                    )
                    if col_in_yaml and "description" in col_in_yaml:
                        merged_info["description"] = col_in_yaml["description"]

                # Merge tags
                existing_tags = knowledge[col_name].get("tags", [])
                new_tags = set(merged_info.pop("tags", [])) | set(existing_tags)
                if new_tags:
                    merged_info["tags"] = list(new_tags)

                # Merge meta
                existing_meta = knowledge[col_name].get("meta", {})
                combined_meta = {**existing_meta, **merged_info.pop("meta", {})}
                if combined_meta:
                    merged_info["meta"] = combined_meta

                # Now unify
                knowledge[col_name].update(merged_info)

    return knowledge


def get_node_columns_with_inherited_knowledge(
    context: YamlRefactorContext, node: ResultNode
) -> _KnowledgeBase:
    """Build a knowledgebase for the node by climbing the ancestor tree and merging column doc info from nearest to farthest ancestors."""
    family_tree = _build_node_ancestor_tree(context.project.manifest, node)
    return inherit_column_level_knowledge(context, family_tree)


def get_prior_knowledge(knowledge: _KnowledgeBase, column: str) -> _ColumnLevelKnowledge:
    """If the user has changed column name's case or prefix, attempt to find the best match among possible variants (lowercase, pascalCase, etc.)

    We sort so that any source/seed is considered first, then models,
    and within each group we sort descending by generation.
    """
    camelcase: str = re.sub(r"_(.)", lambda m: m.group(1).upper(), column)
    pascalcase: str = camelcase[0].upper() + camelcase[1:] if camelcase else camelcase
    variants = (column, column.lower(), camelcase, pascalcase)

    def is_source_or_seed(k: _ColumnLevelKnowledge) -> bool:
        p = k.get("progenitor", "")
        return p.startswith("source") or p.startswith("seed")

    matches: list[_ColumnLevelKnowledge] = []
    for var in variants:
        found = knowledge.get(var)
        if found is not None:
            matches.append(found)

    def _sort_k(k: _ColumnLevelKnowledge) -> tuple[bool, str]:
        return (not is_source_or_seed(k), k.get("generation", ""))

    sorted_matches = sorted(matches, key=_sort_k, reverse=True)
    return sorted_matches[0] if sorted_matches else {}


def merge_knowledge_with_original_knowledge(
    prior_knowledge: _ColumnLevelKnowledge,
    original_knowledge: _ColumnLevelKnowledge,
    add_progenitor_to_meta: bool,
    progenitor: str,
) -> _ColumnLevelKnowledge:
    """Merge two column level knowledge dictionaries."""
    merged = dict(original_knowledge)

    # Unify tags
    if "tags" in prior_knowledge:
        prior_tags = set(prior_knowledge["tags"])
        merged_tags = set(merged.get("tags", []))
        merged["tags"] = list(prior_tags | merged_tags)

    # Unify meta
    if "meta" in prior_knowledge:
        new_meta = {**merged.get("meta", {}), **prior_knowledge["meta"]}
        merged["meta"] = new_meta

    # If the user wants the source or seed name in meta, apply it
    if add_progenitor_to_meta and progenitor:
        merged.setdefault("meta", {})
        merged["meta"]["osmosis_progenitor"] = progenitor

    # If meta says "osmosis_keep_description" => keep the original description
    if merged.get("meta", {}).get("osmosis_keep_description"):
        # Do nothing
        pass
    else:
        # Otherwise if prior knowledge has a non-empty description, override
        if prior_knowledge.get("description"):
            merged["description"] = prior_knowledge["description"]

    # Remove empty tags or meta
    if merged.get("tags") == []:
        merged.pop("tags", None)
    if merged.get("meta") == {}:
        merged.pop("meta", None)

    return merged


def update_undocumented_columns_with_prior_knowledge(
    undocumented_columns: Iterable[str],
    node: ManifestNode,
    yaml_file_model_section: dict[str, t.Any],
    knowledge: _KnowledgeBase,
    skip_add_tags: bool,
    skip_merge_meta: bool,
    add_progenitor_to_meta: bool,
    add_inheritance_for_specified_keys: Iterable[str] = (),
) -> int:
    """For columns that are undocumented, we find prior knowledge in the knowledge dict, merge it with the existing column's knowledge, then assign it to both node and YAML."""
    # Which keys are we allowed to adopt from prior knowledge
    inheritables = ["description"]
    if not skip_add_tags:
        inheritables.append("tags")
    if not skip_merge_meta:
        inheritables.append("meta")
    for k in add_inheritance_for_specified_keys:
        if k not in inheritables:
            inheritables.append(k)

    changes = 0
    for column in undocumented_columns:
        if column not in node.columns:
            node.columns[column] = ColumnInfo.from_dict({"name": column})
        original_dict = node.columns[column].to_dict()

        prior = get_prior_knowledge(knowledge, column)
        progenitor = t.cast(str, prior.pop("progenitor", ""))

        # Only keep keys we want to inherit
        filtered_prior = {kk: vv for kk, vv in prior.items() if kk in inheritables}

        new_knowledge = merge_knowledge_with_original_knowledge(
            filtered_prior,
            original_dict,
            add_progenitor_to_meta,
            progenitor,
        )
        if new_knowledge == original_dict:
            continue

        node.columns[column] = ColumnInfo.from_dict(new_knowledge)
        for col_def in yaml_file_model_section.get("columns", []):
            if col_def.get("name") == column:
                # Only update the keys we are inheriting
                for k2 in filtered_prior:
                    col_def[k2] = new_knowledge.get(k2, col_def.get(k2))
        logger.info(
            "[osmosis] Inherited knowledge for column: '%s' from progenitor '%s' in node '%s'",
            column,
            progenitor,
            node.unique_id,
        )
        changes += 1
    return changes


def get_columns_meta(
    context: YamlRefactorContext, node: ManifestNode, catalog: CatalogArtifact | None
) -> dict[str, ColumnMetadata]:
    cased_cols = OrderedDict()
    blacklist = context.project.config.vars.get("dbt-osmosis", {}).get("_blacklist", [])

    key = _catalog_key_for_node(node)
    if catalog:
        cat_objs = {**catalog.nodes, **catalog.sources}
        matched = [table for k, table in cat_objs.items() if k.split(".")[-1] == key.name]
        if matched:
            for col in matched[0].columns.values():
                if any(re.match(b, col.name) for b in blacklist):
                    continue
                cased = normalize_column_name(
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

    adapter = context.project.adapter
    rel = adapter.get_relation(key.database, key.schema, key.name)
    if not rel:
        return cased_cols
    try:
        for col_ in adapter.get_columns_in_relation(rel):
            if any(re.match(b, col_.name) for b in blacklist):
                continue
            cased = normalize_column_name(
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
                    cased2 = normalize_column_name(
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
    except Exception as ex:
        logger.warning(f"Could not introspect columns for {key}: {ex}")
    return cased_cols


# NOTE: usage example of the more FP style module below


def run_example_compilation_flow() -> None:
    config = DbtConfiguration(target="some_target", threads=2)
    config.vars = {"foo": "bar"}
    proj_ctx = create_dbt_project_context(config)

    node = compile_sql_code(proj_ctx, "select '{{ 1+1 }}' as col")
    print("Compiled =>", node.compiled_code)

    resp = execute_sql_code(proj_ctx, "select '{{ 1+2 }}' as col")
    print("Resp =>", resp)
