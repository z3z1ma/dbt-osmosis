from enum import Enum
from itertools import chain
from pathlib import Path
from typing import (Any, Dict, Iterable, Iterator, List, Mapping,
                    MutableMapping, Optional, Set, Tuple, Union)

import agate
import dbt.config.runtime as dbt_config
import dbt.parser.manifest as dbt_parser
from dbt.adapters.factory import (Adapter, get_adapter, register_adapter,
                                  reset_adapters)
from dbt.contracts.connection import AdapterResponse
from dbt.contracts.graph.manifest import ManifestNode, NodeType
from dbt.contracts.graph.parsed import ColumnInfo, ParsedModelNode
from dbt.exceptions import CompilationException, RuntimeException
from dbt.flags import DEFAULT_PROFILES_DIR, set_from_args
from dbt.tracking import disable_tracking
from pydantic import BaseModel
from rich.progress import track
from ruamel.yaml import YAML

from dbt_osmosis.core.exceptions import (InvalidOsmosisConfig,
                                         MissingOsmosisConfig,
                                         SanitizationRequired)
from dbt_osmosis.core.logging import logger

disable_tracking()

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


class OsmosisConfig(str, Enum):
    SchemaYaml = "schema.yml"
    FolderYaml = "folder.yml"
    ModelYaml = "model.yml"
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

        # Load dbt + verify connection to data warehhouse
        set_from_args(args, args)
        self.project, self.profile = dbt_config.RuntimeConfig.collect_parts(args)
        self.config = dbt_config.RuntimeConfig.from_parts(self.project, self.profile, args)
        reset_adapters()
        register_adapter(self.config)
        self.adapter = self._verify_connection(get_adapter(self.config))

        # Parse project
        self.dbt = dbt_parser.ManifestLoader.get_full_manifest(self.config)

        # Selector Passed in From CLI
        self.fqn = fqn

        # Utilities
        self.yaml = self._build_yaml_parser()
        self.dry_run = dry_run

    @staticmethod
    def _verify_connection(adapter: Adapter) -> Adapter:
        try:
            with adapter.connection_named("debug"):
                adapter.debug_query()
        except Exception as exc:
            raise Exception("Could not connect to Database") from exc
        else:
            return adapter

    @staticmethod
    def _build_yaml_parser() -> YAML:
        yaml = YAML()
        yaml.indent(mapping=2, sequence=4, offset=2)
        yaml.width = 800
        yaml.preserve_quotes = True
        yaml.explicit_start = True
        yaml.explicit_end = True
        yaml.default_flow_style = False
        return yaml

    @property
    def project_name(self) -> str:
        return self.project.project_name

    @property
    def project_root(self) -> str:
        return self.project.project_root

    def rebuild_dbt_manifest(self) -> None:
        self.dbt = dbt_parser.ManifestLoader.get_full_manifest(self.config)

    @property
    def manifest(self) -> Dict[str, Any]:
        return self.dbt.flat_graph

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
