import os
import re
import json
from concurrent.futures import ThreadPoolExecutor, wait
from enum import Enum
from functools import lru_cache
from itertools import chain
from pathlib import Path
from threading import Lock
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    MutableMapping,
    Optional,
    Set,
    Tuple,
)

import ruamel.yaml
from pydantic import BaseModel

from dbt_osmosis.core.exceptions import (
    InvalidOsmosisConfig,
    MissingOsmosisConfig,
)
from dbt_osmosis.core.log_controller import logger
from dbt_osmosis.vendored.dbt_core_interface.project import (
    ColumnInfo,
    DbtProject,
    ManifestNode,
    NodeType,
)

as_path = Path


class YamlHandler(ruamel.yaml.YAML):
    """A `ruamel.yaml` wrapper to handle dbt YAML files with sane defaults"""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.indent(mapping=2, sequence=4, offset=2)
        self.width = 800
        self.preserve_quotes = True
        self.default_flow_style = False


class SchemaFileLocation(BaseModel):
    target: Path
    current: Optional[Path] = None
    node_type: NodeType = NodeType.Model

    @property
    def is_valid(self) -> bool:
        return self.current == self.target


class SchemaFileMigration(BaseModel):
    output: Dict[str, Any] = {}
    supersede: Dict[Path, List[str]] = {}


class DbtYamlManager(DbtProject):
    """The DbtYamlManager class handles developer automation tasks surrounding
    schema yaml files organziation, documentation, and coverage."""

    audit_report = """
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

    # TODO: Let user supply a custom arg / config file / csv of strings which we
    # consider placeholders which are not valid documentation, these are just my own
    # We may well drop the placeholder concept too. It is just a convenience for refactors
    placeholders = [
        "Pending further documentation",
        "Pending further documentation.",
        "No description for this column",
        "No description for this column.",
        "Not documented",
        "Not documented.",
        "Undefined",
        "Undefined.",
        "",  # This is the important one
    ]

    def __init__(
        self,
        target: Optional[str] = None,
        profiles_dir: Optional[str] = None,
        project_dir: Optional[str] = None,
        catalog_file: Optional[str] = None,
        threads: Optional[int] = 1,
        fqn: Optional[str] = None,
        dry_run: bool = False,
        models: Optional[List[str]] = None,
        skip_add_columns: bool = False,
        skip_add_tags: bool = False,
        skip_merge_meta: bool = False,
    ):
        """Initializes the DbtYamlManager class."""
        super().__init__(target, profiles_dir, project_dir, threads)
        self.fqn = fqn
        self.models = models or []
        self.dry_run = dry_run
        self.catalog_file = catalog_file
        self.skip_add_columns = skip_add_columns
        self.skip_add_tags = skip_add_tags
        self.skip_merge_meta = skip_merge_meta

        if len(list(self.filtered_models())) == 0:
            logger().warning(
                "No models found to process. Check your filters: --fqn='%s', pos args %s",
                fqn,
                models,
            )
            logger().info(
                "Please supply a valid fqn segment if using --fqn or a valid model name, path, or"
                " subpath if using positional arguments"
            )
            exit(0)

        self.mutex = Lock()
        self.tpe = ThreadPoolExecutor(max_workers=os.cpu_count() * 2)
        self.mutations = 0

    @property
    def yaml_handler(self):
        """Returns a cached instance of the YAML handler."""
        if not hasattr(self, "_yaml_handler"):
            self._yaml_handler = YamlHandler()
        return self._yaml_handler

    def column_casing(self, column: str) -> str:
        """Converts a column name to the correct casing for the target database."""
        if self.config.credentials.type == "snowflake":
            return column.upper()
        return column

    def _filter_model_by_fqn(self, node: ManifestNode) -> bool:
        """Validates a node as being selected.

        Check FQN length
        Check FQN matches parts
        """
        if not self.fqn:
            return True
        fqn = self.fqn or ".".join(node.fqn[1:])
        fqn_parts = fqn.split(".")
        return len(node.fqn[1:]) >= len(fqn_parts) and all(
            left == right for left, right in zip(fqn_parts, node.fqn[1:])
        )

    def _filter_model_by_models(self, node: ManifestNode) -> bool:
        """Validates a node as being selected.

        Check if the node name matches a model name
        Check if the node path matches a model path
        Check if the node path is a child of a model path
        """
        for model in self.models:
            if node.name == model:
                return True
            node_path = self.get_node_path(node)
            inp_path = as_path(model).resolve()
            if inp_path.is_dir():
                if node_path and inp_path in node_path.parents:
                    return True
            elif inp_path.is_file():
                if node_path and inp_path == node_path:
                    return True
        return False

    def _filter_model(self, node: ManifestNode) -> bool:
        """Validates a node as being actionable.

        Check if the node is a model
        Check if the node is a source
        Check if the node is a model and not ephemeral
        Check if the node is a model and matches the fqn or models filter if supplied
        """
        if self.models:
            filter_method = self._filter_model_by_models
        elif self.fqn:
            filter_method = self._filter_model_by_fqn
        else:
            filter_method = lambda _: True  # noqa: E731
        return (
            node.resource_type in (NodeType.Model, NodeType.Source)
            and node.package_name == self.project_name
            and not (
                node.resource_type == NodeType.Model and node.config.materialized == "ephemeral"
            )
            and filter_method(node)
        )

    @staticmethod
    def get_patch_path(node: ManifestNode) -> Optional[Path]:
        """Returns the patch path for a node if it exists"""
        if node is not None and node.patch_path:
            return as_path(node.patch_path.split("://")[-1])

    def filtered_models(
        self, subset: Optional[MutableMapping[str, ManifestNode]] = None
    ) -> Iterator[Tuple[str, ManifestNode]]:
        """Generates an iterator of valid models"""
        for unique_id, dbt_node in (
            subset.items()
            if subset
            else chain(self.manifest.nodes.items(), self.manifest.sources.items())
        ):
            if self._filter_model(dbt_node):
                yield unique_id, dbt_node

    def get_osmosis_path_spec(self, node: ManifestNode) -> Optional[str]:
        """Validates a config string.

        If input is a source, we return the resource type str instead
        """
        if node.resource_type == NodeType.Source:
            source_specs = self.config.vars.vars.get("dbt-osmosis", {})
            return source_specs.get(node.source_name)
        osmosis_spec = node.unrendered_config.get("dbt-osmosis")
        if not osmosis_spec:
            raise MissingOsmosisConfig(
                f"Config not set for model {node.name}, we recommend setting the config at a"
                " directory level through the `dbt_project.yml`"
            )
        try:
            return osmosis_spec
        except ValueError as exc:
            raise InvalidOsmosisConfig(
                f"Invalid config for model {node.name}: {osmosis_spec}"
            ) from exc

    def get_node_path(self, node: ManifestNode):
        """Resolve absolute file path for a manifest node"""
        return as_path(self.config.project_root, node.original_file_path).resolve()

    def get_schema_path(self, node: ManifestNode) -> Optional[Path]:
        """Resolve absolute schema file path for a manifest node"""
        schema_path = None
        if node.resource_type == NodeType.Model and node.patch_path:
            schema_path: str = node.patch_path.partition("://")[-1]
        elif node.resource_type == NodeType.Source:
            if hasattr(node, "source_name"):
                schema_path: str = node.path
        if schema_path:
            return as_path(self.project_root).joinpath(schema_path)

    def get_target_schema_path(self, node: ManifestNode) -> Path:
        """Resolve the correct schema yml target based on the dbt-osmosis
        config for the model / directory
        """
        osmosis_path_spec = self.get_osmosis_path_spec(node)
        if not osmosis_path_spec:
            # If no config is set, it is a no-op essentially
            return as_path(self.config.project_root, node.original_file_path)
        schema = osmosis_path_spec.format(node=node, model=node.name, parent=node.fqn[-2])
        parts = []

        # Part 1: path from project root to base model directory
        if node.resource_type == NodeType.Source:
            parts += [self.config.model_paths[0]]
        else:
            parts += [as_path(node.original_file_path).parent]

        # Part 2: path from base model directory to file
        parts += [schema if schema.endswith((".yml", ".yaml")) else f"{schema}.yml"]

        # Part 3: join parts relative to project root
        return as_path(self.config.project_root).joinpath(*parts)

    @staticmethod
    def get_database_parts(node: ManifestNode) -> Tuple[str, str, str]:
        """Returns a tuple of database, schema, and alias for a given node."""
        return node.database, node.schema, getattr(node, "alias", node.name)

    def get_base_model(self, node: ManifestNode) -> Dict[str, Any]:
        """Construct a base model object with model name, column names populated from database"""
        columns = self.get_columns(self.get_database_parts(node))
        return {
            "name": node.name,
            "columns": [{"name": column_name, "description": ""} for column_name in columns],
        }

    def augment_existing_model(
        self, documentation: Dict[str, Any], node: ManifestNode
    ) -> Dict[str, Any]:
        """Injects columns from database into existing model if not found"""
        model_columns: List[str] = [c["name"] for c in documentation.get("columns", [])]
        database_columns = self.get_columns(self.get_database_parts(node))
        for column in (
            c for c in database_columns if not any(c.lower() == m.lower() for m in model_columns)
        ):
            logger().info(
                ":syringe: Injecting column %s into dbt schema for %s",
                self.column_casing(column),
                node.unique_id,
            )
            documentation.setdefault("columns", []).append(
                {
                    "name": self.column_casing(column),
                    "description": getattr(column, "description", ""),
                }
            )
        return documentation

    @lru_cache(maxsize=5000)
    def get_columns(self, parts: Tuple[str, str, str]) -> List[str]:
        """Get all columns in a list for a model"""

        # If we provide a catalog, we read from it
        if self.catalog_file:
            file_path = Path(self.catalog_file)

            content = json.loads(file_path.read_text())
            matching_models = [
                model_values
                for model, model_values in content["nodes"].items()
                if model.split(".")[-1] == parts[-1]
            ]
            if matching_models:
                return [col.lower() for col in matching_models[0]["columns"].keys()]
            else:
                return []

        # If we don't provide a catalog we query the warehouse to get the columns
        else:
            with self.adapter.connection_named("dbt-osmosis"):
                table = self.adapter.get_relation(*parts)
                columns = []
                if not table:
                    logger().info(
                        (
                            ":cross_mark: Relation %s.%s.%s does not exist in target database,"
                            " cannot resolve columns"
                        ),
                        *parts,
                    )
                    return columns
                try:
                    for c in self.adapter.get_columns_in_relation(table):
                        columns.append(self.column_casing(c.name))
                        if hasattr(c, "flatten"):
                            columns.extend([self.column_casing(exp.name) for exp in c.flatten()])
                except Exception as error:
                    logger().info(
                        (
                            ":cross_mark: Could not resolve relation %s.%s.%s against database"
                            " active tables during introspective query: %s"
                        ),
                        *parts,
                        str(error),
                    )
                return columns

    def bootstrap_sources(self) -> None:
        """Bootstrap sources from the dbt-osmosis vars config"""
        performed_disk_mutation = False
        for source, spec in self.config.vars.vars.get("dbt-osmosis", {}).items():
            # Parse source config
            if isinstance(spec, str):
                schema = source
                path = spec
            elif isinstance(spec, dict):
                schema = spec.get("schema", source)
                path = spec["path"]
            else:
                raise TypeError(
                    f"Invalid dbt-osmosis var config for source {source}, must be a string or dict"
                )

            # Check if source exists in manifest
            dbt_node = next(
                (s for s in self.manifest.sources.values() if s.source_name == source), None
            )

            if not dbt_node:
                # Create a source file if it doesn't exist
                osmosis_schema_path = as_path(self.config.project_root).joinpath(
                    self.config.model_paths[0], path.lstrip(os.sep)
                )
                relations = self.adapter.list_relations(
                    database=self.config.credentials.database,
                    schema=schema,
                )
                tables = [
                    {
                        "name": relation.identifier,
                        "description": "",
                        "columns": [
                            {
                                "name": self.column_casing(exp.name),
                                "description": getattr(
                                    exp, "description", getattr(c, "description", "")
                                ),
                            }
                            for c in self.adapter.get_columns_in_relation(relation)
                            for exp in getattr(c, "flatten", lambda: [c])()
                        ] if not self.skip_add_columns else [],
                    }
                    for relation in relations
                ]
                osmosis_schema_path.parent.mkdir(parents=True, exist_ok=True)
                with open(osmosis_schema_path, "w") as schema_file:
                    logger().info(
                        ":syringe: Injecting source %s into dbt project",
                        source,
                    )
                    self.yaml_handler.dump(
                        {
                            "version": 2,
                            "sources": [
                                {
                                    "name": source,
                                    "schema": schema,
                                    "tables": tables,
                                }
                            ],
                        },
                        schema_file,
                    )
                    self.mutations += 1
                performed_disk_mutation = True

        if performed_disk_mutation:
            # Reload project to pick up new sources
            logger().info("...reloading project to pick up new sources")
            self.safe_parse_project(reinit=True)

    def build_schema_folder_mapping(self) -> Dict[str, SchemaFileLocation]:
        """Builds a mapping of models or sources to their existing and target schema file paths"""

        # Resolve target nodes
        self.bootstrap_sources()

        # Container for output
        schema_map = {}
        logger().info("...building project structure mapping in memory")

        # Iterate over models and resolve current path vs declarative target path
        for unique_id, dbt_node in self.filtered_models():
            schema_path = self.get_schema_path(dbt_node)
            osmosis_schema_path = self.get_target_schema_path(dbt_node)
            schema_map[unique_id] = SchemaFileLocation(
                target=osmosis_schema_path.resolve(),
                current=schema_path.resolve() if schema_path else None,
                node_type=dbt_node.resource_type,
            )

        return schema_map

    def _draft(self, schema_file: SchemaFileLocation, unique_id: str, blueprint: dict) -> None:
        try:
            with self.mutex:
                blueprint.setdefault(
                    schema_file.target,
                    SchemaFileMigration(
                        output={"version": 2, "models": [], "sources": []}, supersede={}
                    ),
                )
            if schema_file.node_type == NodeType.Model:
                node = self.manifest.nodes[unique_id]
            elif schema_file.node_type == NodeType.Source:
                node = self.manifest.sources[unique_id]
            else:
                return
            if schema_file.current is None:
                # Bootstrapping undocumented NodeType.Model
                # NodeType.Source files are guaranteed to exist by this point
                with self.mutex:
                    assert schema_file.node_type == NodeType.Model
                    blueprint[schema_file.target].output["models"].append(self.get_base_model(node))
            else:
                # Sanity check that the file exists before we try to load it, this should never be false
                assert schema_file.current.exists(), f"File {schema_file.current} does not exist"
                # Model/Source Is Documented but Must be Migrated
                with self.mutex:
                    schema = self.yaml_handler.load(schema_file.current)
                models_in_file: Iterable[Dict[str, Any]] = schema.get("models", [])
                sources_in_file: Iterable[Dict[str, Any]] = schema.get("sources", [])
                for documented_model in (
                    model for model in models_in_file if model["name"] == node.name
                ):
                    # Augment Documented Model
                    augmented_model = self.augment_existing_model(documented_model, node)
                    with self.mutex:
                        blueprint[schema_file.target].output["models"].append(augmented_model)
                        # Target to supersede current
                        blueprint[schema_file.target].supersede.setdefault(
                            schema_file.current, []
                        ).append(node)
                    break
                for documented_model, i in (
                    (table, j)
                    for j, source in enumerate(sources_in_file)
                    if source["name"] == node.source_name
                    for table in source["tables"]
                    if table["name"] == node.name
                ):
                    # Augment Documented Source
                    augmented_model = self.augment_existing_model(documented_model, node)
                    with self.mutex:
                        if not any(
                            s["name"] == node.source_name
                            for s in blueprint[schema_file.target].output["sources"]
                        ):
                            # Add the source if it doesn't exist in the blueprint
                            blueprint[schema_file.target].output["sources"].append(
                                sources_in_file[i]
                            )
                        # Find source in blueprint
                        for src in blueprint[schema_file.target].output["sources"]:
                            if src["name"] == node.source_name:
                                # Find table in blueprint
                                for tbl in src["tables"]:
                                    if tbl["name"] == node.name:
                                        # Augment table
                                        tbl = augmented_model
                                        break
                                break
                        else:
                            # This should never happen
                            raise RuntimeError(f"Source {node.source_name} not found in blueprint?")
                        # Target to supersede current
                        blueprint[schema_file.target].supersede.setdefault(
                            schema_file.current, []
                        ).append(node)
                    break
        except Exception as e:
            with self.mutex:
                logger().error(
                    "Failed to draft project structure update plan for %s: %s", unique_id, e
                )
            raise e

    def draft_project_structure_update_plan(self) -> Dict[Path, SchemaFileMigration]:
        """Build project structure update plan based on `dbt-osmosis:` configs set across
        dbt_project.yml and model files. The update plan includes injection of undocumented models.
        Unless this plan is constructed and executed by the `commit_project_restructure` function,
        dbt-osmosis will only operate on models it is aware of through the existing documentation.

        Returns:
            MutableMapping: Update plan where dict keys consist of targets and contents consist of
                outputs which match the contents of the `models` to be output in the
            target file and supersede lists of what files are superseded by a migration
        """

        # Container for output
        blueprint: Dict[Path, SchemaFileMigration] = {}
        logger().info(
            ":chart_increasing: Searching project stucture for required updates and building action"
            " plan"
        )
        futs = []
        with self.adapter.connection_named("dbt-osmosis"):
            for unique_id, schema_file in self.build_schema_folder_mapping().items():
                if not schema_file.is_valid:
                    futs.append(self.tpe.submit(self._draft, schema_file, unique_id, blueprint))
            wait(futs)
        return blueprint

    def commit_project_restructure_to_disk(
        self, blueprint: Optional[Dict[Path, SchemaFileMigration]] = None
    ) -> bool:
        """Given a project restrucure plan of pathlib Paths to a mapping of output and supersedes
        which is in itself a mapping of Paths to model names, commit changes to filesystem to
        conform project to defined structure as code fully or partially superseding existing models
        as needed.

        Args:
            blueprint (Dict[Path, SchemaFileMigration]): Project restructure plan as typically
                created by `build_project_structure_update_plan`

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
            ":construction_worker: Executing action plan and conforming projecting schemas to"
            " defined structure"
        )
        for target, structure in blueprint.items():
            if not target.exists():
                # Build File
                logger().info(":construction: Building schema file %s", target.name)
                if not self.dry_run:
                    target.parent.mkdir(exist_ok=True, parents=True)
                    target.touch()
                    self.yaml_handler.dump(structure.output, target)
                    self.mutations += 1

            else:
                # Update File
                logger().info(":toolbox: Updating schema file %s", target.name)
                target_schema: Optional[Dict[str, Any]] = self.yaml_handler.load(target)
                # Add version if not present
                if not target_schema:
                    target_schema = {"version": 2}
                elif "version" not in target_schema:
                    target_schema["version"] = 2
                # Add models and sources to target schema
                if structure.output["models"]:
                    target_schema.setdefault("models", []).extend(structure.output["models"])
                if structure.output["sources"]:
                    target_schema.setdefault("sources", []).extend(structure.output["sources"])
                if not self.dry_run:
                    self.yaml_handler.dump(target_schema, target)
                    self.mutations += 1

            # Clean superseded schema files
            for dir, nodes in structure.supersede.items():
                raw_schema: Dict[str, Any] = self.yaml_handler.load(dir)
                # Gather models and sources marked for superseding
                models_marked_for_superseding = set(
                    node.name for node in nodes if node.resource_type == NodeType.Model
                )
                sources_marked_for_superseding = set(
                    (node.source_name, node.name)
                    for node in nodes
                    if node.resource_type == NodeType.Source
                )
                # Gather models and sources in schema file
                models_in_schema = set(m["name"] for m in raw_schema.get("models", []))
                sources_in_schema = set(
                    (s["name"], t["name"])
                    for s in raw_schema.get("sources", [])
                    for t in s.get("tables", [])
                )
                # Set difference to determine non-superseded models and sources
                non_superseded_models = models_in_schema - models_marked_for_superseding
                non_superseded_sources = sources_in_schema - sources_marked_for_superseding
                if len(non_superseded_models) + len(non_superseded_sources) == 0:
                    logger().info(":rocket: Superseded schema file %s", dir.name)
                    if not self.dry_run:
                        dir.unlink(missing_ok=True)
                        if len(list(dir.parent.iterdir())) == 0:
                            dir.parent.rmdir()
                else:
                    # Preserve non-superseded models
                    preserved_models = []
                    for model in raw_schema.get("models", []):
                        if model["name"] in non_superseded_models:
                            preserved_models.append(model)
                    raw_schema["models"] = preserved_models
                    # Preserve non-superseded sources
                    ix = []
                    for i, source in enumerate(raw_schema.get("sources", [])):
                        for j, table in enumerate(source.get("tables", [])):
                            if (source["name"], table["name"]) not in non_superseded_sources:
                                ix.append((i, j))
                    for i, j in reversed(ix):
                        raw_schema["sources"][i]["tables"].pop(j)
                    ix = []
                    for i, source in enumerate(raw_schema.get("sources", [])):
                        if not source["tables"]:
                            ix.append(i)
                    for i in reversed(ix):
                        if not raw_schema["sources"][i]["tables"]:
                            raw_schema["sources"].pop(i)
                    if not self.dry_run:
                        self.yaml_handler.dump(raw_schema, dir)
                        self.mutations += 1
                    logger().info(
                        ":satellite: Model documentation migrated from %s to %s",
                        dir.name,
                        target.name,
                    )
        return True

    @staticmethod
    def pretty_print_restructure_plan(blueprint: Dict[Path, SchemaFileMigration]) -> None:
        logger().info(
            list(
                map(
                    lambda plan: (
                        [s.name for s in blueprint[plan].supersede] or "CREATE",
                        "->",
                        plan,
                    ),
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
        if not hasattr(node, "depends_on"):
            return family_tree
        for parent in getattr(node.depends_on, "nodes", []):
            member = self.manifest.nodes.get(parent, self.manifest.sources.get(parent))
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
        """Inherit knowledge from ancestors in reverse insertion order to ensure that the most
        recent ancestor is always the one to inherit from
        """
        knowledge: Dict[str, Dict[str, Any]] = {}
        for generation in reversed(family_tree):
            for ancestor in family_tree[generation]:
                member: ManifestNode = self.manifest.nodes.get(
                    ancestor, self.manifest.sources.get(ancestor)
                )
                if not member:
                    continue
                for name, info in member.columns.items():
                    knowledge.setdefault(name, {"progenitor": ancestor})
                    deserialized_info = info.to_dict()
                    # Handle Info:
                    # 1. tags are additive
                    # 2. descriptions are overriden
                    # 3. meta is merged
                    # 4. tests are ignored until I am convinced those shouldn't be
                    #       hand curated with love
                    if deserialized_info["description"] in self.placeholders:
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
        undocumented_columns: Columns missing documentation -- descriptions will be inherited and
            injected into schema file where prior knowledge exists
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

    def _run(self, unique_id, node, schema_map, force_inheritance=False):
        try:
            with self.mutex:
                logger().info(":point_right: Processing model: [bold]%s[/bold]", unique_id)
            # Get schema file path, must exist to propagate documentation
            schema_path: Optional[SchemaFileLocation] = schema_map.get(unique_id)
            if schema_path is None or schema_path.current is None:
                with self.mutex:
                    logger().info(
                        ":bow: No valid schema file found for model %s", unique_id
                    )  # We can't take action
                    return

            # Build Sets
            logger().info(":mag: Resolving columns in database")
            database_columns_ordered = self.get_columns(self.get_database_parts(node))
            database_columns: Set[str] = set(database_columns_ordered)
            yaml_columns_ordered = [column for column in node.columns]
            yaml_columns: Set[str] = set(yaml_columns_ordered)

            if not database_columns:
                with self.mutex:
                    logger().info(
                        (
                            ":safety_vest: Unable to resolve columns in database, falling back to"
                            " using yaml columns as base column set for model %s"
                        ),
                        unique_id,
                    )
                database_columns_ordered = yaml_columns_ordered
                database_columns = yaml_columns

            # Get documentated columns
            documented_columns: Set[str] = set(
                column
                for column, info in node.columns.items()
                if info.description and info.description not in self.placeholders
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

            with self.mutex:
                schema_file = self.yaml_handler.load(schema_path.current)
                section = self.maybe_get_section_from_schema_file(schema_file, node)
                if section is None:  # If we can't find the section, we can't take action
                    logger().info(":thumbs_up: No actions needed for %s", node.unique_id)
                    return

                should_dump = False
                n_cols_added, n_cols_doc_inherited, n_cols_removed = 0, 0, 0
                if len(missing_columns) > 0 or len(undocumented_columns) or len(extra_columns) > 0:
                    # Update schema file
                    (
                        n_cols_added,
                        n_cols_doc_inherited,
                        n_cols_removed,
                    ) = self.update_schema_file_and_node(
                        missing_columns,
                        undocumented_columns,
                        extra_columns,
                        node,
                        section,
                    )
                if n_cols_added + n_cols_doc_inherited + n_cols_removed > 0:
                    should_dump = True
                if tuple(database_columns_ordered) != tuple(yaml_columns_ordered):
                    # Sort columns in schema file to match database
                    logger().info(
                        ":wrench: Reordering columns in schema file for model %s", unique_id
                    )

                    last_ix: int = int(1e6) # Arbitrary starting value which increments, ensuring sort order
                    def _sort_columns(column_info: dict) -> int:
                        nonlocal last_ix
                        try:
                            normalized_name = self.column_casing(column_info["name"])
                            return database_columns_ordered.index(normalized_name)
                        except IndexError:
                            last_ix += 1
                            return last_ix

                    section["columns"].sort(key=_sort_columns)
                    should_dump = True
                if should_dump and not self.dry_run:
                    # Dump the mutated schema file back to the disk
                    self.yaml_handler.dump(schema_file, schema_path.current)
                    self.mutations += 1
                    logger().info(
                        ":sparkles: Schema file %s updated",
                        schema_path.current,
                    )
                else:
                    logger().info(
                        ":sparkles: Schema file is up to date for model %s",
                        unique_id,
                    )

            # Print Audit Report
            n_cols = float(len(database_columns))
            n_cols_documented = float(len(documented_columns)) + n_cols_doc_inherited
            perc_coverage = (
                min(100.0 * round(n_cols_documented / n_cols, 3), 100.0)
                if n_cols > 0
                else "Unable to Determine"
            )
            if logger().level <= 10:
                with self.mutex:
                    logger().debug(
                        self.audit_report.format(
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
        except Exception as e:
            with self.mutex:
                logger().error("Error occurred while processing model %s: %s", unique_id, e)
            raise e

    def propagate_documentation_downstream(self, force_inheritance: bool = False) -> None:
        schema_map = self.build_schema_folder_mapping()
        futs = []
        with self.adapter.connection_named("dbt-osmosis"):
            for unique_id, node in self.filtered_models():
                futs.append(
                    self.tpe.submit(self._run, unique_id, node, schema_map, force_inheritance)
                )
            wait(futs)

    @staticmethod
    def remove_columns_not_in_database(
        extra_columns: Iterable[str],
        node: ManifestNode,
        yaml_file_model_section: Dict[str, Any],
    ) -> int:
        """Removes columns found in dbt model that do not exist in database from both node
        and model simultaneously
        THIS MUTATES THE NODE AND MODEL OBJECTS so that state is always accurate"""
        changes_committed = 0
        for column in extra_columns:
            node.columns.pop(column, None)
            yaml_file_model_section["columns"] = [
                c for c in yaml_file_model_section["columns"] if c["name"] != column
            ]
            changes_committed += 1
            logger().info(
                ":wrench: Removing column %s from dbt schema for model %s", column, node.unique_id
            )
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

        inheritables = ["description"]
        if not self.skip_add_tags:
            inheritables.append("tags")
        if not self.skip_merge_meta:
            inheritables.append("meta")

        changes_committed = 0
        for column in undocumented_columns:
            camel_column = re.sub("_(.)", lambda m:m.group(1).upper(), column)
            prior_knowledge = knowledge.get(column, False) or knowledge.get(column.lower(), False) or knowledge.get(camel_column, False) or {}
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
                (
                    ":light_bulb: Column %s is inheriting knowledge from the lineage of progenitor"
                    " (%s) for model %s"
                ),
                column,
                progenitor,
                node.unique_id,
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
            node.columns[column] = ColumnInfo.from_dict({"name": column, "description": ""})
            yaml_file_model_section.setdefault("columns", []).append(
                {"name": column, "description": ""}
            )
            changes_committed += 1
            logger().info(
                ":syringe: Injecting column %s into dbt schema for model %s", column, node.unique_id
            )
        return changes_committed

    def update_schema_file_and_node(
        self,
        missing_columns: Iterable[str],
        undocumented_columns: Iterable[str],
        extra_columns: Iterable[str],
        node: ManifestNode,
        section: Dict[str, Any],
    ) -> Tuple[int, int, int]:
        """Take action on a schema file mirroring changes in the node."""
        logger().info(":microscope: Looking for actions for %s", node.unique_id)
        if not self.skip_add_columns:
            n_cols_added = self.add_missing_cols_to_node_and_model(missing_columns, node, section)
        n_cols_doc_inherited = self.update_undocumented_columns_with_prior_knowledge(
            undocumented_columns, node, section
        )
        n_cols_removed = self.remove_columns_not_in_database(extra_columns, node, section)
        return n_cols_added, n_cols_doc_inherited, n_cols_removed

    @staticmethod
    def maybe_get_section_from_schema_file(
        yaml_file: Dict[str, Any], node: ManifestNode
    ) -> Optional[Dict[str, Any]]:
        """Get the section of a schema file that corresponds to a node."""
        if node.resource_type == NodeType.Source:
            section = next(
                (
                    table
                    for source in yaml_file["sources"]
                    if node.source_name == source["name"]
                    for table in source["tables"]
                    if table["name"] == node.name
                ),
                None,
            )
        else:
            section = next(
                (s for s in yaml_file["models"] if s["name"] == node.name),
                None,
            )
        return section
