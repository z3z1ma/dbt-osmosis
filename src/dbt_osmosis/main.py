"""dbt-osmosis

Primary Objectives

Standardize organization of schema files
    -- Config is be able to be set per directory if desired utilizing `dbt_project.yml`, all directories require direct or inherited config `+dbt-osmosis:`
        If even one dir is missing the config, we close gracefully and inform user to update dbt_project.yml. No assumed defaults.
    - Can be one schema file to one model file sharing the same name and directory 
        -> +dbt-osmosis: "model.yml"
    - Can be one schema file per directory wherever model files reside named schema.yml 
        -> +dbt-osmosis: "schema.yml"
    - Can be one schema file per directory wherever model files reside named after its containing folder 
        -> +dbt-osmosis: "folder.yml"
    - Can be one schema file to one model file sharing the same name nested in a schema subdir wherever model files reside 
        -> +dbt-osmosis: "schema/model.yml"

Bootstrap Non-documented models
    - Will automatically conform to above config per directory based on location of model file 

Propagate existing column level documentation downward to children
    - Build column level knowledge graph accumulated and updated from furthest identifiable origin to immediate parents
    - Will automatically populate undocumented columns of the same name with passed down knowledge
    - Will inquire on mutations in definition while resolving progenitors 
        -- "We have detected X columns with mutated definitions. They can be ignored (default), we can backpopulate parents definitions 
            with the mutation (ie this definition is true upstream/better), or we can mark this column as a new progenitor aka source of 
            truth for the column's definition for the models children (the children would've inherited this definition anyway but the column 
            progenitor would resolve to a further upstream model -- this optimizes how engineers perform impact analysis)."


Order Matters
    1. Conform dbt project
        -- configuration lives in `dbt_project.yml` --> we require our config to run, can be at root level of `models:` to apply a default convention to a project 
        or can be folder by folder, follows dbt config resolution where config is overridden by scope. Lets validate this at the onset informing user of any folders 
        missing config defined in dbt_project.yml models block and gracefully close 
        Config is called +dbt-osmosis: "folder.yml" | "schema.yml" | "model.yml" | "schema/model.yml"
    2. Bootstrap models to ensure all models exist
    3. Recompile Manifest?
    4. Propagate definitions 
        - in a perfect world, we iterate the DAG from left to right mutating the in memory manifest to keep it in sync with yaml mutations

Directives: 
    Organize dbt project () -> non interactive
    Bootstrap models () -> non interactive
    Propagate definitions () -> interactive or non interactive
    Do all of the above () -> interactive or non interactive


Interactively bootstrap sources [lower priority but we have the pieces already]


New workflow enabled!

1.
    build one dbt model or a bunch of them
    run dbt-osmosis
    get your perfectly prepared schema yamls built with as much of the definitions 
        pre-populated as possible in exactly the right directories/files that
        conform to a configured standard upheld and enforced across your dbt 
        project on a dir by dir basis automatically -- using dbt_project.yml, nice
    boom, mic drop

2.
    problem reported by stakeholder with data
    identify column
    run dbt-osmosis impact --model --column
    find the originating model and action

3.
    need to score our documentation
    run dbt-osmosis coverage --docs --min-cov 80
    get a curated list of all the documentation to update
        in your pre-bootstrapped dbt project
    sip coffee and engage in documentation

"""

from dataclasses import dataclass
from pathlib import Path
from typing import (
    Optional,
    Dict,
    Any,
    List,
    Iterable,
    Mapping,
    MutableMapping,
    Tuple,
    Iterator,
    Union,
)
import subprocess

import click
from rich.progress import track
from ruamel.yaml import YAML
import dbt.config.profile
import dbt.config.project
import dbt.config.runtime
import dbt.config.renderer
import dbt.context.base
import dbt.parser.manifest
import dbt.exceptions
from dbt.adapters.factory import get_adapter, register_adapter, reset_adapters, Adapter
from dbt.tracking import disable_tracking

from .utils.logging import logger
from .exceptions.osmosis import (
    InvalidOsmosisConfig,
    MissingOsmosisConfig,
    SanitizationRequired,
)


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

SOURCE_REPORT = """
:white_check_mark: [bold]Audit Report[/bold]
-------------------------------

Database: [bold green]{database}[/bold green]
Schema: [bold green]{schema}[/bold green]
Table: [bold green]{table}[/bold green]

Total Columns in Database: {total_columns}
Total Documentation Coverage: {coverage}%

Action Log:
Columns Added to dbt: {n_cols_added}
Extra Columns Removed: {n_cols_removed}
"""
CONTEXT = {"max_content_width": 800}
UNDOCUMENTED_STR = [
    "Pending further documentation",
    "No description for this column.",
    "",
]
VALID_CONFIGS = [
    "schema.yml",
    "folder.yml",
    "model.yml",
    "schema/model.yml",
]
FILE_ADAPTER_POSTFIX = "://"
DEFAULT_PROFILES_DIR = dbt.config.profile.DEFAULT_PROFILES_DIR


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


@dataclass
class SchemaFile:
    target: Path
    current: Optional[Path] = None

    @property
    def is_valid(self) -> bool:
        return self.current == self.target


def remove_columns_not_in_database(
    extra_columns: Iterable, node: MutableMapping, model: MutableMapping
) -> int:
    """Removes columns found in dbt model that do not exist in database from both node and model simultaneously

    Args:
        extra_columns (Iterable): Iterable of extra columns
        node (MutableMapping): A manifest json node
        model (MutableMapping): A schema file model

    Returns:
        int: Number of actions performed
    """
    n_actions = 0
    for column in extra_columns:
        node["columns"].pop(column, None)
        found_ix = None
        for ix, model_column in enumerate(model["columns"]):
            if model_column["name"] == column:
                found_ix = ix
                break
        else:
            continue
        model["columns"].pop(found_ix)
        n_actions += 1
        logger().info(":wrench: Removing column %s from dbt schema", column)
    return n_actions


def update_undocumented_columns_with_prior_knowledge(
    undocumented_columns: Iterable, knowledge: Mapping, node: MutableMapping, model: MutableMapping
) -> int:
    """Update undocumented columns with prior knowledge in node and model simultaneously

    Args:
        undocumented_columns (Iterable): Iterable of undocumented columns
        knowledge (Mapping): Accumulated and updated knowledge from ancestors
        node (MutableMapping): A manifest.json model node
        model (MutableMapping): A loaded schema file model

    Returns:
        int: Number of actions performed
    """
    n_actions = 0
    for column in undocumented_columns:
        try:
            prior_knowledge = knowledge.get(column, {})
            if prior_knowledge.get("description"):
                node["columns"][column].update(prior_knowledge)
                progenitor = node["columns"][column].pop("progenitor", "Unknown")
                for model_column in model["columns"]:
                    if model_column["name"] == column:
                        model_column["description"] = prior_knowledge["description"]
                n_actions += 1
                logger().info(
                    ":light_bulb: Column %s is inheriting knowledge from a progenitor (%s)",
                    column,
                    progenitor,
                )
        except KeyError:
            pass
    return n_actions


def update_columns_with_prior_attributes(
    knowledge: Mapping, node: MutableMapping, model: MutableMapping
) -> int:
    """Update columns with inheritable tags, descriptions, and tests

    Args:
        undocumented_columns (Iterable): Iterable of undocumented columns
        knowledge (Mapping): Accumulated and updated knowledge from ancestors
        node (MutableMapping): A manifest.json model node
        model (MutableMapping): A loaded schema file model

    Returns:
        int: Number of actions performed
    """
    n_actions = 0
    for column, prior_knowledge in knowledge.items():
        try:
            node["columns"][column].update(prior_knowledge)
            progenitor = node["columns"][column].pop("progenitor", "Unknown")
            for model_column in model["columns"]:
                if model_column["name"] == column:
                    att = {}
                    if prior_knowledge.get("tags"):
                        att["tags"] = prior_knowledge["tags"]
                    if prior_knowledge.get("meta"):
                        att["meta"] = prior_knowledge["meta"]
                    if prior_knowledge.get("tests"):
                        att["tests"] = prior_knowledge["tests"]
                    model_column.update(att)
            n_actions += 1
            logger().info(
                ":light_bulb: Column %s is inheriting attributes from a progenitor (%s)",
                column,
                progenitor,
            )
        except KeyError:
            pass
    return n_actions


def add_missing_cols_to_node_and_model(
    missing_columns: Iterable, node: MutableMapping, model: MutableMapping
) -> int:
    """Add missing columns to node and model simultaneously

    Args:
        missing_columns (Iterable): An iterable of missing columns
        node (MutableMapping): A manifest.json model node
        model (MutableMapping): A loaded schema file model

    Returns:
        int: Number of actions performed
    """
    n_actions = 0
    for column in missing_columns:
        node["columns"][column] = {
            "name": column,
        }
        model.setdefault("columns", []).append({"name": column})
        n_actions += 1
        logger().info(":syringe: Injecting column %s into dbt schema", column)
    return n_actions


def update_schema_file_and_node(
    knowledge: Dict[str, Any],
    missing_columns: Iterable,
    undocumented_columns: Iterable,
    extra_columns: Iterable,
    node: Dict[str, Any],
    schema_file: Dict[str, Any],
) -> Tuple[int, int, int]:
    """Take action on a schema file mirroring changes in the node. This function mutates both the `schema_file` and the
    `node`.

    Args:
        knowledge (Dict[str, Any]): Model column knowledge
        missing_columns (Iterable): Missing columns present in database not in dbt
        undocumented_columns (Iterable): Columns in dbt which are not
        extra_columns (Iterable): Columns which are in dbt but not in database
        node (Dict[str, Any]): Loaded manifest.json node
        schema_file (Dict[str, Any]): Input loaded schema file

    Returns:
        Tuple[int, int, int]: Returns a tuple of actions based on input sets; actions are added columns, inherited columns, removed columns
    """
    # We can extrapolate this to a general func
    if node["resource_type"] == "source":
        _schema_file = None
        for src in schema_file.get("sources", []):
            if src["name"] == node["source_name"]:
                # Scope our pointer to a specific portion of the object
                _schema_file = src
        _schema_key = "tables"
    else:
        _schema_file = schema_file
        _schema_key = "models"
    if _schema_file is None:
        return 0, 0, 0
    for model in _schema_file[_schema_key]:
        if model["name"] == node["name"]:
            logger().info(":microscope: Looking for actions")
            # Empty iterables in the first arg are simply no-ops
            n_cols_added = add_missing_cols_to_node_and_model(missing_columns, node, model)
            n_cols_doc_inherited = update_undocumented_columns_with_prior_knowledge(
                undocumented_columns, knowledge, node, model
            )
            n_cols_inherit_attr = update_columns_with_prior_attributes(knowledge, node, model)
            n_cols_removed = remove_columns_not_in_database(extra_columns, node, model)
            return n_cols_added, n_cols_doc_inherited, n_cols_removed, n_cols_inherit_attr
    else:
        logger().info(":thumbs_up: No actions needed")
        return 0, 0, 0


def commit_project_restructure(restructured_project: MutableMapping[Path, Any]) -> bool:
    """Given a project restrucure plan of pathlib Paths to a mapping of output and supersedes which is in itself a mapping of Paths to model names,
    commit changes to filesystem to conform project to defined structure as code fully or partially superseding existing models as needed.

    Args:
        restructured_project (MutableMapping[Path, Any]): Project restructure plan as typically created by `build_project_structure_update_plan`

    Returns:
        bool: True if the project was restructured, False if no action was required
    """
    yaml = YAML()

    if not restructured_project:
        logger().info(":1st_place_medal: Project structure approved")
        return False

    logger().info(
        list(
            map(
                lambda plan: (restructured_project[plan]["supersede"], "->", plan),
                restructured_project.keys(),
            )
        )
    )

    logger().info(
        ":construction_worker: Executing action plan and conforming projecting schemas to defined structure"
    )
    for target, structure in restructured_project.items():
        if not target.exists():
            # Build File
            logger().info(":construction: Building schema file %s", target.name)
            target.parent.mkdir(exist_ok=True, parents=True)
            target.touch()
            with open(target, "w", encoding="utf-8") as f:
                yaml.dump(structure["output"], f)

        else:
            # Update File
            logger().info(":toolbox: Updating schema file %s", target.name)
            target_schema = yaml.load(target)
            if "version" not in target_schema:
                target_schema["version"] = 2
            target_schema.setdefault("models", []).extend(structure["output"]["models"])
            with open(target, "w", encoding="utf-8") as f:
                yaml.dump(target_schema, f)

        # Clean superseded schema files
        for dir, models in structure["supersede"].items():
            preserved_models = []
            raw_schema = yaml.load(dir)
            models_marked_for_superseding = set(models)
            models_in_schema = set(map(lambda mdl: mdl["name"], raw_schema.get("models", [])))
            non_superseded_models = models_in_schema - models_marked_for_superseding
            if len(non_superseded_models) == 0:
                logger().info(":rocket: Superseded schema file %s", dir.name)
                dir.unlink(missing_ok=True)
            else:
                for model in raw_schema["models"]:
                    if model["name"] in non_superseded_models:
                        preserved_models.append(model)
                raw_schema["models"] = preserved_models
                with open(dir, "w", encoding="utf-8") as f:
                    yaml.dump(raw_schema, f)
                logger().info(
                    ":satellite: Model documentation migrated from %s to %s",
                    dir.name,
                    target.name,
                )

    return True


def assert_schema_has_no_sources(schema: Mapping) -> Mapping:
    """Asserts that a schema does not have a source key

    Args:
        schema (Mapping): Loaded schema file to scheck

    Raises:
        SanitizationRequired: Raises this error if we require the user to sanitize and separate models from sources file

    Returns:
        Mapping: Loaded schema is returned on success
    """
    if schema.get("sources"):
        raise SanitizationRequired(
            "Found `sources:` block in a models schema file. We require you separate sources in order to organize your project."
        )
    return schema


def get_columns(database: str, schema: str, identifier: str, adapter: Adapter) -> Iterable:
    """Get all columns in a list for a model

    Args:
        database (str): Database name
        schema (str): Schema name
        identifier (str): Model name
        adapter (Adapter): dbt database adapter

    Returns:
        Iterable: List of column names for model from database
    """
    table = adapter.get_relation(database, schema, identifier)
    try:
        columns = [c.name for c in adapter.get_columns_in_relation(table)]
    except (dbt.exceptions.CompilationException, AttributeError):
        logger().info(
            ":cross_mark: Could not resolve relation %s.%s.%s against database active tables during introspective query",
            database,
            schema,
            identifier,
        )
        columns = []
    return columns


def bootstrap_existing_model(
    model: MutableMapping, database: str, schema: str, identifier: str, adapter: Adapter
) -> MutableMapping:
    """Injects columns from database into existing model if not found

    Args:
        model (MutableMapping): Model node
        database (str): Database name
        schema (str): Schema name
        identifier (str): Model name
        adapter (Adapter): dbt database adapter

    Returns:
        MutableMapping: Model is returned with injected columns or as is if no new columns found
    """
    model_columns = [c["name"] for c in model.get("columns", [])]
    database_columns = get_columns(database, schema, identifier, adapter)
    for column in database_columns:
        if column not in model_columns:
            logger().info(":syringe: Injecting column %s into dbt schema", column)
            model.setdefault("columns", []).append({"name": column})
    return model


def create_base_model(
    database: str, schema: str, identifier: str, adapter: Adapter
) -> MutableMapping:
    """Creates a base model with model name, column names populated from database

    Args:
        database (str): Database name
        schema (str): Schema name
        identifier (str): Model name
        adapter (Adapter): dbt database adapter

    Returns:
        MutableMapping: Base model with model name and column names populated
    """
    columns = get_columns(database, schema, identifier, adapter)
    return {
        "name": identifier,
        "columns": [{"name": column_name} for column_name in columns],
    }


def build_project_structure_update_plan(
    schema_map: MutableMapping, manifest: MutableMapping, adapter: Adapter
) -> MutableMapping:
    """Build project structure update plan based on `dbt-osmosis:` configs set across dbt_project.yml and model files.
    The update plan includes injection of undocumented models. Unless this plan is constructed and executed by the `commit_project_restructure` function,
    dbt-osmosis will only operate on models it is aware of through the existing documentation.

    Args:
        schema_map (MutableMapping): Mapping of model ids to SchemaFile objects which include path to schema and path to target
        manifest (MutableMapping): A dbt manifest.json artifact
        adapter (Adapter): dbt database adapter

    Returns:
        MutableMapping: Update plan where dict keys consist of targets and contents consist of outputs which match the contents of the `models` to be output in the
        target file and supersede lists of what files are superseded by a migration
    """
    yaml = YAML()
    proj_restructure_plan = {}
    logger().info(
        ":chart_increasing: Searching project stucture for required updates and building action plan"
    )
    with adapter.connection_named("dbt-osmosis"):
        for unique_id, schema_file in schema_map.items():
            if not schema_file.is_valid:
                proj_restructure_plan.setdefault(
                    schema_file.target, {"output": {"version": 2, "models": []}, "supersede": {}}
                )
                node = manifest["nodes"][unique_id]
                if schema_file.current is None:
                    # Bootstrapping Undocumented Model
                    proj_restructure_plan[schema_file.target]["output"]["models"].append(
                        create_base_model(node["database"], node["schema"], node["name"], adapter)
                    )
                else:
                    # Model Is Documented but Must be Migrated
                    if not schema_file.current.exists():
                        continue
                    schema = assert_schema_has_no_sources(yaml.load(schema_file.current))
                    models: Iterable[MutableMapping] = schema.get("models", [])
                    for model in models:
                        if model["name"] == node["name"]:
                            # Bootstrapping Documented Model
                            proj_restructure_plan[schema_file.target]["output"]["models"].append(
                                bootstrap_existing_model(
                                    model, node["database"], node["schema"], node["name"], adapter
                                )
                            )
                            # Target to supersede current
                            proj_restructure_plan[schema_file.target]["supersede"].setdefault(
                                schema_file.current, []
                            ).append(model["name"])
                            break
                    else:
                        ...  # Model not found at patch path -- We should pass on this for now
            else:
                ...  # Valid schema file found for model -- We will update the columns in the `Document` task

    return proj_restructure_plan


def get_documented_columns(dbt_columns: Iterable, node: MutableMapping) -> Iterable:
    """Return all documented columns for a node based on columns existing in dbt schema file

    Args:
        dbt_columns (Iterable): dbt columns extracted from manifest as derived from dbt schema file
        node (MutableMapping): A manifest json node representing a model

    Returns:
        set: A set of column names which are documented in the schema file
    """
    return [
        column_name
        for column_name in dbt_columns
        if node["columns"].get(column_name, {}).get("description", "") not in UNDOCUMENTED_STR
    ]


def build_ancestor_tree(
    node: MutableMapping,
    manifest: MutableMapping,
    family_tree: Optional[MutableMapping] = None,
    members_found: Optional[List[str]] = None,
    depth: int = 0,
) -> MutableMapping:
    """Recursively build dictionary of parents in generational order

    Args:
        node (MutableMapping): A manifest node from a loaded compiled manifest
        manifest (MutableMapping): A fully compile manifest.json
        family_tree (Optional[MutableMapping], optional): Used in recursion. Defaults to None.
        members_found (Optional[List[str]]): Used in recursion. Defaults to None.
        depth (int, optional): Used in recursion. Defaults to 0.

    Returns:
        MutableMapping: Mapping of parents keyed by generations removed
    """
    if family_tree is None:
        family_tree = {}
    if members_found is None:
        members_found = []

    for parent in node.get("depends_on", {}).get("nodes", []):
        member = manifest["nodes"].get(parent, manifest["sources"].get(parent))
        if member and parent not in members_found:
            family_tree.setdefault(f"generation_{depth}", []).append(parent)
            members_found.append(parent)
            family_tree = build_ancestor_tree(
                member, manifest, family_tree, members_found, depth + 1
            )

    return family_tree


def pass_down_knowledge(
    family_tree: MutableMapping,
    manifest: MutableMapping,
) -> MutableMapping:
    """Build a knowledgebase for the model based on iterating through ancestors

    Args:
        family_tree (MutableMapping): Hash map of parents keyed by generation
        manifest (MutableMapping): dbt Manifest

    Returns:
        MutableMapping: Mapping of columns to dbt Column representations
    """
    knowledge = {}
    for generation in reversed(family_tree):
        for ancestor in family_tree[generation]:
            member = manifest["nodes"].get(ancestor, manifest["sources"].get(ancestor))
            if not member:
                continue
            for column_name, column_details in member["columns"].items():
                knowledge.setdefault(column_name, {"progenitor": ancestor})
                if column_details.get("description", "") in UNDOCUMENTED_STR:
                    column_details.pop("description", None)
                knowledge[column_name].update(column_details)

    return knowledge


def resolve_schema_path(root_path: str, node: MutableMapping) -> Optional[Path]:
    """Resolve absolute schema file path for a manifest node

    Args:
        root_path (str): Root dbt project path
        node (MutableMapping): A manifest json node representing a model

    Returns:
        Optional[Path]: Path object representing resolved path to file where schema exists otherwise return None
    """
    schema_path = None
    if node["resource_type"] == "model":
        schema_path: str = node.get("patch_path")
    elif node["resource_type"] == "source":
        schema_path: str = node["source_name"] + FILE_ADAPTER_POSTFIX + node.get("path")
    if schema_path:
        schema_path = Path(root_path).joinpath(schema_path.split(FILE_ADAPTER_POSTFIX, 1)[1])
    return schema_path


def resolve_osmosis_target_schema_path(osmosis_config: str, node: MutableMapping) -> Path:
    """Resolve the correct schema yml target based on the dbt-osmosis config for the model / directory

    Args:
        osmosis_config (str): config string as seen in `VALID_CONFIGS`
        node (MutableMapping): A manifest json node representing a model

    Raises:
        InvalidOsmosisConfig: If no valid config is found

    Returns:
        Path: Path object representing resolved path to file where schema should exists
    """
    if osmosis_config == "schema.yml":
        schema = "schema"
    elif osmosis_config == "folder.yml":
        schema = node["fqn"][-2]
    elif osmosis_config == "model.yml":
        schema = node["name"]
    elif osmosis_config == "schema/model.yml":
        schema = "schema/" + node["name"]
    else:
        raise InvalidOsmosisConfig("Invalid dbt-osmosis config for model")

    return Path(node["root_path"], node["original_file_path"]).parent / Path(f"{schema}.yml")


def validate_osmosis_config(node: MutableMapping) -> None:
    """Validates a config string. If input is a source, we return the resource type str instead

    Args:
        node (MutableMapping): A manifest json node representing a model or source

    Raises:
        MissingOsmosisConfig: Thrown if no config is present for a node
        InvalidOsmosisConfig: Thrown if an invalid config is present on a node under our key `dbt-osmosis`

    Returns:
        str: Validated config str
    """
    if node["resource_type"] == "source":
        return node["resource_type"]
    osmosis_config = node["config"].get("dbt-osmosis")
    if not osmosis_config:
        raise MissingOsmosisConfig(
            f"Config not set for model {node['name']}, we recommend setting the config at a directory level through the `dbt_project.yml`"
        )
    if osmosis_config not in VALID_CONFIGS:
        raise InvalidOsmosisConfig(f"Invalid dbt-osmosis config for model {node['name']}")

    return osmosis_config


def is_valid_model(proj: str, node: MutableMapping, fqn: Optional[str] = None) -> bool:
    """Validates a node as being a targetable model. Validates both models and sources.

    Args:
        proj (str): Project name
        node (MutableMapping): A manifest json node
        fqn (str, optional): Filter target

    Returns:
        bool: returns true if the node is a valid targetable model
    """
    if fqn is None:
        fqn = ".".join(node["fqn"][1:])
    logger().debug("%s: %s -> %s", node["resource_type"], fqn, node["fqn"][1:])
    return (
        node["resource_type"] in ("model", "source")
        and node.get("package_name") == proj
        and node["config"].get("materialized", "source") != "ephemeral"
        and (len(node["fqn"][1:]) >= len(fqn.split(".")))
        and all(p1 == p2 for p1, p2 in zip(fqn.split("."), node["fqn"][1:]))
    )


def iter_valid_models(
    proj: str, nodes: MutableMapping, fqn: Optional[str] = None
) -> Iterator[Tuple[str, MutableMapping]]:
    """Generates an iterator of valid models

    Args:
        proj (str): The name of the target project
        nodes (MutableMapping): Nodes as passed from manifest["nodes"]
        fqn (str, optional): Filter targets

    Yields:
        Iterator[Tuple[str, MutableMapping]]: A generator of model ids and contents
    """
    for unique_id, node in nodes.items():
        if is_valid_model(proj, node, fqn):
            yield unique_id, node


def build_schema_folder_map(
    proj: str,
    root: str,
    manifest: MutableMapping,
    fqn: Optional[str] = None,
    model_type: str = "nodes",
) -> Dict[str, SchemaFile]:
    """Builds a mapping of models or sources to their existing and target schema file paths

    Args:
        proj (str): The name of the project which is being actioned
        root (str): Root dbt project path
        manifest (MutableMapping): dbt manifest json artifact
        fqn (str, optional): Filter targets
        model_type (str, optional): Model type can be one of either `nodes` or `models`

    Returns:
        Dict[str, SchemaFile]: Mapping of models to their existing and target schema file paths
    """
    assert model_type in (
        "nodes",
        "sources",
    ), "Invalid model_type argument passed to build_schema_folder_map, expected one of ('nodes', 'sources')"
    schema_map = {}
    logger().info("...building project structure mapping in memory")
    for unique_id, node in iter_valid_models(proj, manifest[model_type], fqn):
        schema_path = resolve_schema_path(root, node)
        osmosis_config = validate_osmosis_config(node)
        if osmosis_config == "source":
            # For sources, we will keep the current path and target the same. We have no standards by which to adhere to regarding
            # how users wish to organize source yamls
            osmosis_schema_path = schema_path
        else:
            # For models, we will resolve the target path -- perhaps this logical fork moves into the function below
            osmosis_schema_path = resolve_osmosis_target_schema_path(osmosis_config, node)
        schema_map[unique_id] = SchemaFile(target=osmosis_schema_path, current=schema_path)

    return schema_map


def get_raw_profiles(profiles_dir: Optional[str] = None) -> Dict[str, Any]:
    return dbt.config.profile.read_profile(profiles_dir or dbt.config.profile.DEFAULT_PROFILES_DIR)


def load_dbt(
    threads: int = 1,
    project_dir: Optional[str] = None,
    profiles_dir: Optional[str] = None,
    target: Optional[str] = None,
):
    args = PseudoArgs(
        threads=threads,
        project_dir=project_dir,
        profiles_dir=profiles_dir,
        target=target,
    )
    project, profile = dbt.config.runtime.RuntimeConfig.collect_parts(args)
    config = dbt.config.runtime.RuntimeConfig.from_parts(project, profile, args)
    reset_adapters()
    register_adapter(config)
    adapter = verify_connection(get_adapter(config))
    return project, profile, config, adapter


def compile_project_load_manifest(
    cfg: Optional[dbt.config.runtime.RuntimeConfig] = None,
    flat: bool = True,
    project_dir: Optional[str] = None,
    profiles_dir: Optional[str] = None,
    target: Optional[str] = None,
) -> Union[MutableMapping, dbt.parser.manifest.Manifest]:
    """Compiles dbt project and builds manifest json artifact in memory

    Args:
        cfg (Optional[RuntimeConfig]): dbt runtime config

    Returns:
        MutableMapping: Loaded manifest.json artifact
    """
    if not cfg:
        _, _, cfg, _ = load_dbt(1, project_dir, profiles_dir, target)
    if not flat:
        return dbt.parser.manifest.ManifestLoader.get_full_manifest(cfg)
    else:
        return dbt.parser.manifest.ManifestLoader.get_full_manifest(cfg).flat_graph


def verify_connection(adapter: Adapter) -> Adapter:
    try:
        with adapter.connection_named("debug"):
            adapter.debug_query()
    except Exception as exc:
        raise Exception("Could not connect to Database") from exc
    else:
        return adapter


def propagate_documentation_downstream(
    proj: str,
    schema_map: MutableMapping,
    manifest: MutableMapping,
    adapter: Adapter,
    fqn: Optional[str] = None,
    force_inheritance: bool = False,
) -> None:
    yaml = YAML()
    with adapter.connection_named("dbt-osmosis"):
        for unique_id, node in track(list(iter_valid_models(proj, manifest["nodes"], fqn))):
            logger().info("\n:point_right: Processing model: [bold]%s[/bold] \n", unique_id)

            table = node["database"], node["schema"], node.get("alias", node["name"])
            knowledge = pass_down_knowledge(build_ancestor_tree(node, manifest), manifest)
            schema_path = schema_map.get(unique_id)

            if schema_path is None:
                logger().info(
                    ":bow: No schema file found for model %s", unique_id
                )  # We can't take action
                continue

            # Build Sets
            database_columns = set(get_columns(*table, adapter))
            dbt_columns = set(column_name for column_name in node["columns"])
            documented_columns = set(get_documented_columns(dbt_columns, node))

            if not database_columns:
                # Note to user we are using dbt columns as source since we did not resolve them from db
                # Doing this means only `undocumented_columns` can be resolved to a non-empty set
                logger().info(
                    ":safety_vest: Unable to resolve columns in database, falling back to using manifest columns as base column set\n"
                )
                database_columns = dbt_columns

            # Action
            missing_columns = database_columns - dbt_columns
            """Columns in database not in dbt -- will be injected into schema file"""
            undocumented_columns = database_columns - documented_columns
            """Columns missing documentation -- descriptions will be inherited and injected into schema file where prior knowledge exists"""
            extra_columns = dbt_columns - database_columns
            """Columns in schema file not in database -- will be removed from schema file"""

            if force_inheritance:
                # Consider all columns "undocumented" so that inheritance is not selective
                undocumented_columns = database_columns

            n_cols_added = 0
            n_cols_doc_inherited = 0
            n_cols_removed = 0
            if (
                len(missing_columns) > 0 or len(undocumented_columns) or len(extra_columns) > 0
            ) and len(database_columns) > 0:
                schema_file = yaml.load(schema_path.current)
                (
                    n_cols_added,
                    n_cols_doc_inherited,
                    n_cols_removed,
                    n_cols_inherit_attr,
                ) = update_schema_file_and_node(
                    knowledge,
                    missing_columns,
                    undocumented_columns,
                    extra_columns,
                    node,
                    schema_file,
                )
                if n_cols_added + n_cols_doc_inherited + n_cols_removed + n_cols_inherit_attr > 0:
                    with open(schema_path.current, "w", encoding="utf-8") as f:
                        yaml.dump(schema_file, f)
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
                    database=node["database"],
                    schema=node["schema"],
                    table=node["name"],
                    total_columns=n_cols,
                    n_cols_added=n_cols_added,
                    n_cols_doc_inherited=n_cols_doc_inherited,
                    n_cols_removed=n_cols_removed,
                    coverage=perc_coverage,
                )
            )


def synchronize_sources(
    proj: str,
    schema_map: MutableMapping,
    manifest: MutableMapping,
    adapter: Adapter,
    fqn: Optional[str] = None,
) -> None:
    yaml = YAML()
    with adapter.connection_named("dbt-osmosis"):
        for unique_id, node in track(list(iter_valid_models(proj, manifest["sources"], fqn))):
            logger().info("\n:point_right: Processing source: [bold]%s[/bold] \n", unique_id)

            table = node["database"], node["schema"], node.get("identifier", node["name"])
            schema_path = schema_map.get(unique_id)

            if schema_path is None:
                logger().info(
                    ":bow: No schema file found for source %s", unique_id
                )  # We can't take action
                continue

            # Build Sets
            database_columns = set(get_columns(*table, adapter))
            dbt_columns = set(column_name for column_name in node["columns"])
            documented_columns = set(get_documented_columns(dbt_columns, node))

            if not database_columns:
                # Note to user we are using dbt columns as source since we did not resolve them from db
                # Doing this means only `undocumented_columns` can be resolved to a non-empty set
                logger().info(
                    ":safety_vest: Unable to resolve columns in database, skipping this source\n"
                )
                continue

            # Action
            missing_columns = database_columns - dbt_columns
            """Columns in database not in dbt -- will be injected into schema file"""
            extra_columns = dbt_columns - database_columns
            """Columns in schema file not in database -- will be removed from schema file"""

            n_cols_added = 0
            n_cols_removed = 0
            if (len(missing_columns) > 0 or len(extra_columns) > 0) and len(database_columns) > 0:
                schema_file = yaml.load(schema_path.current)
                (
                    n_cols_added,
                    _,
                    n_cols_removed,
                    n_cols_inherit_attr,
                ) = update_schema_file_and_node(
                    {},  # No ancestors to generate knowledge graph
                    missing_columns,
                    [],  # Undocumented columns have nothing upstream to inherit
                    extra_columns,
                    node,
                    schema_file,
                )
                if n_cols_added + n_cols_removed + n_cols_inherit_attr > 0:
                    with open(schema_path.current, "w", encoding="utf-8") as f:
                        yaml.dump(schema_file, f)
                    logger().info(":sparkles: Schema file updated")

            # Print Audit Report
            n_cols = float(len(database_columns))
            n_cols_documented = float(len(documented_columns))
            perc_coverage = (
                min(100.0 * round(n_cols_documented / n_cols, 3), 100.0)
                if n_cols > 0
                else "Unable to Determine"
            )
            logger().info(
                SOURCE_REPORT.format(
                    database=node["database"],
                    schema=node["schema"],
                    table=node["name"],
                    total_columns=n_cols,
                    n_cols_added=n_cols_added,
                    n_cols_removed=n_cols_removed,
                    coverage=perc_coverage,
                )
            )


@click.group()
@click.version_option()
def cli():
    pass


@cli.command(context_settings=CONTEXT)
@click.option(
    "--project-dir",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    help="Which directory to look in for the dbt_project.yml file. Default is the current working directory and its parents.",
)
@click.option(
    "--profiles-dir",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    default=dbt.config.profile.DEFAULT_PROFILES_DIR,
    help="Which directory to look in for the profiles.yml file. Defaults to ~/.dbt",
)
@click.option(
    "--target",
    type=click.STRING,
    help="Which profile to load. Overrides setting in dbt_project.yml.",
)
@click.option(
    "-f",
    "--fqn",
    type=click.STRING,
    help="Specify models based on dbt's FQN. Looks like folder.folder, folder.folder.model, or folder.folder.source.table. Use list command to see the scope of an FQN filter.",
)
@click.option(
    "-F",
    "--force-inheritance",
    is_flag=True,
    help="If specified, forces documentation to be inherited overriding existing column level documentation where applicable.",
)
def run(
    target: Optional[str] = None,
    project_dir: Optional[str] = None,
    profiles_dir: Optional[str] = None,
    fqn: Optional[str] = None,
    force_inheritance: bool = False,
):
    """Compose -> Document -> Audit

    \f
    This command will conform your project as outlined in `dbt_project.yml`, bootstrap undocumented dbt models,
    and propagate column level documentation downwards

    Args:
        target (Optional[str]): Profile target. Defaults to default target set in profile yml
        project_dir (Optional[str], optional): Dbt project directory. Defaults to current working directory.
        profiles_dir (Optional[str], optional): Dbt profile directory. Defaults to ~/.dbt
    """
    logger().info(":water_wave: Executing dbt-osmosis\n")

    # Initialize dbt & prepare database adapter
    project, profile, config, adapter = load_dbt(1, project_dir, profiles_dir, target)
    manifest = compile_project_load_manifest(config)

    # Conform project structure & bootstrap undocumented models injecting columns
    schema_map = build_schema_folder_map(project.project_name, project.project_root, manifest, fqn)
    was_restructured = commit_project_restructure(
        build_project_structure_update_plan(schema_map, manifest, adapter)
    )

    if was_restructured:
        # Recompile on restructure
        manifest = compile_project_load_manifest(config)
        schema_map = build_schema_folder_map(
            project.project_name, project.project_root, manifest, fqn
        )

    # Propagate documentation & inject/remove schema file columns to align with model in database
    propagate_documentation_downstream(
        project.project_name, schema_map, manifest, adapter, fqn, force_inheritance
    )


@cli.command(context_settings=CONTEXT)
@click.option(
    "--project-dir",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    help="Which directory to look in for the dbt_project.yml file. Default is the current working directory and its parents.",
)
@click.option(
    "--profiles-dir",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    default=dbt.config.profile.DEFAULT_PROFILES_DIR,
    help="Which directory to look in for the profiles.yml file. Defaults to ~/.dbt",
)
@click.option(
    "--target",
    type=click.STRING,
    help="Which profile to load. Overrides setting in dbt_project.yml.",
)
@click.option(
    "-f",
    "--fqn",
    type=click.STRING,
    help="Specify models based on FQN. Use dots as separators. Looks like folder.folder.model or folder.folder.source.table. Use list command to see the scope of an FQN filter.",
)
def compose(
    target: Optional[str] = None,
    project_dir: Optional[str] = None,
    profiles_dir: Optional[str] = None,
    fqn: Optional[str] = None,
):
    """Organizes schema ymls based on config and injects undocumented models

    \f
    This command will conform schema ymls in your project as outlined in `dbt_project.yml` & bootstrap undocumented dbt models

    Args:
        target (Optional[str]): Profile target. Defaults to default target set in profile yml
        project_dir (Optional[str], optional): Dbt project directory. Defaults to current working directory.
        profiles_dir (Optional[str], optional): Dbt profile directory. Defaults to ~/.dbt
    """
    logger().info(":water_wave: Executing dbt-osmosis\n")

    # Initialize dbt & prepare database adapter
    project, profile, config, adapter = load_dbt(1, project_dir, profiles_dir, target)
    manifest = compile_project_load_manifest(config)

    # Conform project structure & bootstrap undocumented models injecting columns
    commit_project_restructure(
        build_project_structure_update_plan(
            build_schema_folder_map(project.project_name, project.project_root, manifest, fqn),
            manifest,
            adapter,
        )
    )


@cli.command()
def audit():
    """Audits documentation for coverage with actionable artifact or interactive prompt driving user to document progenitors"""
    click.echo("Executing Audit")


@cli.command(context_settings=CONTEXT)
@click.option(
    "--project-dir",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    help="Which directory to look in for the dbt_project.yml file. Default is the current working directory and its parents.",
)
@click.option(
    "--profiles-dir",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    default=dbt.config.profile.DEFAULT_PROFILES_DIR,
    help="Which directory to look in for the profiles.yml file. Defaults to ~/.dbt",
)
@click.option(
    "--target",
    type=click.STRING,
    help="Which profile to load. Overrides setting in dbt_project.yml.",
)
@click.option(
    "-f",
    "--fqn",
    type=click.STRING,
    help="Specify models based on FQN. Use dots as separators. Looks like folder.folder.model or folder.folder.source.table. Use list command to see the scope of an FQN filter.",
)
@click.option(
    "-F",
    "--force-inheritance",
    is_flag=True,
    help="If specified, forces documentation to be inherited overriding existing column level documentation where applicable.",
)
def document(
    target: Optional[str] = None,
    project_dir: Optional[str] = None,
    profiles_dir: Optional[str] = None,
    fqn: Optional[str] = None,
    force_inheritance: bool = False,
):
    """Column level documentation inheritance for existing models"""
    logger().info(":water_wave: Executing dbt-osmosis\n")

    # Initialize dbt & prepare database adapter
    project, profile, config, adapter = load_dbt(1, project_dir, profiles_dir, target)
    manifest = compile_project_load_manifest(config)
    schema_map = build_schema_folder_map(project.project_name, project.project_root, manifest, fqn)

    # Propagate documentation & inject/remove schema file columns to align with model in database
    propagate_documentation_downstream(
        project.project_name, schema_map, manifest, adapter, fqn, force_inheritance
    )


@cli.group()
def sources():
    """Synchronize schema file sources with database or create/update a source based on pattern"""
    ...


@sources.command(context_settings=CONTEXT)
@click.option(
    "--project-dir",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    help="Which directory to look in for the dbt_project.yml file. Default is the current working directory and its parents.",
)
@click.option(
    "--profiles-dir",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    default=dbt.config.profile.DEFAULT_PROFILES_DIR,
    help="Which directory to look in for the profiles.yml file. Defaults to ~/.dbt",
)
@click.option(
    "--target",
    type=click.STRING,
    help="Which profile to load. Overrides setting in dbt_project.yml.",
)
@click.option(
    "--database",
    type=click.STRING,
    help="The database to search for tables in",
)
@click.option(
    "--schema",
    type=click.STRING,
    help="the schema to search for tables in",
)
@click.option(
    "--table-prefix",
    type=click.STRING,
    help="The pattern used to look for tables",
)
def extract(
    target: Optional[str] = None,
    project_dir: Optional[str] = None,
    profiles_dir: Optional[str] = None,
    **kwargs,
):
    """Extract tables from database based on a pattern and load into source file

    \f
    This command will take a source, query the database for columns, and inject it into either existing yaml or create one if needed

    Args:
        target (Optional[str]): Profile target. Defaults to default target set in profile yml
        project_dir (Optional[str], optional): Dbt project directory. Defaults to current working directory.
        profiles_dir (Optional[str], optional): Dbt profile directory. Defaults to ~/.dbt
    """

    logger().warning(":stop: Not implemented yet")
    raise NotImplementedError("This command is not yet implemented")

    logger().info(":water_wave: Executing dbt-osmosis\n")

    # Initialize dbt & prepare database adapter
    project, profile, config, adapter = load_dbt(1, project_dir, profiles_dir, target)
    manifest = compile_project_load_manifest(config)

    # Conform project structure & bootstrap undocumented models injecting columns
    schema_map = build_schema_folder_map(
        project.project_name, project.project_root, manifest, model_type="sources"
    )

    while True:
        click.echo("Found following files from other formats that you may import:")
        choices = list(schema_map.items())[1:10]
        for i, (model, path) in enumerate(choices):
            click.echo(f"{i}. {model}")
        click.echo(f"9. next")
        click.echo(f"10. previous")
        choice = click.prompt(
            "Please select:",
            type=click.Choice([str(i) for i in range(len(choices) + 1)]),
            show_default=False,
        )
        print(choice)


@sources.command(context_settings=CONTEXT)
@click.option(
    "--project-dir",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    help="Which directory to look in for the dbt_project.yml file. Default is the current working directory and its parents.",
)
@click.option(
    "--profiles-dir",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    default=dbt.config.profile.DEFAULT_PROFILES_DIR,
    help="Which directory to look in for the profiles.yml file. Defaults to ~/.dbt",
)
@click.option(
    "--target",
    type=click.STRING,
    help="Which profile to load. Overrides setting in dbt_project.yml.",
)
@click.option(
    "-f",
    "--fqn",
    type=click.STRING,
    help="Specify models based on FQN. Use dots as separators. Looks like folder.folder.model or folder.folder.source.table. Use list command to see the scope of an FQN filter.",
)
def sync(
    target: Optional[str] = None,
    project_dir: Optional[str] = None,
    profiles_dir: Optional[str] = None,
    fqn: Optional[str] = None,
):
    """Synchronize existing schema file sources with database

    \f
    This command will take a source, query the database for columns, and inject it into the existing yaml removing unused columns if needed

    Args:
        target (Optional[str]): Profile target. Defaults to default target set in profile yml
        project_dir (Optional[str], optional): Dbt project directory. Defaults to current working directory.
        profiles_dir (Optional[str], optional): Dbt profile directory. Defaults to ~/.dbt
    """
    logger().info(":water_wave: Executing dbt-osmosis\n")

    # Initialize dbt & prepare database adapter
    project, profile, config, adapter = load_dbt(1, project_dir, profiles_dir, target)
    manifest = compile_project_load_manifest(config)

    # Conform project structure & bootstrap undocumented models injecting columns
    schema_map = build_schema_folder_map(
        project.project_name, project.project_root, manifest, fqn, "sources"
    )
    synchronize_sources(project.project_name, schema_map, manifest, adapter, fqn)


@cli.command(
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
    )
)
@click.pass_context
def workbench(ctx):
    """Instantiate the dbt-osmosis workbench.

    Pass the --options command to see streamlit specific options that can be passed to the app
    """
    if "--options" in ctx.args:
        subprocess.run(["streamlit", "run", "--help"])
        ctx.exit()
    subprocess.run(["streamlit", "run", str(Path(__file__).parent / "app.py")] + ctx.args)


if __name__ == "__main__":
    cli()
