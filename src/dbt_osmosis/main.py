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
from typing import Optional, Dict, Any, List, Sequence, Iterable, Mapping, MutableMapping, Tuple
from pathlib import Path

import click
from ruamel.yaml import YAML
import dbt.config.profile
import dbt.config.project
import dbt.config.runtime
import dbt.config.renderer
import dbt.context.base
import dbt.parser.manifest
import dbt.exceptions
from dbt.adapters.factory import get_adapter, register_adapter, Adapter
from dbt.tracking import disable_tracking

from exceptions.osmosis import (
    InvalidOsmosisConfig,
    MissingOsmosisConfig,
    SanitizationRequired,
)


disable_tracking()


AUDIT_REPORT = """
Audit Report
----------------------------------------------------

Database: {database}
Schema: {schema}
Table: {table}

Total Columns in Database: {total_columns}
Total Documentation Coverage: {coverage}%

Action Log:
Columns Added to dbt: {n_cols_added}
Column Knowledge Inherited: {n_cols_doc_inherited}
Extra Columns Removed: {n_cols_removed}
"""

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


class PseudoArgs:
    def __init__(
        self,
        threads: Optional[int] = 1,
        target: Optional[str] = None,
        profiles_dir: Optional[str] = dbt.config.profile.DEFAULT_PROFILES_DIR,
        project_dir: Optional[str] = None,
        vars: Optional[str] = "{}",
    ):
        self.threads = threads
        if target:
            self.target = target  # We don't want target in args context if it is None
        self.profiles_dir = profiles_dir
        self.project_dir = project_dir
        self.vars = vars  # json.dumps str


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
        node.pop(column, None)
        found_ix = None
        for ix, model_column in enumerate(model["columns"]):
            if model_column["name"] == column:
                found_ix = ix
                break
        else:
            continue
        model["columns"].pop(found_ix)
        n_actions += 1
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
                node["columns"][column].pop("progenitor", None)
                for model_column in model["columns"]:
                    if model_column["name"] == column:
                        model_column["description"] = prior_knowledge["description"]
                n_actions += 1
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
            "description": None,
        }
        model.setdefault("columns", []).append({"name": column, "description": None})
        n_actions += 1
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
        extra_columns (Iterable): [description]
        node (Dict[str, Any]): [description]
        schema_file (Dict[str, Any]): [description]

    Returns:
        Tuple[int, int, int]: [description]
    """
    for model in schema_file["models"]:
        if model["name"] == node["name"]:
            n_cols_added = add_missing_cols_to_node_and_model(missing_columns, node, model)
            n_cols_doc_inherited = update_undocumented_columns_with_prior_knowledge(
                undocumented_columns, knowledge, node, model
            )
            n_cols_removed = remove_columns_not_in_database(extra_columns, node, model)
            return n_cols_added, n_cols_doc_inherited, n_cols_removed
    else:
        return 0, 0, 0


def commit_project_restructure(restructured_project: MutableMapping[Path, Any]) -> None:
    """Given a project restrucure plan of pathlib Paths to a mapping of output and supersedes, commit changes to filesystem
    to conform project to defined structure as code

    Args:
        restructured_project (MutableMapping[Path, Any]): Project restructure plan as typically created by `build_project_structure_update_plan`
    """
    yaml = YAML()
    for target, structure in restructured_project.items():

        if not target.exists():
            # Build File
            target.parent.mkdir(exist_ok=True, parents=True)
            target.touch()
            with open(target, "w", encoding="utf-8") as f:
                yaml.dump(structure["output"], f)

        else:
            # Update File
            target_schema = yaml.load(target)
            if "version" not in target_schema:
                target_schema["version"] = 2
            target_schema.setdefault("models", []).extend(structure["output"]["models"])
            with open(target, "w", encoding="utf-8") as f:
                yaml.dump(target_schema, f)

        # Clean superseded schema files
        for dir in structure["supersede"]:
            dir.unlink(missing_ok=True)


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
    except dbt.exceptions.CompilationException:
        click.echo(
            f"Could not resolve relation {database}.{schema}.{identifier} against database active tables"
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
    """Build project structure update plan based on `dbt-osmosis:` configs set across dbt_project.yml and model files

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
    with adapter.connection_named("dbt-osmosis"):
        for unique_id, schema_file in schema_map.items():
            if not schema_file.is_valid:
                proj_restructure_plan.setdefault(
                    schema_file.target, {"output": {"version": 2, "models": []}, "supersede": []}
                )
                node = manifest["nodes"][unique_id]
                if schema_file.current is None:
                    # Bootstrapping Undocumented Model
                    proj_restructure_plan[schema_file.target]["output"]["models"].append(
                        create_base_model(node["database"], node["schema"], node["name"], adapter)
                    )
                else:
                    # Model Is Documented but Must be Migrated
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
                            # Only add to supersede list once for brevity
                            if (
                                schema_file.current
                                not in proj_restructure_plan[schema_file.target]["supersede"]
                            ):
                                proj_restructure_plan[schema_file.target]["supersede"].append(
                                    schema_file.current
                                )
                            break
                    else:
                        ...  # Model not found at patch path -- we should pass on this for now
            else:
                ...  # Valid schema file found for model -- we will update the columns in the osmosis task

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
    """Resolve schema path for a manifest node

    Args:
        root_path (str): Root dbt project path
        node (MutableMapping): A manifest json node representing a model

    Returns:
        Optional[Path]: Path object representing resolved path to file where schema exists otherwise return None
    """
    schema_path: str = node.get("patch_path")
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


def validate_osmosis_config(osmosis_config: Optional[str]) -> None:
    """Validates a config string

    Args:
        osmosis_config (Optional[str]): Osmosis config string gathered from a node's config

    Raises:
        MissingOsmosisConfig: Thrown if no config is present for a node
        InvalidOsmosisConfig: Thrown if an invalid config is present on a node under our key `dbt-osmosis`

    Returns:
        str: Validated config str
    """
    if not osmosis_config:
        raise MissingOsmosisConfig(
            "Config not set for model, we recommend setting the config at a directory level through the `dbt_project.yml`"
        )
    if osmosis_config not in VALID_CONFIGS:
        raise InvalidOsmosisConfig("Invalid dbt-osmosis config for model")

    return osmosis_config


def is_valid_model(node: MutableMapping) -> bool:
    """Validates a node as being a targetable model

    Args:
        node (MutableMapping): A manifest json node

    Returns:
        bool: returns true if the node is a valid targetable model
    """
    return node["resource_type"] == "model" and node["config"]["materialized"] != "ephemeral"


def build_schema_folder_map(root: str, manifest: MutableMapping) -> Dict[str, SchemaFile]:
    """Builds a mapping of models to their existing and target schema file paths

    Args:
        root (str): Root dbt project path
        manifest (MutableMapping): dbt manifest json artifact

    Returns:
        Dict[str, SchemaFile]: Mapping of models to their existing and target schema file paths
    """
    schema_map = {}
    for unique_id, node in manifest["nodes"].items():
        if is_valid_model(node):
            schema_path = resolve_schema_path(root, node)
            osmosis_config = validate_osmosis_config(node["config"].get("dbt-osmosis"))
            osmosis_schema_path = resolve_osmosis_target_schema_path(osmosis_config, node)
            schema_map[unique_id] = SchemaFile(target=osmosis_schema_path, current=schema_path)

    return schema_map


def compile_project_load_manifest(
    cfg: Optional[dbt.config.runtime.RuntimeConfig] = None,
) -> MutableMapping:
    """Compiles dbt project and builds manifest json artifact in memory

    Args:
        cfg (Optional[RuntimeConfig]): dbt runtime config

    Returns:
        MutableMapping: Loaded manifest.json artifact
    """
    return dbt.parser.manifest.ManifestLoader.get_full_manifest(cfg).flat_graph


def verify_connection(adapter: Adapter) -> Adapter:
    try:
        with adapter.connection_named("debug"):
            adapter.debug_query()
    except Exception as exc:
        raise Exception("Could not connect to Database") from exc
    else:
        return adapter


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "--project-dir",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    help="Path to the dbt project directory, defaults to current working directory",
)
@click.option(
    "--profiles-dir",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    default=dbt.config.profile.DEFAULT_PROFILES_DIR,
    help="Path to the dbt profiles.yml, defaults to ~/.dbt",
)
@click.option("--target")
def run(
    target: Optional[str] = None,
    project_dir: Optional[str] = None,
    profiles_dir: Optional[str] = None,
):
    """Interactively build documentation

    Args:
        manifest (str): Path to manifest.json artifact
        target (Optional[str]): Profile target. Defaults to default target set in profile yml
        project_dir (Optional[str], optional): Dbt project directory. Defaults to current working directory.
    """
    click.echo("Executing Model Osmosis")

    # Collect/build our args
    args = PseudoArgs(
        threads=1,
        project_dir=project_dir,
        target=target,
        profiles_dir=profiles_dir,
    )

    # Initialize dbt & prepare database adapter
    project, profile = dbt.config.runtime.RuntimeConfig.collect_parts(args)
    config = dbt.config.runtime.RuntimeConfig.from_parts(project, profile, args)
    register_adapter(config)

    adapter = verify_connection(get_adapter(config))
    pre_manifest = compile_project_load_manifest(config)
    commit_project_restructure(
        build_project_structure_update_plan(
            build_schema_folder_map(project.project_root, pre_manifest), pre_manifest, adapter
        )
    )
    manifest = compile_project_load_manifest(config)
    schema_map = build_schema_folder_map(project.project_root, manifest)

    # Below is focused on doumentation propagation
    yaml = YAML()
    with adapter.connection_named("dbt-osmosis"):
        for unique_id, node in manifest["nodes"].items():
            if is_valid_model(node):
                table = node["database"], node["schema"], node["name"]
                knowledge = pass_down_knowledge(build_ancestor_tree(node, manifest), manifest)
                schema_path = schema_map.get(unique_id)

                if schema_path is None:
                    ...  # We can't take action

                # Build Sets
                database_columns = set(get_columns(*table, adapter))
                dbt_columns = set(column_name for column_name in node["columns"])
                documented_columns = set(get_documented_columns(dbt_columns, node))

                if not database_columns:
                    # Note to user we are using dbt columns as source since we did not resolve them from db
                    # Doing this means only `undocumented_columns` can be resolved to a non-empty set
                    click.echo("Falling back to using manifest columns as base column set")
                    database_columns = dbt_columns

                # Action
                missing_columns = database_columns - dbt_columns
                """Columns in database not in dbt -- will be injected into schema file"""
                undocumented_columns = database_columns - documented_columns
                """Columns missing documentation -- descriptions will be inherited and injected into schema file where prior knowledge exists"""
                extra_columns = dbt_columns - database_columns
                """Columns in schema file not in database -- will be removed from schema file"""

                if (
                    len(missing_columns) > 0 or len(undocumented_columns) or len(extra_columns) > 0
                ) and len(database_columns) > 0:
                    schema_file = yaml.load(schema_path.current)
                    (
                        n_cols_added,
                        n_cols_doc_inherited,
                        n_cols_removed,
                    ) = update_schema_file_and_node(
                        knowledge,
                        missing_columns,
                        undocumented_columns,
                        extra_columns,
                        node,
                        schema_file,
                    )
                    if n_cols_added + n_cols_doc_inherited + n_cols_removed > 0:
                        with open(schema_path.current, "w", encoding="utf-8") as f:
                            yaml.dump(schema_file, f)

                # Print Audit Report
                n_cols = float(len(database_columns))
                n_cols_documented = float(len(documented_columns)) + n_cols_doc_inherited
                perc_coverage = (
                    min(100.0 * round(n_cols_documented / n_cols, 3), 100.0)
                    if n_cols > 0
                    else "Unable to Determine"
                )
                click.echo(
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


def list_extra_details(
    nonexistent_columns: Optional[Sequence[str]] = None,
    mutated_columns: Optional[Sequence[str]] = None,
    progenitors: Optional[Sequence[str]] = None,
) -> None:
    """Write some additional details to the console

    Args:
        nonexistent_columns (Optional[Sequence[str]], optional): List of columns which are found in dbt yaml files not in database. Defaults to None.
        mutated_columns (Optional[Sequence[str]], optional): List of columns which have mutated from their progenitor's definition. Defaults to None.
        progenitors (Optional[Sequence[str]], optional): List of columns with identified progenitors. Defaults to None.
    """
    if nonexistent_columns is None:
        nonexistent_columns = []
    if mutated_columns is None:
        mutated_columns = []
    if progenitors is None:
        progenitors = []

    for column in nonexistent_columns:
        click.echo(f"Column {nonexistent_columns} resolved in manifest not present in database")
    for column in mutated_columns:
        click.echo(f"Column {column['name']} has mutated: {column['prior']} -> {column['current']}")
    for column in progenitors:
        click.echo(f"Column {column['name']} progenitor: {column['progenitor']}")


@cli.command()
def audit():
    click.echo("Executing Audit")


if __name__ == "__main__":
    cli()
