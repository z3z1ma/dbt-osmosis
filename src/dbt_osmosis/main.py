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
"""

import subprocess
from pathlib import Path
from typing import Optional

import click

from dbt_osmosis.core.diff import diff_and_print_to_console
from dbt_osmosis.core.log_controller import logger
from dbt_osmosis.core.macros import inject_macros
from dbt_osmosis.core.osmosis import DEFAULT_PROFILES_DIR, DbtOsmosis

CONTEXT = {"max_content_width": 800}


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
    default=DEFAULT_PROFILES_DIR,
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
@click.option(
    "-d",
    "--dry-run",
    is_flag=True,
    help="If specified, no changes are committed to disk.",
)
def run(
    target: Optional[str] = None,
    project_dir: Optional[str] = None,
    profiles_dir: Optional[str] = None,
    fqn: Optional[str] = None,
    force_inheritance: bool = False,
    dry_run: bool = False,
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

    runner = DbtOsmosis(
        project_dir=project_dir,
        profiles_dir=profiles_dir,
        target=target,
        fqn=fqn,
        dry_run=dry_run,
    )

    # Conform project structure & bootstrap undocumented models injecting columns
    if runner.commit_project_restructure_to_disk():
        runner.rebuild_dbt_manifest()
    runner.propagate_documentation_downstream(force_inheritance=force_inheritance)


@cli.command(context_settings=CONTEXT)
@click.option(
    "--project-dir",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    help="Which directory to look in for the dbt_project.yml file. Default is the current working directory and its parents.",
)
@click.option(
    "--profiles-dir",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    default=DEFAULT_PROFILES_DIR,
    help="Which directory to look in for the profiles.yml file. Defaults to ~/.dbt",
)
@click.option(
    "-t",
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
    "-d",
    "--dry-run",
    is_flag=True,
    help="If specified, no changes are committed to disk.",
)
def compose(
    target: Optional[str] = None,
    project_dir: Optional[str] = None,
    profiles_dir: Optional[str] = None,
    fqn: Optional[str] = None,
    dry_run: bool = False,
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

    runner = DbtOsmosis(
        project_dir=project_dir,
        profiles_dir=profiles_dir,
        target=target,
        fqn=fqn,
        dry_run=dry_run,
    )

    # Conform project structure & bootstrap undocumented models injecting columns
    runner.commit_project_restructure_to_disk()


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
    default=DEFAULT_PROFILES_DIR,
    help="Which directory to look in for the profiles.yml file. Defaults to ~/.dbt",
)
@click.option(
    "-t",
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
@click.option(
    "-d",
    "--dry-run",
    is_flag=True,
    help="If specified, no changes are committed to disk.",
)
def document(
    target: Optional[str] = None,
    project_dir: Optional[str] = None,
    profiles_dir: Optional[str] = None,
    fqn: Optional[str] = None,
    force_inheritance: bool = False,
    dry_run: bool = False,
):
    """Column level documentation inheritance for existing models"""
    logger().info(":water_wave: Executing dbt-osmosis\n")

    runner = DbtOsmosis(
        project_dir=project_dir,
        profiles_dir=profiles_dir,
        target=target,
        fqn=fqn,
        dry_run=dry_run,
    )

    # Propagate documentation & inject/remove schema file columns to align with model in database
    runner.propagate_documentation_downstream(force_inheritance)


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
    default=DEFAULT_PROFILES_DIR,
    help="Which directory to look in for the profiles.yml file. Defaults to ~/.dbt",
)
@click.option(
    "--t",
    "--target",
    type=click.STRING,
    help="Which profile to load. Overrides setting in dbt_project.yml.",
)
@click.option(
    "--schema",
    type=click.STRING,
    help="the schema to search for tables in",
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


@cli.command(
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
    )
)
@click.option(
    "--project-dir",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    help="Which directory to look in for the dbt_project.yml file. Default is the current working directory and its parents.",
)
@click.option(
    "--profiles-dir",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    default=DEFAULT_PROFILES_DIR,
    help="Which directory to look in for the profiles.yml file. Defaults to ~/.dbt",
)
@click.option(
    "-m",
    "--model",
    type=click.STRING,
    required=True,
    help="The model to edit in the workbench",
)
@click.pass_context
def workbench(
    ctx, model: str, profiles_dir: Optional[str] = None, project_dir: Optional[str] = None
):
    """Instantiate the dbt-osmosis workbench and begin architecting a model.
        --model argument is required

    \f
    Pass the --options command to see streamlit specific options that can be passed to the app,

    """

    logger().info(":water_wave: Executing dbt-osmosis\n")

    if "--options" in ctx.args:
        subprocess.run(["streamlit", "run", "--help"])
        ctx.exit()

    script_args = ["--"]
    if project_dir:
        script_args.append("--project-dir")
        script_args.append(project_dir)
    if profiles_dir:
        script_args.append("--profiles-dir")
        script_args.append(profiles_dir)
    if model:
        script_args.append("--model")
        script_args.append(model)

    import os

    subprocess.run(
        ["streamlit", "run", Path(__file__).parent / "app.py"] + ctx.args + script_args,
        env=os.environ,
        cwd=Path.cwd(),
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
    default=DEFAULT_PROFILES_DIR,
    help="Which directory to look in for the profiles.yml file. Defaults to ~/.dbt",
)
@click.option(
    "-t",
    "--target",
    type=click.STRING,
    help="Which profile to load. Overrides setting in dbt_project.yml.",
)
@click.option(
    "-m",
    "--model",
    type=click.STRING,
    required=True,
    help="The model to edit in the workbench, must be a valid model as would be selected by `ref`",
)
@click.option(
    "--pk",
    type=click.STRING,
    required=True,
    help="The primary key of the model with which to base the diff",
)
@click.option(
    "--temp-table",
    is_flag=True,
    help="If specified, temp tables are used to stage the queries.",
)
@click.option(
    "--agg/--no-agg",
    default=True,
    help="Use --no-agg to show sample results, by default we agg for a summary view.",
)
@click.option(
    "-o",
    "--output",
    default="table",
    help="Output format can be one of table, chart/bar, or csv. CSV is saved to a file named dbt-osmosis-diff in working dir",
)
def diff(
    model: str,
    pk: str,
    target: Optional[str] = None,
    project_dir: Optional[str] = None,
    profiles_dir: Optional[str] = None,
    temp_table: bool = False,
    agg: bool = True,
    output: str = "table",
):
    """Diff a model based on git HEAD to working copy on disk"""

    logger().info(":water_wave: Executing dbt-osmosis\n")

    runner = DbtOsmosis(
        project_dir=project_dir,
        profiles_dir=profiles_dir,
        target=target,
    )
    inject_macros(runner)
    diff_and_print_to_console(model, pk, runner, temp_table, agg, output)


if __name__ == "__main__":
    cli()
