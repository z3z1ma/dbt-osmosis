# pyright: reportUnreachable=false
import functools
import io
import subprocess
import sys
import typing as t
from pathlib import Path

import click

import dbt_osmosis.core.logger as logger
from dbt_osmosis.core.osmosis import (
    DbtConfiguration,
    YamlRefactorContext,
    YamlRefactorSettings,
    apply_restructure_plan,
    commit_yamls,
    compile_sql_code,
    create_dbt_project_context,
    create_missing_source_yamls,
    discover_profiles_dir,
    discover_project_dir,
    draft_restructure_delta_plan,
    execute_sql_code,
    inherit_upstream_column_knowledge,
    inject_missing_columns,
    remove_columns_not_in_database,
    sort_columns_as_in_database,
    sync_node_to_yaml,
)

T = t.TypeVar("T")
if sys.version_info >= (3, 9):
    P = t.ParamSpec("P")
else:
    import typing_extensions as te

    P = te.ParamSpec("P")

_CONTEXT = {"max_content_width": 800}


@click.group()
@click.version_option()
def cli() -> None:
    """dbt-osmosis is a CLI tool for dbt that helps you manage, document, and organize your dbt yaml files"""
    pass


@cli.group()
def yaml():
    """Manage, document, and organize dbt YAML files"""


@cli.group()
def sql():
    """Execute and compile dbt SQL statements"""


def shared_opts(func: t.Callable[P, T]) -> t.Callable[P, T]:
    """Options common across subcommands"""

    @click.option(
        "--project-dir",
        type=click.Path(exists=True, dir_okay=True, file_okay=False),
        default=discover_project_dir,
        help=(
            "Which directory to look in for the dbt_project.yml file. Default is the current"
            " working directory and its parents."
        ),
    )
    @click.option(
        "--profiles-dir",
        type=click.Path(exists=True, dir_okay=True, file_okay=False),
        default=discover_profiles_dir,
        help="Which directory to look in for the profiles.yml file. Defaults to ~/.dbt",
    )
    @click.option(
        "-t",
        "--target",
        type=click.STRING,
        help="Which target to load. Overrides default target in the profiles.yml.",
    )
    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        return func(*args, **kwargs)

    return wrapper


def yaml_opts(func: t.Callable[P, T]) -> t.Callable[P, T]:
    """Options common to YAML operations."""

    @click.option(
        "-f",
        "--fqn",
        multiple=True,
        type=click.STRING,
        help="Specify models based on dbt's FQN. Mostly useful when combined with dbt ls.",
    )
    @click.option(
        "-d",
        "--dry-run",
        is_flag=True,
        help="If specified, no changes are committed to disk.",
    )
    @click.option(
        "-C",
        "--check",
        is_flag=True,
        help="If specified, will return a non-zero exit code if any files are changed or would have changed.",
    )
    @click.option(
        "--catalog-path",
        type=click.Path(exists=True),
        help="If specified, will read the list of columns from the catalog.json file instead of querying the warehouse.",
    )
    @click.option(
        "--profile",
        type=click.STRING,
        help="Which profile to load. Overrides setting in dbt_project.yml.",
    )
    @click.option(
        "--vars",
        type=click.STRING,
        help='Supply variables to the project. This argument overrides variables defined in your dbt_project.yml file. This argument should be a YAML string, eg. \'{"foo": "bar"}\'',
    )
    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        return func(*args, **kwargs)

    return wrapper


@yaml.command(context_settings=_CONTEXT)
@shared_opts
@yaml_opts
@click.option(
    "-F",
    "--force-inherit-descriptions",
    is_flag=True,
    help="If specified, forces descriptions to be inherited from an upstream source if possible.",
)
@click.option(
    "--skip-add-columns",
    is_flag=True,
    help="If specified, we will skip adding columns to the models. This is useful if you want to document your models without adding columns present in the database.",
)
@click.option(
    "--skip-add-tags",
    is_flag=True,
    help="If specified, we will skip adding upstream tags to the model columns.",
)
@click.option(
    "--skip-merge-meta",
    is_flag=True,
    help="If specified, we will skip merging upstrean meta keys to the model columns.",
)
@click.option(
    "--skip-add-data-types",  # TODO: make sure this is implemented
    is_flag=True,
    help="If specified, we will skip adding data types to the models.",
)
@click.option(
    "--numeric-precision",
    is_flag=True,
    help="If specified, numeric types will have precision and scale, e.g. Number(38, 8).",
)
@click.option(
    "--char-length",
    is_flag=True,
    help="If specified, character types will have length, e.g. Varchar(128).",
)
@click.option(
    "--add-progenitor-to-meta",
    is_flag=True,
    help="If specified, progenitor information will be added to the meta information of a column. This is useful if you want to know which model is the progenitor (origin) of a specific model's column.",
)
@click.option(
    "--use-unrendered-descriptions",
    is_flag=True,
    help="If specified, will use unrendered column descriptions in the documentation. This is the only way to propogate docs blocks",
)
@click.option(
    "--add-inheritance-for-specified-keys",
    multiple=True,
    type=click.STRING,
    help="If specified, will add inheritance for the specified keys. IE policy_tags",
)
@click.option(
    "--output-to-lower",  # TODO: validate this is implemented
    is_flag=True,
    help="If specified, output yaml file columns and data types in lowercase if possible.",
)
@click.option(
    "--auto-apply",
    is_flag=True,
    help="If specified, will automatically apply the restructure plan without confirmation.",
)
@click.argument("models", nargs=-1)
def refactor(
    target: str | None = None,
    project_dir: str | None = None,
    profiles_dir: str | None = None,
    catalog_path: str | None = None,
    fqn: list[str] | None = None,
    force_inherit_descriptions: bool = False,
    dry_run: bool = False,
    check: bool = False,
    skip_add_columns: bool = False,
    skip_add_tags: bool = False,
    skip_add_data_types: bool = False,
    numeric_precision: bool = False,
    char_length: bool = False,
    skip_merge_meta: bool = False,
    add_progenitor_to_meta: bool = False,
    models: list[str] | None = None,
    profile: str | None = None,
    vars: str | None = None,
    use_unrendered_descriptions: bool = False,
    add_inheritance_for_specified_keys: list[str] | None = None,
    output_to_lower: bool = False,
    auto_apply: bool = False,
) -> None:
    """Executes organize which syncs yaml files with database schema and organizes the dbt models
    directory, reparses the project, then executes document passing down inheritable documentation

    \f
    This command will conform your project as outlined in `dbt_project.yml`, bootstrap undocumented
    dbt models, and propagate column level documentation downwards once all yamls are accounted for
    """
    logger.info(":water_wave: Executing dbt-osmosis\n")
    settings = DbtConfiguration(
        project_dir=t.cast(str, project_dir),
        profiles_dir=t.cast(str, profiles_dir),
        target=target,
        profile=profile,
    )
    context = YamlRefactorContext(
        project=create_dbt_project_context(settings),
        settings=YamlRefactorSettings(
            fqns=fqn or [],
            models=models or [],
            dry_run=dry_run,
            skip_add_columns=skip_add_columns,
            skip_add_tags=skip_add_tags,
            skip_merge_meta=skip_merge_meta,
            skip_add_data_types=skip_add_data_types,
            numeric_precision=numeric_precision,
            char_length=char_length,
            add_progenitor_to_meta=add_progenitor_to_meta,
            use_unrendered_descriptions=use_unrendered_descriptions,
            add_inheritance_for_specified_keys=add_inheritance_for_specified_keys or [],
            output_to_lower=output_to_lower,
            force_inherit_descriptions=force_inherit_descriptions,
            catalog_path=catalog_path,
            create_catalog_if_not_exists=False,  # TODO: allow enabling if ready
        ),
    )
    if vars:
        settings.vars = context.yaml_handler.load(io.StringIO(vars))  # pyright: ignore[reportUnknownMemberType]

    create_missing_source_yamls(context=context)
    apply_restructure_plan(
        context=context, plan=draft_restructure_delta_plan(context), confirm=not auto_apply
    )
    inject_missing_columns(context=context)
    remove_columns_not_in_database(context=context)
    inherit_upstream_column_knowledge(context=context)
    sort_columns_as_in_database(context=context)
    sync_node_to_yaml(context=context)
    commit_yamls(context=context)

    if check and context.mutated:
        exit(1)


@yaml.command(context_settings=_CONTEXT)
@shared_opts
@yaml_opts
@click.argument("models", nargs=-1)
@click.option(
    "--auto-apply",
    is_flag=True,
    help="If specified, will automatically apply the restructure plan without confirmation.",
)
def organize(
    target: str | None = None,
    project_dir: str | None = None,
    profiles_dir: str | None = None,
    catalog_path: str | None = None,
    fqn: list[str] | None = None,
    dry_run: bool = False,
    check: bool = False,
    models: list[str] | None = None,
    profile: str | None = None,
    vars: str | None = None,
    auto_apply: bool = False,
) -> None:
    """Organizes schema ymls based on config and injects undocumented models

    \f
    This command will conform schema ymls in your project as outlined in `dbt_project.yml` &
    bootstrap undocumented dbt models
    """
    logger.info(":water_wave: Executing dbt-osmosis\n")
    settings = DbtConfiguration(
        project_dir=t.cast(str, project_dir),
        profiles_dir=t.cast(str, profiles_dir),
        target=target,
        profile=profile,
    )
    context = YamlRefactorContext(
        project=create_dbt_project_context(settings),
        settings=YamlRefactorSettings(
            fqns=fqn or [],
            models=models or [],
            dry_run=dry_run,
            catalog_path=catalog_path,
            create_catalog_if_not_exists=False,
        ),
    )
    if vars:
        settings.vars = context.yaml_handler.load(io.StringIO(vars))  # pyright: ignore[reportUnknownMemberType]

    create_missing_source_yamls(context=context)
    apply_restructure_plan(
        context=context, plan=draft_restructure_delta_plan(context), confirm=not auto_apply
    )

    if check and context.mutated:
        exit(1)


@yaml.command(context_settings=_CONTEXT)
@shared_opts
@yaml_opts
@click.option(
    "-F",
    "--force-inherit-descriptions",
    is_flag=True,
    help="If specified, forces descriptions to be inherited from an upstream source if possible.",
)
@click.option(
    "--skip-add-tags",
    is_flag=True,
    help="If specified, we will skip adding upstream tags to the model columns.",
)
@click.option(
    "--skip-merge-meta",
    is_flag=True,
    help="If specified, we will skip merging upstrean meta keys to the model columns.",
)
@click.option(
    "--skip-add-data-types",  # TODO: make sure this is implemented
    is_flag=True,
    help="If specified, we will skip adding data types to the models.",
)
@click.option(
    "--skip-add-columns",
    is_flag=True,
    help="If specified, we will skip adding columns to the models. This is useful if you want to document your models without adding columns present in the database.",
)
@click.option(
    "--numeric-precision",
    is_flag=True,
    help="If specified, numeric types will have precision and scale, e.g. Number(38, 8).",
)
@click.option(
    "--char-length",
    is_flag=True,
    help="If specified, character types will have length, e.g. Varchar(128).",
)
@click.option(
    "--add-progenitor-to-meta",
    is_flag=True,
    help="If specified, progenitor information will be added to the meta information of a column. This is useful if you want to know which model is the progenitor (origin) of a specific model's column.",
)
@click.option(
    "--use-unrendered-descriptions",
    is_flag=True,
    help="If specified, will use unrendered column descriptions in the documentation. This is the only way to propogate docs blocks",
)
@click.option(
    "--add-inheritance-for-specified-keys",
    multiple=True,
    type=click.STRING,
    help="If specified, will add inheritance for the specified keys. IE policy_tags",
)
@click.option(
    "--output-to-lower",  # TODO: validate this is implemented
    is_flag=True,
    help="If specified, output yaml file columns and data types in lowercase if possible.",
)
@click.argument("models", nargs=-1)
def document(
    target: str | None = None,
    project_dir: str | None = None,
    profiles_dir: str | None = None,
    models: list[str] | None = None,
    fqn: list[str] | None = None,
    dry_run: bool = False,
    check: bool = False,
    skip_merge_meta: bool = False,
    skip_add_tags: bool = False,
    skip_add_data_types: bool = False,
    skip_add_columns: bool = False,
    add_progenitor_to_meta: bool = False,
    add_inheritance_for_specified_keys: list[str] | None = None,
    use_unrendered_descriptions: bool = False,
    force_inherit_descriptions: bool = False,
    output_to_lower: bool = False,
    char_length: bool = False,
    numeric_precision: bool = False,
    catalog_path: str | None = None,
    vars: str | None = None,
    profile: str | None = None,
) -> None:
    """Column level documentation inheritance for existing models

    \f
    This command will conform schema ymls in your project as outlined in `dbt_project.yml` &
    bootstrap undocumented dbt models

    Args:
        target (Optional[str]): Profile target. Defaults to default target set in profile yml
        project_dir (Optional[str], optional): Dbt project directory. Defaults to working directory.
        profiles_dir (Optional[str], optional): Dbt profile directory. Defaults to ~/.dbt
    """
    logger.info(":water_wave: Executing dbt-osmosis\n")
    settings = DbtConfiguration(
        project_dir=t.cast(str, project_dir),
        profiles_dir=t.cast(str, profiles_dir),
        target=target,
        profile=profile,
    )
    context = YamlRefactorContext(
        project=create_dbt_project_context(settings),
        settings=YamlRefactorSettings(
            fqns=fqn or [],
            models=models or [],
            dry_run=dry_run,
            skip_add_tags=skip_add_tags,
            skip_merge_meta=skip_merge_meta,
            skip_add_data_types=skip_add_data_types,
            skip_add_columns=skip_add_columns,
            numeric_precision=numeric_precision,
            char_length=char_length,
            add_progenitor_to_meta=add_progenitor_to_meta,
            use_unrendered_descriptions=use_unrendered_descriptions,
            add_inheritance_for_specified_keys=add_inheritance_for_specified_keys or [],
            output_to_lower=output_to_lower,
            force_inherit_descriptions=force_inherit_descriptions,
            catalog_path=catalog_path,
        ),
    )
    if vars:
        settings.vars = context.yaml_handler.load(io.StringIO(vars))  # pyright: ignore[reportUnknownMemberType]

    inject_missing_columns(context=context)
    inherit_upstream_column_knowledge(context=context)
    sort_columns_as_in_database(context=context)
    sync_node_to_yaml(context=context)
    commit_yamls(context=context)

    if check and context.mutated:
        exit(1)


@cli.command(
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
    )
)
@click.option(
    "--project-dir",
    default=discover_project_dir,
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    help="Which directory to look in for the dbt_project.yml file. Default is the current working directory and its parents.",
)
@click.option(
    "--profiles-dir",
    default=discover_profiles_dir,
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    help="Which directory to look in for the profiles.yml file. Defaults to ~/.dbt",
)
@click.option(
    "--host",
    type=click.STRING,
    help="The host to serve the server on",
    default="localhost",
)
@click.option(
    "--port",
    type=click.INT,
    help="The port to serve the server on",
    default=8501,
)
@click.pass_context
def workbench(
    ctx: click.Context,
    profiles_dir: str | None = None,
    project_dir: str | None = None,
    host: str = "localhost",
    port: int = 8501,
) -> None:
    """Start the dbt-osmosis workbench

    \f
    Pass the --options command to see streamlit specific options that can be passed to the app,
    pass --config to see the output of streamlit config show
    """
    logger.info(":water_wave: Executing dbt-osmosis\n")

    if "--options" in ctx.args:
        proc = subprocess.run(["streamlit", "run", "--help"])
        ctx.exit(proc.returncode)

    import os

    if "--config" in ctx.args:
        proc = subprocess.run(
            ["streamlit", "config", "show"],
            env=os.environ,
            cwd=Path.cwd(),
        )
        ctx.exit(proc.returncode)

    script_args = ["--"]
    if project_dir:
        script_args.append("--project-dir")
        script_args.append(project_dir)
    if profiles_dir:
        script_args.append("--profiles-dir")
        script_args.append(profiles_dir)

    proc = subprocess.run(
        [
            "streamlit",
            "run",
            "--runner.magicEnabled=false",
            f"--browser.serverAddress={host}",
            f"--browser.serverPort={port}",
            Path(__file__).parent.parent / "workbench" / "app.py",
            *ctx.args,
            *script_args,
        ],
        env=os.environ,
        cwd=Path.cwd(),
    )

    ctx.exit(proc.returncode)


@sql.command(context_settings=_CONTEXT)
@shared_opts
@click.argument("sql")
def run(
    sql: str = "",
    project_dir: str | None = None,
    profiles_dir: str | None = None,
    target: str | None = None,
) -> None:
    """Executes a dbt SQL statement writing results to stdout"""
    settings = DbtConfiguration(
        project_dir=t.cast(str, project_dir), profiles_dir=t.cast(str, profiles_dir), target=target
    )
    project = create_dbt_project_context(settings)
    _, table = execute_sql_code(project, sql)

    getattr(table, "print_table")(
        max_rows=50,
        max_columns=6,
        output=sys.stdout,
        max_column_width=20,
        locale=None,
        max_precision=3,
    )


@sql.command(context_settings=_CONTEXT)
@shared_opts
@click.argument("sql")
def compile(
    sql: str = "",
    project_dir: str | None = None,
    profiles_dir: str | None = None,
    target: str | None = None,
) -> None:
    """Executes a dbt SQL statement writing results to stdout"""
    settings = DbtConfiguration(
        project_dir=t.cast(str, project_dir), profiles_dir=t.cast(str, profiles_dir), target=target
    )
    project = create_dbt_project_context(settings)
    node = compile_sql_code(project, sql)

    print(node.compiled_code)


if __name__ == "__main__":
    cli()
