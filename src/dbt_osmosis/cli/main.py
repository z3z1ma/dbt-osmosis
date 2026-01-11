# pyright: reportUnreachable=false, reportAny=false

from __future__ import annotations

import functools
import subprocess
import sys
import typing as t
from pathlib import Path

import click
import yaml as yaml_handler

import dbt_osmosis.core.logger as logger
from dbt_osmosis.core.osmosis import (
    DbtConfiguration,
    YamlRefactorContext,
    YamlRefactorSettings,
    apply_restructure_plan,
    compile_sql_code,
    create_dbt_project_context,
    create_missing_source_yamls,
    discover_profiles_dir,
    discover_project_dir,
    draft_restructure_delta_plan,
    execute_sql_code,
    generate_dbt_model_from_nl,
    generate_sql_from_nl,
    inherit_upstream_column_knowledge,
    inject_missing_columns,
    remove_columns_not_in_database,
    sort_columns_as_configured,
    synchronize_data_types,
    synthesize_missing_documentation_with_openai,
)

T = t.TypeVar("T")
if sys.version_info >= (3, 10):
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


def test_llm_connection(llm_client=None) -> None:
    """Test the connection to the LLM client."""
    import os

    from dbt_osmosis.core.llm import get_llm_client

    llm_client = os.getenv("LLM_PROVIDER")
    if not llm_client:
        click.echo(
            "ERROR: LLM_PROVIDER environment variable is not set. Please set it to one of the available providers."
        )
        return

    client, model_engine = get_llm_client()
    if not client or not model_engine:
        click.echo(
            f"Connection ERROR: The environment variables for LLM provider {llm_client} are not set correctly."
        )
        return

    click.echo(
        f"LLM client connection successful. Provider: {llm_client}, Model Engine: {model_engine}"
    )


@cli.command()
def test_llm() -> None:
    """Test the connection to the LLM client"""
    logger.info("INFO: Invoking test_llm_connection...")
    from dbt_osmosis.core.llm import get_llm_client

    llm_client = get_llm_client()
    test_llm_connection(llm_client)
    click.echo("LLM client connection test completed.")


@cli.group()
def yaml():
    """Manage, document, and organize dbt YAML files"""


def logging_opts(func: t.Callable[P, T]) -> t.Callable[P, T]:
    """Options common across subcommands"""

    @click.option(
        "--log-level",
        type=click.STRING,
        default="INFO",
        help="The log level to use. Default is INFO.",
    )
    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        # NOTE: Remove log_level from kwargs so it's not passed to the function.
        log_level = kwargs.pop("log_level")
        logger.set_log_level(str(log_level).upper())
        return func(*args, **kwargs)

    return wrapper


@cli.group()
def sql():
    """Execute and compile dbt SQL statements"""


def dbt_opts(func: t.Callable[P, T]) -> t.Callable[P, T]:
    """Options common across subcommands"""

    @click.option(
        "--project-dir",
        type=click.Path(exists=True, dir_okay=True, file_okay=False),
        default=discover_project_dir,
        help="Which directory to look in for the dbt_project.yml file. Default is the current working directory and its parents.",
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
    @click.option(
        "--threads",
        type=click.INT,
        envvar="DBT_THREADS",
        help="How many threads to use when executing.",
    )
    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        return func(*args, **kwargs)

    return wrapper


def yaml_opts(func: t.Callable[P, T]) -> t.Callable[P, T]:
    """Options common to YAML operations."""

    @click.argument("models", nargs=-1)
    @click.option(
        "-f",
        "--fqn",
        multiple=True,
        type=click.STRING,
        help="Specify models based on dbt's FQN. Mostly useful when combined with dbt ls and command interpolation.",
    )
    @click.option(
        "-d",
        "--dry-run",
        is_flag=True,
        help="No changes are committed to disk. Works well with --check as check will still exit with a code.",
    )
    @click.option(
        "-C",
        "--check",
        is_flag=True,
        help="Return a non-zero exit code if any files are changed or would have changed.",
    )
    @click.option(
        "--catalog-path",
        type=click.Path(exists=True),
        help="Read the list of columns from the catalog.json file instead of querying the warehouse.",
    )
    @click.option(
        "--profile",
        type=click.STRING,
        help="Which profile to load. Overrides setting in dbt_project.yml.",
    )
    @click.option(
        "--vars",
        type=click.STRING,
        help='Supply variables to the project. Override variables defined in your dbt_project.yml file. This argument should be a YAML string, eg. \'{"foo": "bar"}\'',
    )
    @click.option(
        "--disable-introspection",
        is_flag=True,
        help="Allows running of program without a database connection, it is recommended to use the --catalog-path option if using this.",
    )
    @click.option(
        "--scaffold-empty-configs/--no-scaffold-empty-configs",
        default=False,
        help="When disabled, avoid writing empty/placeholder fields (e.g., empty descriptions) to YAML.",
    )
    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        if kwargs.get("disable_introspection") and not kwargs.get("catalog_path"):
            logger.warning(
                ":construction: You have disabled introspection without providing a catalog path. This will result in some features not working as expected."
            )
        return func(*args, **kwargs)

    return wrapper


@yaml.command(context_settings=_CONTEXT)
@dbt_opts
@yaml_opts
@logging_opts
@click.option(
    "-F",
    "--force-inherit-descriptions",
    is_flag=True,
    help="Force descriptions to be inherited from an upstream source if possible.",
)
@click.option(
    "--use-unrendered-descriptions",
    is_flag=True,
    help="Use unrendered column descriptions in the documentation. This is the only way to propogate docs blocks",
)
@click.option(
    "--prefer-yaml-values",
    is_flag=True,
    help="Prefer YAML values as-is for ALL fields, preserving unrendered jinja templates like {{ var(...) }}, {{ env_var(...) }}, etc. Takes precedence over use-unrendered-descriptions.",
)
@click.option(
    "--skip-add-columns",
    is_flag=True,
    help="Skip adding missing columns to any yaml. Useful if you want to document your models without adding large volume of columns present in the database.",
)
@click.option(
    "--skip-add-source-columns",
    is_flag=True,
    help="Skip adding missing columns to source yamls. Useful if you want to document your models without adding large volume of columns present in the database.",
)
@click.option(
    "--skip-add-tags",
    is_flag=True,
    help="Skip adding upstream tags to the model columns.",
)
@click.option(
    "--skip-merge-meta",
    is_flag=True,
    help="Skip merging upstrean meta keys to the model columns.",
)
@click.option(
    "--skip-add-data-types",
    is_flag=True,
    help="Skip adding data types to the models.",
)
@click.option(
    "--add-progenitor-to-meta",
    is_flag=True,
    help="Progenitor information will be added to the meta information of a column. Useful to understand which model is the progenitor (origin) of a specific model's column.",
)
@click.option(
    "--add-inheritance-for-specified-keys",
    multiple=True,
    type=click.STRING,
    help="Add inheritance for the specified keys. IE policy_tags",
)
@click.option(
    "--numeric-precision-and-scale",
    is_flag=True,
    help="Numeric types will have precision and scale, e.g. Number(38, 8).",
)
@click.option(
    "--string-length",
    is_flag=True,
    help="Character types will have length, e.g. Varchar(128).",
)
@click.option(
    "--output-to-lower",
    is_flag=True,
    help="Output yaml file columns and data types in lowercase if possible.",
)
@click.option(
    "--output-to-upper",
    is_flag=True,
    help="Output yaml file columns and data types in uppercase if possible.",
)
@click.option(
    "--auto-apply",
    is_flag=True,
    help="Automatically apply the restructure plan without confirmation.",
)
@click.option(
    "--synthesize",
    is_flag=True,
    help="Automatically synthesize missing documentation with OpenAI.",
)
@click.option(
    "--include-external",
    is_flag=True,
    help="Include models and sources from external dbt packages in the processing.",
)
def refactor(
    target: str | None = None,
    profile: str | None = None,
    project_dir: str | None = None,
    profiles_dir: str | None = None,
    vars: str | None = None,
    auto_apply: bool = False,
    check: bool = False,
    threads: int | None = None,
    disable_introspection: bool = False,
    synthesize: bool = False,
    **kwargs: t.Any,
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
        threads=threads,
        vars=yaml_handler.safe_load(vars) if vars else None,
        disable_introspection=disable_introspection,
    )

    with YamlRefactorContext(
        project=create_dbt_project_context(settings),
        settings=YamlRefactorSettings(
            **{k: v for k, v in kwargs.items() if v is not None}, create_catalog_if_not_exists=False
        ),
    ) as context:
        create_missing_source_yamls(context=context)
        apply_restructure_plan(
            context=context, plan=draft_restructure_delta_plan(context), confirm=not auto_apply
        )

        transform = (
            inject_missing_columns
            >> remove_columns_not_in_database
            >> inherit_upstream_column_knowledge
            >> sort_columns_as_configured
            >> synchronize_data_types
        )
        if synthesize:
            transform >>= synthesize_missing_documentation_with_openai

        _ = transform(context=context)

        if check and context.mutated:
            exit(1)


@yaml.command(context_settings=_CONTEXT)
@dbt_opts
@yaml_opts
@logging_opts
@click.option(
    "--auto-apply",
    is_flag=True,
    help="If specified, will automatically apply the restructure plan without confirmation.",
)
def organize(
    target: str | None = None,
    project_dir: str | None = None,
    profiles_dir: str | None = None,
    check: bool = False,
    profile: str | None = None,
    vars: str | None = None,
    auto_apply: bool = False,
    threads: int | None = None,
    disable_introspection: bool = False,
    **kwargs: t.Any,
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
        threads=threads,
        vars=yaml_handler.safe_load(vars) if vars else None,
        disable_introspection=disable_introspection,
    )

    with YamlRefactorContext(
        project=create_dbt_project_context(settings),
        settings=YamlRefactorSettings(
            **{k: v for k, v in kwargs.items() if v is not None}, create_catalog_if_not_exists=False
        ),
    ) as context:
        create_missing_source_yamls(context=context)
        apply_restructure_plan(
            context=context, plan=draft_restructure_delta_plan(context), confirm=not auto_apply
        )

        if check and context.mutated:
            exit(1)


@yaml.command(context_settings=_CONTEXT)
@dbt_opts
@yaml_opts
@logging_opts
@click.option(
    "-F",
    "--force-inherit-descriptions",
    is_flag=True,
    help="Force descriptions to be inherited from an upstream source if possible.",
)
@click.option(
    "--use-unrendered-descriptions",
    is_flag=True,
    help="Use unrendered column descriptions in the documentation. This is the only way to propogate docs blocks",
)
@click.option(
    "--prefer-yaml-values",
    is_flag=True,
    help="Prefer YAML values as-is for ALL fields, preserving unrendered jinja templates like {{ var(...) }}, {{ env_var(...) }}, etc. Takes precedence over use-unrendered-descriptions.",
)
@click.option(
    "--skip-add-columns",
    is_flag=True,
    help="Skip adding missing columns to any yaml. Useful if you want to document your models without adding large volume of columns present in the database.",
)
@click.option(
    "--skip-add-source-columns",
    is_flag=True,
    help="Skip adding missing columns to source yamls. Useful if you want to document your models without adding large volume of columns present in the database.",
)
@click.option(
    "--skip-add-tags",
    is_flag=True,
    help="Skip adding upstream tags to the model columns.",
)
@click.option(
    "--skip-merge-meta",
    is_flag=True,
    help="Skip merging upstrean meta keys to the model columns.",
)
@click.option(
    "--skip-add-data-types",
    is_flag=True,
    help="Skip adding data types to the models.",
)
@click.option(
    "--add-progenitor-to-meta",
    is_flag=True,
    help="Progenitor information will be added to the meta information of a column. Useful to understand which model is the progenitor (origin) of a specific model's column.",
)
@click.option(
    "--add-inheritance-for-specified-keys",
    multiple=True,
    type=click.STRING,
    help="Add inheritance for the specified keys. IE policy_tags",
)
@click.option(
    "--numeric-precision-and-scale",
    is_flag=True,
    help="Numeric types will have precision and scale, e.g. Number(38, 8).",
)
@click.option(
    "--string-length",
    is_flag=True,
    help="Character types will have length, e.g. Varchar(128).",
)
@click.option(
    "--output-to-lower",
    is_flag=True,
    help="Output yaml file columns and data types in lowercase if possible.",
)
@click.option(
    "--output-to-upper",
    is_flag=True,
    help="Output yaml file columns and data types in uppercase if possible.",
)
@click.option(
    "--synthesize",
    is_flag=True,
    help="Automatically synthesize missing documentation with OpenAI.",
)
@click.option(
    "--include-external",
    is_flag=True,
    help="Include models and sources from external dbt packages in the processing.",
)
def document(
    target: str | None = None,
    profile: str | None = None,
    project_dir: str | None = None,
    profiles_dir: str | None = None,
    vars: str | None = None,
    check: bool = False,
    threads: int | None = None,
    disable_introspection: bool = False,
    synthesize: bool = False,
    **kwargs: t.Any,
) -> None:
    """Column level documentation inheritance for existing models

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
        threads=threads,
        vars=yaml_handler.safe_load(vars) if vars else None,
        disable_introspection=disable_introspection,
    )

    with YamlRefactorContext(
        project=create_dbt_project_context(settings),
        settings=YamlRefactorSettings(
            **{k: v for k, v in kwargs.items() if v is not None}, create_catalog_if_not_exists=False
        ),
    ) as context:
        transform = (
            inject_missing_columns
            >> inherit_upstream_column_knowledge
            >> sort_columns_as_configured
        )
        if synthesize:
            transform >>= synthesize_missing_documentation_with_openai

        _ = transform(context=context)

        if check and context.mutated:
            exit(1)


@cli.group()
def nl():
    """Natural language interface for dbt model generation and SQL queries"""


@nl.command(context_settings=_CONTEXT)
@dbt_opts
@logging_opts
@click.argument("query")
@click.option(
    "--model-name",
    type=click.STRING,
    help="Optional name for the generated model (auto-generated if not provided)",
)
@click.option(
    "--output-path",
    type=click.Path(),
    help="Path to save the generated model SQL file",
)
@click.option(
    "--schema-yml",
    type=click.Path(),
    help="Path to save the generated schema.yml file",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Print the generated model without writing to disk",
)
def generate(
    query: str = "",
    model_name: str | None = None,
    output_path: str | None = None,
    schema_yml: str | None = None,
    dry_run: bool = False,
    project_dir: str | None = None,
    profiles_dir: str | None = None,
    target: str | None = None,
    **kwargs: t.Any,
) -> None:
    """Generate a dbt model from a natural language description.

    \f
    Example:
        dbt-osmosis nl generate "Show me customers who churned in the last 30 days"

    The AI will analyze your query, understand your available models and sources,
    and generate a complete dbt model with SQL and documentation.
    """
    logger.info(":water_wave: Executing dbt-osmosis natural language generation\n")
    settings = DbtConfiguration(
        project_dir=t.cast(str, project_dir),
        profiles_dir=t.cast(str, profiles_dir),
        target=target,
        **kwargs,
    )
    project = create_dbt_project_context(settings)

    # Gather available sources and models from the manifest
    available_sources: list[dict[str, t.Any]] = []

    # Add models from manifest
    for node_id, node in project.manifest.nodes.items():
        if hasattr(node, "resource_type") and node.resource_type == "model":
            columns = list(node.columns.keys()) if hasattr(node, "columns") else []
            available_sources.append({
                "name": node.name,
                "type": "model",
                "description": getattr(node, "description", ""),
                "columns": columns,
            })

    # Add sources from manifest
    for source_id, source in project.manifest.sources.items():
        if hasattr(source, "resource_type") and source.resource_type == "source":
            columns = list(source.columns.keys()) if hasattr(source, "columns") else []
            available_sources.append({
                "name": f"{source.source_name}.{source.name}",
                "type": "source",
                "description": getattr(source, "description", ""),
                "columns": columns,
            })

    logger.info(f":crystal_ball: Found {len(available_sources)} available sources/models")

    # Generate the model specification
    try:
        model_spec = generate_dbt_model_from_nl(query, available_sources)
    except Exception as e:
        logger.error(f":x: Failed to generate model: {e}")
        raise

    # Override model name if provided
    if model_name:
        model_spec["model_name"] = model_name

    click.echo(f"\n:sparkles: Generated model: {model_spec['model_name']}")
    click.echo(f"Description: {model_spec['description']}")
    click.echo(f"Materialized: {model_spec['materialized']}")

    # Generate SQL content
    sql_content = f"-- {model_spec['description']}\n"
    sql_content += f"-- Materialized: {model_spec['materialized']}\n\n"
    sql_content += model_spec["sql"]

    if dry_run:
        click.echo("\n" + "=" * 80)
        click.echo("SQL:")
        click.echo("=" * 80)
        click.echo(sql_content)
        click.echo("\n" + "=" * 80)
        click.echo("Columns:")
        click.echo("=" * 80)
        for col in model_spec["columns"]:
            click.echo(f"  - {col['name']}: {col['description']}")
        return

    # Write SQL file
    if output_path is None:
        models_dir = Path(project_dir) / "models"
        output_path = str(models_dir / f"{model_spec['model_name']}.sql")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(sql_content)
    click.echo(f"\n:white_check_mark: Wrote SQL to: {output_path}")

    # Write schema.yml if requested
    if schema_yml or output_path:
        schema_path = schema_yml or str(
            Path(output_path).parent / f"{model_spec['model_name']}.yml"
        )

        import yaml

        schema_content = {
            "version": 2,
            "models": [
                {
                    "name": model_spec["model_name"],
                    "description": model_spec["description"],
                    "columns": [
                        {"name": col["name"], "description": col["description"]}
                        for col in model_spec["columns"]
                    ],
                }
            ],
        }

        Path(schema_path).write_text(
            yaml.dump(schema_content, default_flow_style=False, sort_keys=False)
        )
        click.echo(f":white_check_mark: Wrote schema.yml to: {schema_path}")


@nl.command(context_settings=_CONTEXT)
@dbt_opts
@logging_opts
@click.argument("query")
@click.option(
    "--execute",
    is_flag=True,
    help="Execute the generated SQL and display results",
)
def query(
    query: str = "",
    execute: bool = False,
    project_dir: str | None = None,
    profiles_dir: str | None = None,
    target: str | None = None,
    **kwargs: t.Any,
) -> None:
    """Generate SQL from a natural language query.

    \f
    Example:
        dbt-osmosis nl query "Show me the top 10 customers by lifetime value"

    The AI will translate your natural language query into SQL using dbt's ref() syntax.
    """
    logger.info(":water_wave: Executing dbt-osmosis natural language SQL generation\n")
    settings = DbtConfiguration(
        project_dir=t.cast(str, project_dir),
        profiles_dir=t.cast(str, profiles_dir),
        target=target,
        **kwargs,
    )
    project = create_dbt_project_context(settings)

    # Gather available sources and models from the manifest
    available_sources: list[dict[str, t.Any]] = []

    for node_id, node in project.manifest.nodes.items():
        if hasattr(node, "resource_type") and node.resource_type == "model":
            columns = list(node.columns.keys()) if hasattr(node, "columns") else []
            available_sources.append({
                "name": node.name,
                "type": "model",
                "description": getattr(node, "description", ""),
                "columns": columns,
            })

    for source_id, source in project.manifest.sources.items():
        if hasattr(source, "resource_type") and source.resource_type == "source":
            columns = list(source.columns.keys()) if hasattr(source, "columns") else []
            available_sources.append({
                "name": f"{source.source_name}.{source.name}",
                "type": "source",
                "description": getattr(source, "description", ""),
                "columns": columns,
            })

    logger.info(f":crystal_ball: Found {len(available_sources)} available sources/models")

    # Generate SQL
    try:
        sql = generate_sql_from_nl(query, available_sources)
    except Exception as e:
        logger.error(f":x: Failed to generate SQL: {e}")
        raise

    click.echo("\n" + "=" * 80)
    click.echo("Generated SQL:")
    click.echo("=" * 80)
    click.echo(sql)

    if execute:
        click.echo("\n" + "=" * 80)
        click.echo("Executing SQL...")
        click.echo("=" * 80)
        _, table = execute_sql_code(project, sql)

        getattr(table, "print_table")(
            max_rows=50,
            max_columns=6,
            output=sys.stdout,
            max_column_width=20,
            locale=None,
            max_precision=3,
        )


@cli.command(
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
    )
)
@logging_opts
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
@dbt_opts
@logging_opts
@click.argument("sql")
def run(
    sql: str = "",
    project_dir: str | None = None,
    profiles_dir: str | None = None,
    target: str | None = None,
    **kwargs: t.Any,
) -> None:
    """Executes a dbt SQL statement writing results to stdout"""
    settings = DbtConfiguration(
        project_dir=t.cast(str, project_dir),
        profiles_dir=t.cast(str, profiles_dir),
        target=target,
        **kwargs,
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
@dbt_opts
@logging_opts
@click.argument("sql")
def compile(
    sql: str = "",
    project_dir: str | None = None,
    profiles_dir: str | None = None,
    target: str | None = None,
    **kwargs: t.Any,
) -> None:
    """Executes a dbt SQL statement writing results to stdout"""
    settings = DbtConfiguration(
        project_dir=t.cast(str, project_dir),
        profiles_dir=t.cast(str, profiles_dir),
        target=target,
        **kwargs,
    )
    project = create_dbt_project_context(settings)
    node = compile_sql_code(project, sql)

    print(node.compiled_code)


if __name__ == "__main__":
    cli()
