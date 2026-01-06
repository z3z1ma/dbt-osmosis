# pyright: reportUnreachable=false, reportAny=false

from __future__ import annotations

import functools
import subprocess
import sys
import typing as t
from pathlib import Path

import click
import yaml as yaml_handler

from dbt_osmosis.core import logger
from dbt_osmosis.core.exceptions import OsmosisError
from dbt_osmosis.core.osmosis import (
    DbtConfiguration,
    ModelTestAnalysis,
    ValidationReport,
    YamlRefactorContext,
    YamlRefactorSettings,
    apply_restructure_plan,
    apply_semantic_analysis,
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
    suggest_tests_for_project,
    synchronize_data_types,
    synthesize_missing_documentation_with_openai,
    validate_models,
)
from dbt_osmosis.core.staging import (
    generate_staging_for_all_sources,
    generate_staging_for_source,
    write_staging_files,
)
from dbt_osmosis.core.node_filters import _iter_candidate_nodes

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


def test_llm_connection(llm_client=None) -> None:
    """Test the connection to the LLM client."""
    import os

    from dbt_osmosis.core.llm import get_llm_client

    llm_client = os.getenv("LLM_PROVIDER")
    if not llm_client:
        click.echo(
            "ERROR: LLM_PROVIDER environment variable is not set. Please set it to one of the available providers.",
        )
        return

    client, model_engine = get_llm_client()
    if not client or not model_engine:
        click.echo(
            f"Connection ERROR: The environment variables for LLM provider {llm_client} are not set correctly.",
        )
        return

    click.echo(
        f"LLM client connection successful. Provider: {llm_client}, Model Engine: {model_engine}",
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


def handle_errors(func: t.Callable[P, T]) -> t.Callable[P, T]:
    """Decorator to handle and display dbt-osmosis errors gracefully."""

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        try:
            return func(*args, **kwargs)
        except OsmosisError as e:
            logger.error(f":x: {type(e).__name__}: {e}")
            raise click.ClickException(str(e)) from None
        except click.ClickException:
            # Re-raise Click exceptions as-is
            raise
        except Exception as e:
            logger.error(f":boom: Unexpected error: {e}")
            raise click.ClickException(f"An unexpected error occurred: {e}") from None

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
        help="Allows running the program without a database connection, it is recommended to use the --catalog-path option if using this.",
    )
    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        if kwargs.get("disable_introspection") and not kwargs.get("catalog_path"):
            logger.warning(
                ":construction: You have disabled introspection without providing a catalog path. This will result in some features not working as expected.",
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
@handle_errors
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
        project_dir=t.cast("str", project_dir),
        profiles_dir=t.cast("str", profiles_dir),
        target=target,
        profile=profile,
        threads=threads,
        vars=yaml_handler.safe_load(vars) if vars else None,
        disable_introspection=disable_introspection,
    )

    with YamlRefactorContext(
        project=create_dbt_project_context(settings),
        settings=YamlRefactorSettings(
            **{k: v for k, v in kwargs.items() if v is not None},
            create_catalog_if_not_exists=False,
        ),
    ) as context:
        create_missing_source_yamls(context=context)
        apply_restructure_plan(
            context=context,
            plan=draft_restructure_delta_plan(context),
            confirm=not auto_apply,
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
@handle_errors
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
        project_dir=t.cast("str", project_dir),
        profiles_dir=t.cast("str", profiles_dir),
        target=target,
        profile=profile,
        threads=threads,
        vars=yaml_handler.safe_load(vars) if vars else None,
        disable_introspection=disable_introspection,
    )

    with YamlRefactorContext(
        project=create_dbt_project_context(settings),
        settings=YamlRefactorSettings(
            **{k: v for k, v in kwargs.items() if v is not None},
            create_catalog_if_not_exists=False,
        ),
    ) as context:
        create_missing_source_yamls(context=context)
        apply_restructure_plan(
            context=context,
            plan=draft_restructure_delta_plan(context),
            confirm=not auto_apply,
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
    "--semantic-analysis",
    is_flag=True,
    help="Use AI semantic analysis to infer business meaning and relationships for columns.",
)
@click.option(
    "--include-external",
    is_flag=True,
    help="Include models and sources from external dbt packages in the processing.",
)
@handle_errors
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
    semantic_analysis: bool = False,
    **kwargs: t.Any,
) -> None:
    """Column level documentation inheritance for existing models

    \f
    This command will conform schema ymls in your project as outlined in `dbt_project.yml` &
    bootstrap undocumented dbt models
    """
    logger.info(":water_wave: Executing dbt-osmosis\n")
    settings = DbtConfiguration(
        project_dir=t.cast("str", project_dir),
        profiles_dir=t.cast("str", profiles_dir),
        target=target,
        profile=profile,
        threads=threads,
        vars=yaml_handler.safe_load(vars) if vars else None,
        disable_introspection=disable_introspection,
    )

    with YamlRefactorContext(
        project=create_dbt_project_context(settings),
        settings=YamlRefactorSettings(
            **{k: v for k, v in kwargs.items() if v is not None},
            create_catalog_if_not_exists=False,
        ),
    ) as context:
        transform = (
            inject_missing_columns
            >> inherit_upstream_column_knowledge
            >> sort_columns_as_configured
        )
        if semantic_analysis:
            transform >>= apply_semantic_analysis
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


@yaml.command(context_settings=_CONTEXT)
@dbt_opts
@logging_opts
@click.option(
    "-s",
    "--source",
    type=tuple([str, str]),
    multiple=False,
    help="Generate staging for a specific source as 'source_name table_name'",
)
@click.option(
    "--pattern",
    type=str,
    help="Filter sources by pattern (e.g., 'raw_*')",
)
@click.option(
    "--exclude",
    type=str,
    multiple=True,
    help="Exclude sources matching these patterns",
)
@click.option(
    "-d",
    "--dry-run",
    is_flag=True,
    help="Generate staging models without writing files",
)
@click.option(
    "--temperature",
    type=float,
    default=0.3,
    help="LLM temperature for generation (0.0-1.0). Lower is more deterministic.",
)
def stage(
    target: str | None = None,
    profile: str | None = None,
    project_dir: str | None = None,
    profiles_dir: str | None = None,
    vars: str | None = None,
    source: tuple[str, str] | None = None,
    pattern: str | None = None,
    exclude: tuple[str, ...] = (),
    dry_run: bool = False,
    temperature: float = 0.3,
    threads: int | None = None,
    **kwargs: t.Any,
) -> None:
    """Generate AI-powered staging models from source tables.

    \f
    This command analyzes your source tables and automatically generates
    staging models with appropriate column renaming, type casting, and
    basic data cleaning using AI.

    Examples:
        # Generate staging for all sources
        dbt-osmosis yaml stage

        # Generate for a specific source
        dbt-osmosis yaml stage --source raw_stripe customers

        # Generate for sources matching a pattern
        dbt-osmosis yaml stage --pattern "raw_*"
    """
    logger.info(":water_wave: Executing dbt-osmosis staging generation\n")
    settings = DbtConfiguration(
        project_dir=t.cast(str, project_dir),
        profiles_dir=t.cast(str, profiles_dir),
        target=target,
        profile=profile,
        threads=threads,
        vars=yaml_handler.safe_load(vars) if vars else None,
    )

    with YamlRefactorContext(
        project=create_dbt_project_context(settings),
        settings=YamlRefactorSettings(
            **{k: v for k, v in kwargs.items() if v is not None}, create_catalog_if_not_exists=False
        ),
    ) as context:
        if source:
            # Generate for a specific source
            source_name, table_name = source
            result = generate_staging_for_source(
                project=context.project,
                settings=context.settings,
                source_name=source_name,
                table_name=table_name,
                temperature=temperature,
            )

            if result.spec:
                write_staging_files(result, dry_run=dry_run)
                logger.info(
                    ":white_check_mark: Generated staging model %s for source %s.%s",
                    result.spec.staging_name,
                    source_name,
                    table_name,
                )
            elif result.error:
                logger.error(":boom: Failed to generate staging: %s", result.error)
                exit(1)
        else:
            # Generate for all sources (with optional filtering)
            results = generate_staging_for_all_sources(
                project=context.project,
                settings=context.settings,
                source_pattern=pattern,
                exclude_patterns=list(exclude) if exclude else None,
                temperature=temperature,
            )

            successful = sum(1 for r in results if r.spec is not None)
            failed = sum(1 for r in results if r.error is not None)

            logger.info(
                ":bar_chart: Generated %d staging models, %d failed",
                successful,
                failed,
            )

            # Write all successful results
            for result in results:
                write_staging_files(result, dry_run=dry_run)

            if failed > 0:
                logger.warning(":warning: Some staging models failed to generate")
                for result in results:
                    if result.error:
                        logger.warning("  - %s: %s", result.source_name, result.error)


@yaml.command(context_settings=_CONTEXT)
@dbt_opts
@logging_opts
@click.option(
    "-f",
    "--fix",
    is_flag=True,
    help="Automatically fix fixable issues",
)
@click.option(
    "--format",
    type=click.Choice(["table", "json", "yaml"]),
    default="table",
    help="Output format for validation results (default: table).",
)
@click.option(
    "--include-formatting",
    is_flag=True,
    help="Include formatting validation (trailing whitespace, line endings).",
)
@click.option(
    "--fail-on-warning",
    is_flag=True,
    help="Exit with non-zero code on warnings in addition to errors.",
)
@handle_errors
def validate(
    project_dir: str | None = None,
    profiles_dir: str | None = None,
    target: str | None = None,
    vars: str | None = None,
    threads: int | None = None,
    fix: bool = False,
    format: str = "table",
    include_formatting: bool = False,
    fail_on_warning: bool = False,
    **kwargs: t.Any,
) -> None:
    """Validate dbt YAML schemas against project conventions.

    \f
    This command validates your dbt YAML files for:
    - Missing required fields (version, name)
    - Invalid test configurations
    - Malformed YAML structure
    - Formatting issues (optional)

    Examples:

        # Validate all YAML files
        dbt-osmosis yaml validate

        # Validate and auto-fix issues
        dbt-osmosis yaml validate --fix

        # Output as JSON
        dbt-osmosis yaml validate --format json

        # Include formatting checks
        dbt-osmosis yaml validate --include-formatting
    """
    import json as json_module

    from dbt_osmosis.core.schema import (
        FormattingValidator,
        ModelValidator,
        SourceValidator,
        StructureValidator,
        validate_yaml_file,
    )

    logger.info(":mag: Validating dbt YAML schemas\n")

    # Build list of validators
    validators = [StructureValidator(), ModelValidator(), SourceValidator()]
    if include_formatting:
        validators.append(FormattingValidator())

    # Find all YAML files in the project
    yaml_files = list(Path(project_dir).rglob("*.yml")) + list(Path(project_dir).rglob("*.yaml"))

    # Filter out dbt_project.yml and profiles.yml
    yaml_files = [
        f for f in yaml_files if f.name not in ("dbt_project.yml", "profiles.yml", "packages.yml")
    ]

    logger.info(f":page_facing_up: Found {len(yaml_files)} YAML files to validate\n")

    all_results: list[tuple[Path, t.Any]] = []
    for yaml_file in yaml_files:
        try:
            raw_content = yaml_file.read_text() if include_formatting else None
            result = validate_yaml_file(yaml_file, raw_content, validators)
            all_results.append((yaml_file, result))

            # Auto-fix if requested
            if fix and result.get_fixable():
                import threading

                from dbt_osmosis.core.schema import auto_fix_yaml
                from dbt_osmosis.core.schema.parser import create_yaml_instance
                from dbt_osmosis.core.schema.reader import _read_yaml
                from dbt_osmosis.core.schema.writer import _write_yaml

                yaml_handler = create_yaml_instance()
                yaml_handler_lock = threading.Lock()
                data = _read_yaml(yaml_handler, yaml_handler_lock, yaml_file)
                fixed_data = auto_fix_yaml(data, result)
                _write_yaml(yaml_handler, yaml_handler_lock, yaml_file, fixed_data)
                logger.info(f":wrench: Fixed {len(result.fixes_applied)} issues in {yaml_file}")

        except Exception as e:
            logger.error(f":x: Error validating {yaml_file}: {e}")
            continue

    # Aggregate results
    total_errors = sum(len(r.get_errors()) for _, r in all_results)
    total_warnings = sum(len(r.get_warnings()) for _, r in all_results)
    total_fixable = sum(len(r.get_fixable()) for _, r in all_results)

    # Format output
    if format == "json":
        output = []
        for yaml_file, result in all_results:
            output.append({
                "file": str(yaml_file),
                "valid": result.is_valid,
                "issues": [issue.to_dict() for issue in result.issues],
            })
        click.echo(json_module.dumps(output, indent=2))
    elif format == "yaml":
        output = []
        for yaml_file, result in all_results:
            output.append({
                "file": str(yaml_file),
                "valid": result.is_valid,
                "issues": [issue.to_dict() for issue in result.issues],
            })
        click.echo(yaml_handler.dump(output, default_flow_style=False))
    else:
        # Table format
        for yaml_file, result in all_results:
            if result.issues:
                click.echo(f"\n:file_folder: {yaml_file}")
                for issue in result.issues:
                    click.echo(f"  {issue}")
            else:
                click.echo(f":white_check_mark: {yaml_file}")

        # Summary
        click.echo(f"\n{'=' * 60}")
        click.echo("Validation Summary")
        click.echo(f"{'=' * 60}")
        click.echo(f"Files validated: {len(yaml_files)}")
        click.echo(f"Errors: {total_errors}")
        click.echo(f"Warnings: {total_warnings}")
        click.echo(f"Fixable: {total_fixable}")

        is_valid = total_errors == 0
        if is_valid:
            click.echo("\n:white_check_mark: All validations passed!")
        else:
            click.echo(f"\n:x: Validation failed with {total_errors} error(s)")

    # Exit with appropriate code
    if not is_valid or (fail_on_warning and total_warnings > 0):
        exit(1)


@cli.command(
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
    ),
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
        proc = subprocess.run(["streamlit", "run", "--help"], check=False)
        ctx.exit(proc.returncode)

    import os

    if "--config" in ctx.args:
        proc = subprocess.run(
            ["streamlit", "config", "show"],
            check=False,
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
        check=False,
        env=os.environ,
        cwd=Path.cwd(),
    )

    ctx.exit(proc.returncode)


@sql.command(context_settings=_CONTEXT)
@dbt_opts
@logging_opts
@click.argument("sql")
@handle_errors
def run(
    sql: str = "",
    project_dir: str | None = None,
    profiles_dir: str | None = None,
    target: str | None = None,
    **kwargs: t.Any,
) -> None:
    """Executes a dbt SQL statement writing results to stdout"""
    settings = DbtConfiguration(
        project_dir=t.cast("str", project_dir),
        profiles_dir=t.cast("str", profiles_dir),
        target=target,
        **kwargs,
    )
    project = create_dbt_project_context(settings)
    _, table = execute_sql_code(project, sql)

    table.print_table(
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
@handle_errors
def compile(
    sql: str = "",
    project_dir: str | None = None,
    profiles_dir: str | None = None,
    target: str | None = None,
    **kwargs: t.Any,
) -> None:
    """Executes a dbt SQL statement writing results to stdout"""
    settings = DbtConfiguration(
        project_dir=t.cast("str", project_dir),
        profiles_dir=t.cast("str", profiles_dir),
        target=target,
        **kwargs,
    )
    project = create_dbt_project_context(settings)
    node = compile_sql_code(project, sql)

    print(node.compiled_code)


@cli.group()
def test():
    """AI-powered test suggestion and generation for dbt models"""


@test.command(context_settings=_CONTEXT)
@dbt_opts
@yaml_opts
@logging_opts
@click.option(
    "--use-ai",
    is_flag=True,
    default=True,
    help="Use AI for test suggestions (default: True). Set to False for pattern-based only.",
)
@click.option(
    "--temperature",
    type=click.FLOAT,
    default=0.3,
    help="LLM temperature for test generation (0.0-1.0, default: 0.3). Higher values are more creative.",
)
@click.option(
    "--format",
    type=click.Choice(["table", "json", "yaml"]),
    default="table",
    help="Output format for test suggestions (default: table).",
)
def suggest(
    target: str | None = None,
    profile: str | None = None,
    project_dir: str | None = None,
    profiles_dir: str | None = None,
    vars: str | None = None,
    threads: int | None = None,
    disable_introspection: bool = False,
    use_ai: bool = True,
    temperature: float = 0.3,
    format: str = "table",
    **kwargs: t.Any,
) -> None:
    """Suggest appropriate tests for dbt models based on project patterns and AI analysis

    \f
    This command analyzes your dbt project to suggest appropriate tests for your models.
    It learns from existing test patterns in your project and uses AI to generate
    contextually relevant test suggestions.

    Examples:

        # Suggest tests for all models (uses AI by default)
        dbt-osmosis test suggest

        # Suggest tests without AI (pattern-based only)
        dbt-osmosis test suggest --no-use-ai

        # Suggest tests for specific models
        dbt-osmosis test suggest --models my_model

        # Output as JSON
        dbt-osmosis test suggest --format json

        # Output as YAML (ready to paste into schema.yml)
        dbt-osmosis test suggest --format yaml
    """
    import json as json_module

    logger.info(":robot: Analyzing models and suggesting tests\n")

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
            **{
                k: v
                for k, v in kwargs.items()
                if v is not None and k not in ("check", "models", "fqn")
            },
            create_catalog_if_not_exists=False,
        ),
    ) as context:
        # Get test suggestions for all models
        suggestions = suggest_tests_for_project(
            context=context, use_ai=use_ai, temperature=temperature
        )

        # Format output
        if format == "json":
            output = _format_suggestions_as_json(suggestions)
            click.echo(json_module.dumps(output, indent=2))
        elif format == "yaml":
            output = _format_suggestions_as_yaml(suggestions)
            click.echo(yaml_handler.dump(output, default_flow_style=False))
        else:
            # Default: table format
            _print_suggestions_as_table(suggestions)


def _format_suggestions_as_json(
    suggestions: dict[str, "ModelTestAnalysis"],
) -> dict[str, t.Any]:
    """Format test suggestions as JSON."""
    output: dict[str, t.Any] = {"models": []}
    for model_name, analysis in suggestions.items():
        model_data: dict[str, t.Any] = {
            "name": model_name,
            "summary": analysis.get_test_summary(),
            "suggested_tests": {},
        }
        for col_name, tests in analysis.suggested_tests.items():
            model_data["suggested_tests"][col_name] = [
                {
                    "test_type": t.test_type,
                    "reason": t.reason,
                    "config": t.config,
                    "confidence": t.confidence,
                }
                for t in tests
            ]
        output["models"].append(model_data)
    return output


def _format_suggestions_as_yaml(
    suggestions: dict[str, "ModelTestAnalysis"],
) -> str:
    """Format test suggestions as YAML ready to paste into schema.yml files.

    This generates YAML in dbt's test format that can be directly pasted
    into your schema.yml files.
    """
    output = []  # type: ignore[var-annotated]

    for model_name, analysis in suggestions.items():
        model_yaml = {"models": []}  # type: ignore[var-annotated]

        model_entry = {"name": model_name, "columns": []}  # type: ignore[var-annotated]

        for col_name, tests in analysis.suggested_tests.items():
            col_entry = {"name": col_name, "tests": []}  # type: ignore[var-annotated]

            for test in tests:
                test_yaml = test.to_yaml_dict()
                col_entry["tests"].append(test_yaml)

            model_entry["columns"].append(col_entry)

        model_yaml["models"].append(model_entry)
        output.append(model_yaml)

    return str(yaml_handler.dump(output[0] if output else {}, default_flow_style=False))


def _print_suggestions_as_table(
    suggestions: dict[str, "ModelTestAnalysis"],
) -> None:
    """Print test suggestions in a human-readable table format."""
    for model_name, analysis in suggestions.items():
        click.echo(f"\n{'=' * 60}")
        click.echo(f"Model: {model_name}")
        click.echo(f"{'=' * 60}")

        summary = analysis.get_test_summary()
        click.echo(f"Columns: {summary['total_columns']}")
        click.echo(f"Columns with tests: {summary['columns_with_tests']}")
        click.echo(f"Suggested tests: {summary['total_suggested_tests']}")

        if not analysis.suggested_tests:
            click.echo("  No new test suggestions.")
            continue

        for col_name, tests in analysis.suggested_tests.items():
            click.echo(f"\n  :pushpin: Column: {col_name}")
            for test in tests:
                conf_pct = int(test.confidence * 100)
                click.echo(f"    - {test.test_type} ({conf_pct}% confidence)")
                if test.reason:
                    click.echo(f"      Reason: {test.reason}")
                if test.config:
                    click.echo(f"      Config: {test.config}")

    click.echo(f"\n{'=' * 60}")
    click.echo(f"Summary: {len(suggestions)} models analyzed")
    click.echo(f"{'=' * 60}\n")


@cli.group()
def validate():
    """Validate dbt models against production data without materializing"""


@validate.command(context_settings=_CONTEXT)
@dbt_opts
@yaml_opts
@logging_opts
@click.option(
    "--timeout",
    type=float,
    default=None,
    help="Maximum execution time per query in seconds (default: no timeout).",
)
@click.option(
    "-q",
    "--quiet",
    is_flag=True,
    help="Suppress progress output, only show summary.",
)
@click.option(
    "--fail-on-error",
    is_flag=True,
    help="Exit with non-zero status if any model fails validation.",
)
@click.option(
    "--include-external",
    is_flag=True,
    help="Include models from external dbt packages in the validation.",
)
@handle_errors
def dry_run(
    target: str | None = None,
    profile: str | None = None,
    project_dir: str | None = None,
    profiles_dir: str | None = None,
    vars: str | None = None,
    threads: int | None = None,
    timeout: float | None = None,
    quiet: bool = False,
    fail_on_error: bool = False,
    include_external: bool = False,
    disable_introspection: bool = False,
    **kwargs: t.Any,
) -> None:
    """Run dbt models against production data without materializing.

    \f
    This command validates dbt models by compiling and executing them against
    the production database without creating any tables or views. It's useful for:

    - Validating model logic before deployment
    - Catching SQL errors early
    - Estimating query costs (execution time, rows processed)

    Examples:

        # Validate all models
        dbt-osmosis validate dry-run

        # Validate specific models
        dbt-osmosis validate dry-run my_model another_model

        # Validate with timeout
        dbt-osmosis validate dry-run --timeout 60

        # Validate specific FQN
        dbt-osmosis validate dry-run --fqn my_project.staging
    """
    logger.info(":microscope: Executing dbt-osmosis dry-run validation\n")

    settings = DbtConfiguration(
        project_dir=t.cast(str, project_dir),
        profiles_dir=t.cast(str, profiles_dir),
        target=target,
        profile=profile,
        threads=threads,
        vars=yaml_handler.safe_load(vars) if vars else None,
        disable_introspection=disable_introspection,
    )

    project = create_dbt_project_context(settings)

    # Create a minimal YamlRefactorSettings for filtering
    from dbt_osmosis.core.settings import YamlRefactorSettings

    refactor_settings = YamlRefactorSettings(
        models=list(kwargs.get("models", ())),
        fqn=list(kwargs.get("fqn", ())),
    )

    # Create a minimal context for node filtering
    with YamlRefactorContext(
        project=project,
        settings=refactor_settings,
    ) as context:
        # Collect models to validate
        models_to_validate = list(_iter_candidate_nodes(context, include_external=include_external))

        if not models_to_validate:
            logger.warning(":warning: No models found matching the specified criteria.")
            return

        # Run validation
        report: ValidationReport = validate_models(
            context=context.project,
            models=models_to_validate,
            timeout_seconds=timeout,
            quiet=quiet,
        )

        # Exit with error if any models failed and fail-on-error is set
        if fail_on_error and report.failed > 0:
            logger.error(":x: Validation failed for %d model(s)", report.failed)
            for failed_result in report.get_failed_models():
                logger.error(
                    "  - %s: %s",
                    failed_result.model_name,
                    failed_result.error_message or failed_result.status.value,
                )
            exit(1)


if __name__ == "__main__":
    cli()
