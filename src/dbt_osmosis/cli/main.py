# pyright: reportUnreachable=false

from __future__ import annotations

import functools
import importlib.util
import os
import shutil
import subprocess
import sys
import threading
import typing as t
from pathlib import Path

import click
import yaml as yaml_handler

import dbt_osmosis.core.logger as logger
from dbt_osmosis.core.config import (
    DbtConfiguration,
    create_dbt_project_context,
    discover_profiles_dir,
    discover_project_dir,
)
from dbt_osmosis.core.diff import SchemaDiff
from dbt_osmosis.core.generators import (
    generate_sources_from_database,
    generate_staging_from_source,
)
from dbt_osmosis.core.llm import generate_dbt_model_from_nl, generate_sql_from_nl
from dbt_osmosis.core.path_management import create_missing_source_yamls
from dbt_osmosis.core.restructuring import (
    apply_restructure_plan,
    draft_restructure_delta_plan,
)
from dbt_osmosis.core.schema.parser import create_yaml_instance
from dbt_osmosis.core.schema.reader import _read_yaml
from dbt_osmosis.core.schema.writer import _write_yaml
from dbt_osmosis.core.settings import YamlRefactorContext, YamlRefactorSettings
from dbt_osmosis.core.sql_lint import (
    LintLevel,
    LintResult,
    LintViolation,
    SQLLinter,
    lint_sql_code,
)
from dbt_osmosis.core.sql_operations import compile_sql_code, execute_sql_code
from dbt_osmosis.core.test_suggestions import suggest_tests_for_model, suggest_tests_for_project
from dbt_osmosis.core.transforms import (
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
_WORKBENCH_EXTRA_HINT = "pip install dbt-osmosis[workbench]"
_WORKBENCH_APP_MODULES = (
    "feedparser",
    "pandas",
    "streamlit",
    "streamlit_elements_fluence",
    "ydata_profiling",
)


def _missing_streamlit_error() -> click.ClickException:
    return click.ClickException(
        "Streamlit is required to run dbt-osmosis workbench. "
        f"Install the optional workbench extra with `{_WORKBENCH_EXTRA_HINT}`."
    )


def _streamlit_executable() -> str:
    executable = shutil.which("streamlit")
    if executable is None:
        raise _missing_streamlit_error()
    return executable


def _check_workbench_app_dependencies() -> None:
    missing = [
        module for module in _WORKBENCH_APP_MODULES if importlib.util.find_spec(module) is None
    ]
    if missing:
        missing_modules = ", ".join(missing)
        raise click.ClickException(
            "Workbench optional dependencies are missing: "
            f"{missing_modules}. Install them with `{_WORKBENCH_EXTRA_HINT}`."
        )


def _run_streamlit_command(
    args: list[t.Any], executable: str | None = None
) -> subprocess.CompletedProcess[t.Any]:
    try:
        return subprocess.run(
            [executable or _streamlit_executable(), *args],
            env=os.environ,
            cwd=Path.cwd(),
        )
    except FileNotFoundError as e:
        raise _missing_streamlit_error() from e


@click.group()
@click.version_option()
def cli() -> None:
    """dbt-osmosis is a CLI tool for dbt that helps you manage, document, and organize your dbt yaml files"""

    pass


def test_llm_connection(llm_client: tuple[t.Any, str] | None = None) -> None:
    """Test the connection to the LLM client."""
    import os

    if llm_client is None:
        from dbt_osmosis.core.llm import get_llm_client

        llm_client = get_llm_client()

    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    client, model_engine = llm_client
    if not client or not model_engine:
        raise click.ClickException(
            f"The environment variables for LLM provider {provider} are not set correctly."
        )

    click.echo(
        f"LLM client connection successful. Provider: {provider}, Model Engine: {model_engine}"
    )


@cli.command()
def test_llm() -> None:
    """Test the connection to the LLM client"""
    logger.info("INFO: Invoking test_llm_connection...")
    from dbt_osmosis.core.exceptions import LLMConfigurationError
    from dbt_osmosis.core.llm import get_llm_client

    try:
        llm_client = get_llm_client()
        test_llm_connection(llm_client)
    except (ImportError, LLMConfigurationError) as e:
        raise click.ClickException(str(e)) from e

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


@cli.group()
def test():
    """Suggest and generate dbt tests"""


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
        type=click.Path(dir_okay=True, file_okay=False),
        default=discover_profiles_dir,
        help="Which directory to look in for the profiles.yml file. Defaults to DBT_PROFILES_DIR, the current directory, the discovered project root, or ~/.dbt.",
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
    @click.option(
        "--strip-eof-blank-lines/--keep-eof-blank-lines",
        default=False,
        help="Remove trailing blank lines at EOF when writing YAML.",
    )
    @click.option(
        "--fusion-compat/--no-fusion-compat",
        default=None,
        help="Output Fusion-compatible YAML with meta/tags nested inside config blocks. Auto-detects from known Fusion manifest evidence or dbt >= 1.9.6 if not specified.",
    )
    @click.option(
        "--formatter",
        type=click.STRING,
        default=None,
        help='External command to format written YAML files (e.g. "prettier --write", "yamlfmt"). '
        "File paths are appended as arguments. Skipped during --dry-run.",
    )
    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        if kwargs.get("disable_introspection") and not kwargs.get("catalog_path"):
            logger.warning(
                ":construction: You have disabled introspection without providing a catalog path. This will result in some features not working as expected."
            )
        return func(*args, **kwargs)

    return wrapper


def _run_formatter_if_configured(context: YamlRefactorContext) -> None:
    """Run the external formatter on written files if configured and applicable."""
    formatter = context.resolved_formatter
    if formatter and not context.settings.dry_run and context.written_files:
        from dbt_osmosis.core.formatting import run_external_formatter

        run_external_formatter(formatter, context.written_files, context.project_root)


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
    "--skip-inherit-descriptions",
    is_flag=True,
    help="Skip inheriting descriptions from upstream sources while preserving tag and meta inheritance.",
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
    "--skip-inheritance-for-meta-keys",
    multiple=True,
    type=click.STRING,
    help="Skip inheriting the specified upstream column meta keys while preserving other meta keys.",
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
        vars=yaml_handler.safe_load(vars) if vars else {},
        disable_introspection=disable_introspection,
    )

    with YamlRefactorContext(
        project=create_dbt_project_context(settings),
        settings=YamlRefactorSettings(
            **{k: v for k, v in kwargs.items() if v is not None}, create_catalog_if_not_exists=False
        ),
    ) as context:
        typed_context: t.Any = context
        create_missing_source_yamls(context=context)
        apply_restructure_plan(
            context=typed_context,
            plan=draft_restructure_delta_plan(typed_context),
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

        _ = transform(context=typed_context)

        _run_formatter_if_configured(context)

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
        vars=yaml_handler.safe_load(vars) if vars else {},
        disable_introspection=disable_introspection,
    )

    with YamlRefactorContext(
        project=create_dbt_project_context(settings),
        settings=YamlRefactorSettings(
            **{k: v for k, v in kwargs.items() if v is not None}, create_catalog_if_not_exists=False
        ),
    ) as context:
        typed_context: t.Any = context
        create_missing_source_yamls(context=context)
        apply_restructure_plan(
            context=typed_context,
            plan=draft_restructure_delta_plan(typed_context),
            confirm=not auto_apply,
        )

        _run_formatter_if_configured(context)

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
    "--skip-inherit-descriptions",
    is_flag=True,
    help="Skip inheriting descriptions from upstream sources while preserving tag and meta inheritance.",
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
    "--skip-inheritance-for-meta-keys",
    multiple=True,
    type=click.STRING,
    help="Skip inheriting the specified upstream column meta keys while preserving other meta keys.",
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
        vars=yaml_handler.safe_load(vars) if vars else {},
        disable_introspection=disable_introspection,
    )

    with YamlRefactorContext(
        project=create_dbt_project_context(settings),
        settings=YamlRefactorSettings(
            **{k: v for k, v in kwargs.items() if v is not None}, create_catalog_if_not_exists=False
        ),
    ) as context:
        typed_context: t.Any = context
        transform = (
            inject_missing_columns
            >> inherit_upstream_column_knowledge
            >> sort_columns_as_configured
        )
        if synthesize:
            transform >>= synthesize_missing_documentation_with_openai

        _ = transform(context=typed_context)

        _run_formatter_if_configured(context)

        if check and context.mutated:
            exit(1)


@cli.group()
def nl():
    """Natural language interface for dbt model generation and SQL queries"""


@cli.group()
def generate():
    """Generate dbt artifacts: sources, staging models, and more"""


_GENERATED_YAML_HANDLER_LOCK = threading.Lock()


def _get_generated_project_root(project: t.Any, project_dir: str | None) -> Path:
    runtime_cfg = getattr(project, "runtime_cfg", None)
    project_root = getattr(runtime_cfg, "project_root", None)
    if not isinstance(project_root, (str, os.PathLike)):
        project_root = None
    if project_root is None:
        config = getattr(project, "config", None)
        project_root = getattr(config, "project_dir", None)
    if not isinstance(project_root, (str, os.PathLike)):
        project_root = None
    if project_root is None:
        project_root = project_dir or "."
    project_root_path = os.fspath(project_root)
    if isinstance(project_root_path, bytes):
        project_root_path = project_root_path.decode()
    return Path(project_root_path).resolve()


def _resolve_project_yaml_output_path(path: Path | str, project_root: Path) -> Path:
    output_path = Path(path)
    if not output_path.is_absolute():
        output_path = project_root / output_path
    resolved = output_path.resolve()
    try:
        resolved.relative_to(project_root)
    except ValueError as e:
        raise click.ClickException(
            f"Refusing to write YAML outside the dbt project root: {resolved} "
            f"(project root: {project_root})"
        ) from e
    return resolved


def _resolve_generated_file_path(path: Path | str, project_root: Path) -> Path:
    output_path = Path(path)
    if not output_path.is_absolute():
        output_path = project_root / output_path
    resolved = output_path.resolve()
    try:
        resolved.relative_to(project_root)
    except ValueError as e:
        raise click.ClickException(
            f"Refusing to write generated output outside the dbt project root: {resolved} "
            f"(project root: {project_root})"
        ) from e
    return resolved


def _model_schema_data(model_spec: dict[str, t.Any]) -> dict[str, t.Any]:
    return {
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


def _load_generated_yaml_data(yaml_content: str) -> dict[str, t.Any]:
    yaml_handler = create_yaml_instance()
    data = yaml_handler.load(yaml_content) or {}
    if not isinstance(data, dict):
        raise click.ClickException("Generated YAML root must be a mapping.")
    return t.cast("dict[str, t.Any]", data)


def _prepare_generated_yaml_write(
    *,
    project: t.Any,
    project_dir: str | None,
    yaml_path: Path | str,
    yaml_data: dict[str, t.Any] | None = None,
    yaml_content: str | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> tuple[Path, dict[str, t.Any]]:
    project_root = _get_generated_project_root(project, project_dir)
    resolved_path = _resolve_project_yaml_output_path(yaml_path, project_root)

    if resolved_path.exists() and not overwrite:
        raise click.ClickException(
            f"Refusing to overwrite existing schema YAML at {resolved_path}. "
            "Pass --overwrite to replace it."
        )

    data = yaml_data if yaml_data is not None else _load_generated_yaml_data(yaml_content or "")
    if resolved_path.is_file():
        yaml_handler = create_yaml_instance()
        _read_yaml(yaml_handler, _GENERATED_YAML_HANDLER_LOCK, resolved_path)

    return resolved_path, data


def _write_prepared_generated_yaml(
    prepared_write: tuple[Path, dict[str, t.Any]],
    *,
    dry_run: bool = False,
    overwrite: bool = False,
) -> None:
    yaml_handler = create_yaml_instance()
    path, data = prepared_write
    _write_yaml(
        yaml_handler=yaml_handler,
        yaml_handler_lock=_GENERATED_YAML_HANDLER_LOCK,
        path=path,
        data=data,
        dry_run=dry_run,
        allow_overwrite=overwrite,
    )


def _echo_planned_writes(paths: t.Iterable[Path | None]) -> None:
    planned_paths = [path for path in paths if path is not None]
    if not planned_paths:
        return
    click.echo("\nPlanned writes:")
    for path in planned_paths:
        click.echo(f"  - {path}")


@generate.command(context_settings=_CONTEXT)
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
@click.option(
    "--overwrite",
    is_flag=True,
    help="Allow generated schema YAML to replace an existing file.",
)
def model(
    query: str = "",
    model_name: str | None = None,
    output_path: str | None = None,
    schema_yml: str | None = None,
    dry_run: bool = False,
    overwrite: bool = False,
    project_dir: str | None = None,
    profiles_dir: str | None = None,
    target: str | None = None,
    **kwargs: t.Any,
) -> None:
    """Generate a dbt model from a natural language description.

    \f
    Example:
        dbt-osmosis generate model "Show me customers who churned in the last 30 days"

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

    try:
        model_spec = generate_dbt_model_from_nl(query, available_sources)
    except Exception as e:
        logger.error(f":x: Failed to generate model: {e}")
        raise

    if model_name:
        model_spec["model_name"] = model_name

    click.echo(f"\n:sparkles: Generated model: {model_spec['model_name']}")
    click.echo(f"Description: {model_spec['description']}")
    click.echo(f"Materialized: {model_spec['materialized']}")

    sql_content = f"-- {model_spec['description']}\n"
    sql_content += f"-- Materialized: {model_spec['materialized']}\n\n"
    sql_content += model_spec["sql"]

    project_root = _get_generated_project_root(project, project_dir)
    if output_path is None:
        output_path_obj = _resolve_generated_file_path(
            project_root / "models" / f"{model_spec['model_name']}.sql",
            project_root,
        )
    else:
        output_path_obj = _resolve_generated_file_path(output_path, project_root)
    schema_path = schema_yml or output_path_obj.parent / f"{model_spec['model_name']}.yml"
    schema_write = _prepare_generated_yaml_write(
        project=project,
        project_dir=project_dir,
        yaml_path=schema_path,
        yaml_data=_model_schema_data(model_spec),
        overwrite=overwrite,
        dry_run=dry_run,
    )

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
        _write_prepared_generated_yaml(schema_write, dry_run=True, overwrite=overwrite)
        _echo_planned_writes([output_path_obj, schema_write[0]])
        return

    _write_prepared_generated_yaml(schema_write, overwrite=overwrite)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    output_path_obj.write_text(sql_content)
    click.echo(f"\n:white_check_mark: Wrote SQL to: {output_path_obj}")
    click.echo(f":white_check_mark: Wrote schema.yml to: {schema_write[0]}")


@generate.command(context_settings=_CONTEXT)
@dbt_opts
@logging_opts
@click.option(
    "--source-name",
    type=click.STRING,
    default="raw",
    help="Name for the source (default: 'raw')",
)
@click.option(
    "--schema-name",
    type=click.STRING,
    default=None,
    help="Specific schema to scan (None = all schemas in database)",
)
@click.option(
    "--exclude-schemas",
    multiple=True,
    type=click.STRING,
    help="Schemas to exclude from scanning",
)
@click.option(
    "--exclude-tables",
    multiple=True,
    type=click.STRING,
    help="Tables to exclude from generation",
)
@click.option(
    "--quote-identifiers",
    is_flag=True,
    help="Quote identifiers in generated YAML",
)
@click.option(
    "--output-path",
    type=click.Path(),
    help="Path where YAML file should be written (default: models/sources/{source_name}.yml)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Print the generated YAML without writing to disk",
)
@click.option(
    "--overwrite",
    is_flag=True,
    help="Allow generated source YAML to replace an existing file.",
)
def sources(
    source_name: str = "raw",
    schema_name: str | None = None,
    exclude_schemas: tuple[str, ...] = (),
    exclude_tables: tuple[str, ...] = (),
    quote_identifiers: bool = False,
    output_path: str | None = None,
    dry_run: bool = False,
    overwrite: bool = False,
    project_dir: str | None = None,
    profiles_dir: str | None = None,
    target: str | None = None,
    **kwargs: t.Any,
) -> None:
    """Generate source definitions from database introspection.

    \f
    Example:
        dbt-osmosis generate sources --source-name raw --schema-name my_schema

    This command discovers tables in your database and generates dbt source YAML definitions.
    """
    logger.info(":water_wave: Executing dbt-osmosis source generation\n")
    settings = DbtConfiguration(
        project_dir=t.cast(str, project_dir),
        profiles_dir=t.cast(str, profiles_dir),
        target=target,
        **kwargs,
    )
    project = create_dbt_project_context(settings)

    result = generate_sources_from_database(
        context=project,
        source_name=source_name,
        schema_name=schema_name,
        exclude_schemas=list(exclude_schemas) if exclude_schemas else None,
        exclude_tables=list(exclude_tables) if exclude_tables else None,
        quote_identifiers=quote_identifiers,
        output_path=Path(output_path) if output_path else None,
    )

    yaml_write = None
    if result.yaml_content:
        yaml_write = _prepare_generated_yaml_write(
            project=project,
            project_dir=project_dir,
            yaml_path=result.yaml_path,
            yaml_content=result.yaml_content,
            overwrite=overwrite,
            dry_run=dry_run,
        )

    if dry_run:
        click.echo("\n" + "=" * 80)
        click.echo("Generated YAML:")
        click.echo("=" * 80)
        click.echo(result.yaml_content)
        if yaml_write is not None:
            _write_prepared_generated_yaml(yaml_write, dry_run=True, overwrite=overwrite)
            _echo_planned_writes([yaml_write[0]])
        return

    if result.yaml_content and yaml_write is not None:
        _write_prepared_generated_yaml(yaml_write, overwrite=overwrite)
        click.echo(f":white_check_mark: Wrote source YAML to: {yaml_write[0]}")
    else:
        click.echo(":warning: No sources found with given configuration")


@generate.command(context_settings=_CONTEXT)
@dbt_opts
@logging_opts
@click.argument("source_name")
@click.argument("table_name")
@click.option(
    "--ai",
    is_flag=True,
    help="Use AI-based generation (intelligent staging with business logic)",
)
@click.option(
    "--staging-path",
    type=click.Path(),
    help="Directory where staging models should be written (default: models/staging/)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Print the generated files without writing to disk",
)
@click.option(
    "--overwrite",
    is_flag=True,
    help="Allow generated staging YAML to replace an existing file.",
)
def staging(
    source_name: str = "",
    table_name: str = "",
    ai: bool = False,
    staging_path: str | None = None,
    dry_run: bool = False,
    overwrite: bool = False,
    project_dir: str | None = None,
    profiles_dir: str | None = None,
    target: str | None = None,
    **kwargs: t.Any,
) -> None:
    """Generate a staging model from a source table.

    \f
    Example:
        dbt-osmosis generate staging raw customers --ai
        dbt-osmosis generate staging raw stripe_transactions

    This command generates staging models from source tables. Use --ai flag for
    intelligent staging with AI-powered business logic, or omit for deterministic
    generation via dbt-core-interface.
    """
    logger.info(":water_wave: Executing dbt-osmosis staging generation\n")
    settings = DbtConfiguration(
        project_dir=t.cast(str, project_dir),
        profiles_dir=t.cast(str, profiles_dir),
        target=target,
        **kwargs,
    )
    project = create_dbt_project_context(settings)

    try:
        result = generate_staging_from_source(
            context=project,
            source_name=source_name,
            table_name=table_name,
            use_ai=ai,
            staging_path=Path(staging_path) if staging_path else None,
        )

        yaml_write = None
        if result.yaml_content and result.yaml_path:
            yaml_write = _prepare_generated_yaml_write(
                project=project,
                project_dir=project_dir,
                yaml_path=result.yaml_path,
                yaml_content=result.yaml_content,
                overwrite=overwrite,
                dry_run=dry_run,
            )
        resolved_sql_path = (
            _resolve_generated_file_path(
                result.sql_path, _get_generated_project_root(project, project_dir)
            )
            if result.sql_content and result.sql_path
            else None
        )

        if dry_run:
            click.echo("\n" + "=" * 80)
            click.echo("Generated SQL:")
            click.echo("=" * 80)
            click.echo(result.sql_content)
            click.echo("\n" + "=" * 80)
            click.echo("Generated YAML:")
            click.echo("=" * 80)
            click.echo(result.yaml_content)
            if yaml_write is not None:
                _write_prepared_generated_yaml(yaml_write, dry_run=True)
            _echo_planned_writes([
                resolved_sql_path,
                yaml_write[0] if yaml_write is not None else None,
            ])
            return

        click.echo(f"\n:sparkles: Generated staging model: {result.staging_name}")

        if result.sql_content and resolved_sql_path:
            resolved_sql_path.parent.mkdir(parents=True, exist_ok=True)
            resolved_sql_path.write_text(result.sql_content, encoding="utf-8")
            click.echo(f":white_check_mark: Wrote SQL to: {resolved_sql_path}")
        elif result.sql_content:
            raise click.ClickException("Generated SQL content is missing a target path.")

        if result.yaml_content and yaml_write is not None:
            _write_prepared_generated_yaml(yaml_write, overwrite=overwrite)
            click.echo(f":white_check_mark: Wrote YAML to: {yaml_write[0]}")
        elif result.yaml_content:
            raise click.ClickException("Generated YAML content is missing a target path.")

    except Exception as e:
        logger.error(f":x: Failed to generate staging model: {e}")
        raise


@generate.command(context_settings=_CONTEXT, name="query")
@dbt_opts
@logging_opts
@click.argument("query")
@click.option(
    "--execute",
    is_flag=True,
    help="Execute the generated SQL and display results",
)
def generate_query(
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
        dbt-osmosis generate query "Show me the top 10 customers by lifetime value"

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


@nl.command(context_settings=_CONTEXT, name="generate")
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
@click.option(
    "--overwrite",
    is_flag=True,
    help="Allow generated schema YAML to replace an existing file.",
)
def nl_generate_deprecated(
    query: str = "",
    model_name: str | None = None,
    output_path: str | None = None,
    schema_yml: str | None = None,
    dry_run: bool = False,
    overwrite: bool = False,
    project_dir: str | None = None,
    profiles_dir: str | None = None,
    target: str | None = None,
    **kwargs: t.Any,
) -> None:
    """Generate a dbt model from a natural language description.

    \f
    DEPRECATED: Use `dbt-osmosis generate model` instead.

    Example:
        dbt-osmosis nl generate "Show me customers who churned in the last 30 days"

    The AI will analyze your query, understand your available models and sources,
    and generate a complete dbt model with SQL and documentation.
    """
    logger.warning(
        ":warning: The `nl generate` command is deprecated. "
        "Use `dbt-osmosis generate model` instead."
    )
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

    project_root = _get_generated_project_root(project, project_dir)
    if output_path is None:
        output_path_obj = _resolve_generated_file_path(
            project_root / "models" / f"{model_spec['model_name']}.sql",
            project_root,
        )
    else:
        output_path_obj = _resolve_generated_file_path(output_path, project_root)
    schema_path = schema_yml or output_path_obj.parent / f"{model_spec['model_name']}.yml"
    schema_write = _prepare_generated_yaml_write(
        project=project,
        project_dir=project_dir,
        yaml_path=schema_path,
        yaml_data=_model_schema_data(model_spec),
        overwrite=overwrite,
        dry_run=dry_run,
    )

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
        _write_prepared_generated_yaml(schema_write, dry_run=True, overwrite=overwrite)
        _echo_planned_writes([output_path_obj, schema_write[0]])
        return

    _write_prepared_generated_yaml(schema_write, overwrite=overwrite)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    output_path_obj.write_text(sql_content)
    click.echo(f"\n:white_check_mark: Wrote SQL to: {output_path_obj}")
    click.echo(f":white_check_mark: Wrote schema.yml to: {schema_write[0]}")


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
    type=click.Path(dir_okay=True, file_okay=False),
    help="Which directory to look in for the profiles.yml file. Defaults to DBT_PROFILES_DIR, the current directory, the discovered project root, or ~/.dbt.",
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
@click.option(
    "--enable-external-feed",
    is_flag=True,
    help="Opt in to fetching the external Hacker News RSS feed in the workbench.",
)
@click.pass_context
def workbench(
    ctx: click.Context,
    profiles_dir: str | None = None,
    project_dir: str | None = None,
    host: str = "localhost",
    port: int = 8501,
    enable_external_feed: bool = False,
) -> None:
    """Start the dbt-osmosis workbench

    \f
    Pass the --options command to see streamlit specific options that can be passed to the app,
    pass --config to see the output of streamlit config show
    """
    logger.info(":water_wave: Executing dbt-osmosis\n")

    if "--options" in ctx.args:
        proc = _run_streamlit_command(["run", "--help"])
        ctx.exit(proc.returncode)

    if "--config" in ctx.args:
        proc = _run_streamlit_command(["config", "show"])
        ctx.exit(proc.returncode)

    script_args = ["--"]
    if project_dir:
        script_args.append("--project-dir")
        script_args.append(project_dir)
    if profiles_dir:
        script_args.append("--profiles-dir")
        script_args.append(profiles_dir)
    if enable_external_feed:
        script_args.append("--enable-external-feed")

    streamlit_executable = _streamlit_executable()
    _check_workbench_app_dependencies()
    proc = _run_streamlit_command(
        [
            "run",
            "--runner.magicEnabled=false",
            f"--server.address={host}",
            f"--server.port={port}",
            *ctx.args,
            Path(__file__).parent.parent / "workbench" / "app.py",
            *script_args,
        ],
        executable=streamlit_executable,
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
    """Compiles a dbt SQL statement and writes the result to stdout"""
    settings = DbtConfiguration(
        project_dir=t.cast(str, project_dir),
        profiles_dir=t.cast(str, profiles_dir),
        target=target,
        **kwargs,
    )
    project = create_dbt_project_context(settings)
    node = compile_sql_code(project, sql)

    print(node.compiled_code)


@cli.group()
def diff():
    """Detect and report schema changes between YAML definitions and database"""


@diff.command(context_settings=_CONTEXT)
@dbt_opts
@yaml_opts
@logging_opts
@click.option(
    "--output-format",
    type=click.Choice(["text", "json", "markdown"], case_sensitive=False),
    default="text",
    help="Output format for the diff results.",
)
@click.option(
    "--severity",
    type=click.Choice(["safe", "moderate", "breaking", "all"], case_sensitive=False),
    default="all",
    help="Filter changes by severity level.",
)
@click.option(
    "--fuzzy-match-threshold",
    type=click.FLOAT,
    default=85.0,
    help="Threshold for detecting column renames (0-100).",
)
@click.option(
    "--detect-column-renames/--no-detect-column-renames",
    default=True,
    help="Enable or disable fuzzy matching for column rename detection.",
)
@click.option(
    "--include-external",
    is_flag=True,
    help="Include models and sources from external dbt packages in the diff.",
)
def schema(
    target: str | None = None,
    profile: str | None = None,
    project_dir: str | None = None,
    profiles_dir: str | None = None,
    vars: str | None = None,
    threads: int | None = None,
    disable_introspection: bool = False,
    fqn: tuple[str, ...] = (),
    output_format: str = "text",
    severity: str = "all",
    fuzzy_match_threshold: float = 85.0,
    detect_column_renames: bool = True,
    include_external: bool = False,
    models: tuple[str, ...] = (),
    **kwargs: t.Any,
) -> None:
    """Detect schema changes between YAML definitions and the database.

    \f
    This command compares your YAML schema definitions with the actual database
    schema and reports:
    - Columns added to the database but not in YAML
    - Columns in YAML but missing from the database
    - Column renames (detected via fuzzy matching)
    - Column data type changes

    Example:
        dbt-osmosis diff schema
        dbt-osmosis diff schema --severity breaking
        dbt-osmosis diff schema -f my_project.my_model --output-format json
    """
    logger.info(":mag: Executing dbt-osmosis schema diff\n")

    settings = DbtConfiguration(
        project_dir=t.cast(str, project_dir),
        profiles_dir=t.cast(str, profiles_dir),
        target=target,
        profile=profile,
        threads=threads,
        vars=yaml_handler.safe_load(vars) if vars else {},
        disable_introspection=disable_introspection,
    )

    with YamlRefactorContext(
        project=create_dbt_project_context(settings),
        settings=YamlRefactorSettings(
            **{
                k: v
                for k, v in kwargs.items()
                if v is not None and k not in {"check", "dry_run", "models"}
            },
            create_catalog_if_not_exists=False,
            fqn=list(fqn),
            models=list(models),
            include_external=include_external,
        ),
    ) as context:
        typed_context: t.Any = context

        # Initialize the schema diff engine
        differ = SchemaDiff(
            typed_context,
            fuzzy_match_threshold=fuzzy_match_threshold,
            detect_column_renames=detect_column_renames,
        )

        results = differ.compare_all()

        # Output the results
        if output_format == "json":
            _output_diff_json(results, severity)
        elif output_format == "markdown":
            _output_diff_markdown(results, severity)
        else:
            _output_diff_text(results, severity)


def _output_diff_text(results: dict[str, t.Any], severity_filter: str) -> None:
    """Output diff results in human-readable text format."""
    if not results:
        click.echo(":white_check_mark: No schema changes detected")
        return

    total_changes = sum(len(r.changes) for r in results.values())
    click.echo(f":warning: Detected {total_changes} schema changes across {len(results)} node(s)\n")

    # Group changes by node
    for node_id, result in results.items():
        # Filter by severity if needed
        changes = result.changes
        if severity_filter != "all":
            from dbt_osmosis.core.diff import ChangeSeverity

            severity_map = {
                "safe": ChangeSeverity.SAFE,
                "moderate": ChangeSeverity.MODERATE,
                "breaking": ChangeSeverity.BREAKING,
            }
            changes = [c for c in changes if c.severity == severity_map[severity_filter]]

        if not changes:
            continue

        # Node header
        node = result.node
        click.echo(f":page_facing_up: {node.name} ({node.resource_type})")
        click.echo(f"   Unique ID: {node.unique_id}")
        click.echo(f"   Path: {node.original_file_path}")

        # Summary
        summary = result.summary
        if summary:
            click.echo(f"   Summary: {', '.join(f'{k}: {v}' for k, v in summary.items())}")

        # Changes list
        for change in changes:
            click.echo(f"\n   {change}")

        # Add extra info for renames
        from dbt_osmosis.core.diff import ColumnRenamed

        for change in changes:
            if isinstance(change, ColumnRenamed):
                click.echo(f"      Similarity: {change.similarity_score:.1f}%")

        click.echo("\n" + "-" * 80 + "\n")

    # Overall summary
    breaking_count = sum(
        1 for r in results.values() for c in r.changes if c.severity.value == "breaking"
    )
    moderate_count = sum(
        1 for r in results.values() for c in r.changes if c.severity.value == "moderate"
    )
    safe_count = sum(1 for r in results.values() for c in r.changes if c.severity.value == "safe")

    click.echo("Overall Summary:")
    click.echo(f"  Breaking changes: {breaking_count}")
    click.echo(f"  Moderate changes: {moderate_count}")
    click.echo(f"  Safe changes: {safe_count}")

    if breaking_count > 0:
        click.echo("\n:rotating_light: Breaking changes detected. Review required before applying.")


def _output_diff_json(results: dict[str, t.Any], severity_filter: str) -> None:
    """Output diff results in JSON format."""
    import json
    from datetime import datetime

    nodes: list[dict[str, object]] = []
    output: dict[str, object] = {
        "timestamp": datetime.utcnow().isoformat(),
        "total_nodes": len(results),
        "total_changes": sum(len(r.changes) for r in results.values()),
        "nodes": nodes,
    }

    for node_id, result in results.items():
        # Filter by severity if needed
        changes = result.changes
        if severity_filter != "all":
            from dbt_osmosis.core.diff import ChangeSeverity

            severity_map = {
                "safe": ChangeSeverity.SAFE,
                "moderate": ChangeSeverity.MODERATE,
                "breaking": ChangeSeverity.BREAKING,
            }
            changes = [c for c in changes if c.severity == severity_map[severity_filter]]

        if not changes:
            continue

        node_data: dict[str, object] = {
            "unique_id": node_id,
            "name": result.node.name,
            "resource_type": str(result.node.resource_type),
            "path": result.node.original_file_path,
            "summary": result.summary,
            "changes": [
                {
                    "category": c.category.value,
                    "severity": c.severity.value,
                    "description": c.description,
                }
                for c in changes
            ],
        }
        nodes.append(node_data)

    click.echo(json.dumps(output, indent=2))


def _output_diff_markdown(results: dict[str, t.Any], severity_filter: str) -> None:
    """Output diff results in Markdown format."""
    if not results:
        click.echo("## Schema Diff Results\n\n:white_check_mark: No schema changes detected")
        return

    total_changes = sum(len(r.changes) for r in results.values())
    click.echo(
        f"# Schema Diff Results\n\n**Detected {total_changes} changes across {len(results)} node(s)**\n"
    )

    for node_id, result in results.items():
        # Filter by severity if needed
        changes = result.changes
        if severity_filter != "all":
            from dbt_osmosis.core.diff import ChangeSeverity

            severity_map = {
                "safe": ChangeSeverity.SAFE,
                "moderate": ChangeSeverity.MODERATE,
                "breaking": ChangeSeverity.BREAKING,
            }
            changes = [c for c in changes if c.severity == severity_map[severity_filter]]

        if not changes:
            continue

        node = result.node
        click.echo(f"## {node.name}\n\n")
        click.echo(f"- **Unique ID**: `{node.unique_id}`\n")
        click.echo(f"- **Type**: {node.resource_type}\n")
        click.echo(f"- **Path**: `{node.original_file_path}`\n")

        # Summary
        if result.summary:
            summary_items = ", ".join(f"{k}: {v}" for k, v in result.summary.items())
            click.echo(f"- **Summary**: {summary_items}\n")

        click.echo("### Changes\n\n")

        # Changes list
        for change in changes:
            severity_emoji = {
                "safe": ":white_check_mark:",
                "moderate": ":warning:",
                "breaking": ":rotating_light:",
            }.get(change.severity.value, "")

            click.echo(
                f"#### {severity_emoji} {change.category.value.replace('_', ' ').title()}\n\n"
            )
            click.echo(f"{change.description}\n\n")

            # Add extra info for renames
            from dbt_osmosis.core.diff import ColumnRenamed

            if isinstance(change, ColumnRenamed):
                click.echo(f"- **Similarity**: {change.similarity_score:.1f}%\n\n")

        click.echo("---\n\n")


@test.command(context_settings=_CONTEXT)
@dbt_opts
@logging_opts
@click.argument("models", nargs=-1)
@click.option(
    "-f",
    "--fqn",
    multiple=True,
    type=click.STRING,
    help="Specify models based on dbt's FQN to analyze.",
)
@click.option(
    "--use-ai",
    is_flag=True,
    default=True,
    help=(
        "Use AI for test suggestions (enabled by default; requires OpenAI). "
        "AI failures fall back to pattern-based suggestions."
    ),
)
@click.option(
    "--pattern-only",
    is_flag=True,
    help="Use pattern-based suggestions only (no AI).",
)
@click.option(
    "--temperature",
    type=click.FLOAT,
    default=0.3,
    help="LLM temperature for AI suggestions (0.0-1.0). Default is 0.3.",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    help="Write suggestions to file instead of stdout.",
)
@click.option(
    "--format",
    type=click.Choice(["json", "yaml", "table"]),
    default="table",
    help="Output format. Default is table.",
)
def suggest(
    target: str | None = None,
    profile: str | None = None,
    project_dir: str | None = None,
    profiles_dir: str | None = None,
    vars: str | None = None,
    threads: int | None = None,
    disable_introspection: bool = False,
    fqn: tuple[str, ...] = (),
    use_ai: bool = True,
    pattern_only: bool = False,
    temperature: float = 0.3,
    output: str | None = None,
    format: str = "table",
    models: tuple[str, ...] = (),
) -> None:
    """Suggest dbt tests for models based on patterns and AI analysis.

    \f
    This command analyzes your dbt project and suggests appropriate tests for each model.
    It can use AI-powered analysis (requires OpenAI) or pattern-based analysis.

    Examples:
        dbt-osmosis test suggest
        dbt-osmosis test suggest --fqn my_project.my_model --use-ai
        dbt-osmosis test suggest --pattern-only --format json
        dbt-osmosis test suggest --output suggestions.json
    """
    logger.info(":water_wave: Executing dbt-osmosis test suggestions\n")

    settings = DbtConfiguration(
        project_dir=t.cast(str, project_dir),
        profiles_dir=t.cast(str, profiles_dir),
        target=target,
        profile=profile,
        threads=threads,
        vars=yaml_handler.safe_load(vars) if vars else {},
        disable_introspection=disable_introspection,
    )

    project = create_dbt_project_context(settings)

    # Determine if we should use AI
    use_ai_for_suggestions = use_ai and not pattern_only
    if use_ai_for_suggestions:
        click.echo(
            "AI test suggestions are enabled by default; if AI configuration fails, "
            "dbt-osmosis falls back to pattern-based suggestions.",
            err=True,
        )
    else:
        click.echo(
            "Pattern-only test suggestions enabled; AI will not be used.",
            err=True,
        )

    # Check if specific models are requested via FQN or models args
    if fqn or models:
        # Suggest tests for specific models
        from dbt.artifacts.resources.types import NodeType

        selected_nodes = []
        for node in project.manifest.nodes.values():
            if getattr(node, "resource_type", None) != NodeType.Model:
                continue

            node_fqn = ".".join(getattr(node, "fqn", []))
            node_name = getattr(node, "name", "")

            # Check if node matches any FQN or model name
            if any(f in node_fqn for f in fqn) or any(m == node_name for m in models):
                selected_nodes.append(node)

        if not selected_nodes:
            click.echo("No models found matching the specified criteria.")
            return

        results: dict[str, t.Any] = {}
        for node in selected_nodes:
            model_name = getattr(node, "name", "unknown")
            try:
                analysis = suggest_tests_for_model(
                    context=YamlRefactorContext(
                        project=project,
                        settings=YamlRefactorSettings(),
                    ),
                    node=node,
                    use_ai=use_ai_for_suggestions,
                    temperature=temperature,
                )
                results[model_name] = analysis
            except Exception as e:
                logger.error(f":x: Failed to suggest tests for {model_name}: {e}")
    else:
        # Suggest tests for all models
        try:
            context = YamlRefactorContext(
                project=project,
                settings=YamlRefactorSettings(),
            )
            results = suggest_tests_for_project(
                context=context,
                use_ai=use_ai_for_suggestions,
                temperature=temperature,
            )
        except Exception as e:
            logger.error(f":x: Failed to suggest tests: {e}")
            raise

    # Format and output results
    if format == "json":
        _output_as_json(results, output)
    elif format == "yaml":
        _output_as_yaml(results, output)
    else:
        _output_as_table(results, output)


def _output_as_json(results: dict[str, t.Any], output_path: str | None = None) -> None:
    """Output results as JSON."""
    import json

    output_data = {}
    for model_name, analysis in results.items():
        summary = analysis.get_test_summary()
        output_data[model_name] = {
            "summary": summary,
            "suggested_tests": {
                col: [
                    {
                        "test_type": t.test_type,
                        "reason": t.reason,
                        "config": t.config,
                        "confidence": t.confidence,
                    }
                    for t in tests
                ]
                for col, tests in analysis.suggested_tests.items()
            },
        }

    json_str = json.dumps(output_data, indent=2)

    if output_path:
        Path(output_path).write_text(json_str, encoding="utf-8")
        click.echo(f":white_check_mark: Wrote suggestions to: {output_path}")
    else:
        click.echo(json_str)


def _output_as_yaml(results: dict[str, t.Any], output_path: str | None = None) -> None:
    """Output results as YAML."""
    import yaml

    output_data = {}
    for model_name, analysis in results.items():
        summary = analysis.get_test_summary()
        output_data[model_name] = {
            "summary": summary,
            "suggested_tests": {
                col: [
                    {
                        "test_type": t.test_type,
                        "reason": t.reason,
                        "config": t.config,
                        "confidence": t.confidence,
                    }
                    for t in tests
                ]
                for col, tests in analysis.suggested_tests.items()
            },
        }

    yaml_str = yaml.dump(output_data, default_flow_style=False, sort_keys=False)

    if output_path:
        Path(output_path).write_text(yaml_str, encoding="utf-8")
        click.echo(f":white_check_mark: Wrote suggestions to: {output_path}")
    else:
        click.echo(yaml_str)


def _output_as_table(results: dict[str, t.Any], output_path: str | None = None) -> None:
    """Output results as a formatted table."""
    lines = []

    for model_name, analysis in results.items():
        summary = analysis.get_test_summary()
        lines.append(f"\n:file_folder: Model: {model_name}")
        lines.append(f"  Columns: {summary['total_columns']}")
        lines.append(f"  Columns with tests: {summary['columns_with_tests']}")
        lines.append(f"  Existing tests: {summary['total_existing_tests']}")
        lines.append(f"  Suggested tests: {summary['total_suggested_tests']}")

        if analysis.suggested_tests:
            lines.append("\n  :bulb: Suggested tests:")
            for col_name, suggestions in analysis.suggested_tests.items():
                lines.append(f"    - {col_name}:")
                for suggestion in suggestions:
                    conf_pct = int(suggestion.confidence * 100)
                    lines.append(f"      • {suggestion.test_type} (confidence: {conf_pct}%)")
                    if suggestion.reason:
                        lines.append(f"        Reason: {suggestion.reason}")
                    if suggestion.config:
                        lines.append(f"        Config: {suggestion.config}")

    output_text = "\n".join(lines)

    if output_path:
        Path(output_path).write_text(output_text, encoding="utf-8")
        click.echo(f":white_check_mark: Wrote suggestions to: {output_path}")
    else:
        click.echo(output_text)


@cli.group()
def lint():
    """Lint SQL code for style and anti-patterns"""


def _lint_violation_groups(
    result: LintResult,
) -> tuple[list[LintViolation], list[LintViolation], list[LintViolation]]:
    """Return lint violations grouped by error, warning, and other levels."""
    errors = result.errors
    warnings = result.warnings
    other = [
        violation
        for violation in result.violations
        if violation.level not in (LintLevel.ERROR, LintLevel.WARNING)
    ]
    return errors, warnings, other


@lint.command(context_settings=_CONTEXT, name="file")
@dbt_opts
@logging_opts
@click.argument("sql")
@click.option(
    "--rules",
    multiple=True,
    type=click.STRING,
    help="Specific rules to enable (default: all)",
)
@click.option(
    "--disable-rules",
    multiple=True,
    type=click.STRING,
    help="Specific rules to disable",
)
@click.option(
    "--dialect",
    type=click.STRING,
    help="SQL dialect (e.g., postgres, duckdb, snowflake)",
)
def lint_file(
    sql: str = "",
    project_dir: str | None = None,
    profiles_dir: str | None = None,
    target: str | None = None,
    rules: tuple[str, ...] = (),
    disable_rules: tuple[str, ...] = (),
    dialect: str | None = None,
    **kwargs: t.Any,
) -> None:
    """Lint a SQL string or file for style and anti-patterns.

    \f
    Example:
        dbt-osmosis lint file "SELECT * FROM users"
        dbt-osmosis lint file "$(cat models/my_model.sql)" --rules keyword-case line-length

    This command analyzes SQL code for style issues, anti-patterns, and potential bugs.
    """
    logger.info(":water_wave: Executing dbt-osmosis SQL linting\n")
    settings = DbtConfiguration(
        project_dir=t.cast(str, project_dir),
        profiles_dir=t.cast(str, profiles_dir),
        target=target,
        **kwargs,
    )
    project = create_dbt_project_context(settings)

    # Use provided dialect or get from adapter
    sql_dialect = dialect or project.adapter.type()

    # Prepare rules list
    enabled_rules = list(rules) if rules else None
    disabled_rules = list(disable_rules) if disable_rules else None

    # Lint the SQL
    result = lint_sql_code(
        context=project,
        raw_sql=sql,
        dialect=sql_dialect,
        rules=enabled_rules,
        disabled_rules=disabled_rules,
    )

    # Display results
    click.echo(f"\n:sparkles: Lint Results: {result.summary()}\n")

    if result.violations:
        # Group by level
        errors, warnings, other = _lint_violation_groups(result)

        if errors:
            click.echo(":no_entry: Errors:")
            for violation in errors:
                click.echo(f"  {violation}")
            click.echo()

        if warnings:
            click.echo(":warning: Warnings:")
            for violation in warnings:
                click.echo(f"  {violation}")
            click.echo()

        if other:
            click.echo(":information_source: Other:")
            for violation in other:
                click.echo(f"  {violation}")
            click.echo()

        # Exit with error code if there are errors or warnings
        if errors or warnings:
            exit(1)
    else:
        click.echo(":white_check_mark: No issues found!")


@lint.command(context_settings=_CONTEXT, name="model")
@dbt_opts
@logging_opts
@click.argument("model_name")
@click.option(
    "--rules",
    multiple=True,
    type=click.STRING,
    help="Specific rules to enable (default: all)",
)
@click.option(
    "--disable-rules",
    multiple=True,
    type=click.STRING,
    help="Specific rules to disable",
)
@click.option(
    "--dialect",
    type=click.STRING,
    help="SQL dialect (e.g., postgres, duckdb, snowflake)",
)
def lint_model_command(
    model_name: str = "",
    project_dir: str | None = None,
    profiles_dir: str | None = None,
    target: str | None = None,
    rules: tuple[str, ...] = (),
    disable_rules: tuple[str, ...] = (),
    dialect: str | None = None,
    **kwargs: t.Any,
) -> None:
    """Lint a dbt model's SQL code.

    \f
    Example:
        dbt-osmosis lint model my_model
        dbt-osmosis lint model my_model --rules keyword-case select-star

    This command analyzes a dbt model's SQL for style issues, anti-patterns, and potential bugs.
    """
    logger.info(":water_wave: Executing dbt-osmosis SQL linting\n")
    settings = DbtConfiguration(
        project_dir=t.cast(str, project_dir),
        profiles_dir=t.cast(str, profiles_dir),
        target=target,
        **kwargs,
    )
    project = create_dbt_project_context(settings)

    # Use provided dialect or get from adapter
    sql_dialect = dialect or project.adapter.type()

    # Create linter
    enabled_rules = list(rules) if rules else None
    disabled_rules = list(disable_rules) if disable_rules else None
    linter = SQLLinter(
        dialect=sql_dialect,
        enabled_rules=enabled_rules,
        disabled_rules=disabled_rules,
    )

    # Lint the model
    result = linter.lint_model(project, model_name)

    # Display results
    click.echo(f"\n:sparkles: Lint Results for {model_name}: {result.summary()}\n")

    if result.violations:
        # Group by level
        errors, warnings, other = _lint_violation_groups(result)

        if errors:
            click.echo(":no_entry: Errors:")
            for violation in errors:
                click.echo(f"  {violation}")
            click.echo()

        if warnings:
            click.echo(":warning: Warnings:")
            for violation in warnings:
                click.echo(f"  {violation}")
            click.echo()

        if other:
            click.echo(":information_source: Other:")
            for violation in other:
                click.echo(f"  {violation}")
            click.echo()

        # Exit with error code if there are errors or warnings
        if errors or warnings:
            exit(1)
    else:
        click.echo(":white_check_mark: No issues found!")


@lint.command(context_settings=_CONTEXT, name="project")
@dbt_opts
@logging_opts
@click.option(
    "-f",
    "--fqn",
    multiple=True,
    type=click.STRING,
    help="Filter models by FQN pattern",
)
@click.option(
    "--rules",
    multiple=True,
    type=click.STRING,
    help="Specific rules to enable (default: all)",
)
@click.option(
    "--disable-rules",
    multiple=True,
    type=click.STRING,
    help="Specific rules to disable",
)
@click.option(
    "--dialect",
    type=click.STRING,
    help="SQL dialect (e.g., postgres, duckdb, snowflake)",
)
def lint_project_command(
    project_dir: str | None = None,
    profiles_dir: str | None = None,
    target: str | None = None,
    fqn: tuple[str, ...] = (),
    rules: tuple[str, ...] = (),
    disable_rules: tuple[str, ...] = (),
    dialect: str | None = None,
    **kwargs: t.Any,
) -> None:
    """Lint all models in a dbt project.

    \f
    Example:
        dbt-osmosis lint project
        dbt-osmosis lint project --fqn my_project.staging
        dbt-osmosis lint project --rules keyword-case select-star

    This command analyzes all dbt models' SQL for style issues, anti-patterns, and potential bugs.
    """
    logger.info(":water_wave: Executing dbt-osmosis SQL linting\n")
    settings = DbtConfiguration(
        project_dir=t.cast(str, project_dir),
        profiles_dir=t.cast(str, profiles_dir),
        target=target,
        **kwargs,
    )
    project = create_dbt_project_context(settings)

    # Use provided dialect or get from adapter
    sql_dialect = dialect or project.adapter.type()

    # Create linter
    enabled_rules = list(rules) if rules else None
    disabled_rules = list(disable_rules) if disable_rules else None
    linter = SQLLinter(
        dialect=sql_dialect,
        enabled_rules=enabled_rules,
        disabled_rules=disabled_rules,
    )

    # Lint the project
    fqn_filter = list(fqn) if fqn else None
    results = linter.lint_project(project, fqn_filter=fqn_filter)

    # Display results
    grouped_results = {name: _lint_violation_groups(result) for name, result in results.items()}
    total_errors = sum(len(errors) for errors, _, _ in grouped_results.values())
    total_warnings = sum(len(warnings) for _, warnings, _ in grouped_results.values())
    total_other = sum(len(other) for _, _, other in grouped_results.values())

    click.echo(f"\n:sparkles: Lint Results for {len(results)} models\n")
    click.echo(
        f"  Total: {total_errors} error(s), {total_warnings} warning(s), {total_other} info\n"
    )

    # Show models with issues
    models_with_issues = {name: r for name, r in results.items() if r.violations}

    if models_with_issues:
        for model_name, result in models_with_issues.items():
            click.echo(f"\n:page_facing_up: {model_name} ({result.summary()})")
            errors, warnings, other = grouped_results[model_name]

            if errors:
                for violation in errors:
                    click.echo(f"  :no_entry: {violation}")

            if warnings:
                for violation in warnings:
                    click.echo(f"  :warning: {violation}")

            if other:
                for violation in other:
                    click.echo(f"  :information_source: {violation}")

        # Exit with error code if there are errors or warnings
        if total_errors or total_warnings:
            exit(1)
    else:
        click.echo(":white_check_mark: No issues found across all models!")


if __name__ == "__main__":
    cli()
