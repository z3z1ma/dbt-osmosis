# pyright: reportUnreachable=false

from __future__ import annotations

import dataclasses
import functools
import importlib
import os
import shutil
import subprocess
import sys
import threading
import typing as t
from pathlib import Path

import click
import yaml as yaml_handler

from dbt_osmosis.core import logger
from dbt_osmosis.core.config import (
    DbtConfiguration,
    DbtProjectContext,
    create_dbt_project_context,
    discover_profiles_dir,
    discover_project_dir,
)
from dbt_osmosis.core.diff import SchemaDiff
from dbt_osmosis.core.discovery import (
    DiscoveryResult,
    discover_undocumented_columns,
    discover_undocumented_models,
)
from dbt_osmosis.core.generators import (
    DocumentationCheckResult,
    check_documentation,
    generate_sources_from_database,
    generate_staging_from_source,
)
from dbt_osmosis.core.llm import generate_dbt_model_from_nl, generate_sql_from_nl
from dbt_osmosis.core.migration import MigrationPlan, MigrationPlanner
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
from dbt_osmosis.core.validation import (
    ModelValidationStatus,
    ValidationReport,
    validate_models,
)
from dbt_osmosis.core.voice_learning import (
    ProjectStyleProfile,
    analyze_project_documentation_style,
)

T = t.TypeVar("T")
P = t.ParamSpec("P")

_CONTEXT = {"max_content_width": 800}
_WORKBENCH_EXTRA_HINT = "pip install dbt-osmosis[workbench]"


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


def _record_missing_workbench_module(
    missing: list[str],
    module: str,
    error: ImportError,
) -> None:
    if isinstance(error, ModuleNotFoundError) and error.name == module:
        missing.append(module)
    else:
        missing.append(f"{module} ({error})")


def _check_workbench_app_dependencies() -> None:
    missing: list[str] = []
    try:
        importlib.import_module("feedparser")
    except ImportError as e:
        _record_missing_workbench_module(missing, "feedparser", e)
    try:
        importlib.import_module("pandas")
    except ImportError as e:
        _record_missing_workbench_module(missing, "pandas", e)
    try:
        importlib.import_module("streamlit")
    except ImportError as e:
        _record_missing_workbench_module(missing, "streamlit", e)
    try:
        importlib.import_module("streamlit_elements_fluence")
    except ImportError as e:
        _record_missing_workbench_module(missing, "streamlit_elements_fluence", e)
    try:
        importlib.import_module("ydata_profiling")
    except ImportError as e:
        _record_missing_workbench_module(missing, "ydata_profiling", e)

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
            check=False,
        )
    except FileNotFoundError as e:
        raise _missing_streamlit_error() from e


@click.group()
@click.version_option()
def cli() -> None:
    """dbt-osmosis is a CLI tool for dbt that helps you manage, document, and organize your dbt yaml files"""


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
        default=None,
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
        wrapper_kwargs = t.cast("dict[str, t.Any]", kwargs)
        wrapper_kwargs["profiles_dir"] = _resolve_profiles_dir(
            project_dir=t.cast(str | None, wrapper_kwargs.get("project_dir")),
            profiles_dir=t.cast(str | None, wrapper_kwargs.get("profiles_dir")),
        )
        return func(*args, **wrapper_kwargs)

    return wrapper


def _resolve_profiles_dir(
    project_dir: str | None,
    profiles_dir: str | None,
) -> str:
    if profiles_dir is not None:
        return profiles_dir
    return discover_profiles_dir(project_dir)


def _create_cli_project_context(
    project_dir: str | None,
    profiles_dir: str | None,
    target: str | None,
    **kwargs: t.Any,
) -> DbtProjectContext:
    settings = DbtConfiguration(
        project_dir=t.cast(str, project_dir),
        profiles_dir=t.cast(str, profiles_dir),
        target=target,
        **kwargs,
    )
    return create_dbt_project_context(settings)


def _parsed_cli_vars(vars_value: str | None) -> dict[str, t.Any]:
    parsed = yaml_handler.safe_load(vars_value) if vars_value else {}
    if parsed is None:
        return {}
    if not isinstance(parsed, dict):
        raise click.ClickException("--vars must parse to a YAML mapping.")
    return t.cast("dict[str, t.Any]", parsed)


def _create_yaml_refactor_context(
    *, project: DbtProjectContext, settings: YamlRefactorSettings
) -> YamlRefactorContext:
    """Translate explicit catalog input errors for every CLI context consumer."""
    from dbt_osmosis.core.exceptions import CatalogValidationError

    try:
        return YamlRefactorContext(project=project, settings=settings)
    except CatalogValidationError as exc:
        raise click.ClickException(str(exc)) from exc


def _create_cli_yaml_context(
    *,
    project_dir: str | None,
    profiles_dir: str | None,
    target: str | None,
    profile: str | None = None,
    threads: int | None = None,
    vars_value: str | None = None,
    disable_introspection: bool = False,
    fqn: tuple[str, ...] = (),
    models: tuple[str, ...] = (),
    include_external: bool = False,
    catalog_path: str | None = None,
) -> YamlRefactorContext:
    settings = DbtConfiguration(
        project_dir=t.cast(str, project_dir),
        profiles_dir=t.cast(str, profiles_dir),
        target=target,
        profile=profile,
        threads=threads,
        vars=_parsed_cli_vars(vars_value),
        disable_introspection=disable_introspection,
    )
    return _create_yaml_refactor_context(
        project=create_dbt_project_context(settings),
        settings=YamlRefactorSettings(
            create_catalog_if_not_exists=False,
            fqn=list(fqn),
            models=list(models),
            include_external=include_external,
            catalog_path=catalog_path,
        ),
    )


def _write_or_echo(text: str, output_path: str | None, *, label: str = "output") -> None:
    if output_path:
        Path(output_path).write_text(text, encoding="utf-8")
        click.echo(f":white_check_mark: Wrote {label} to: {output_path}")
        return
    click.echo(text)


def _json_text(data: t.Any) -> str:
    import json

    return json.dumps(data, indent=2, default=str)


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

    with _create_yaml_refactor_context(
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
            sys.exit(1)


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

    with _create_yaml_refactor_context(
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
            sys.exit(1)


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

    with _create_yaml_refactor_context(
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
            sys.exit(1)


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


def _node_columns(node: t.Any) -> list[str]:
    return list(node.columns.keys()) if hasattr(node, "columns") else []


def _available_sources_from_manifest(project: t.Any) -> list[dict[str, t.Any]]:
    available_sources: list[dict[str, t.Any]] = []
    for node in project.manifest.nodes.values():
        if getattr(node, "resource_type", None) == "model":
            available_sources.append({
                "name": node.name,
                "type": "model",
                "description": getattr(node, "description", ""),
                "columns": _node_columns(node),
            })

    for source in project.manifest.sources.values():
        if getattr(source, "resource_type", None) == "source":
            available_sources.append({
                "name": f"{source.source_name}.{source.name}",
                "type": "source",
                "description": getattr(source, "description", ""),
                "columns": _node_columns(source),
            })
    return available_sources


def _log_available_sources(available_sources: list[dict[str, t.Any]]) -> None:
    logger.info(f":crystal_ball: Found {len(available_sources)} available sources/models")


def _model_sql_content(model_spec: dict[str, t.Any]) -> str:
    return (
        f"-- {model_spec['description']}\n"
        f"-- Materialized: {model_spec['materialized']}\n\n"
        f"{model_spec['sql']}"
    )


def _generated_model_sql_path(
    project: t.Any,
    project_dir: str | None,
    model_spec: dict[str, t.Any],
    output_path: str | None,
) -> Path:
    project_root = _get_generated_project_root(project, project_dir)
    if output_path is not None:
        return _resolve_generated_file_path(output_path, project_root)
    return _resolve_generated_file_path(
        project_root / "models" / f"{model_spec['model_name']}.sql",
        project_root,
    )


def _prepare_model_generation_outputs(
    project: t.Any,
    project_dir: str | None,
    model_spec: dict[str, t.Any],
    output_path: str | None,
    schema_yml: str | None,
    overwrite: bool,
    dry_run: bool,
) -> tuple[str, Path, tuple[Path, dict[str, t.Any]]]:
    sql_content = _model_sql_content(model_spec)
    output_path_obj = _generated_model_sql_path(project, project_dir, model_spec, output_path)
    schema_path = schema_yml or output_path_obj.parent / f"{model_spec['model_name']}.yml"
    schema_write = _prepare_generated_yaml_write(
        project=project,
        project_dir=project_dir,
        yaml_path=schema_path,
        yaml_data=_model_schema_data(model_spec),
        overwrite=overwrite,
        dry_run=dry_run,
    )
    return sql_content, output_path_obj, schema_write


def _echo_generated_model_header(model_spec: dict[str, t.Any]) -> None:
    click.echo(f"\n:sparkles: Generated model: {model_spec['model_name']}")
    click.echo(f"Description: {model_spec['description']}")
    click.echo(f"Materialized: {model_spec['materialized']}")


def _echo_model_dry_run(
    model_spec: dict[str, t.Any],
    sql_content: str,
    output_path: Path,
    schema_write: tuple[Path, dict[str, t.Any]],
    overwrite: bool,
) -> None:
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
    _echo_planned_writes([output_path, schema_write[0]])


def _write_model_outputs(
    sql_content: str,
    output_path: Path,
    schema_write: tuple[Path, dict[str, t.Any]],
    overwrite: bool,
) -> None:
    _write_prepared_generated_yaml(schema_write, overwrite=overwrite)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(sql_content)
    click.echo(f"\n:white_check_mark: Wrote SQL to: {output_path}")
    click.echo(f":white_check_mark: Wrote schema.yml to: {schema_write[0]}")


def _run_model_generation(
    project: t.Any,
    project_dir: str | None,
    query: str,
    model_name: str | None,
    output_path: str | None,
    schema_yml: str | None,
    dry_run: bool,
    overwrite: bool,
) -> None:
    available_sources = _available_sources_from_manifest(project)
    _log_available_sources(available_sources)

    try:
        model_spec = generate_dbt_model_from_nl(query, available_sources)
    except Exception as e:
        logger.error(f":x: Failed to generate model: {e}")
        raise

    if model_name:
        model_spec["model_name"] = model_name

    _echo_generated_model_header(model_spec)
    sql_content, output_path_obj, schema_write = _prepare_model_generation_outputs(
        project,
        project_dir,
        model_spec,
        output_path,
        schema_yml,
        overwrite,
        dry_run,
    )

    if dry_run:
        _echo_model_dry_run(model_spec, sql_content, output_path_obj, schema_write, overwrite)
        return

    _write_model_outputs(sql_content, output_path_obj, schema_write, overwrite)


def _echo_generated_sql(sql: str) -> None:
    click.echo("\n" + "=" * 80)
    click.echo("Generated SQL:")
    click.echo("=" * 80)
    click.echo(sql)


def _print_table(table: t.Any) -> None:
    table.print_table(
        max_rows=50,
        max_columns=6,
        output=sys.stdout,
        max_column_width=20,
        locale=None,
        max_precision=3,
    )


def _run_sql_generation(project: t.Any, query: str, execute: bool) -> None:
    available_sources = _available_sources_from_manifest(project)
    _log_available_sources(available_sources)

    try:
        sql = generate_sql_from_nl(query, available_sources)
    except Exception as e:
        logger.error(f":x: Failed to generate SQL: {e}")
        raise

    _echo_generated_sql(sql)
    if execute:
        click.echo("\n" + "=" * 80)
        click.echo("Executing SQL...")
        click.echo("=" * 80)
        _, table = execute_sql_code(project, sql)
        _print_table(table)


def _prepare_staging_outputs(
    project: t.Any,
    project_dir: str | None,
    result: t.Any,
    overwrite: bool,
    dry_run: bool,
) -> tuple[tuple[Path, dict[str, t.Any]] | None, Path | None]:
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

    sql_path = None
    if result.sql_content and result.sql_path:
        sql_path = _resolve_generated_file_path(
            result.sql_path,
            _get_generated_project_root(project, project_dir),
        )
    return yaml_write, sql_path


def _echo_staging_dry_run(
    result: t.Any,
    yaml_write: tuple[Path, dict[str, t.Any]] | None,
    sql_path: Path | None,
) -> None:
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
    _echo_planned_writes([sql_path, yaml_write[0] if yaml_write is not None else None])


def _write_staging_outputs(
    result: t.Any,
    yaml_write: tuple[Path, dict[str, t.Any]] | None,
    sql_path: Path | None,
    overwrite: bool,
) -> None:
    click.echo(f"\n:sparkles: Generated staging model: {result.staging_name}")

    if result.sql_content and sql_path:
        sql_path.parent.mkdir(parents=True, exist_ok=True)
        sql_path.write_text(result.sql_content, encoding="utf-8")
        click.echo(f":white_check_mark: Wrote SQL to: {sql_path}")
    elif result.sql_content:
        raise click.ClickException("Generated SQL content is missing a target path.")

    if result.yaml_content and yaml_write is not None:
        _write_prepared_generated_yaml(yaml_write, overwrite=overwrite)
        click.echo(f":white_check_mark: Wrote YAML to: {yaml_write[0]}")
    elif result.yaml_content:
        raise click.ClickException("Generated YAML content is missing a target path.")


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
    project = _create_cli_project_context(project_dir, profiles_dir, target, **kwargs)
    _run_model_generation(
        project,
        project_dir,
        query,
        model_name,
        output_path,
        schema_yml,
        dry_run,
        overwrite,
    )


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
    project = _create_cli_project_context(project_dir, profiles_dir, target, **kwargs)

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
    project = _create_cli_project_context(project_dir, profiles_dir, target, **kwargs)

    try:
        result = generate_staging_from_source(
            context=project,
            source_name=source_name,
            table_name=table_name,
            use_ai=ai,
            staging_path=Path(staging_path) if staging_path else None,
        )

        yaml_write, resolved_sql_path = _prepare_staging_outputs(
            project,
            project_dir,
            result,
            overwrite,
            dry_run,
        )

        if dry_run:
            _echo_staging_dry_run(result, yaml_write, resolved_sql_path)
            return

        _write_staging_outputs(result, yaml_write, resolved_sql_path, overwrite)

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
    project = _create_cli_project_context(project_dir, profiles_dir, target, **kwargs)
    _run_sql_generation(project, query, execute)


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
    project = _create_cli_project_context(project_dir, profiles_dir, target, **kwargs)
    _run_model_generation(
        project,
        project_dir,
        query,
        model_name,
        output_path,
        schema_yml,
        dry_run,
        overwrite,
    )


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
    project = _create_cli_project_context(project_dir, profiles_dir, target, **kwargs)
    _run_sql_generation(project, query, execute)


@cli.command(
    context_settings={
        "ignore_unknown_options": True,
        "allow_extra_args": True,
    }
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
    default=None,
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
    profiles_dir = _resolve_profiles_dir(project_dir, profiles_dir)

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
    project = _create_cli_project_context(project_dir, profiles_dir, target, **kwargs)
    _, table = execute_sql_code(project, sql)

    t.cast("t.Any", table).print_table(
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
    project = _create_cli_project_context(project_dir, profiles_dir, target, **kwargs)
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

    with _create_yaml_refactor_context(
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


def _output_diff_json(results: dict[str, t.Any], severity_filter: str) -> None:
    """Output diff results in JSON format."""
    import json
    from datetime import datetime, timezone

    nodes: list[dict[str, object]] = []
    output: dict[str, object] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
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


def _diff_changes_for_severity(result: t.Any, severity_filter: str) -> list[t.Any]:
    changes = result.changes
    if severity_filter == "all":
        return list(changes)

    from dbt_osmosis.core.diff import ChangeSeverity

    severity_map = {
        "safe": ChangeSeverity.SAFE,
        "moderate": ChangeSeverity.MODERATE,
        "breaking": ChangeSeverity.BREAKING,
    }
    return [change for change in changes if change.severity == severity_map[severity_filter]]


def _diff_change_counts(results: dict[str, t.Any]) -> tuple[int, int, int]:
    breaking_count = sum(
        1
        for result in results.values()
        for change in result.changes
        if change.severity.value == "breaking"
    )
    moderate_count = sum(
        1
        for result in results.values()
        for change in result.changes
        if change.severity.value == "moderate"
    )
    safe_count = sum(
        1
        for result in results.values()
        for change in result.changes
        if change.severity.value == "safe"
    )
    return breaking_count, moderate_count, safe_count


def _echo_diff_text_result(result: t.Any, changes: list[t.Any]) -> None:
    from dbt_osmosis.core.diff import ColumnRenamed

    node = result.node
    click.echo(f":page_facing_up: {node.name} ({node.resource_type})")
    click.echo(f"   Unique ID: {node.unique_id}")
    click.echo(f"   Path: {node.original_file_path}")

    if result.summary:
        click.echo(f"   Summary: {', '.join(f'{k}: {v}' for k, v in result.summary.items())}")

    for change in changes:
        click.echo(f"\n   {change}")

    for change in changes:
        if isinstance(change, ColumnRenamed):
            click.echo(f"      Similarity: {change.similarity_score:.1f}%")

    click.echo("\n" + "-" * 80 + "\n")


def _echo_diff_text_summary(results: dict[str, t.Any]) -> None:
    breaking_count, moderate_count, safe_count = _diff_change_counts(results)
    click.echo("Overall Summary:")
    click.echo(f"  Breaking changes: {breaking_count}")
    click.echo(f"  Moderate changes: {moderate_count}")
    click.echo(f"  Safe changes: {safe_count}")

    if breaking_count > 0:
        click.echo("\n:rotating_light: Breaking changes detected. Review required before applying.")


def _severity_emoji(severity_value: str) -> str:
    return {
        "safe": ":white_check_mark:",
        "moderate": ":warning:",
        "breaking": ":rotating_light:",
    }.get(severity_value, "")


def _echo_diff_markdown_change(change: t.Any) -> None:
    from dbt_osmosis.core.diff import ColumnRenamed

    click.echo(
        f"#### {_severity_emoji(change.severity.value)} "
        f"{change.category.value.replace('_', ' ').title()}\n\n"
    )
    click.echo(f"{change.description}\n\n")
    if isinstance(change, ColumnRenamed):
        click.echo(f"- **Similarity**: {change.similarity_score:.1f}%\n\n")


def _echo_diff_markdown_result(result: t.Any, changes: list[t.Any]) -> None:
    node = result.node
    click.echo(f"## {node.name}\n\n")
    click.echo(f"- **Unique ID**: `{node.unique_id}`\n")
    click.echo(f"- **Type**: {node.resource_type}\n")
    click.echo(f"- **Path**: `{node.original_file_path}`\n")

    if result.summary:
        summary_items = ", ".join(f"{k}: {v}" for k, v in result.summary.items())
        click.echo(f"- **Summary**: {summary_items}\n")

    click.echo("### Changes\n\n")
    for change in changes:
        _echo_diff_markdown_change(change)
    click.echo("---\n\n")


def _iter_diff_results_with_changes(
    results: dict[str, t.Any],
    severity_filter: str,
) -> t.Iterator[tuple[t.Any, list[t.Any]]]:
    for result in results.values():
        changes = _diff_changes_for_severity(result, severity_filter)
        if changes:
            yield result, changes


def _output_diff_text(results: dict[str, t.Any], severity_filter: str) -> None:
    """Output diff results in human-readable text format."""
    if not results:
        click.echo(":white_check_mark: No schema changes detected")
        return

    total_changes = sum(len(r.changes) for r in results.values())
    click.echo(f":warning: Detected {total_changes} schema changes across {len(results)} node(s)\n")

    for result, changes in _iter_diff_results_with_changes(results, severity_filter):
        _echo_diff_text_result(result, changes)

    _echo_diff_text_summary(results)


def _output_diff_markdown(results: dict[str, t.Any], severity_filter: str) -> None:
    """Output diff results in Markdown format."""
    if not results:
        click.echo("## Schema Diff Results\n\n:white_check_mark: No schema changes detected")
        return

    total_changes = sum(len(r.changes) for r in results.values())
    click.echo(
        f"# Schema Diff Results\n\n**Detected {total_changes} changes across {len(results)} node(s)**\n"
    )

    for result, changes in _iter_diff_results_with_changes(results, severity_filter):
        _echo_diff_markdown_result(result, changes)


@cli.group()
def migration():
    """Plan database migrations from schema diffs"""


def _filtered_diff_results(
    results: dict[str, t.Any],
    severity_filter: str,
) -> dict[str, t.Any]:
    if severity_filter == "all":
        return results
    filtered_results = {}
    for node_id, result in results.items():
        changes = _diff_changes_for_severity(result, severity_filter)
        if changes:
            filtered_results[node_id] = dataclasses.replace(result, changes=changes)
    return filtered_results


def _migration_plan_summary(plan: MigrationPlan) -> dict[str, t.Any]:
    return {
        "node_id": plan.node_id,
        "node_name": plan.node_name,
        "total_steps": len(plan.steps),
        "safe_steps": len(plan.safe_steps),
        "breaking_steps": len(plan.breaking_steps),
    }


def _migration_plans_with_steps(plans: dict[str, MigrationPlan]) -> dict[str, MigrationPlan]:
    return {node_id: plan for node_id, plan in plans.items() if plan.steps}


def _render_migration_plans(
    plans: dict[str, MigrationPlan],
    output_format: str,
    *,
    include_rollback: bool,
) -> str:
    planned = _migration_plans_with_steps(plans)
    if output_format == "json":
        return _json_text({
            "total_nodes": len(planned),
            "plans": {node_id: plan.to_dict() for node_id, plan in planned.items()},
        })

    if not planned:
        return ":white_check_mark: No migration steps generated"

    if output_format == "markdown":
        summary = "\n".join(
            f"- `{plan.node_name}`: {len(plan.steps)} step(s), {len(plan.breaking_steps)} breaking"
            for plan in planned.values()
        )
        plan_sections = "\n\n".join(plan.to_markdown() for plan in planned.values())
        return f"# Migration Plans\n\n## Summary\n\n{summary}\n\n{plan_sections}"

    header = "\n".join([
        "-- dbt-osmosis migration plans",
        *[
            f"-- {item['node_name']}: {item['total_steps']} step(s), "
            f"{item['breaking_steps']} breaking"
            for item in (_migration_plan_summary(plan) for plan in planned.values())
        ],
        "",
    ])
    return header + "\n\n".join(
        plan.to_sql(include_rollback=include_rollback) for plan in planned.values()
    )


@migration.command(context_settings=_CONTEXT, name="plan")
@dbt_opts
@logging_opts
@click.argument("models", nargs=-1)
@click.option(
    "-f",
    "--fqn",
    multiple=True,
    type=click.STRING,
    help="Filter models by dbt fully qualified name.",
)
@click.option(
    "--profile",
    type=click.STRING,
    help="Which profile to load. Overrides setting in dbt_project.yml.",
)
@click.option(
    "--vars",
    type=click.STRING,
    help="Supply project variables as a YAML mapping.",
)
@click.option(
    "--catalog-path",
    type=click.Path(exists=True),
    help="Read database columns from a catalog.json file instead of querying the warehouse.",
)
@click.option(
    "--disable-introspection",
    is_flag=True,
    help="Load the project without live database introspection.",
)
@click.option(
    "--include-external",
    is_flag=True,
    help="Include models and sources from external dbt packages.",
)
@click.option(
    "--output-format",
    type=click.Choice(["sql", "json", "markdown"], case_sensitive=False),
    default="sql",
    help="Output format for generated migration plans.",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    help="Write the migration plan to a file instead of stdout.",
)
@click.option(
    "--severity",
    type=click.Choice(["safe", "moderate", "breaking", "all"], case_sensitive=False),
    default="all",
    help="Plan only schema changes at this severity.",
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
    "--include-rollback/--no-rollback",
    default=True,
    help="Include rollback SQL comments in SQL output.",
)
def migration_plan_command(
    target: str | None = None,
    profile: str | None = None,
    project_dir: str | None = None,
    profiles_dir: str | None = None,
    vars: str | None = None,
    threads: int | None = None,
    disable_introspection: bool = False,
    fqn: tuple[str, ...] = (),
    catalog_path: str | None = None,
    include_external: bool = False,
    output_format: str = "sql",
    output: str | None = None,
    severity: str = "all",
    fuzzy_match_threshold: float = 85.0,
    detect_column_renames: bool = True,
    include_rollback: bool = True,
    models: tuple[str, ...] = (),
) -> None:
    """Generate migration SQL, JSON, or Markdown from schema diff results."""
    logger.info(":water_wave: Executing dbt-osmosis migration planning\n")
    with _create_cli_yaml_context(
        project_dir=project_dir,
        profiles_dir=profiles_dir,
        target=target,
        profile=profile,
        threads=threads,
        vars_value=vars,
        disable_introspection=disable_introspection,
        fqn=fqn,
        models=models,
        include_external=include_external,
        catalog_path=catalog_path,
    ) as context:
        differ = SchemaDiff(
            t.cast(t.Any, context),
            fuzzy_match_threshold=fuzzy_match_threshold,
            detect_column_renames=detect_column_renames,
        )
        results = _filtered_diff_results(differ.compare_all(), severity)
        plans = MigrationPlanner(t.cast(t.Any, context)).plan_for_results(results)

    rendered = _render_migration_plans(plans, output_format, include_rollback=include_rollback)
    _write_or_echo(rendered, output, label="migration plan")


def _suggestion_ai_enabled(use_ai: bool, pattern_only: bool) -> bool:
    return use_ai and not pattern_only


def _echo_suggestion_mode(use_ai_for_suggestions: bool) -> None:
    if use_ai_for_suggestions:
        click.echo(
            "AI test suggestions are enabled by default; if AI configuration fails, "
            "dbt-osmosis falls back to pattern-based suggestions.",
            err=True,
        )
    else:
        click.echo("Pattern-only test suggestions enabled; AI will not be used.", err=True)


def _selected_test_nodes(
    project: t.Any, fqn: tuple[str, ...], models: tuple[str, ...]
) -> list[t.Any]:
    from dbt.artifacts.resources.types import NodeType

    selected_nodes = []
    for node in project.manifest.nodes.values():
        if getattr(node, "resource_type", None) != NodeType.Model:
            continue
        node_fqn = ".".join(getattr(node, "fqn", []))
        node_name = getattr(node, "name", "")
        if any(selector in node_fqn for selector in fqn) or any(
            model == node_name for model in models
        ):
            selected_nodes.append(node)
    return selected_nodes


def _suggest_tests_for_nodes(
    project: t.Any,
    selected_nodes: list[t.Any],
    use_ai_for_suggestions: bool,
    temperature: float,
) -> dict[str, t.Any]:
    results: dict[str, t.Any] = {}
    for node in selected_nodes:
        model_name = getattr(node, "name", "unknown")
        try:
            analysis = suggest_tests_for_model(
                context=_create_yaml_refactor_context(
                    project=project, settings=YamlRefactorSettings()
                ),
                node=node,
                use_ai=use_ai_for_suggestions,
                temperature=temperature,
            )
            results[model_name] = analysis
        except Exception as e:  # noqa: BLE001
            logger.error(f":x: Failed to suggest tests for {model_name}: {e}")
    return results


def _suggestion_results(
    project: t.Any,
    fqn: tuple[str, ...],
    models: tuple[str, ...],
    use_ai_for_suggestions: bool,
    temperature: float,
) -> dict[str, t.Any] | None:
    if fqn or models:
        selected_nodes = _selected_test_nodes(project, fqn, models)
        if not selected_nodes:
            click.echo("No models found matching the specified criteria.")
            return None
        return _suggest_tests_for_nodes(
            project, selected_nodes, use_ai_for_suggestions, temperature
        )

    try:
        context = _create_yaml_refactor_context(project=project, settings=YamlRefactorSettings())
        return suggest_tests_for_project(
            context=context,
            use_ai=use_ai_for_suggestions,
            temperature=temperature,
        )
    except Exception as e:
        logger.error(f":x: Failed to suggest tests: {e}")
        raise


def _output_suggestion_results(
    results: dict[str, t.Any], format_name: str, output: str | None
) -> None:
    if format_name == "json":
        _output_as_json(results, output)
    elif format_name == "yaml":
        _output_as_yaml(results, output)
    else:
        _output_as_table(results, output)


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

    use_ai_for_suggestions = _suggestion_ai_enabled(use_ai, pattern_only)
    _echo_suggestion_mode(use_ai_for_suggestions)
    results = _suggestion_results(project, fqn, models, use_ai_for_suggestions, temperature)
    if results is not None:
        _output_suggestion_results(results, format, output)


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


def _suggestion_table_lines(col_name: str, suggestions: t.Iterable[t.Any]) -> list[str]:
    lines = [f"    - {col_name}:"]
    for suggestion in suggestions:
        conf_pct = int(suggestion.confidence * 100)
        lines.append(f"      • {suggestion.test_type} (confidence: {conf_pct}%)")
        if suggestion.reason:
            lines.append(f"        Reason: {suggestion.reason}")
        if suggestion.config:
            lines.append(f"        Config: {suggestion.config}")
    return lines


def _analysis_table_lines(model_name: str, analysis: t.Any) -> list[str]:
    summary = analysis.get_test_summary()
    lines = [
        f"\n:file_folder: Model: {model_name}",
        f"  Columns: {summary['total_columns']}",
        f"  Columns with tests: {summary['columns_with_tests']}",
        f"  Existing tests: {summary['total_existing_tests']}",
        f"  Suggested tests: {summary['total_suggested_tests']}",
    ]
    if analysis.suggested_tests:
        lines.append("\n  :bulb: Suggested tests:")
        for col_name, suggestions in analysis.suggested_tests.items():
            lines.extend(_suggestion_table_lines(col_name, suggestions))
    return lines


def _output_as_table(results: dict[str, t.Any], output_path: str | None = None) -> None:
    """Output results as a formatted table."""
    lines = []

    for model_name, analysis in results.items():
        lines.extend(_analysis_table_lines(model_name, analysis))

    output_text = "\n".join(lines)

    if output_path:
        Path(output_path).write_text(output_text, encoding="utf-8")
        click.echo(f":white_check_mark: Wrote suggestions to: {output_path}")
    else:
        click.echo(output_text)


@cli.group()
def validate():
    """Validate dbt models without materializing them"""


def _selected_model_nodes(context: YamlRefactorContext) -> list[tuple[str, t.Any]]:
    from dbt.artifacts.resources.types import NodeType

    from dbt_osmosis.core.node_filters import _iter_candidate_nodes

    return [
        (uid, node)
        for uid, node in _iter_candidate_nodes(context)
        if getattr(node, "resource_type", None) == NodeType.Model
    ]


def _validation_result_dict(result: t.Any) -> dict[str, t.Any]:
    return {
        "model_name": result.model_name,
        "unique_id": result.unique_id,
        "status": result.status.value
        if isinstance(result.status, ModelValidationStatus)
        else str(result.status),
        "error_message": result.error_message,
        "execution_time_seconds": result.execution_time_seconds,
        "row_count": result.row_count,
        "bytes_processed": result.bytes_processed,
    }


def _validation_report_dict(report: ValidationReport) -> dict[str, t.Any]:
    return {
        "summary": {
            "total_models": report.total_models,
            "successful": report.successful,
            "failed": report.failed,
            "success_rate": report.get_success_rate(),
            "total_execution_time": report.total_execution_time,
        },
        "results": [_validation_result_dict(result) for result in report.results],
    }


def _validation_report_text(report: ValidationReport) -> str:
    lines = [
        "Model validation summary",
        f"  Total models: {report.total_models}",
        f"  Successful: {report.successful}",
        f"  Failed: {report.failed}",
        f"  Success rate: {report.get_success_rate():.1f}%",
        f"  Total execution time: {report.total_execution_time:.2f}s",
    ]
    for result in report.results:
        status = (
            result.status.value
            if isinstance(result.status, ModelValidationStatus)
            else str(result.status)
        )
        detail = f" ({result.error_message})" if result.error_message else ""
        lines.append(
            f"  - {result.model_name}: {status}, "
            f"{result.execution_time_seconds:.2f}s, rows={result.row_count or 0}{detail}"
        )
    return "\n".join(lines)


@validate.command(context_settings=_CONTEXT, name="models")
@dbt_opts
@logging_opts
@click.argument("models", nargs=-1)
@click.option(
    "-f",
    "--fqn",
    multiple=True,
    type=click.STRING,
    help="Filter models by dbt fully qualified name.",
)
@click.option(
    "--profile",
    type=click.STRING,
    help="Which profile to load. Overrides setting in dbt_project.yml.",
)
@click.option(
    "--vars",
    type=click.STRING,
    help="Supply project variables as a YAML mapping.",
)
@click.option(
    "--timeout",
    type=click.FLOAT,
    default=None,
    help="Best-effort local timeout in seconds for each model query.",
)
@click.option(
    "--include-external",
    is_flag=True,
    help="Include models from external dbt packages.",
)
@click.option(
    "--quiet",
    is_flag=True,
    help="Suppress per-model validation progress logs.",
)
@click.option(
    "--format",
    "format_name",
    type=click.Choice(["table", "json"]),
    default="table",
    help="Output format. Default is table.",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    help="Write validation results to a file instead of stdout.",
)
def validate_models_command(
    target: str | None = None,
    profile: str | None = None,
    project_dir: str | None = None,
    profiles_dir: str | None = None,
    vars: str | None = None,
    threads: int | None = None,
    fqn: tuple[str, ...] = (),
    timeout: float | None = None,
    include_external: bool = False,
    quiet: bool = False,
    format_name: str = "table",
    output: str | None = None,
    models: tuple[str, ...] = (),
) -> None:
    """Compile and execute selected models as validation queries."""
    logger.info(":water_wave: Executing dbt-osmosis model validation\n")
    with _create_cli_yaml_context(
        project_dir=project_dir,
        profiles_dir=profiles_dir,
        target=target,
        profile=profile,
        threads=threads,
        vars_value=vars,
        fqn=fqn,
        models=models,
        include_external=include_external,
    ) as context:
        selected_models = _selected_model_nodes(context)
        report = validate_models(
            context.project,
            selected_models,
            timeout_seconds=timeout,
            quiet=quiet,
        )

    rendered = (
        _json_text(_validation_report_dict(report))
        if format_name == "json"
        else _validation_report_text(report)
    )
    _write_or_echo(rendered, output, label="validation results")
    if report.failed:
        sys.exit(1)


@cli.group()
def analyze():
    """Analyze documentation coverage, gaps, and style"""


def _documentation_column_coverage(result: DocumentationCheckResult) -> float:
    if result.total_columns == 0:
        return 100.0
    return (result.documented_columns / result.total_columns) * 100


def _gap_to_dict(gap: t.Any) -> dict[str, t.Any]:
    if dataclasses.is_dataclass(gap) and not isinstance(gap, type):
        return {field.name: getattr(gap, field.name) for field in dataclasses.fields(gap)}
    if isinstance(gap, dict):
        return gap
    return {
        name: getattr(gap, name)
        for name in (
            "model_name",
            "column_name",
            "resource_name",
            "gap_type",
            "description",
            "message",
            "priority",
            "severity",
        )
        if hasattr(gap, name)
    }


def _documentation_check_dict(result: DocumentationCheckResult) -> dict[str, t.Any]:
    return {
        "total_models": result.total_models,
        "models_with_descriptions": result.models_with_descriptions,
        "models_without_descriptions": result.models_without_descriptions,
        "total_columns": result.total_columns,
        "documented_columns": result.documented_columns,
        "undocumented_columns": result.undocumented_columns,
        "column_coverage_percent": _documentation_column_coverage(result),
        "gaps": [_gap_to_dict(gap) for gap in result.gaps],
    }


def _documentation_check_text(result: DocumentationCheckResult) -> str:
    lines = [
        "Documentation check summary",
        f"  Models described: {result.models_with_descriptions}/{result.total_models}",
        (
            f"  Columns documented: {result.documented_columns}/{result.total_columns} "
            f"({_documentation_column_coverage(result):.1f}%)"
        ),
        f"  Gaps: {len(result.gaps)}",
    ]
    for gap in result.gaps[:20]:
        gap_data = _gap_to_dict(gap)
        description = gap_data.get("description") or gap_data.get("message") or str(gap)
        lines.append(f"  - {description}")
    return "\n".join(lines)


@analyze.command(context_settings=_CONTEXT, name="docs")
@dbt_opts
@logging_opts
@click.option(
    "--profile",
    type=click.STRING,
    help="Which profile to load. Overrides setting in dbt_project.yml.",
)
@click.option(
    "--vars",
    type=click.STRING,
    help="Supply project variables as a YAML mapping.",
)
@click.option(
    "--model-filter",
    type=click.STRING,
    default=None,
    help="Optional model name filter for documentation checking.",
)
@click.option(
    "--min-model-length",
    type=click.INT,
    default=10,
    help="Minimum model description length.",
)
@click.option(
    "--min-column-length",
    type=click.INT,
    default=5,
    help="Minimum column description length.",
)
@click.option(
    "--min-column-coverage",
    type=click.FLOAT,
    default=100.0,
    help="Minimum documented-column percentage required for exit zero.",
)
@click.option(
    "--fail-on-gaps/--allow-gaps",
    default=True,
    help="Exit non-zero when documentation gaps are reported.",
)
@click.option(
    "--format",
    "format_name",
    type=click.Choice(["table", "json"]),
    default="table",
    help="Output format. Default is table.",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    help="Write documentation check results to a file instead of stdout.",
)
def analyze_docs_command(
    target: str | None = None,
    profile: str | None = None,
    project_dir: str | None = None,
    profiles_dir: str | None = None,
    vars: str | None = None,
    threads: int | None = None,
    model_filter: str | None = None,
    min_model_length: int = 10,
    min_column_length: int = 5,
    min_column_coverage: float = 100.0,
    fail_on_gaps: bool = True,
    format_name: str = "table",
    output: str | None = None,
) -> None:
    """Check project documentation completeness."""
    logger.info(":water_wave: Executing dbt-osmosis documentation check\n")
    project = _create_cli_project_context(
        project_dir,
        profiles_dir,
        target,
        profile=profile,
        threads=threads,
        vars=_parsed_cli_vars(vars),
    )
    result = check_documentation(
        project,
        model_filter=model_filter,
        min_model_length=min_model_length,
        min_column_length=min_column_length,
    )
    rendered = (
        _json_text(_documentation_check_dict(result))
        if format_name == "json"
        else _documentation_check_text(result)
    )
    _write_or_echo(rendered, output, label="documentation check results")
    if _documentation_column_coverage(result) < min_column_coverage or (
        fail_on_gaps and result.gaps
    ):
        sys.exit(1)


def _style_profile_text(profile: ProjectStyleProfile) -> str:
    lines = ["Documentation style profile"]
    if profile.description_length_stats:
        avg = profile.description_length_stats.get("avg_length", 0)
        lines.append(f"  Average description length: {avg:.1f} words")
    if profile.common_phrases:
        phrases = ", ".join(phrase for phrase, _ in profile.common_phrases[:5])
        lines.append(f"  Common phrases: {phrases}")
    if profile.terminology_preferences:
        terms = ", ".join(
            f"{preferred} over {alternative}"
            for preferred, alternative in list(profile.terminology_preferences.items())[:5]
        )
        lines.append(f"  Terminology: {terms}")
    if profile.tone_markers:
        tones = ", ".join(f"{name}: {count}" for name, count in profile.tone_markers.items())
        lines.append(f"  Tone markers: {tones}")
    lines.append(f"  Model examples: {len(profile.model_description_samples)}")
    lines.append(f"  Column examples: {len(profile.column_description_samples)}")
    return "\n".join(lines)


@analyze.command(context_settings=_CONTEXT, name="style")
@dbt_opts
@logging_opts
@click.argument("models", nargs=-1)
@click.option(
    "-f",
    "--fqn",
    multiple=True,
    type=click.STRING,
    help="Filter models by dbt fully qualified name.",
)
@click.option(
    "--profile",
    type=click.STRING,
    help="Which profile to load. Overrides setting in dbt_project.yml.",
)
@click.option(
    "--vars",
    type=click.STRING,
    help="Supply project variables as a YAML mapping.",
)
@click.option(
    "--include-external",
    is_flag=True,
    help="Include models and sources from external dbt packages.",
)
@click.option(
    "--max-nodes",
    type=click.INT,
    default=50,
    help="Maximum number of nodes to analyze.",
)
@click.option(
    "--max-columns-per-node",
    type=click.INT,
    default=10,
    help="Maximum columns to analyze per node.",
)
@click.option(
    "--max-examples",
    type=click.INT,
    default=3,
    help="Maximum examples to include in prompt output.",
)
@click.option(
    "--format",
    "format_name",
    type=click.Choice(["table", "json", "prompt"]),
    default="table",
    help="Output format. Default is table.",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    help="Write style analysis to a file instead of stdout.",
)
def analyze_style_command(
    target: str | None = None,
    profile: str | None = None,
    project_dir: str | None = None,
    profiles_dir: str | None = None,
    vars: str | None = None,
    threads: int | None = None,
    fqn: tuple[str, ...] = (),
    include_external: bool = False,
    max_nodes: int = 50,
    max_columns_per_node: int = 10,
    max_examples: int = 3,
    format_name: str = "table",
    output: str | None = None,
    models: tuple[str, ...] = (),
) -> None:
    """Analyze existing documentation style and examples."""
    logger.info(":water_wave: Executing dbt-osmosis documentation style analysis\n")
    with _create_cli_yaml_context(
        project_dir=project_dir,
        profiles_dir=profiles_dir,
        target=target,
        profile=profile,
        threads=threads,
        vars_value=vars,
        fqn=fqn,
        models=models,
        include_external=include_external,
    ) as context:
        profile_result = analyze_project_documentation_style(
            t.cast(t.Any, context),
            max_nodes=max_nodes,
            max_columns_per_node=max_columns_per_node,
        )

    if format_name == "json":
        rendered = _json_text(dataclasses.asdict(profile_result))
    elif format_name == "prompt":
        rendered = profile_result.to_prompt_context(max_examples=max_examples)
    else:
        rendered = _style_profile_text(profile_result)
    _write_or_echo(rendered, output, label="style analysis")


def _discovery_result_text(label: str, result: DiscoveryResult, max_gaps: int) -> list[str]:
    lines = [
        f"{label} discovery summary",
        f"  Coverage: {result.coverage_percent:.1f}%",
        f"  Total gaps: {len(result.gaps)}",
        f"  High priority: {len(result.high_priority_gaps)}",
        f"  Medium priority: {len(result.medium_priority_gaps)}",
        f"  Low priority: {len(result.low_priority_gaps)}",
    ]
    for gap in result.gaps[:max_gaps]:
        lines.append(f"  - {gap.description} ({gap.priority:.1f}): {gap.reason}")
    return lines


@analyze.command(context_settings=_CONTEXT, name="discover")
@dbt_opts
@logging_opts
@click.argument("models", nargs=-1)
@click.option(
    "-f",
    "--fqn",
    multiple=True,
    type=click.STRING,
    help="Filter models by dbt fully qualified name.",
)
@click.option(
    "--profile",
    type=click.STRING,
    help="Which profile to load. Overrides setting in dbt_project.yml.",
)
@click.option(
    "--vars",
    type=click.STRING,
    help="Supply project variables as a YAML mapping.",
)
@click.option(
    "--include-external",
    is_flag=True,
    help="Include models and sources from external dbt packages.",
)
@click.option(
    "--scope",
    type=click.Choice(["models", "columns", "all"]),
    default="all",
    help="Documentation gap scope to discover.",
)
@click.option(
    "--min-columns",
    type=click.INT,
    default=3,
    help="Minimum model columns for model-level gap discovery.",
)
@click.option(
    "--include-sources/--exclude-sources",
    default=False,
    help="Include source definitions in model-level discovery.",
)
@click.option(
    "--min-priority",
    type=click.FLOAT,
    default=0.0,
    help="Minimum column-gap priority to include.",
)
@click.option(
    "--max-gaps",
    type=click.INT,
    default=20,
    help="Maximum gaps to print in table output.",
)
@click.option(
    "--check",
    is_flag=True,
    help="Exit non-zero when any selected discovery scope reports gaps.",
)
@click.option(
    "--format",
    "format_name",
    type=click.Choice(["table", "json"]),
    default="table",
    help="Output format. Default is table.",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    help="Write discovery results to a file instead of stdout.",
)
def analyze_discover_command(
    target: str | None = None,
    profile: str | None = None,
    project_dir: str | None = None,
    profiles_dir: str | None = None,
    vars: str | None = None,
    threads: int | None = None,
    fqn: tuple[str, ...] = (),
    include_external: bool = False,
    scope: str = "all",
    min_columns: int = 3,
    include_sources: bool = False,
    min_priority: float = 0.0,
    max_gaps: int = 20,
    check: bool = False,
    format_name: str = "table",
    output: str | None = None,
    models: tuple[str, ...] = (),
) -> None:
    """Discover prioritized documentation gaps."""
    logger.info(":water_wave: Executing dbt-osmosis documentation discovery\n")
    results: dict[str, DiscoveryResult] = {}
    with _create_cli_yaml_context(
        project_dir=project_dir,
        profiles_dir=profiles_dir,
        target=target,
        profile=profile,
        threads=threads,
        vars_value=vars,
        fqn=fqn,
        models=models,
        include_external=include_external,
    ) as context:
        typed_context = t.cast(t.Any, context)
        if scope in ("models", "all"):
            results["models"] = discover_undocumented_models(
                typed_context,
                min_columns=min_columns,
                exclude_sources=not include_sources,
            )
        if scope in ("columns", "all"):
            results["columns"] = discover_undocumented_columns(
                typed_context,
                min_priority=min_priority,
            )

    if format_name == "json":
        rendered = _json_text({name: result.to_dict() for name, result in results.items()})
    else:
        lines: list[str] = []
        for name, result in results.items():
            lines.extend(_discovery_result_text(name.title(), result, max_gaps))
            lines.append("")
        rendered = "\n".join(lines).rstrip()
    _write_or_echo(rendered, output, label="discovery results")
    if check and any(result.gaps for result in results.values()):
        sys.exit(1)


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


def _rule_options(
    rules: tuple[str, ...],
    disable_rules: tuple[str, ...],
) -> tuple[list[str] | None, list[str] | None]:
    enabled_rules = list(rules) if rules else None
    disabled_rules = list(disable_rules) if disable_rules else None
    return enabled_rules, disabled_rules


def _sql_lint_inputs(
    project_dir: str | None,
    profiles_dir: str | None,
    target: str | None,
    dialect: str | None,
    rules: tuple[str, ...],
    disable_rules: tuple[str, ...],
    **kwargs: t.Any,
) -> tuple[DbtProjectContext, str, list[str] | None, list[str] | None]:
    project = _create_cli_project_context(project_dir, profiles_dir, target, **kwargs)
    sql_dialect = dialect or project.adapter.type()
    enabled_rules, disabled_rules = _rule_options(rules, disable_rules)
    return project, sql_dialect, enabled_rules, disabled_rules


def _sql_linter(
    sql_dialect: str,
    enabled_rules: list[str] | None,
    disabled_rules: list[str] | None,
) -> SQLLinter:
    return SQLLinter(
        dialect=sql_dialect,
        enabled_rules=enabled_rules,
        disabled_rules=disabled_rules,
    )


def _echo_violation_section(title: str, violations: list[LintViolation]) -> None:
    if not violations:
        return
    click.echo(title)
    for violation in violations:
        click.echo(f"  {violation}")
    click.echo()


def _echo_lint_result(header: str, result: LintResult) -> tuple[int, int]:
    click.echo(header)
    if not result.violations:
        click.echo(":white_check_mark: No issues found!")
        return 0, 0

    errors, warnings, other = _lint_violation_groups(result)
    _echo_violation_section(":no_entry: Errors:", errors)
    _echo_violation_section(":warning: Warnings:", warnings)
    _echo_violation_section(":information_source: Other:", other)
    return len(errors), len(warnings)


def _exit_on_lint_failures(error_count: int, warning_count: int) -> None:
    if error_count or warning_count:
        sys.exit(1)


def _project_lint_counts(
    grouped_results: dict[
        str, tuple[list[LintViolation], list[LintViolation], list[LintViolation]]
    ],
) -> tuple[int, int, int]:
    total_errors = sum(len(errors) for errors, _, _ in grouped_results.values())
    total_warnings = sum(len(warnings) for _, warnings, _ in grouped_results.values())
    total_other = sum(len(other) for _, _, other in grouped_results.values())
    return total_errors, total_warnings, total_other


def _echo_project_lint_model(
    model_name: str,
    result: LintResult,
    groups: tuple[list[LintViolation], list[LintViolation], list[LintViolation]],
) -> None:
    click.echo(f"\n:page_facing_up: {model_name} ({result.summary()})")
    errors, warnings, other = groups
    for violation in errors:
        click.echo(f"  :no_entry: {violation}")
    for violation in warnings:
        click.echo(f"  :warning: {violation}")
    for violation in other:
        click.echo(f"  :information_source: {violation}")


def _echo_project_lint_results(results: dict[str, LintResult]) -> tuple[int, int]:
    grouped_results = {name: _lint_violation_groups(result) for name, result in results.items()}
    total_errors, total_warnings, total_other = _project_lint_counts(grouped_results)

    click.echo(f"\n:sparkles: Lint Results for {len(results)} models\n")
    click.echo(
        f"  Total: {total_errors} error(s), {total_warnings} warning(s), {total_other} info\n"
    )

    models_with_issues = {name: result for name, result in results.items() if result.violations}
    if not models_with_issues:
        click.echo(":white_check_mark: No issues found across all models!")
        return 0, 0

    for model_name, result in models_with_issues.items():
        _echo_project_lint_model(model_name, result, grouped_results[model_name])
    return total_errors, total_warnings


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
    project, sql_dialect, enabled_rules, disabled_rules = _sql_lint_inputs(
        project_dir,
        profiles_dir,
        target,
        dialect,
        rules,
        disable_rules,
        **kwargs,
    )

    # Lint the SQL
    result = lint_sql_code(
        context=project,
        raw_sql=sql,
        dialect=sql_dialect,
        rules=enabled_rules,
        disabled_rules=disabled_rules,
    )

    # Display results
    error_count, warning_count = _echo_lint_result(
        f"\n:sparkles: Lint Results: {result.summary()}\n",
        result,
    )
    _exit_on_lint_failures(error_count, warning_count)


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
    project, sql_dialect, enabled_rules, disabled_rules = _sql_lint_inputs(
        project_dir,
        profiles_dir,
        target,
        dialect,
        rules,
        disable_rules,
        **kwargs,
    )
    linter = _sql_linter(sql_dialect, enabled_rules, disabled_rules)

    # Lint the model
    result = linter.lint_model(project, model_name)

    # Display results
    error_count, warning_count = _echo_lint_result(
        f"\n:sparkles: Lint Results for {model_name}: {result.summary()}\n",
        result,
    )
    _exit_on_lint_failures(error_count, warning_count)


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
    project, sql_dialect, enabled_rules, disabled_rules = _sql_lint_inputs(
        project_dir,
        profiles_dir,
        target,
        dialect,
        rules,
        disable_rules,
        **kwargs,
    )
    linter = _sql_linter(sql_dialect, enabled_rules, disabled_rules)

    # Lint the project
    fqn_filter = list(fqn) if fqn else None
    results = linter.lint_project(project, fqn_filter=fqn_filter)

    # Display results
    total_errors, total_warnings = _echo_project_lint_results(results)
    _exit_on_lint_failures(total_errors, total_warnings)


if __name__ == "__main__":
    cli()
