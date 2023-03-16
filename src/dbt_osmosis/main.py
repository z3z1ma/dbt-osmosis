import functools
import importlib.util
import multiprocessing
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional, Union
from urllib.parse import urlencode
from dataclasses import asdict

import click
import requests

from dbt_osmosis.core.diff import diff_and_print_to_console
from dbt_osmosis.core.log_controller import logger
from dbt_osmosis.core.macros import inject_macros
from dbt_osmosis.core.osmosis import DbtYamlManager
from dbt_osmosis.vendored.dbt_core_interface import DEFAULT_PROFILES_DIR, DbtProject, run_server

CONTEXT = {"max_content_width": 800}


@click.group()
@click.version_option()
def cli():
    pass


@cli.group()
def yaml():
    """Manage, document, and organize dbt YAML files"""


@cli.group()
def sql():
    """Execute and compile dbt SQL statements"""


@cli.group()
def server():
    """Manage dbt osmosis server"""


def shared_opts(func: Callable) -> Callable:
    """Here we define the options shared across subcommands
    Args:
        func (Callable): Wraps a subcommand
    Returns:
        Callable: Subcommand with added options
    """

    @click.option(
        "--project-dir",
        type=click.Path(exists=True, dir_okay=True, file_okay=False),
        default=str(Path.cwd()),
        help=(
            "Which directory to look in for the dbt_project.yml file. Default is the current"
            " working directory and its parents."
        ),
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
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


@yaml.command(context_settings=CONTEXT)
@shared_opts
@click.option(
    "-f",
    "--fqn",
    type=click.STRING,
    help=(
        "Specify models based on dbt's FQN. Looks like folder.folder, folder.folder.model, or"
        " folder.folder.source.table. Use list command to see the scope of an FQN filter."
    ),
)
@click.option(
    "-F",
    "--force-inheritance",
    is_flag=True,
    help=(
        "If specified, forces documentation to be inherited overriding existing column level"
        " documentation where applicable."
    ),
)
@click.option(
    "-d",
    "--dry-run",
    is_flag=True,
    help="If specified, no changes are committed to disk.",
)
def refactor(
    target: Optional[str] = None,
    project_dir: Optional[str] = None,
    profiles_dir: Optional[str] = None,
    fqn: Optional[str] = None,
    force_inheritance: bool = False,
    dry_run: bool = False,
):
    """Executes organize which syncs yaml files with database schema and organizes the dbt models
    directory, reparses the project, then executes document passing down inheritable documentation

    \f
    This command will conform your project as outlined in `dbt_project.yml`, bootstrap undocumented
    dbt models, and propagate column level documentation downwards once all yamls are accounted for

    Args:
        target (Optional[str]): Profile target. Defaults to default target set in profile yml
        project_dir (Optional[str], optional): Dbt project directory. Defaults to working directory.
        profiles_dir (Optional[str], optional): Dbt profile directory. Defaults to ~/.dbt
    """
    logger().info(":water_wave: Executing dbt-osmosis\n")

    runner = DbtYamlManager(
        project_dir=project_dir,
        profiles_dir=profiles_dir,
        target=target,
        fqn=fqn,
        dry_run=dry_run,
    )

    # Conform project structure & bootstrap undocumented models injecting columns
    if runner.commit_project_restructure_to_disk():
        runner.safe_parse_project()
    runner.propagate_documentation_downstream(force_inheritance=force_inheritance)


@yaml.command(context_settings=CONTEXT)
@shared_opts
@click.option(
    "-f",
    "--fqn",
    type=click.STRING,
    help=(
        "Specify models based on FQN. Use dots as separators. Looks like folder.folder.model or"
        " folder.folder.source.table. Use list command to see the scope of an FQN filter."
    ),
)
@click.option(
    "-d",
    "--dry-run",
    is_flag=True,
    help="If specified, no changes are committed to disk.",
)
def organize(
    target: Optional[str] = None,
    project_dir: Optional[str] = None,
    profiles_dir: Optional[str] = None,
    fqn: Optional[str] = None,
    dry_run: bool = False,
):
    """Organizes schema ymls based on config and injects undocumented models

    \f
    This command will conform schema ymls in your project as outlined in `dbt_project.yml` &
    bootstrap undocumented dbt models

    Args:
        target (Optional[str]): Profile target. Defaults to default target set in profile yml
        project_dir (Optional[str], optional): Dbt project directory. Defaults to working directory.
        profiles_dir (Optional[str], optional): Dbt profile directory. Defaults to ~/.dbt
    """
    logger().info(":water_wave: Executing dbt-osmosis\n")

    runner = DbtYamlManager(
        project_dir=project_dir,
        profiles_dir=profiles_dir,
        target=target,
        fqn=fqn,
        dry_run=dry_run,
    )

    # Conform project structure & bootstrap undocumented models injecting columns
    runner.commit_project_restructure_to_disk()


@yaml.command(context_settings=CONTEXT)
@shared_opts
@click.option(
    "-f",
    "--fqn",
    type=click.STRING,
    help=(
        "Specify models based on FQN. Use dots as separators. Looks like folder.folder.model or"
        " folder.folder.source.table. Use list command to see the scope of an FQN filter."
    ),
)
@click.option(
    "-F",
    "--force-inheritance",
    is_flag=True,
    help=(
        "If specified, forces documentation to be inherited overriding existing column level"
        " documentation where applicable."
    ),
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
    """Column level documentation inheritance for existing models

    \f
    This command will conform schema ymls in your project as outlined in `dbt_project.yml` &
    bootstrap undocumented dbt models

    Args:
        target (Optional[str]): Profile target. Defaults to default target set in profile yml
        project_dir (Optional[str], optional): Dbt project directory. Defaults to working directory.
        profiles_dir (Optional[str], optional): Dbt profile directory. Defaults to ~/.dbt
    """
    logger().info(":water_wave: Executing dbt-osmosis\n")

    runner = DbtYamlManager(
        project_dir=project_dir,
        profiles_dir=profiles_dir,
        target=target,
        fqn=fqn,
        dry_run=dry_run,
    )

    # Propagate documentation & inject/remove schema file columns to align with model in database
    runner.propagate_documentation_downstream(force_inheritance)


class ServerRegisterThread(threading.Thread):
    """Thin container to capture errors in project registration"""

    def run(self):
        try:
            threading.Thread.run(self)
        except Exception as err:
            self.err = err
            pass
        else:
            self.err = None


def _health_check(host: str, port: int):
    """Performs health check on server,
    raises ConnectionError otherwise returns result"""
    t, max_t, i = 0.25, 10, 0
    address = f"http://{host}:{port}/health"
    error_msg = f"Server at {address} is not healthy"
    while True:
        try:
            resp = requests.get(address)
        except Exception:
            time.sleep(t)
            i += 1
            if t * i > max_t:
                logger().critical(error_msg, address)
                raise ConnectionError(error_msg)
            else:
                continue
        if resp.ok:
            break
        else:
            logger().critical(error_msg, address)
            raise ConnectionError(error_msg)
    return resp.json()


@server.command(context_settings=CONTEXT)
@shared_opts
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
    default=8581,
)
@click.option(
    "--register-project",
    is_flag=True,
    help=(
        "Try to register a dbt project on init as specified by --project-dir, --profiles-dir or"
        " their defaults if not passed explicitly"
    ),
)
@click.option(
    "--exit-on-error",
    is_flag=True,
    help=(
        "A flag which indicates the program should terminate on registration failure if"
        " --register-project was unsuccessful"
    ),
)
def serve(
    project_dir: str,
    profiles_dir: str,
    target: str,
    host: str = "localhost",
    port: int = 8581,
    register_project: bool = False,
    exit_on_error: bool = False,
):
    """Runs a lightweight server compatible with dbt-power-user and convenient for interactively
    running or compile dbt SQL queries with two simple endpoints accepting POST messages"""
    if importlib.util.find_spec("sqlfluff_templater_dbt"):
        logger().error(
            "sqlfluff-templater-dbt is not compatible with dbt-osmosis server. "
            "Please uninstall it to continue."
        )
        sys.exit(1)

    logger().info(":water_wave: Executing dbt-osmosis\n")

    def _register_project():
        """Background job which registers the first project on the server automatically"""

        # Wait
        _health_check(host, port)

        # Register
        params = {"project_dir": project_dir, "profiles_dir": profiles_dir}
        if target:
            params["target"] = target
        endpoint = f"http://{host}:{port}/register?{urlencode(params)}"
        logger().info("Registering project: %s", endpoint)
        res = requests.post(
            endpoint,
            headers={"X-dbt-Project": str(Path(project_dir).absolute())},
        ).json()

        # Log
        logger().info(res)
        if "error" in res:
            raise ConnectionError(res["error"]["message"])

    server = multiprocessing.Process(target=run_server, args=(None, host, port))
    server.start()

    import atexit

    atexit.register(lambda: server.terminate())

    register_handler: Optional[ServerRegisterThread] = None
    if register_project and project_dir and profiles_dir:
        register_handler = ServerRegisterThread(target=_register_project)
        register_handler.start()

    register_exit = None
    if register_handler is not None:
        register_handler.join()
        if register_handler.err is not None and exit_on_error:
            register_exit = 1
            server.kill()

    server.join()
    sys.exit(register_exit or server.exitcode)


@server.command(context_settings=CONTEXT)
@shared_opts
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
    default=8581,
)
@click.option(
    "--project-name",
    type=click.STRING,
    help=(
        "The name to register the project with. By default, it is a string value representing the"
        " absolute directory of the project on disk"
    ),
)
def register_project(
    project_dir: str,
    profiles_dir: str,
    target: str,
    host: str = "localhost",
    port: int = 8581,
    project_name: Optional[str] = None,
):
    """Convenience method to allow user to register project on the running server from the CLI"""
    logger().info(":water_wave: Executing dbt-osmosis\n")

    # Wait
    _health_check(host, port)

    # Register
    params = {"project_dir": project_dir, "profiles_dir": profiles_dir, "force": True}
    if target:
        params["target"] = target
    endpoint = f"http://{host}:{port}/register?{urlencode(params)}"
    logger().info("Registering project: %s", endpoint)
    res = requests.post(
        endpoint,
        headers={"X-dbt-Project": project_name or str(Path(project_dir).absolute())},
    )

    # Log
    logger().info(res.json())


@server.command(context_settings=CONTEXT)
@click.option(
    "--project-name",
    type=click.STRING,
    help="The name of the registered project to remove.",
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
    default=8581,
)
def unregister_project(
    project_name: str,
    host: str = "localhost",
    port: int = 8581,
):
    """Convenience method to allow user to unregister project on the running server from the CLI"""
    logger().info(":water_wave: Executing dbt-osmosis\n")

    # Wait
    _health_check(host, port)

    # Unregister
    endpoint = f"http://{host}:{port}/unregister"
    logger().info("Unregistering project: %s", endpoint)
    res = requests.post(
        endpoint,
        headers={"X-dbt-Project": project_name},
    )

    # Log
    logger().info(res.json())


@cli.command(
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
    )
)
@click.option(
    "--project-dir",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    help=(
        "Which directory to look in for the dbt_project.yml file. Default is the current working"
        " directory and its parents."
    ),
)
@click.option(
    "--profiles-dir",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    default=DEFAULT_PROFILES_DIR,
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
    ctx,
    profiles_dir: Optional[str] = None,
    project_dir: Optional[str] = None,
    host: str = "localhost",
    port: int = 8501,
):
    """Start the dbt-osmosis workbench

    \f
    Pass the --options command to see streamlit specific options that can be passed to the app,
    pass --config to see the output of streamlit config show
    """
    raise NotImplementedError("Workbench is not yet implemented for new dbt-osmosis")
    logger().info(":water_wave: Executing dbt-osmosis\n")

    if "--options" in ctx.args:
        subprocess.run(["streamlit", "run", "--help"])
        ctx.exit()

    import os

    if "--config" in ctx.args:
        subprocess.run(
            ["streamlit", "config", "show"],
            env=os.environ,
            cwd=Path.cwd(),
        )
        ctx.exit()

    script_args = ["--"]
    if project_dir:
        script_args.append("--project-dir")
        script_args.append(project_dir)
    if profiles_dir:
        script_args.append("--profiles-dir")
        script_args.append(profiles_dir)

    subprocess.run(
        [
            "streamlit",
            "run",
            "--runner.magicEnabled=false",
            f"--browser.serverAddress={host}",
            f"--browser.serverPort={port}",
            Path(__file__).parent / "app.py",
        ]
        + ctx.args
        + script_args,
        env=os.environ,
        cwd=Path.cwd(),
    )


@cli.command(context_settings=CONTEXT)
@shared_opts
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
    help=(
        "Output format can be one of table, chart/bar, or csv. CSV is saved to a file named"
        " dbt-osmosis-diff in working dir"
    ),
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
    """Diff dbt models at different git revisions"""

    logger().info(":water_wave: Executing dbt-osmosis\n")

    runner = DbtProject(
        project_dir=project_dir,
        profiles_dir=profiles_dir,
        target=target,
    )
    inject_macros(runner)
    diff_and_print_to_console(model, pk, runner, temp_table, agg, output)


@sql.command(context_settings=CONTEXT)
@shared_opts
@click.argument("sql")
def run(
    sql: str = "",
    project_dir: Optional[str] = None,
    profiles_dir: Optional[str] = None,
    target: Optional[str] = None,
):
    """Executes a dbt SQL statement writing an OsmosisRunResult | OsmosisErrorContainer to stdout"""
    from dbt_osmosis.vendored.dbt_core_interface.project import (
        ServerError,
        ServerErrorCode,
        ServerErrorContainer,
        ServerRunResult,
    )

    rv: Union[ServerRunResult, ServerErrorContainer] = None

    try:
        runner = DbtProject(
            project_dir=project_dir,
            profiles_dir=profiles_dir,
            target=target,
        )
    except Exception as init_err:
        rv = ServerErrorContainer(
            error=ServerError(
                code=ServerErrorCode.ProjectParseFailure,
                message=str(init_err),
                data=init_err.__dict__,
            )
        )

    if rv is not None:
        print(asdict(rv))
        return rv

    try:
        result = runner.execute_code("\n".join(sys.stdin.readlines()) if sql == "-" else sql)
    except Exception as execution_err:
        rv = ServerErrorContainer(
            error=ServerError(
                code=ServerErrorCode.ExecuteSqlFailure,
                message=str(execution_err),
                data=execution_err.__dict__,
            )
        )
    else:
        rv = ServerRunResult(
            rows=[list(row) for row in result.table.rows],
            column_names=result.table.column_names,
            executed_code=result.compiled_code,
            raw_code=result.raw_code,
        )

    print(asdict(rv))
    return rv


@sql.command(context_settings=CONTEXT)
@shared_opts
@click.argument("sql")
def compile(
    sql: str,
    project_dir: Optional[str] = None,
    profiles_dir: Optional[str] = None,
    target: Optional[str] = None,
):
    """Compiles dbt SQL statement writing an OsmosisCompileResult | OsmosisErrorContainer to stdout"""
    from dbt_osmosis.vendored.dbt_core_interface.project import (
        ServerCompileResult,
        ServerError,
        ServerErrorCode,
        ServerErrorContainer,
    )

    rv: Union[ServerCompileResult, ServerErrorContainer] = None

    try:
        runner = DbtProject(
            project_dir=project_dir,
            profiles_dir=profiles_dir,
            target=target,
        )
    except Exception as init_err:
        rv = ServerErrorContainer(
            error=ServerError(
                code=ServerErrorCode.ProjectParseFailure,
                message=str(init_err),
                data=init_err.__dict__,
            )
        )

    if rv is not None:
        print(asdict(rv))
        return rv

    try:
        result = runner.compile_code("\n".join(sys.stdin.readlines()) if sql == "-" else sql)
    except Exception as compilation_err:
        rv = ServerErrorContainer(
            error=ServerError(
                code=ServerErrorCode.CompileSqlFailure,
                message=str(compilation_err),
                data=compilation_err.__dict__,
            )
        )
    else:
        rv = ServerCompileResult(result=result.compiled_code)

    print(asdict(rv))
    return rv


if __name__ == "__main__":
    cli()
