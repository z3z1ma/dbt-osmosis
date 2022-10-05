import asyncio
import datetime
import json
import logging
import os
import re
import sys
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import Any, cast, Dict, List, Optional, Tuple, Union

import click
import sqlfluff.core
import sqlfluff.cli.commands
import uvicorn
import yaml
from fastapi import BackgroundTasks, FastAPI, Header, Request, Response, status
from fastapi.testclient import TestClient
from pydantic import BaseModel
from sqlfluff.cli import EXIT_SUCCESS
from sqlfluff.cli.commands import dump_file_payload, get_config, make_output_stream, get_linter_and_formatter, progress_bar_configuration, set_logging_level, format_linting_result_header, PathAndUserErrorHandler
from sqlfluff.core.enums import FormatType, Color

from dbt_osmosis.core.osmosis import DbtProject, DbtProjectContainer, has_jinja

DEFAULT = "__default__"


app = FastAPI()
app.state.dbt_project_container = DbtProjectContainer()


class OsmosisRunResult(BaseModel):
    column_names: List[str]
    rows: List[List[Any]]
    raw_sql: str
    compiled_sql: str


class OsmosisCompileResult(BaseModel):
    result: str


class OsmosisLintError(BaseModel):
    # Based on SQLFluff.lint() result format
    code: str
    description: str
    line_no: int
    line_pos: int


class OsmosisLintResult(BaseModel):
    result: List[OsmosisLintError]


class OsmosisResetResult(BaseModel):
    result: str


class OsmosisRegisterResult(BaseModel):
    added: str
    projects: List[str]


class OsmosisUnregisterResult(BaseModel):
    removed: str
    projects: List[str]


class OsmosisErrorCode(int, Enum):
    FailedToReachServer = -1
    CompileSqlFailure = 1
    ExecuteSqlFailure = 2
    ProjectParseFailure = 3
    ProjectNotRegistered = 4
    ProjectHeaderNotSupplied = 5


class OsmosisError(BaseModel):
    code: OsmosisErrorCode
    message: str
    data: Dict[str, Any]


class OsmosisErrorContainer(BaseModel):
    error: OsmosisError


@contextmanager
def set_directory(path: Path):
    """Sets the cwd within the context

    Args:
        path (Path): The path to the cwd

    Yields:
        None
    """

    origin = Path().absolute()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(origin)

@app.post(
    "/run",
    response_model=Union[OsmosisRunResult, OsmosisErrorContainer],
    responses={status.HTTP_404_NOT_FOUND: {"model": OsmosisErrorContainer}},
    openapi_extra={
        "requestBody": {
            "content": {"text/plain": {"schema": {"type": "string"}}},
            "required": True,
        },
    },
)
async def run_sql(
    request: Request,
    response: Response,
    limit: int = 100,
    x_dbt_project: str = Header(default=DEFAULT),
) -> Union[OsmosisRunResult, OsmosisErrorContainer]:
    """Execute dbt SQL against a registered project as determined by X-dbt-Project header"""
    dbt: DbtProjectContainer = app.state.dbt_project_container
    if x_dbt_project == DEFAULT:
        project = dbt.get_default_project()
    else:
        project = dbt.get_project(x_dbt_project)
    if project is None:
        response.status_code = status.HTTP_404_NOT_FOUND
        return OsmosisErrorContainer(
            error=OsmosisError(
                code=OsmosisErrorCode.ProjectNotRegistered,
                message="Project is not registered. Make a POST request to the /register endpoint first to register a runner",
                data={"registered_projects": dbt.registered_projects()},
            )
        )

    # Query Construction
    query = (await request.body()).decode("utf-8").strip()
    if has_jinja(query):
        query = f"\n(\n{{# PAD #}}{query}{{# PAD #}}\n)\n"
    query_with_limit = (
        # we need to support `TOP` too
        f"select * from ({query}) as osmosis_query limit {limit}"
    )

    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, project.fn_threaded_conn(project.execute_sql, query_with_limit)
        )
    except Exception as execution_err:
        return OsmosisErrorContainer(
            error=OsmosisError(
                code=OsmosisErrorCode.ExecuteSqlFailure,
                message=str(execution_err),
                data=execution_err.__dict__,
            )
        )

    # Re-extract compiled query and return data structure
    compiled_query = re.search(
        r"select \* from \(([\w\W]+)\) as osmosis_query", result.compiled_sql
    ).groups()[0]
    return OsmosisRunResult(
        rows=[list(row) for row in result.table.rows],
        column_names=result.table.column_names,
        compiled_sql=compiled_query.strip()[1:-1],
        raw_sql=query,
    )


@app.post(
    "/compile",
    response_model=Union[OsmosisCompileResult, OsmosisErrorContainer],
    responses={
        status.HTTP_404_NOT_FOUND: {"model": OsmosisErrorContainer},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": OsmosisErrorContainer},
    },
    openapi_extra={
        "requestBody": {
            "content": {"text/plain": {"schema": {"type": "string"}}},
            "required": True,
        },
    },
)
async def compile_sql(
    request: Request,
    response: Response,
    x_dbt_project: str = Header(default=DEFAULT),
) -> Union[OsmosisCompileResult, OsmosisErrorContainer]:
    """Compile dbt SQL against a registered project as determined by X-dbt-Project header"""
    dbt: DbtProjectContainer = app.state.dbt_project_container
    if x_dbt_project == DEFAULT:
        project = dbt.get_default_project()
    else:
        project = dbt.get_project(x_dbt_project)
    if project is None:
        response.status_code = status.HTTP_404_NOT_FOUND
        return OsmosisErrorContainer(
            error=OsmosisError(
                code=OsmosisErrorCode.ProjectNotRegistered,
                message="Project is not registered. Make a POST request to the /register endpoint first to register a runner",
                data={"registered_projects": dbt.registered_projects()},
            )
        )

    # Query Compilation
    query: str = (await request.body()).decode("utf-8").strip()
    if has_jinja(query):
        try:
            compiled_query = project.compile_sql(query).compiled_sql
        except Exception as compile_err:
            response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
            return OsmosisErrorContainer(
                error=OsmosisError(
                    code=OsmosisErrorCode.CompileSqlFailure,
                    message=str(compile_err),
                    data=compile_err.__dict__,
                )
            )
    else:
        compiled_query = query

    return OsmosisCompileResult(result=compiled_query)


def lint_command(
    paths: Tuple[str],
    format: str,
    write_output: Optional[str],
    annotation_level: str,
    nofail: bool,
    disregard_sqlfluffignores: bool,
    logger: Optional[logging.Logger] = None,
    bench: bool = False,
    processes: Optional[int] = None,
    disable_progress_bar: Optional[bool] = False,
    extra_config_path: Optional[str] = None,
    ignore_local_config: bool = False,
    **kwargs,
) -> int:
    """Lint SQL files via passing a list of files or using stdin.

    PATH is the path to a sql file or directory to lint. This can be either a
    file ('path/to/file.sql'), a path ('directory/of/sql/files'), a single ('-')
    character to indicate reading from *stdin* or a dot/blank ('.'/' ') which will
    be interpreted like passing the current working directory as a path argument.

    Linting SQL files:

        sqlfluff lint path/to/file.sql
        sqlfluff lint directory/of/sql/files

    Linting a file via stdin (note the lone '-' character):

        cat path/to/file.sql | sqlfluff lint -
        echo 'select col from tbl' | sqlfluff lint -

    """
    config = get_config(
        extra_config_path, ignore_local_config, require_dialect=False, **kwargs
    )
    non_human_output = (format != FormatType.human.value) or (write_output is not None)
    file_output = None
    output_stream = make_output_stream(config, format, write_output)
    lnt, formatter = get_linter_and_formatter(config, output_stream)

    verbose = config.get("verbose")
    progress_bar_configuration.disable_progress_bar = disable_progress_bar

    formatter.dispatch_config(lnt)

    # Set up logging.
    set_logging_level(
        verbosity=verbose,
        formatter=formatter,
        logger=logger,
        stderr_output=non_human_output,
    )

    # Output the results as we go
    if verbose >= 1:
        click.echo(format_linting_result_header())

    with PathAndUserErrorHandler(formatter, paths):
        # add stdin if specified via lone '-'
        if ("-",) == paths:
            result = lnt.lint_string_wrapped(sys.stdin.read(), fname="stdin")
        else:
            result = lnt.lint_paths(
                paths,
                ignore_non_existent_files=False,
                ignore_files=not disregard_sqlfluffignores,
                processes=processes,
            )

    # Output the final stats
    if verbose >= 1:
        click.echo(formatter.format_linting_stats(result, verbose=verbose))

    if format == FormatType.json.value:
        file_output = json.dumps(result.as_records())
    elif format == FormatType.yaml.value:
        file_output = yaml.dump(result.as_records(), sort_keys=False)
    elif format == FormatType.github_annotation.value:
        if annotation_level == "error":
            annotation_level = "failure"

        github_result = []
        for record in result.as_records():
            filepath = record["filepath"]
            for violation in record["violations"]:
                # NOTE: The output format is designed for this GitHub action:
                # https://github.com/yuzutech/annotations-action
                # It is similar, but not identical, to the native GitHub format:
                # https://docs.github.com/en/rest/reference/checks#annotations-items
                github_result.append(
                    {
                        "file": filepath,
                        "line": violation["line_no"],
                        "start_column": violation["line_pos"],
                        "end_column": violation["line_pos"],
                        "title": "SQLFluff",
                        "message": f"{violation['code']}: {violation['description']}",
                        "annotation_level": annotation_level,
                    }
                )
        file_output = json.dumps(github_result)
    elif format == FormatType.github_annotation_native.value:
        if annotation_level == "failure":
            annotation_level = "error"

        github_result_native = []
        for record in result.as_records():
            filepath = record["filepath"]
            for violation in record["violations"]:
                # NOTE: The output format is designed for GitHub action:
                # https://docs.github.com/en/actions/using-workflows/workflow-commands-for-github-actions#setting-a-notice-message
                line = f"::{annotation_level} "
                line += "title=SQLFluff,"
                line += f"file={filepath},"
                line += f"line={violation['line_no']},"
                line += f"col={violation['line_pos']}"
                line += "::"
                line += f"{violation['code']}: {violation['description']}"

                github_result_native.append(line)

        file_output = "\n".join(github_result_native)

    if file_output:
        dump_file_payload(write_output, cast(str, file_output))

    output_stream.close()
    if bench:
        click.echo("==== overall timings ====")
        click.echo(formatter.cli_table([("Clock time", result.total_time)]))
        timing_summary = result.timing_summary()
        for step in timing_summary:
            click.echo(f"=== {step} ===")
            click.echo(formatter.cli_table(timing_summary[step].items()))

    if not nofail:
        if not non_human_output:
            formatter.completion_message()
        #sys.exit(result.stats()["exit code"])
        return result.stats()["exit code"]
    else:
        #sys.exit(EXIT_SUCCESS)
        return EXIT_SUCCESS

@app.post(
    "/lint",
    response_model=Union[OsmosisLintResult, OsmosisErrorContainer],
    responses={
        status.HTTP_404_NOT_FOUND: {"model": OsmosisErrorContainer},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": OsmosisErrorContainer},
    },
)
async def lint_sql(
    response: Response,
    query: str,
    # TODO: Should config_path be part of /register instead?
    config_path: Optional[str],
    x_dbt_project: str = Header(default=DEFAULT),
) -> Union[OsmosisLintResult, OsmosisErrorContainer]:
    """Lint dbt SQL against a registered project as determined by X-dbt-Project header"""
    dbt: DbtProjectContainer = app.state.dbt_project_container
    if x_dbt_project == DEFAULT:
        project = dbt.get_default_project()
    else:
        project = dbt.get_project(x_dbt_project)
    if project is None:
        response.status_code = status.HTTP_404_NOT_FOUND
        return OsmosisErrorContainer(
            error=OsmosisError(
                code=OsmosisErrorCode.ProjectNotRegistered,
                message="Project is not registered. Make a POST request to the /register endpoint first to register a runner",
                data={"registered_projects": dbt.registered_projects()},
            )
        )

    # Query Linting
    try:
        with set_directory(Path("/Users/bhart/dev/dbt-osmosis/tests/sqlfluff_templater/fixtures/dbt/dbt_project")):
            #result = sqlfluff.lint(query, config_path=config_path)
            #result = sqlfluff.lint(query)
            exit_code = lint_command(
                paths=("models/my_new_project/issue_1608.sql",),
                format="json",
                write_output="/tmp/lint_result.json",
                annotation_level="notice",
                nofail=False,
                disregard_sqlfluffignores=False,
                nocolor=False,
            )
            entire_result = json.load(open("/tmp/lint_result.json"))
            result = entire_result[0]["violations"]
    except Exception as lint_err:
        logging.exception("Linting failed")
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return OsmosisErrorContainer(
            error=OsmosisError(
                code=OsmosisErrorCode.CompileSqlFailure,
                message=str(lint_err),
                data=lint_err.__dict__,
            )
        )
    else:
        lint_result = OsmosisLintResult(result=[OsmosisLintError(**error) for error in result])
    return lint_result


@app.get(
    "/parse",
    response_model=Union[OsmosisResetResult, OsmosisErrorContainer],
    responses={
        status.HTTP_404_NOT_FOUND: {"model": OsmosisErrorContainer},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": OsmosisErrorContainer},
    },
)
async def reset(
    background_tasks: BackgroundTasks,
    response: Response,
    reset: bool = False,
    target: Optional[str] = None,
    x_dbt_project: str = Header(default=DEFAULT),
) -> Union[OsmosisResetResult, OsmosisErrorContainer]:
    """Reparse a registered project on disk as determined by X-dbt-Project header writing
    manifest.json to target directory"""
    dbt: DbtProjectContainer = app.state.dbt_project_container
    if x_dbt_project == DEFAULT:
        project = dbt.get_default_project()
    else:
        project = dbt.get_project(x_dbt_project)
    if project is None:
        response.status_code = status.HTTP_404_NOT_FOUND
        return OsmosisErrorContainer(
            error=OsmosisError(
                code=OsmosisErrorCode.ProjectNotRegistered,
                message="Project is not registered. Make a POST request to the /register endpoint first to register a runner",
                data={"registered_projects": dbt.registered_projects()},
            )
        )

    # Get targets
    old_target = getattr(project.args, "target", project.config.target_name)
    new_target = target or old_target

    if not reset and old_target == new_target:
        # Async (target same)
        if project.mutex.acquire(blocking=False):
            background_tasks.add_task(_reset, project, reset, old_target, new_target)
            return OsmosisResetResult(result="Initializing project parsing")
        else:
            return OsmosisResetResult(result="Currently reparsing project")
    else:
        # Sync (target changed or reset is true)
        if project.mutex.acquire(blocking=old_target != new_target):
            rv = _reset(project, reset, old_target, new_target)
            if isinstance(rv, OsmosisErrorContainer):
                response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
            return rv
        else:
            return OsmosisResetResult(result="Currently reparsing project")


def _reset(
    runner: DbtProject, reset: bool, old_target: str, new_target: str
) -> Union[OsmosisResetResult, OsmosisErrorContainer]:
    """Use a mutex to ensure only a single reset can be running for any
    given project at any given time synchronously or asynchronously"""
    target_did_change = old_target != new_target
    try:
        runner.args.target = new_target
        runner.safe_parse_project(reset=reset or target_did_change)
    except Exception as reparse_err:
        runner.args.target = old_target
        rv = OsmosisErrorContainer(
            error=OsmosisError(
                code=OsmosisErrorCode.ProjectParseFailure,
                message=str(reparse_err),
                data=reparse_err.__dict__,
            )
        )
    else:
        runner._version += 1
        rv = OsmosisResetResult(
            result=(
                f"Profile target changed from {old_target} to {new_target}!"
                if target_did_change
                else f"Reparsed project with profile {old_target}!"
            )
        )
    finally:
        runner.mutex.release()
    return rv


@app.post(
    "/register",
    responses={status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": OsmosisErrorContainer}},
)
async def register(
    response: Response,
    project_dir: str,
    profiles_dir: str,
    target: Optional[str] = None,
    x_dbt_project: str = Header(),
) -> Union[OsmosisResetResult, OsmosisErrorContainer]:
    """Register a new project. This will parse the project on disk and load it into memory"""
    dbt: DbtProjectContainer = app.state.dbt_project_container
    if x_dbt_project in dbt:
        # We ask for X-dbt-Project header here to provide early-exit if we registered the project already
        # though adding the same project is idempotent, better to avoid the parse
        return OsmosisRegisterResult(added=x_dbt_project, projects=dbt.registered_projects())

    try:
        dbt.add_project(
            name_override=x_dbt_project,
            project_dir=project_dir,
            profiles_dir=profiles_dir,
            target=target,
        )
    except Exception as init_err:
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return OsmosisErrorContainer(
            error=OsmosisError(
                code=OsmosisErrorCode.ProjectParseFailure,
                message=str(init_err),
                data=init_err.__dict__,
            )
        )

    return OsmosisRegisterResult(added=x_dbt_project, projects=dbt.registered_projects())


@app.post(
    "/unregister",
    responses={status.HTTP_404_NOT_FOUND: {"model": OsmosisErrorContainer}},
)
async def unregister(
    response: Response,
    x_dbt_project: str = Header(),
) -> Union[OsmosisResetResult, OsmosisErrorContainer]:
    """Unregister a project. This drop a project from memory"""
    dbt: DbtProjectContainer = app.state.dbt_project_container
    project = dbt.get_project(x_dbt_project)
    if project is None:
        response.status_code = status.HTTP_404_NOT_FOUND
        return OsmosisErrorContainer(
            error=OsmosisError(
                code=OsmosisErrorCode.ProjectNotRegistered,
                message="Project is not registered. Make a POST request to the /register endpoint first to register a runner",
                data={"registered_projects": dbt.registered_projects()},
            )
        )
    dbt.drop_project(project)
    return OsmosisUnregisterResult(removed=project, projects=dbt.registered_projects())


@app.get("/health")
async def health_check(
    x_dbt_project: str = Header(default=DEFAULT),
) -> dict:
    """Checks if the server is running and accepting requests

    TODO: We will likely iterate on this, it is mostly for
    kubernetes liveness probes"""
    dbt: DbtProjectContainer = app.state.dbt_project_container
    if x_dbt_project == DEFAULT:
        project = dbt.get_default_project()
    else:
        project = dbt.get_project(x_dbt_project)
    return {
        "result": {
            "status": "ready",
            **(
                {
                    "project_name": project.config.project_name,
                    "target_name": project.config.target_name,
                    "profile_name": project.config.project_name,
                    "logs": project.config.log_path,
                    "runner_parse_iteration": project._version,
                }
                if project is not None
                else {}
            ),
            "timestamp": datetime.datetime.utcnow(),
            "error": None,
        },
        "dbt-osmosis-server": __name__,
    }


def run_server(host="localhost", port=8581):
    uvicorn.run(
        "dbt_osmosis.core.server_v2:app",
        host=host,
        port=port,
        log_level="info",
        reload=False,
    )


def test_server():
    """Some quick and dirty functional tests for the server"""
    import random
    import time
    from concurrent.futures import ThreadPoolExecutor

    client = TestClient(app)

    SIMULATED_CLIENTS = 50
    DUCKDB_PROJECTS = [
        "j_shop_1_duckdb",
        "j_shop_2_duckdb",
        "h_niceserver_1_duckdb",
        "h_niceserver_2_duckdb",
    ]
    SQLITE_PROJECTS = [
        "j_shop_1_sqlite",
        "j_shop_2_sqlite",
        "j_shop_3_sqlite",
        "j_shop_4_sqlite",
        "h_niceserver_1_sqlite",
    ]
    PROJECTS = DUCKDB_PROJECTS + SQLITE_PROJECTS

    e = ThreadPoolExecutor(max_workers=SIMULATED_CLIENTS)
    for proj in SQLITE_PROJECTS:
        register_response = client.post(
            "/register",
            params={
                "project_dir": "./demo_sqlite",
                "profiles_dir": "./demo_sqlite",
                "target": "dev",
            },
            headers={"X-dbt-Project": proj},
        )
        print(register_response.json())
    for proj in DUCKDB_PROJECTS:
        register_response = client.post(
            "/register",
            params={
                "project_dir": "./demo_duckdb",
                "profiles_dir": "./demo_duckdb",
                "target": "dev",
            },
            headers={"X-dbt-Project": proj},
        )
        print(register_response.json())

    # print("SQLITE")
    # response = client.post("/run", data="SELECT 1", headers={"X-dbt-Project": "jaffle_shop_sqlite"})
    # assert response.status_code == 200
    # print(response.json())
    # print("DUCKDB")
    # for i in range(10):
    #     response = client.post(
    #         "/run", data="SELECT 1", headers={"X-dbt-Project": "jaffle_shop_duckdb"}
    #     )
    #     assert response.status_code == 200
    #     print(response.json())

    STATEMENT = f"""
    {{% set payment_methods = ['credit_card', 'coupon', 'bank_transfer', 'gift_card'] %}}

    with orders as (

        select * from {{{{ ref('stg_orders') }}}}

    ),

    payments as (

        select * from {{{{ ref('stg_payments') }}}}

    ),

    order_payments as (

        select
            order_id,

            {{% for payment_method in payment_methods -%}}
            sum(case when payment_method = '{{{{ payment_method }}}}' then amount else 0 end) as {{{{ payment_method }}}}_amount,
            {{% endfor -%}}

            sum(amount) as total_amount

        from payments

        group by order_id

    ),

    final as (

        select
            orders.order_id,
            orders.customer_id,
            orders.order_date,
            orders.status,

            {{% for payment_method in payment_methods -%}}

            order_payments.{{{{ payment_method }}}}_amount,

            {{% endfor -%}}

            order_payments.total_amount as amount

        from orders


        left join order_payments
            on orders.order_id = order_payments.order_id

    )

    select * from final
    """
    LOAD_TEST_SIZE = 1000

    print("\n", "=" * 20)

    print("TEST COMPILE")
    t1 = time.perf_counter()
    futs = e.map(
        lambda i: client.post(
            "/compile",
            data=f"--> select {{{{ 1 + {i} }}}} \n{STATEMENT}",
            headers={"X-dbt-Project": random.choice(PROJECTS)},
        ).ok,
        range(LOAD_TEST_SIZE),
    )
    print("All Successful:", all(futs))
    t2 = time.perf_counter()
    print(
        (t2 - t1) / LOAD_TEST_SIZE,
        f"seconds per `/compile` across {LOAD_TEST_SIZE} calls from {SIMULATED_CLIENTS} simulated "
        f"clients randomly distributed between {len(PROJECTS)} different projects with a sql statement of ~{len(STATEMENT)} chars",
    )

    time.sleep(2.5)
    print("\n", "=" * 20)
    print(STATEMENT[:200], "\n...\n", STATEMENT[-200:])
    print("\n", "=" * 20)
    time.sleep(2.5)

    print("TEST RUN")
    t1 = time.perf_counter()
    futs = e.map(
        lambda i: client.post(
            "/run",
            data=f"-->> select {{{{ 1 + {i} }}}} \n{STATEMENT}",
            headers={"X-dbt-Project": random.choice(PROJECTS)},
        ).ok,
        range(LOAD_TEST_SIZE),
    )
    print("All Successful:", all(futs))
    t2 = time.perf_counter()
    print(
        (t2 - t1) / LOAD_TEST_SIZE,
        f"seconds per `/run` across {LOAD_TEST_SIZE} calls from {SIMULATED_CLIENTS} simulated "
        f"clients randomly distributed between {len(PROJECTS)} different projects with a sql statement of ~{len(STATEMENT)} chars",
    )
    e.shutdown(wait=True)


if __name__ == "__main__":
    test_server()
