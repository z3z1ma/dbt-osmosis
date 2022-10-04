"""Server Interface Notes
Executing `dbt-osmosis server` will seed a "default" project based on
initially supplied --project-dir / --profiles-dir args
TODO: we will probably gut this and migrate it all to Fast API in the near-term

/run [?limit=...]
    POST
    Header: X-dbt-Project=projectName Content-Type=text/plain
    Body: dbt SQL statement
    Action: Executes SQL (uses default project if X-dbt-Project not supplied)

/compile
    POST
    Header: X-dbt-Project=projectName Content-Type=text/plain
    Body: dbt SQL statement
    Action: Compiles SQL (uses default project if X-dbt-Project not supplied)

/parse [?target=...&reset=true/false]
    GET
    Header: X-dbt-Project
    reparses project and writes manifest.json (uses default project if X-dbt-Project not supplied)

/register
    POST
    Header: X-dbt-Project [required] Content-Type=application/json
    adds new project runner

/unregister 
    POST
    Header: X-dbt-Project [required] Content-Type=application/json
    removes project runner
"""
import datetime
import decimal
import re
import threading
import time
import uuid
from enum import Enum
from typing import Any, Dict, List, Union

import orjson
from bottle import JSONPlugin, install, request, response, route, run
from pydantic import BaseModel

from dbt_osmosis.core.log_controller import logger
from dbt_osmosis.core.osmosis import DbtProject, has_jinja

DEFAULT = "__default__"
MUTEX = threading.Lock()


def default(obj):
    if isinstance(obj, decimal.Decimal):
        return float(obj)
    raise TypeError


class OsmosisRunResult(BaseModel):
    column_names: List[str]
    rows: List[List[Any]]
    raw_sql: str
    compiled_sql: str


class OsmosisCompileResult(BaseModel):
    result: str


class OsmosisResetResult(BaseModel):
    result: str


class OsmosisRegisterResult(BaseModel):
    added: str
    projects: List[str]


class OsmosisUnregisterResult(BaseModel):
    removed: str
    projects: List[str]


class OsmosisErrorCode(Enum):
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


class DbtOsmosisPlugin:
    name = "dbt-osmosis"
    api = 2

    def __init__(self, runner: DbtProject):
        self.runners: Dict[str, DbtProject] = {
            DEFAULT: runner,
            runner.config.project_name: runner,
        }

    def apply(self, callback, route):
        def wrapper(*args, **kwargs):
            start = time.time()
            # Headers are read-only so we can't inject it, handle it at the route level
            # if not request.get_header("X-dbt-Project"):
            #     request.headers["X-dbt-Project"] = self.runners[DEFAULT].config.project_name
            body = callback(*args, **kwargs, runners=self.runners)
            end = time.time()
            response.headers["X-dbt-Exec-Time"] = str(end - start)
            return body

        return wrapper


@route("/run", method="POST")
def run_sql(runners: Dict[str, DbtProject]) -> Union[OsmosisRunResult, OsmosisErrorContainer, str]:
    # Project Support
    project = request.get_header("X-dbt-Project", DEFAULT)
    project_runner = runners.get(project)
    if project != DEFAULT and not project_runner:
        return OsmosisErrorContainer(
            error=OsmosisError(
                code=OsmosisErrorCode.ProjectNotRegistered,
                message="Project is not registered. Make a POST request to the /register endpoint first to register a runner",
                data={"registered_projects": runners.keys()},
            )
        ).json()
    elif not project_runner:
        project_runner = runners[DEFAULT]

    # Query Construction
    query = f'\n(\n{{# PAD #}}{request.body.read().decode("utf-8").strip()}{{# PAD #}}\n)\n'
    limit = request.query.get("limit", 200)
    query_with_limit = (
        # we need to support `TOP` too
        f"select * from ({query}) as osmosis_query limit {limit}"
    )

    try:
        result = project_runner.execute_sql(query_with_limit)
    except Exception as execution_err:
        return OsmosisErrorContainer(
            error=OsmosisError(
                code=OsmosisErrorCode.ExecuteSqlFailure,
                message=str(execution_err),
                data=execution_err.__dict__,
            )
        ).json()

    # Re-extract compiled query and return data structure
    compiled_query = re.search(
        r"select \* from \(([\w\W]+)\) as osmosis_query", result.compiled_sql
    ).groups()[0]
    return OsmosisRunResult(
        rows=[list(row) for row in result.table.rows],
        column_names=result.table.column_names,
        compiled_sql=compiled_query.strip()[1:-1],
        raw_sql=query,
    ).json()


@route("/compile", method="POST")
def compile_sql(
    runners: Dict[str, DbtProject]
) -> Union[OsmosisCompileResult, OsmosisErrorContainer, str]:
    # Project Support
    project = request.get_header("X-dbt-Project", DEFAULT)
    project_runner = runners.get(project)
    if project != DEFAULT and not project_runner:
        return OsmosisErrorContainer(
            error=OsmosisError(
                code=OsmosisErrorCode.ProjectNotRegistered,
                message="Project is not registered. Make a POST request to the /register endpoint first to register a runner",
                data={"registered_projects": runners.keys()},
            )
        ).json()
    elif not project_runner:
        project_runner = runners[DEFAULT]

    # Query Compilation
    query: str = request.body.read().decode("utf-8").strip()
    if has_jinja(query):
        try:
            compiled_query = project_runner.compile_sql(query).compiled_sql
        except Exception as compile_err:
            return OsmosisErrorContainer(
                error=OsmosisError(
                    code=OsmosisErrorCode.CompileSqlFailure,
                    message=str(compile_err),
                    data=compile_err.__dict__,
                )
            ).json()
    else:
        compiled_query = query

    return OsmosisCompileResult(result=compiled_query).json()


@route(["/parse", "/reset"])
def reset(runners: Dict[str, DbtProject]) -> Union[OsmosisResetResult, OsmosisErrorContainer, str]:
    # Project Support
    project = request.get_header("X-dbt-Project", DEFAULT)
    project_runner = runners.get(project)
    if project != DEFAULT and not project_runner:
        return OsmosisErrorContainer(
            error=OsmosisError(
                code=OsmosisErrorCode.ProjectNotRegistered,
                message="Project is not registered. Make a POST request to the /register endpoint first to register a runner",
                data={"registered_projects": runners.keys()},
            )
        ).json()
    elif not project_runner:
        project_runner = runners[DEFAULT]

    # Determines if we should clear caches and reset config before re-seeding runner
    reset = str(request.query.get("reset", "false")).lower() == "true"

    # Get targets
    old_target = getattr(project_runner.args, "target", project_runner.config.target_name)
    new_target = request.query.get("target", old_target)

    if not reset and old_target == new_target:
        # Async (target same)
        if MUTEX.acquire(blocking=False):
            logger().debug("Mutex locked")
            parse_job = threading.Thread(
                target=_reset, args=(project_runner, reset, old_target, new_target)
            )
            parse_job.start()
            return OsmosisResetResult(result="Initializing project parsing").json()
        else:
            logger().debug("Mutex is locked, reparse in progress")
            return OsmosisResetResult(result="Currently reparsing project").json()
    else:
        # Sync (target changed or reset is true)
        if MUTEX.acquire(blocking=old_target != new_target):
            logger().debug("Mutex locked")
            return _reset(project_runner, reset, old_target, new_target).json()
        else:
            logger().debug("Mutex is locked, reparse in progress")
            return OsmosisResetResult(result="Currently reparsing project").json()


def _reset(
    runner: DbtProject, reset: bool, old_target: str, new_target: str
) -> Union[OsmosisResetResult, OsmosisErrorContainer]:
    target_did_change = old_target != new_target
    try:
        runner.args.target = new_target
        logger().debug("Starting reparse")
        runner.rebuild_dbt_manifest(reset=reset or target_did_change)
    except Exception as reparse_err:
        logger().debug("Reparse error")
        runner.args.target = old_target
        rv = OsmosisErrorContainer(
            error=OsmosisError(
                code=OsmosisErrorCode.ProjectParseFailure,
                message=str(reparse_err),
                data=reparse_err.__dict__,
            )
        )
    else:
        logger().debug("Reparse success")
        runner._version += 1
        rv = OsmosisResetResult(
            result=(
                f"Profile target changed from {old_target} to {new_target}!"
                if target_did_change
                else f"Reparsed project with profile {old_target}!"
            )
        )
    finally:
        logger().debug("Unlocking mutex")
        MUTEX.release()
    return rv


@route("/register", method="POST")
def register(
    runners: Dict[str, DbtProject]
) -> Union[OsmosisResetResult, OsmosisErrorContainer, str]:
    # Project Support
    project = request.get_header("X-dbt-Project")
    if not project:
        return OsmosisErrorContainer(
            error=OsmosisError(
                code=OsmosisErrorCode.ProjectHeaderNotSupplied,
                message="Project header `X-dbt-Project` was not supplied but is required for this endpoint",
                data=dict(request.headers),
            )
        ).json()
    if project in runners:
        # Idempotent
        return OsmosisRegisterResult(added=project, projects=runners.keys()).json()

    # Inputs
    project_dir = request.json["project_dir"]
    profiles_dir = request.json["profiles_dir"]
    target = request.json.get("target")

    try:
        new_runner = DbtProject(
            project_dir=project_dir,
            profiles_dir=profiles_dir,
            target=target,
        )
    except Exception as init_err:
        return OsmosisErrorContainer(
            error=OsmosisError(
                code=OsmosisErrorCode.ProjectParseFailure,
                message=str(init_err),
                data=init_err.__dict__,
            )
        ).json()

    runners[project] = new_runner
    return OsmosisRegisterResult(added=project, projects=runners.keys()).json()


@route("/unregister", method="POST")
def unregister(
    runners: Dict[str, DbtProject]
) -> Union[OsmosisResetResult, OsmosisErrorContainer, str]:
    # Project Support
    project = request.get_header("X-dbt-Project")
    if not project:
        return OsmosisErrorContainer(
            error=OsmosisError(
                code=OsmosisErrorCode.ProjectHeaderNotSupplied,
                message="Project header `X-dbt-Project` was not supplied but is required for this endpoint",
                data=dict(request.headers),
            )
        ).json()
    if project not in runners:
        return OsmosisErrorContainer(
            error=OsmosisError(
                code=OsmosisErrorCode.ProjectNotRegistered,
                message="Project is not registered. Make a POST request to the /register endpoint first to register a runner",
                data={"registered_projects": runners.keys()},
            )
        ).json()
    runners.pop(project)
    return OsmosisUnregisterResult(removed=project, projects=runners.keys()).json()


@route(["/health", "/api/health"], methods="GET")
def health_check(runner: DbtProject) -> dict:
    return {
        "result": {
            "status": "ready",
            "project_name": runner.config.project_name,
            "target_name": runner.config.target_name,
            "profile_name": runner.config.project_name,
            "logs": runner.config.log_path,
            "timestamp": str(datetime.datetime.utcnow()),
            "runner_parse_iteration": runner._version,
            "error": None,
        },
        "id": str(uuid.uuid4()),
        "dbt-osmosis-server": __name__,
    }


def run_server(runner: DbtProject, host="localhost", port=8581):
    install(DbtOsmosisPlugin(runner=runner))
    install(JSONPlugin(json_dumps=lambda body: orjson.dumps(body, default=default).decode("utf-8")))
    run(host=host, port=port)
