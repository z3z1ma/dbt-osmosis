import datetime
import re
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import uvicorn
from fastapi import FastAPI, Request, Header, BackgroundTasks
from fastapi.testclient import TestClient
from pydantic import BaseModel

from dbt_osmosis.core.osmosis import DbtProject, DbtProjectContainer

DEFAULT = "__default__"
JINJA_CH = ["{{", "}}", "{%", "%}"]


app = FastAPI()
app.state.dbt_project_container = DbtProjectContainer()


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


@app.post(
    "/run",
    response_model=Union[OsmosisRunResult, OsmosisErrorContainer],
    openapi_extra={
        "requestBody": {
            "content": "text/plain",
            "required": True,
        },
    },
)
async def run_sql(
    request: Request,
    limit: int = 100,
    x_dbt_project: str = Header(default=DEFAULT),
) -> Union[OsmosisRunResult, OsmosisErrorContainer]:
    dbt: DbtProjectContainer = app.state.dbt_project_container
    if x_dbt_project == DEFAULT:
        project = dbt.get_default_project()
    else:
        project = dbt.get_project(x_dbt_project)
    if project is None:
        return OsmosisErrorContainer(
            error=OsmosisError(
                code=OsmosisErrorCode.ProjectNotRegistered,
                message="Project is not registered. Make a POST request to the /register endpoint first to register a runner",
                data={"registered_projects": dbt.registered_projects()},
            )
        )

    # Query Construction
    query = f'\n(\n{{# PAD #}}{(await request.body()).decode("utf-8").strip()}{{# PAD #}}\n)\n'
    query_with_limit = (
        # we need to support `TOP` too
        f"select * from ({query}) as osmosis_query limit {limit}"
    )

    try:
        result = project.execute_sql(query_with_limit)
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
        r"select \* from \(([\w\W]+)\) as osmosis_query", result.node.compiled_sql
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
    openapi_extra={
        "requestBody": {
            "content": "text/plain",
            "required": True,
        },
    },
)
async def compile_sql(
    request: Request,
    x_dbt_project: str = Header(default=DEFAULT),
) -> Union[OsmosisCompileResult, OsmosisErrorContainer]:
    dbt: DbtProjectContainer = app.state.dbt_project_container
    if x_dbt_project == DEFAULT:
        project = dbt.get_default_project()
    else:
        project = dbt.get_project(x_dbt_project)
    if project is None:
        return OsmosisErrorContainer(
            error=OsmosisError(
                code=OsmosisErrorCode.ProjectNotRegistered,
                message="Project is not registered. Make a POST request to the /register endpoint first to register a runner",
                data={"registered_projects": dbt.registered_projects()},
            )
        )

    # Query Compilation
    query: str = (await request.body()).decode("utf-8").strip()
    if any(CH in query for CH in JINJA_CH):
        try:
            compiled_query = project.compile_sql_cached(query)
        except Exception as compile_err:
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


@app.get(
    "/parse",
    response_model=Union[OsmosisResetResult, OsmosisErrorContainer],
)
async def reset(
    background_tasks: BackgroundTasks,
    reset: bool = False,
    target: Optional[str] = None,
    x_dbt_project: str = Header(default=DEFAULT),
) -> Union[OsmosisResetResult, OsmosisErrorContainer]:
    dbt: DbtProjectContainer = app.state.dbt_project_container
    if x_dbt_project == DEFAULT:
        project = dbt.get_default_project()
    else:
        project = dbt.get_project(x_dbt_project)
    if project is None:
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
            return _reset(project, reset, old_target, new_target)
        else:
            return OsmosisResetResult(result="Currently reparsing project")


def _reset(
    runner: DbtProject, reset: bool, old_target: str, new_target: str
) -> Union[OsmosisResetResult, OsmosisErrorContainer]:
    """Use a MUTEX to ensure"""
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


@app.post("/register")
async def register(
    project_dir: str,
    profiles_dir: str,
    target: Optional[str] = None,
    x_dbt_project: str = Header(),
) -> Union[OsmosisResetResult, OsmosisErrorContainer]:
    dbt: DbtProjectContainer = app.state.dbt_project_container
    if x_dbt_project in dbt:
        # We ask for X-dbt-Project header here to provide early-exit if we registered the project already
        # though adding the same project is idempotent, better to avoid the parse
        return OsmosisRegisterResult(added=x_dbt_project, projects=dbt.registered_projects())

    try:
        project = dbt.add_project(
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
        )

    return OsmosisRegisterResult(
        added=project.config.project_name, projects=dbt.registered_projects()
    )


@app.post("/unregister")
async def unregister(
    x_dbt_project: str = Header(),
) -> Union[OsmosisResetResult, OsmosisErrorContainer]:
    dbt: DbtProjectContainer = app.state.dbt_project_container
    project = dbt.get_project(x_dbt_project)
    if project is None:
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
    """TODO: We will likely iterate on this, it is mostly for
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
    client = TestClient(app)
    register_response = client.post(
        "/register",
        params={
            "project_dir": "./demo_sqlite",
            "profiles_dir": "./demo_sqlite",
            "target": "test",
        },
        headers={"X-dbt-Project": "jaffle_shop_sqlite"},
    )
    print(register_response.json())
    register_response = client.post(
        "/register",
        params={
            "project_dir": "./demo_duckdb",
            "profiles_dir": "./demo_duckdb",
            "target": "test",
        },
        headers={"X-dbt-Project": "jaffle_shop_duckdb"},
    )
    print(register_response.json())
    print("SQLITE")
    response = client.post("/run", data="SELECT 1", headers={"X-dbt-Project": "jaffle_shop_sqlite"})
    assert response.status_code == 200
    print(response.json())
    print("DUCKDB")
    for i in range(10):
        response = client.post(
            "/run", data="SELECT 1", headers={"X-dbt-Project": "jaffle_shop_duckdb"}
        )
        assert response.status_code == 200
        print(response.json())


if __name__ == "__main__":
    test_server()
