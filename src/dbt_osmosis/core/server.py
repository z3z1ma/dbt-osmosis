import datetime
import decimal
import re
import threading
import time
import uuid

import orjson
from bottle import JSONPlugin, install, request, response, route, run, uninstall

from dbt_osmosis.core.log_controller import logger
from dbt_osmosis.core.osmosis import DbtOsmosis

JINJA_CH = ["{{", "}}", "{%", "%}"]
MUTEX = threading.Lock()


def default(obj):
    if isinstance(obj, decimal.Decimal):
        return float(obj)
    raise TypeError


class DbtOsmosisPlugin:
    name = "dbt-osmosis"
    api = 2

    def __init__(self, runner: DbtOsmosis):
        self.runner = runner

    def apply(self, callback, route):
        def wrapper(*args, **kwargs):
            start = time.time()
            body = callback(*args, **kwargs, runner=self.runner)
            end = time.time()
            response.headers["X-dbt-Exec-Time"] = str(end - start)
            return body

        return wrapper


@route("/run", method="POST")
def run_sql(runner: DbtOsmosis):
    query = f'\n{request.body.read().decode("utf-8").strip()}\n'
    limit = request.query.get("limit", 200)
    query_with_limit = f"select * from ({query}) as osmosis_query limit {limit}"
    try:
        result = runner.execute_sql(query_with_limit)
    except Exception as exc:
        return {"error": {"code": 2, "message": str(exc), "data": exc.__dict__}}
    compiled_query = re.search(
        r"select \* from \(([\w\W]+)\) as osmosis_query", result.node.compiled_sql
    ).groups()[0]
    return {
        "rows": [list(row) for row in result.table.rows],
        "column_names": result.table.column_names,
        "compiled_sql": compiled_query,
        "raw_sql": query,
    }


@route("/compile", method="POST")
def compile_sql(runner: DbtOsmosis):
    query = request.body.read().decode("utf-8").strip()
    if any(CH in query for CH in JINJA_CH):
        try:
            compiled_query = runner.compile_sql(query)
        except Exception as exc:
            return {"error": {"code": 1, "message": str(exc), "data": exc.__dict__}}
    else:
        compiled_query = query
    return {"result": compiled_query}


@route(["/parse", "/reset"])
def reset(runner: DbtOsmosis):
    reset = str(request.query.get("reset", "false")).lower() == "true"
    old_target = getattr(runner.args, "target", runner.config.target_name)
    new_target = request.query.get("target", old_target)
    target_did_change = old_target != new_target
    if not reset or not target_did_change:
        # Async (target same)
        if MUTEX.acquire(blocking=False):
            logger().debug("Mutex locked")
            parse_job = threading.Thread(
                target=_reset, args=(runner, reset, old_target, new_target)
            )
            parse_job.start()
            return {"result": "Initializing project parsing"}
        else:
            logger().debug("Mutex is locked, reparse in progress")
            return {"result": "Currently reparsing project"}
    else:
        # Sync (target changed)
        if MUTEX.acquire(blocking=False):
            logger().debug("Mutex locked")
            return _reset(runner, reset, old_target, new_target)
        else:
            logger().debug("Mutex is locked, reparse in progress")
            return {"result": "Currently reparsing project"}


def _reset(runner: DbtOsmosis, reset: bool, old_target: str, new_target: str):
    target_did_change = old_target != new_target
    rv = {}
    try:
        runner.args.target = new_target
        logger().debug("Starting reparse")
        runner.rebuild_dbt_manifest(reset=reset or target_did_change)
    except Exception as reparse_err:
        logger().debug("Reparse error")
        runner.args.target = old_target
        rv["error"] = {"code": 3, "message": str(reparse_err), "data": reparse_err.__dict__}
    else:
        logger().debug("Reparse success")
        runner._version += 1
        uninstall(DbtOsmosisPlugin), install(DbtOsmosisPlugin(runner=runner))
        rv["result"] = (
            f"Profile target changed from {old_target} to {new_target}!"
            if target_did_change
            else f"Reparsed project with profile {old_target}!"
        )
    finally:
        logger().debug("Unlocking mutex")
        MUTEX.release()
    return rv


@route(["/health", "/api/health"], methods="GET")
def health_check(runner: DbtOsmosis) -> dict:
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


def run_server(runner: DbtOsmosis, host="localhost", port=8581):
    install(DbtOsmosisPlugin(runner=runner))
    install(JSONPlugin(json_dumps=lambda body: orjson.dumps(body, default=default).decode("utf-8")))
    run(host=host, port=port)
