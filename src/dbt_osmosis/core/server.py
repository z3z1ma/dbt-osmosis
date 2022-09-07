import datetime
import decimal
import time
import uuid
from functools import partial
from typing import Callable

import orjson
from bottle import JSONPlugin, install, request, response, route, run

from dbt_osmosis.core.osmosis import DbtOsmosis

JINJA_CH = ["{{", "}}", "{%", "%}"]


def default(obj):
    if isinstance(obj, decimal.Decimal):
        return float(obj)
    raise TypeError


def _dbt_query_engine(callback: Callable, runner: DbtOsmosis):
    def wrapper(*args, **kwargs):
        start = time.time()
        body = callback(*args, **kwargs, runner=runner)
        end = time.time()
        response.headers["X-dbt-Exec-Time"] = str(end - start)
        return body

    return wrapper


@route("/run", method="POST")
def run_sql(runner: DbtOsmosis):
    query = request.body.read().decode("utf-8")
    limit = request.query.get("limit", 200)
    if any(CH in query for CH in JINJA_CH):
        try:
            compiled_query = runner.compile_sql(query).compiled_sql
        except Exception as exc:
            return {"error": {"code": 1, "message": str(exc), "data": exc.__dict__}}
    else:
        compiled_query = query

    query_with_limit = f"select * from ({compiled_query}) as osmosis_query limit {limit}"
    try:
        _, table = runner.execute_sql(query_with_limit, fetch=True)
    except Exception as exc:
        return {"error": {"code": 2, "message": str(exc), "data": exc.__dict__}}

    return {
        "rows": [list(row) for row in table.rows],
        "column_names": table.column_names,
        "compiled_sql": compiled_query,
        "raw_sql": query,
    }


@route("/compile", method="POST")
def compile_sql(runner: DbtOsmosis):
    query = request.body.read().decode("utf-8")
    if any(CH in query for CH in JINJA_CH):
        try:
            compiled_query = runner.compile_sql(query).compiled_sql
        except Exception as exc:
            return {"error": {"code": 1, "message": str(exc), "data": exc.__dict__}}
    else:
        compiled_query = query
    return {"result": compiled_query}


@route("/reset")
def reset(runner: DbtOsmosis):
    target = request.query.get("target", runner.config.target_name)
    runner.profile.target_name = target
    runner.config.target_name = target
    try:
        runner.rebuild_dbt_manifest()
    except Exception as exc:
        return {"error": {"code": 3, "message": str(exc), "data": exc.__dict__}}
    else:
        return {"result": "success"}


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
            "error": None,
        },
        "id": str(uuid.uuid4()),
        "dbt-osmosis-server": __name__,
    }


def run_server(runner: DbtOsmosis, host="localhost", port=8581):
    dbt_query_engine = partial(_dbt_query_engine, runner=runner)
    install(dbt_query_engine)
    install(JSONPlugin(json_dumps=lambda body: orjson.dumps(body, default=default).decode("utf-8")))
    run(host=host, port=port)
