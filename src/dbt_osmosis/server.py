from typing import Dict

from flask import Flask, request
from dbt_client import DbtClient
import click


app = Flask(__name__)


STATE: Dict[str, DbtClient] = {}


@app.route("/run", methods=["POST"])
def run_sql():
    data = request.json
    if data is None:
        return {"error": "Content-Type header must be provided and set to application/json"}
    if not data.get("sql"):
        return {"error": "Expected `sql` key in request body"}
    print("\nRUN QUERY")
    print("==========\n")
    print(data)
    print("\n")
    try:
        # Lets consider memoization
        result = STATE["server"].run_sql("dbt-osmosis", data["sql"], sync=True)
    except Exception as err:
        return {"error": err}
    else:
        return result["result"]["results"][0]["table"]


@app.route("/compile", methods=["POST"])
def compile_sql():
    data = request.json
    if data is None:
        return {"error": "Content-Type header must be provided and set to application/json"}
    if not data.get("sql"):
        return {"error": "Expected `sql` key in request body"}
    print("\nCOMPILING QUERY")
    print("================\n")
    print(data)
    print("\n")
    try:
        # Lets consider memoization
        result = STATE["server"].compile_sql("dbt-osmosis", data["sql"], sync=True)
    except Exception as err:
        return {"error": err}
    else:
        return {"result": result["result"]["results"][0]["compiled_sql"]}


@click.group()
def cli():
    pass


@cli.command()
@click.argument("port", type=click.INT, default=8581)
@click.argument("rpc_port", type=click.INT, default=8580)
def serve(port: int, rpc_port: int):
    # TODO: We can consider osmosis handling RPC spin up? Or user/ts
    STATE["server"] = DbtClient(port=rpc_port)
    app.run("localhost", port)


if __name__ == "__main__":
    cli()
