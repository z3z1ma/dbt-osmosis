from __future__ import annotations

import asyncio
import importlib
import sys
import types
from types import SimpleNamespace


def _import_proxy_with_fake_mysql_mimic(monkeypatch):
    """Import the proxy module without installing the optional proxy extra."""
    for module_name in (
        "dbt_osmosis.sql.proxy",
        "mysql_mimic",
        "mysql_mimic.errors",
        "mysql_mimic.results",
        "mysql_mimic.schema",
        "mysql_mimic.session",
    ):
        monkeypatch.delitem(sys.modules, module_name, raising=False)

    mysql_mimic = types.ModuleType("mysql_mimic")
    errors = types.ModuleType("mysql_mimic.errors")
    results = types.ModuleType("mysql_mimic.results")
    schema = types.ModuleType("mysql_mimic.schema")
    session = types.ModuleType("mysql_mimic.session")

    class MysqlError(Exception):
        pass

    class MysqlServer:
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs

        async def serve_forever(self) -> None:
            return None

    class Session:
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs
            self.middlewares = []

    class Column:
        def __init__(self, **kwargs) -> None:
            self.__dict__.update(kwargs)

    class InfoSchema:
        def __init__(self, tables) -> None:
            self.tables = tables

    class Query:
        pass

    def dict_depth(value) -> int:
        if not isinstance(value, dict) or not value:
            return 0
        return 1 + max(dict_depth(item) for item in value.values())

    def info_schema_tables(columns):
        return list(columns)

    mysql_mimic.MysqlError = MysqlError
    mysql_mimic.MysqlServer = MysqlServer
    mysql_mimic.Session = Session
    errors.MysqlError = MysqlError
    results.AllowedResult = object
    schema.Column = Column
    schema.InfoSchema = InfoSchema
    schema.dict_depth = dict_depth
    schema.info_schema_tables = info_schema_tables
    session.Query = Query

    monkeypatch.setitem(sys.modules, "mysql_mimic", mysql_mimic)
    monkeypatch.setitem(sys.modules, "mysql_mimic.errors", errors)
    monkeypatch.setitem(sys.modules, "mysql_mimic.results", results)
    monkeypatch.setitem(sys.modules, "mysql_mimic.schema", schema)
    monkeypatch.setitem(sys.modules, "mysql_mimic.session", session)

    return importlib.import_module("dbt_osmosis.sql.proxy")


def test_query_preserves_original_sql_instead_of_reserializing(monkeypatch) -> None:
    proxy = _import_proxy_with_fake_mysql_mimic(monkeypatch)
    captured: dict[str, object] = {}

    class Adapter:
        def type(self) -> str:
            return "duckdb"

    class ExpressionThatMustNotReserialize:
        def sql(self, **kwargs) -> str:
            raise AssertionError("DbtSession.query() must pass the original SQL string")

    class Row(dict):
        pass

    def fake_execute_sql_code(project, sql: str):
        captured["project"] = project
        captured["sql"] = sql
        table = SimpleNamespace(rows={0: Row(id=1)}, column_names=("id",))
        return SimpleNamespace(code=None), table

    monkeypatch.setattr(proxy, "execute_sql_code", fake_execute_sql_code)

    project = SimpleNamespace(adapter=Adapter())
    session = proxy.DbtSession(project)
    original_sql = 'select $1 as "MixedCase" -- preserve client text'

    result = asyncio.run(session.query(ExpressionThatMustNotReserialize(), original_sql, {}))

    assert captured == {"project": project, "sql": original_sql}
    assert [list(values) for values in result[0]] == [[1]]
    assert result[1] == ("id",)


def test_comment_middleware_is_documented_and_limited_to_in_memory_updates(monkeypatch) -> None:
    proxy = _import_proxy_with_fake_mysql_mimic(monkeypatch)
    column = SimpleNamespace(name="customer_id", description="old column description")
    node = SimpleNamespace(
        schema="analytics",
        name="orders",
        description="old table description",
        columns={"customer_id": column},
    )
    project = SimpleNamespace(
        manifest=SimpleNamespace(sources={}, nodes={"model.project.orders": node})
    )
    session = proxy.DbtSession(project)

    async def unexpected_next():
        raise AssertionError("comment middleware should handle ALTER COMMENT statements")

    column_comment = SimpleNamespace(
        expression=proxy.exp.Command(this="ALTER"),
        sql="ALTER TABLE analytics.orders MODIFY COLUMN customer_id TEXT COMMENT 'Customer identifier';",
        next=unexpected_next,
    )
    table_comment = SimpleNamespace(
        expression=proxy.exp.Command(this="ALTER"),
        sql="ALTER TABLE analytics.orders COMMENT = 'Orders table';",
        next=unexpected_next,
    )

    asyncio.run(session._alter_table_comment_middleware(column_comment))
    asyncio.run(session._alter_table_comment_middleware(table_comment))

    assert column.description == "Customer identifier"
    assert node.description == "Orders table"

    docstring = proxy.DbtSession._alter_table_comment_middleware.__doc__ or ""
    assert "in-memory" in docstring.lower()
    assert "write the changes to disk" not in docstring.lower()
