"""Proxy server experiment that any MySQL client (including BI tools) can connect to."""

# pyright: reportMissingTypeStubs=false, reportAny=false, reportImplicitOverride=false, reportUnknownMemberType=false, reportUnusedImport=false, reportUnknownParameterType=false
import asyncio
import functools
import re
import typing as t
from collections import defaultdict
from collections.abc import Iterator

from dbt.adapters.contracts.connection import AdapterResponse
from mysql_mimic import MysqlServer, Session
from mysql_mimic.errors import MysqlError
from mysql_mimic.schema import (
    Column,
    InfoSchema,
    dict_depth,  # pyright: ignore[reportUnknownVariableType, reportPrivateLocalImportUsage]
    info_schema_tables,
)
from sqlglot import exp

import dbt_osmosis.core.logger as logger
from dbt_osmosis.core.osmosis import (
    DbtConfiguration,
    DbtProjectContext,
    _has_jinja,  # pyright: ignore[reportPrivateUsage]
    compile_sql_code,
    create_dbt_project_context,
    execute_sql_code,
)

# TODO: this doesn't capture comment body consistently
ALTER_MODIFY_COL_PATTERN = re.compile(
    r"""
    ^\s*                              # start, allow leading whitespace
    ALTER\s+TABLE                     # "ALTER TABLE" (case-insensitive via flags=)
    \s+
    (?:                               # optional schema part:
       (?:"(?P<schema>[^"]+)"         # double-quoted schema
         |(?P<schema_unquoted>\w+)    # or unquoted schema
       )
       \s*\.
    )?
    (?:"(?P<table>[^"]+)"             # table in double-quotes
      |(?P<table_unquoted>\w+)        # or unquoted table
    )
    \s+
    MODIFY\s+COLUMN\s+                # "MODIFY COLUMN" (case-insensitive via flags=)
    (?:"(?P<column>[^"]+)"            # column in double-quotes
      |(?P<column_unquoted>\w+)       # or unquoted column
    )
    .*?                               # lazily consume anything until we might see COMMENT
    (?:                                # optional comment group
       COMMENT\s+                     # must have "COMMENT" then space(s)
       (["'])                         # capture the quote symbol in group 1
       (?P<comment>
         (?:.|[^"'])*              # any escaped char or anything that isn't ' or "
       )
       \1                             # match the same quote symbol (group 1)
    )?
    \s*;?\s*$                         # optional whitespace, optional semicolon
    """,
    flags=re.IGNORECASE | re.DOTALL | re.VERBOSE,
)


def parse_alter_modify_column(sql: str) -> dict[str, str] | None:
    """
    Attempt to parse a statement like:
        ALTER TABLE schema.table MODIFY COLUMN col TYPE ... COMMENT 'some text';

    Returns None if the pattern does not match, otherwise a dict with:
       {
         "schema": ... or None,
         "table":  ...,
         "column": ...,
         "comment": ... or None
       }
    """
    match = ALTER_MODIFY_COL_PATTERN.match(sql)
    if not match:
        return None

    # Because we have both quoted and unquoted named groups, pick whichever matched:
    schema = match.group("schema") or match.group("schema_unquoted")
    table = match.group("table") or match.group("table_unquoted")
    column = match.group("column") or match.group("column_unquoted")
    comment = match.group("comment")  # can be None if COMMENT was not present

    return {"schema": schema, "table": table, "column": column, "comment": comment}


class QueryException(MysqlError):
    def __init__(self, response: AdapterResponse) -> None:
        self.response: AdapterResponse = response
        super().__init__(response._message)  # pyright: ignore[reportPrivateUsage]


class DbtSession(Session):
    def __init__(self, project: DbtProjectContext, *args: t.Any, **kwargs: t.Any) -> None:
        self.project: DbtProjectContext = project
        super().__init__(*args, **kwargs)

    def _parse(self, sql: str) -> list[exp.Expression]:
        if _has_jinja(sql):
            node = compile_sql_code(self.project, sql)
            sql = node.compiled_code or node.raw_code
        return [e for e in self.dialect().parse(sql) if e]

    async def query(self, expression: exp.Expression, sql: str, attrs: dict[str, t.Any]):
        logger.info("Query: %s", sql)
        if isinstance(expression, exp.Command):
            cmd = f"{expression.this} {expression.expression}"
            doc_update = "alter" in sql.lower() and parse_alter_modify_column(cmd)
            if doc_update:
                logger.info("Will update doc: %s", doc_update)
            else:
                logger.info("Ignoring command %s", sql)
            return (), []  # pyright: ignore[reportUnknownVariableType]
        resp, table = await asyncio.to_thread(
            execute_sql_code, self.project, expression.sql(dialect=self.project.adapter.type())
        )
        if resp.code:
            raise QueryException(resp)
        logger.info(table)
        return [
            t.cast(tuple[t.Any], row.values()) for row in t.cast(tuple[t.Any], table.rows.values())
        ], t.cast(tuple[str], table.column_names)

    async def schema(self):
        schema: defaultdict[str, dict[str, dict[str, tuple[str, str | None]]]] = defaultdict(dict)
        for source in self.project.manifest.sources.values():
            schema[source.schema][source.name] = {
                c.name: (c.data_type or "UNKOWN", c.description) for c in source.columns.values()
            }
        for node in self.project.manifest.nodes.values():
            schema[node.schema][node.name] = {
                c.name: (c.data_type or "UNKOWN", c.description) for c in node.columns.values()
            }
        iter_columns = mapping_to_columns(schema)
        return InfoSchema(info_schema_tables(iter_columns))


def mapping_to_columns(schema: dict[str, t.Any]) -> Iterator[Column]:
    """Convert a schema mapping into a list of Column instances"""
    depth = dict_depth(schema)
    if depth < 2:
        return
    if depth == 2:
        # {table: {col: type}}
        schema = {"": schema}
        depth += 1
    if depth == 3:
        # {db: {table: {col: type}}}
        schema = {"def": schema}  # def is the default MySQL catalog
        depth += 1
    if depth != 4:
        raise MysqlError("Invalid schema mapping")

    for catalog, dbs in schema.items():
        for db, tables in dbs.items():
            for table, cols in tables.items():
                for column, (coltype, comment) in cols.items():
                    yield Column(
                        name=column,
                        type=coltype,
                        table=table,
                        schema=db,
                        catalog=catalog,
                        comment=comment,
                    )


if __name__ == "__main__":
    c = DbtConfiguration()
    server = MysqlServer(
        session_factory=functools.partial(DbtSession, create_dbt_project_context(c))
    )
    asyncio.run(server.serve_forever())
