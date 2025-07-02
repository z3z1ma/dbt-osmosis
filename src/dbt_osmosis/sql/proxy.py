# pyright: reportMissingTypeStubs=false, reportAny=false, reportImplicitOverride=false, reportUnknownMemberType=false, reportUnusedImport=false, reportUnknownParameterType=false
"""Proxy server experiment that any MySQL client (including BI tools) can connect to."""

import asyncio
import functools
import re
import typing as t
from collections import defaultdict
from collections.abc import Iterator
from itertools import chain

from dbt.adapters.contracts.connection import AdapterResponse
from mysql_mimic import MysqlServer, Session
from mysql_mimic.errors import MysqlError
from mysql_mimic.results import AllowedResult
from mysql_mimic.schema import (
    Column,
    InfoSchema,
    dict_depth,  # pyright: ignore[reportUnknownVariableType, reportPrivateLocalImportUsage]
    info_schema_tables,
)
from mysql_mimic.session import Query
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

ALTER_TABLE_MODIFY_COLUMN_COMMENT = re.compile(
    r"(?i)(?:/\*.*?\*/\s*)?ALTER TABLE\s+(?:(?P<schema>[^\s\.]+)\.)?(?P<table>[^\s\.]+)\s+MODIFY COLUMN\s+(?P<column>[^\s]+)\s+.*?COMMENT\s+'(?P<comment>[^']*)';?"
)

ALTER_TABLE_COMMENT = re.compile(
    r"(?i)(?:/\*.*?\*/\s*)?ALTER TABLE\s+(?:(?P<schema>[^\s\.]+)\.)?(?P<table>[^\s\.]+)\s+COMMENT\s*=\s*'(?P<comment>[^']*)';"
)


def _regex_parse_to_complete_dict(sql: str, pattern: re.Pattern[str]) -> dict[str, str] | None:
    """Parse a SQL statement using a regex pattern and return a dict with the matched groups ensuring all are present"""
    if match := pattern.match(sql):
        result = match.groupdict()
        if all(result.values()):
            return result


class QueryException(MysqlError):
    def __init__(self, response: AdapterResponse) -> None:
        super().__init__(response._message)  # pyright: ignore[reportPrivateUsage]
        self.response: AdapterResponse = response


class DbtSession(Session):
    def __init__(self, project: DbtProjectContext, *args: t.Any, **kwargs: t.Any) -> None:
        super().__init__(*args, **kwargs)
        self.project: DbtProjectContext = project
        self.middlewares.append(self._alter_table_comment_middleware)

    def _parse(self, sql: str) -> list[exp.Expression]:
        if _has_jinja(sql):
            node = compile_sql_code(self.project, sql)
            sql = node.compiled_code or node.raw_code
        return [e for e in self.dialect().parse(sql) if e]

    async def _alter_table_comment_middleware(self, q: Query) -> AllowedResult:
        """Intercept ALTER TABLE ... MODIFY COLUMN ... COMMENT statements

        This middleware will update the column description in the dbt project manifest. Eventually
        it could use our Yaml context class to actually write the changes to disk.
        """
        if isinstance(q.expression, exp.Command):
            lower_sql = q.sql.lower()
            likely_alter_column_comment = all(
                k in lower_sql for k in ("alter", "table", "modify", "column", "comment")
            )
            if doc_update_req := (
                likely_alter_column_comment
                and _regex_parse_to_complete_dict(q.sql, ALTER_TABLE_MODIFY_COLUMN_COMMENT)
            ):
                ref = (doc_update_req["schema"], doc_update_req["table"])
                for node in chain(
                    self.project.manifest.sources.values(), self.project.manifest.nodes.values()
                ):
                    if ref == (node.schema, node.name):
                        for column in node.columns.values():
                            if column.name == doc_update_req["column"]:
                                column.description = doc_update_req["comment"]
                                break
            likely_alter_table_comment = all(k in lower_sql for k in ("alter", "table", "comment"))
            if doc_update_req := (
                likely_alter_table_comment
                and _regex_parse_to_complete_dict(q.sql, ALTER_TABLE_COMMENT)
            ):
                ref = (doc_update_req["schema"], doc_update_req["table"])
                for node in chain(
                    self.project.manifest.sources.values(), self.project.manifest.nodes.values()
                ):
                    if ref == (node.schema, node.name):
                        node.description = doc_update_req["comment"]
            return [], []
        return await q.next()

    async def query(
        self, expression: exp.Expression, sql: str, attrs: dict[str, t.Any]
    ) -> AllowedResult:
        logger.info("Query: %s", sql)
        resp, table = await asyncio.to_thread(
            execute_sql_code, self.project, expression.sql(dialect=self.project.adapter.type())
        )
        if resp.code:
            raise QueryException(resp)
        rows = t.cast(tuple[t.Any], table.rows.values())
        return [row.values() for row in rows], t.cast(tuple[str], table.column_names)

    async def schema(self):
        schema: defaultdict[str, dict[str, dict[str, tuple[str, t.Optional[str]]]]] = defaultdict(
            dict
        )
        for node in chain(
            self.project.manifest.sources.values(), self.project.manifest.nodes.values()
        ):
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
