"""Conservative, bounded candidate identities; never compiles SQL or copies metadata."""

from dataclasses import dataclass
from typing import Literal

from dbt.contracts.graph.nodes import ModelNode, SeedNode, SourceDefinition
from sqlglot import exp, parse
from sqlglot.errors import OptimizeError, SqlglotError
from sqlglot.lineage import Node, lineage
from sqlglot.schema import MappingSchema

Donor = ModelNode | SeedNode | SourceDefinition

MAX_SQL_BYTES = 65536
MAX_AST_NODES = 4096
MAX_PATH_NODES = 256
MAX_COLUMNS = 4096
MAX_RELATIONS = 256


@dataclass(frozen=True)
class Candidate:
    status: Literal["candidate", "rejected", "unavailable"]
    reason: str
    node_id: str = ""
    column: str = ""


def _relation(node: Donor) -> exp.Table:
    return exp.Table(
        this=exp.Identifier(
            this=node.identifier if isinstance(node, SourceDefinition) else node.alias, quoted=True
        ),
        db=exp.Identifier(this=node.schema, quoted=True),
        catalog=exp.Identifier(this=node.database, quoted=True) if node.database else None,
    )


def _key(table: exp.Table) -> tuple[str, ...]:
    # DuckDB and Trino identifiers are case insensitive even when quoted.
    return tuple(part.name.casefold() for part in table.parts)


def resolve_candidates(
    sql: str | None, dialect: str, columns: list[str], ancestors: list[Donor]
) -> dict[str, Candidate]:
    """Resolve a direct donor for each requested column, without persistent caching.

    Structural budgets are not a wall-clock timeout. Unsupported input preserves
    legacy fallback; observed ambiguity or changed meaning explicitly rejects it.
    """

    def all_results(
        status: Literal["rejected", "unavailable"], reason: str
    ) -> dict[str, Candidate]:
        return {name: Candidate(status, reason) for name in columns}

    if dialect not in {"duckdb", "trino"} or not sql:
        return all_results("unavailable", "missing compiled SQL or unsupported adapter")
    if len(sql) > MAX_SQL_BYTES or len(sql.encode()) > MAX_SQL_BYTES or len(columns) > MAX_COLUMNS:
        return all_results("unavailable", "input budget exceeded")
    if len(ancestors) > MAX_RELATIONS or sum(len(n.columns) for n in ancestors) > MAX_COLUMNS:
        return all_results("unavailable", "schema budget exceeded")
    try:
        queries = parse(sql, dialect=dialect)
        if len(queries) != 1 or not isinstance(queries[0], exp.Query):
            return all_results("rejected", "not one query")
        query = queries[0]
        for index, expression in enumerate(query.walk()):
            if index >= MAX_AST_NODES:
                return all_results("unavailable", "AST budget exceeded")
            if isinstance(expression, exp.SetOperation):
                return all_results("rejected", "set operation")
        schema = MappingSchema(dialect=dialect)
        relations: dict[tuple[str, ...], list[Donor]] = {}
        for ancestor in ancestors:
            table = _relation(ancestor)
            relations.setdefault(_key(table), []).append(ancestor)
            schema.add_table(table, {name: "UNKNOWN" for name in ancestor.columns})
        results = {}
        for name in columns:
            try:
                root = lineage(
                    name,
                    query,
                    schema=schema,
                    dialect=dialect,
                    validate_qualify_columns=True,
                    trim_selects=False,
                )
                nodes = []
                for index, item in enumerate(root.walk()):
                    if index >= MAX_PATH_NODES:
                        break
                    nodes.append(item)
                else:
                    results[name] = _candidate(nodes, relations)
                    continue
                results[name] = Candidate("unavailable", "lineage budget exceeded")
            except OptimizeError:
                results[name] = Candidate("rejected", "unresolved or ambiguous column")
            except (SqlglotError, RecursionError, ValueError):
                results[name] = Candidate("unavailable", "unsupported lineage")
        return results
    except (SqlglotError, RecursionError, ValueError):
        return all_results("unavailable", "unsupported SQL or schema")


def _candidate(nodes: list[Node], relations: dict[tuple[str, ...], list[Donor]]) -> Candidate:
    leaves = [n for n in nodes if not n.downstream]
    if len(leaves) != 1 or not isinstance(leaves[0].expression, exp.Table):
        return Candidate("rejected", "unresolved or multiple origins")
    for node in nodes:
        if not isinstance(node.expression, exp.Table):
            if not isinstance(node.source, exp.Query):
                return Candidate("rejected", "unresolved projection source")
            output = node.expression.alias_or_name.casefold()
            if (
                sum(select.alias_or_name.casefold() == output for select in node.source.selects)
                != 1
            ):
                return Candidate("rejected", "ambiguous output column")
        projection = node.expression.unalias()
        if not isinstance(projection, (exp.Table, exp.Column)) or projection.is_star:
            return Candidate("rejected", "transformed or unresolved projection")
    leaf = leaves[0]
    if not isinstance(leaf.expression, exp.Table):
        return Candidate("rejected", "unresolved origin")
    matches = relations.get(_key(leaf.expression), [])
    if len(matches) != 1:
        return Candidate("rejected", "ambiguous or unknown relation")
    donor = matches[0]
    column = exp.to_column(leaf.name).name
    names = [name for name in donor.columns if name.casefold() == column.casefold()]
    if len(names) != 1:
        return Candidate("rejected", "ambiguous or unknown donor column")
    return Candidate("candidate", "direct column", donor.unique_id, names[0])
