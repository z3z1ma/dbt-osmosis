# pyright: reportPrivateImportUsage=false, reportPrivateUsage=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportAny=false, reportUnknownMemberType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportArgumentType=false, reportFunctionMemberAccess=false, reportUnknownVariableType=false, reportUnusedParameter=false

"""Tests for database introspection and column metadata retrieval.

This module contains:
- Behavior tests: test_get_columns_simple() validates end-to-end column retrieval
- Unit tests for utility functions with clear behavior contracts

The utility function tests (_find_first, _maybe_use_precise_dtype, etc.) test
well-defined behaviors of helper functions that have clear input/output contracts.
"""

from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest import mock

import pytest

pytestmark = pytest.mark.usefixtures("fresh_caches")
from dbt.adapters.base.column import Column
from dbt.adapters.base.relation import BaseRelation
from dbt.artifacts.schemas.catalog import CatalogArtifact

from dbt_osmosis.core.introspection import (
    PropertyAccessor,
    PropertySource,
    _find_first,
    _get_setting_for_node,
    _maybe_use_precise_dtype,
    get_columns,
    normalize_column_name,
)
from dbt_osmosis.core.settings import YamlRefactorContext, YamlRefactorSettings


@dataclass(frozen=True)
class _FakeRelation:
    rendered: str
    meta: dict = field(default_factory=dict)
    config: SimpleNamespace = field(
        default_factory=lambda: SimpleNamespace(extra={}, meta={}),
    )
    columns: dict = field(default_factory=dict)
    unrendered_config: dict = field(default_factory=dict)

    def render(self) -> str:
        return self.rendered

    def __str__(self) -> str:
        return self.rendered


class _RecordingAdapter:
    Relation = SimpleNamespace(create_from=staticmethod(lambda config, relation: relation))
    config = object()

    def __init__(
        self,
        runtime_cfg,
        columns_by_target,
        calls,
    ) -> None:
        self._runtime_cfg = runtime_cfg
        self._columns_by_target = columns_by_target
        self.calls = calls

    def get_columns_in_relation(self, relation):
        key = (self._runtime_cfg.target_name, relation.render())
        self.calls.append(key)
        return self._columns_by_target[key]


def _make_introspection_context(
    columns_by_target,
    calls,
    *,
    target_name: str = "dev",
    string_length: bool = False,
    numeric_precision_and_scale: bool = False,
    ignore_patterns=None,
):
    runtime_cfg = SimpleNamespace(
        project_root="/tmp/project",
        profile_name="default",
        target_name=target_name,
        credentials=SimpleNamespace(type="duckdb"),
    )
    adapter = _RecordingAdapter(runtime_cfg, columns_by_target, calls)
    project = SimpleNamespace(
        runtime_cfg=runtime_cfg,
        adapter=adapter,
        config=SimpleNamespace(disable_introspection=False),
    )
    return SimpleNamespace(
        project=project,
        settings=YamlRefactorSettings(
            string_length=string_length,
            numeric_precision_and_scale=numeric_precision_and_scale,
        ),
        ignore_patterns=list(ignore_patterns or []),
        read_catalog=lambda: None,
    )


def test_get_columns_simple(yaml_context: YamlRefactorContext):
    """Tests the get_columns flow on a known table, e.g., 'customers'."""
    node = yaml_context.project.manifest.nodes["model.jaffle_shop_duckdb.customers"]
    cols = get_columns(yaml_context, node)
    assert "customer_id" in cols


def test_get_columns_reprocesses_cached_columns_when_settings_change():
    """Changing dtype settings must not reuse stale processed column metadata."""
    relation = _FakeRelation("db.schema.orders")
    calls = []
    columns_by_target = {
        ("dev", relation.render()): [Column.from_description("description", "varchar(256)")],
    }
    context = _make_introspection_context(columns_by_target, calls, string_length=False)

    initial = get_columns(context, relation)
    assert initial["description"].type == "varchar"
    assert calls == [("dev", relation.render())]

    context.settings.string_length = True
    updated = get_columns(context, relation)

    assert updated["description"].type == "character varying(256)"
    assert calls == [("dev", relation.render())]


def test_get_columns_honors_context_project_vars_for_precise_dtype():
    """Project config should affect dtype precision during real column introspection."""
    relation = _FakeRelation("db.schema.orders")
    calls = []
    columns_by_target = {
        ("dev", relation.render()): [Column.from_description("description", "varchar(256)")],
    }
    context = _make_introspection_context(columns_by_target, calls, string_length=False)
    context.project.runtime_cfg.vars = {"dbt-osmosis": {"string-length": True}}

    columns = get_columns(context, relation)

    assert columns["description"].type == "character varying(256)"
    assert calls == [("dev", relation.render())]


def test_get_columns_scopes_cache_by_target():
    """Switching targets must not reuse warehouse metadata from another target."""
    relation = _FakeRelation("db.schema.orders")
    calls = []
    columns_by_target = {
        ("dev", relation.render()): [Column.from_description("description", "varchar(256)")],
        ("prod", relation.render()): [Column.from_description("amount", "decimal(18,3)")],
    }
    dev_context = _make_introspection_context(columns_by_target, calls, target_name="dev")
    prod_context = _make_introspection_context(columns_by_target, calls, target_name="prod")

    dev_columns = get_columns(dev_context, relation)
    prod_columns = get_columns(prod_context, relation)

    assert set(dev_columns) == {"description"}
    assert set(prod_columns) == {"amount"}
    assert calls == [("dev", relation.render()), ("prod", relation.render())]


def test_get_columns_reprocesses_cached_columns_for_new_context_filters():
    """A new context must be able to apply different ignore patterns truthfully."""
    relation = _FakeRelation("db.schema.orders")
    calls = []
    columns_by_target = {
        ("dev", relation.render()): [
            Column.from_description("visible", "varchar(256)"),
            Column.from_description("skip_me", "varchar(256)"),
        ],
    }
    initial_context = _make_introspection_context(columns_by_target, calls)
    filtered_context = _make_introspection_context(
        columns_by_target,
        calls,
        ignore_patterns=[r"^skip_"],
    )

    initial = get_columns(initial_context, relation)
    filtered = get_columns(filtered_context, relation)

    assert set(initial) == {"visible", "skip_me"}
    assert set(filtered) == {"visible"}
    assert calls == [("dev", relation.render())]


def _catalog_table(identity, column_name="right_column"):
    return {
        "metadata": {"type": "table", **identity},
        "columns": {
            column_name: {
                "name": column_name,
                "type": "INTEGER" if column_name == "right_column" else "VARCHAR",
                "index": 1,
                "comment": f"Documentation for {column_name}",
            },
        },
        "stats": {},
    }


def _catalog_context(data):
    calls = []
    context = _make_introspection_context({}, calls)
    catalog = CatalogArtifact.from_dict(data)
    context.read_catalog = lambda: catalog
    return context, catalog, calls


@pytest.mark.parametrize("component", ["database", "schema", "name"])
@pytest.mark.parametrize(
    "exact_section,other_section",
    [("nodes", "nodes"), ("sources", "sources"), ("nodes", "sources"), ("sources", "nodes")],
)
@pytest.mark.parametrize("exact_first", [False, True])
def test_get_columns_prefers_exact_catalog_identity(
    component, exact_section, other_section, exact_first
):
    """Case-folded matches must not hide an exact identity anywhere in the catalog."""
    exact = {"database": "db", "schema": "public", "name": "orders"}
    exact[component] = exact[component].capitalize()
    other = {**exact, component: exact[component].lower()}
    data = {"metadata": {}, "nodes": {}, "sources": {}}
    entries = [
        (other_section, "other", _catalog_table(other, "wrong_column")),
        (exact_section, "exact", _catalog_table(exact)),
    ]
    if exact_first:
        entries.reverse()
    for section, key, table in entries:
        data[section][key] = table
    context, catalog, calls = _catalog_context(data)
    before = catalog.to_dict()
    relation = BaseRelation.create(
        database=exact["database"],
        schema=exact["schema"],
        identifier=exact["name"],
    )

    columns = get_columns(context, relation)

    assert list(columns) == ["right_column"]
    assert columns["right_column"].type == "INTEGER"
    assert columns["right_column"].comment == "Documentation for right_column"
    assert catalog.to_dict() == before
    assert calls == []


@pytest.mark.parametrize("database", ["absent", None, "db"])
@pytest.mark.parametrize("catalog_name", ["Orders", "orders"])
def test_get_columns_preserves_catalog_fallback_and_optional_database(database, catalog_name):
    """Retain legacy case-insensitive and omitted database lookup behavior."""
    identity = {"schema": "public", "name": catalog_name}
    if database != "absent":
        identity["database"] = database
    context, catalog, calls = _catalog_context({
        "metadata": {},
        "nodes": {},
        "sources": {"source.pkg.orders": _catalog_table(identity)},
    })
    before = catalog.to_dict()
    relation = BaseRelation.create(database="db", schema="public", identifier="Orders")

    columns = get_columns(context, relation)

    assert list(columns) == ["right_column"]
    assert columns["right_column"].type == "INTEGER"
    assert columns["right_column"].comment == "Documentation for right_column"
    assert catalog.to_dict() == before
    assert calls == []


# PostgresRelation inherits BaseRelation matching unchanged; the installed adapter
# equivalence and actual-object reproduction are recorded in the 2026-11-23 ticket.
# These constructed catalogs are adversarial inputs, not producer fixtures.
@pytest.mark.parametrize("section", ["nodes", "sources"])
@pytest.mark.parametrize("component", ["database", "schema", "name"])
@pytest.mark.parametrize("requested_upper", [False, True])
@pytest.mark.parametrize("disable_introspection", [False, True])
def test_postgres_catalog_rejects_quoted_case_distinct_entry(
    section, component, requested_upper, disable_introspection
):
    requested = {"database": "db", "schema": "public", "name": "orders"}
    candidate = dict(requested)
    (requested if requested_upper else candidate)[component] = requested[component].capitalize()
    context, catalog, calls = _catalog_context({
        "metadata": {},
        "nodes": {},
        "sources": {},
        section: {"constructed": _catalog_table(candidate, "wrong_column")},
    })
    context.project.runtime_cfg.credentials.type = "postgres"
    context.project.config.disable_introspection = disable_introspection
    relation = BaseRelation.create(
        database=requested["database"], schema=requested["schema"], identifier=requested["name"]
    )
    context.project.adapter._columns_by_target[("dev", relation.render())] = [
        Column.from_description("warehouse_column", "integer")
    ]
    before = catalog.to_dict()

    columns = get_columns(context, relation)

    assert list(columns) == ([] if disable_introspection else ["warehouse_column"])
    assert calls == ([] if disable_introspection else [("dev", relation.render())])
    assert catalog.to_dict() == before


@pytest.mark.parametrize("section", ["nodes", "sources"])
@pytest.mark.parametrize("database", ["absent", None, "db"])
def test_postgres_catalog_keeps_quoted_identity_with_unquoted_folding(section, database):
    identity = {"schema": "public", "name": "Orders"}
    if database != "absent":
        identity["database"] = database
    context, catalog, calls = _catalog_context({
        "metadata": {},
        "nodes": {},
        "sources": {},
        section: {"constructed": _catalog_table(identity)},
    })
    context.project.runtime_cfg.credentials.type = "postgres"
    # The identifier is quoted, the schema is not. Folding the latter must not
    # prevent the correctly cased quoted identifier from matching.
    relation = BaseRelation.create(
        database="db", schema="PUBLIC", identifier="Orders", quote_policy={"schema": False}
    )
    before = catalog.to_dict()

    columns = get_columns(context, relation)

    assert list(columns) == ["right_column"]
    assert (columns["right_column"].type, columns["right_column"].comment) == (
        "INTEGER",
        "Documentation for right_column",
    )
    assert calls == []
    assert catalog.to_dict() == before


@pytest.mark.parametrize("adapter_type", ["postgres", "duckdb", "trino", "snowflake"])
def test_catalog_retains_adapter_specific_unquoted_fallback(adapter_type):
    context, _, calls = _catalog_context({
        "metadata": {},
        "nodes": {},
        "sources": {
            "constructed": _catalog_table({"database": "db", "schema": "public", "name": "orders"})
        },
    })
    context.project.runtime_cfg.credentials.type = adapter_type
    relation = BaseRelation.create(
        database="db", schema="public", identifier="Orders", quote_policy={"identifier": False}
    )

    columns = get_columns(context, relation)

    assert list(columns) == (["RIGHT_COLUMN"] if adapter_type == "snowflake" else ["right_column"])
    assert calls == []


@pytest.mark.parametrize("adapter_type", ["duckdb", "trino", "snowflake"])
def test_catalog_retains_other_adapters_quoted_fallback(adapter_type):
    context, _, calls = _catalog_context({
        "metadata": {},
        "nodes": {},
        "sources": {
            "constructed": _catalog_table({"database": "db", "schema": "public", "name": "orders"})
        },
    })
    context.project.runtime_cfg.credentials.type = adapter_type
    relation = BaseRelation.create(database="db", schema="public", identifier="Orders")

    columns = get_columns(context, relation)

    assert list(columns) == (["RIGHT_COLUMN"] if adapter_type == "snowflake" else ["right_column"])
    assert calls == []


def test_postgres_catalog_continues_after_quoted_mismatch_to_valid_source():
    # No entry has the exact raw triple because the requested schema is unquoted.
    context, catalog, calls = _catalog_context({
        "metadata": {},
        "nodes": {
            "wrong": _catalog_table(
                {"database": "db", "schema": "public", "name": "orders"}, "wrong_column"
            )
        },
        "sources": {
            "right": _catalog_table({"database": "db", "schema": "public", "name": "Orders"})
        },
    })
    context.project.runtime_cfg.credentials.type = "postgres"
    relation = BaseRelation.create(
        database="db", schema="PUBLIC", identifier="Orders", quote_policy={"schema": False}
    )
    before = catalog.to_dict()

    columns = get_columns(context, relation)

    assert list(columns) == ["right_column"]
    assert columns["right_column"].comment == "Documentation for right_column"
    assert calls == []
    assert catalog.to_dict() == before


@pytest.mark.parametrize("component", ["schema", "identifier"])
@pytest.mark.parametrize(
    "canonical_section,literal_section",
    [("nodes", "nodes"), ("sources", "sources"), ("nodes", "sources"), ("sources", "nodes")],
)
@pytest.mark.parametrize("canonical_first", [False, True])
def test_postgres_unquoted_prefers_sql_identity(
    component, canonical_section, literal_section, canonical_first
):
    requested = {"database": "db", "schema": "public", "identifier": "orders"}
    requested[component] = requested[component].capitalize()
    literal = {
        "database": requested["database"],
        "schema": requested["schema"],
        "name": requested["identifier"],
    }
    canonical = {
        **literal,
        ("name" if component == "identifier" else component): requested[component].lower(),
    }
    data = {"metadata": {}, "nodes": {}, "sources": {}}
    entries = [
        (literal_section, "literal", _catalog_table(literal, "wrong_column")),
        (canonical_section, "canonical", _catalog_table(canonical)),
    ]
    if canonical_first:
        entries.reverse()
    for section, key, table in entries:
        data[section][key] = table
    context, catalog, calls = _catalog_context(data)
    context.project.runtime_cfg.credentials.type = "postgres"
    relation = BaseRelation.create(**requested, quote_policy={component: False})
    before, relation_before = catalog.to_dict(), relation.to_dict()

    columns = get_columns(context, relation)

    assert list(columns) == ["right_column"]
    assert (columns["right_column"].type, columns["right_column"].comment) == (
        "INTEGER",
        "Documentation for right_column",
    )
    assert calls == []
    assert catalog.to_dict() == before
    assert relation.to_dict() == relation_before


@pytest.mark.parametrize("section", ["nodes", "sources"])
@pytest.mark.parametrize(
    "schema_quoted,identifier_quoted", [(False, True), (True, False), (False, False), (True, True)]
)
def test_postgres_unquoted_respects_each_quote_policy(section, schema_quoted, identifier_quoted):
    identity = {
        "database": "db",
        "schema": "Public" if schema_quoted else "public",
        "name": "Orders" if identifier_quoted else "orders",
    }
    context, catalog, calls = _catalog_context({
        "metadata": {},
        "nodes": {},
        "sources": {},
        section: {
            "unrelated": _catalog_table(
                {
                    "database": "db",
                    "schema": "public" if schema_quoted else "Public",
                    "name": "orders" if identifier_quoted else "Orders",
                },
                "wrong_column",
            ),
            "canonical": _catalog_table(identity),
        },
    })
    context.project.runtime_cfg.credentials.type = "postgres"
    relation = BaseRelation.create(
        database="db",
        schema="Public",
        identifier="Orders",
        quote_policy={"schema": schema_quoted, "identifier": identifier_quoted},
    )
    before = catalog.to_dict()

    columns = get_columns(context, relation)

    assert list(columns) == ["right_column"]
    assert columns["right_column"].comment == "Documentation for right_column"
    assert calls == []
    assert catalog.to_dict() == before


@pytest.mark.parametrize("section", ["nodes", "sources"])
@pytest.mark.parametrize("component", ["schema", "identifier"])
@pytest.mark.parametrize("requested_lower", [False, True])
@pytest.mark.parametrize("disable_introspection", [False, True])
def test_postgres_unquoted_missing_sql_identity_keeps_fallback_policy(
    section, component, requested_lower, disable_introspection
):
    requested = {"database": "db", "schema": "public", "identifier": "orders"}
    requested[component] = (
        requested[component] if requested_lower else requested[component].capitalize()
    )
    candidate = {"database": "db", "schema": "public", "name": "orders"}
    key = "name" if component == "identifier" else component
    candidate[key] = candidate[key].capitalize()
    context, catalog, calls = _catalog_context({
        "metadata": {},
        "nodes": {},
        "sources": {},
        section: {"wrong_literal": _catalog_table(candidate, "wrong_column")},
    })
    context.project.runtime_cfg.credentials.type = "postgres"
    context.project.config.disable_introspection = disable_introspection
    relation = BaseRelation.create(**requested, quote_policy={component: False})
    context.project.adapter._columns_by_target[("dev", relation.render())] = [
        Column.from_description("warehouse_column", "integer")
    ]
    before = catalog.to_dict()

    columns = get_columns(context, relation)

    assert list(columns) == ([] if disable_introspection else ["warehouse_column"])
    assert calls == ([] if disable_introspection else [("dev", relation.render())])
    assert catalog.to_dict() == before


@pytest.mark.parametrize("section", ["nodes", "sources"])
@pytest.mark.parametrize("database", ["absent", None, "db"])
def test_postgres_unquoted_keeps_optional_database_matching(section, database):
    identity = {"schema": "public", "name": "orders"}
    if database != "absent":
        identity["database"] = database
    context, catalog, calls = _catalog_context({
        "metadata": {},
        "nodes": {},
        "sources": {},
        section: {
            "literal": _catalog_table({**identity, "name": "Orders"}, "wrong_column"),
            "canonical": _catalog_table(identity),
        },
    })
    context.project.runtime_cfg.credentials.type = "postgres"
    relation = BaseRelation.create(
        database="db", schema="public", identifier="Orders", quote_policy={"identifier": False}
    )
    before = catalog.to_dict()

    columns = get_columns(context, relation)

    assert list(columns) == ["right_column"]
    assert calls == []
    assert catalog.to_dict() == before


@pytest.mark.parametrize("database_included", [False, True])
@pytest.mark.parametrize("database_quoted", [False, True])
def test_postgres_unquoted_preserves_database_identity(database_included, database_quoted):
    context, catalog, calls = _catalog_context({
        "metadata": {},
        "nodes": {
            "wrong_database": _catalog_table(
                {"database": "db", "schema": "public", "name": "orders"}, "wrong_column"
            ),
            "literal": _catalog_table(
                {"database": "Db", "schema": "public", "name": "Orders"}, "wrong_literal"
            ),
        },
        "sources": {
            "canonical": _catalog_table({"database": "Db", "schema": "public", "name": "orders"})
        },
    })
    context.project.runtime_cfg.credentials.type = "postgres"
    relation = BaseRelation.create(
        database="Db",
        schema="public",
        identifier="Orders",
        quote_policy={"database": database_quoted, "identifier": False},
    ).include(database=database_included)
    before = catalog.to_dict()

    columns = get_columns(context, relation)

    assert list(columns) == ["right_column"]
    assert calls == []
    assert catalog.to_dict() == before


@pytest.mark.parametrize("component", ["schema", "identifier"])
@pytest.mark.parametrize("non_ascii", [False, True])
def test_postgres_unquoted_preserves_unscoped_components(component, non_ascii):
    requested = {"database": "db", "schema": "public", "identifier": "orders"}
    requested[component] = "Örders" if non_ascii else "Orders"
    key = "name" if component == "identifier" else component
    identity = {"database": "db", "schema": requested["schema"], "name": requested["identifier"]}
    context, catalog, calls = _catalog_context({
        "metadata": {},
        "nodes": {
            "folded": _catalog_table({**identity, key: identity[key].lower()}, "wrong_column"),
        },
        "sources": {"literal": _catalog_table(identity)},
    })
    context.project.runtime_cfg.credentials.type = "postgres"
    relation = BaseRelation.create(**requested, quote_policy={component: False})
    if not non_ascii:
        relation = relation.include(**{component: False})
    before = catalog.to_dict()

    columns = get_columns(context, relation)

    assert list(columns) == ["right_column"]
    assert calls == []
    assert catalog.to_dict() == before


@pytest.mark.parametrize("adapter_type", ["duckdb", "trino", "snowflake"])
def test_postgres_unquoted_change_preserves_other_adapter_precedence(adapter_type):
    context, catalog, calls = _catalog_context({
        "metadata": {},
        "nodes": {
            "folded": _catalog_table(
                {"database": "db", "schema": "public", "name": "orders"}, "wrong_column"
            ),
        },
        "sources": {
            "literal": _catalog_table({"database": "db", "schema": "public", "name": "Orders"})
        },
    })
    context.project.runtime_cfg.credentials.type = adapter_type
    relation = BaseRelation.create(
        database="db", schema="public", identifier="Orders", quote_policy={"identifier": False}
    )
    before = catalog.to_dict()

    columns = get_columns(context, relation)

    assert list(columns) == (["RIGHT_COLUMN"] if adapter_type == "snowflake" else ["right_column"])
    assert calls == []
    assert catalog.to_dict() == before


@pytest.mark.parametrize("has_catalog", [False, True])
def test_get_columns_retains_missing_catalog_relation_warehouse_fallback(has_catalog):
    """No catalog match must still use the existing adapter path."""
    relation = BaseRelation.create(database="db", schema="public", identifier="orders")
    calls = []
    context = _make_introspection_context(
        {("dev", relation.render()): [Column.from_description("warehouse_column", "integer")]},
        calls,
    )
    if has_catalog:
        catalog = CatalogArtifact.from_dict({
            "metadata": {},
            "sources": {},
            "nodes": {
                "model.pkg.other": _catalog_table({
                    "database": "db",
                    "schema": "public",
                    "name": "other",
                })
            },
        })
        context.read_catalog = lambda: catalog

    columns = get_columns(context, relation)

    assert list(columns) == ["warehouse_column"]
    assert calls == [("dev", relation.render())]


def test_find_first():
    """Test the _find_first utility function."""
    data = [1, 2, 3, 4]
    assert _find_first(data, lambda x: x > 2) == 3
    assert _find_first(data, lambda x: x > 4) is None
    assert _find_first(data, lambda x: x > 4, default=999) == 999


@pytest.mark.parametrize(
    "input_col,expected",
    [
        ('"My_Col"', "My_Col"),
        ("my_col", "MY_COL"),
    ],
)
def test_normalize_column_name_snowflake(input_col, expected):
    """Test column name normalization for Snowflake adapter."""
    # For snowflake, if quoted - we preserve case but strip quotes, otherwise uppercase
    assert normalize_column_name(input_col, "snowflake") == expected


def test_normalize_column_name_others():
    """Test column name normalization for other adapters."""
    # For other adapters, we only strip outer quotes but do not uppercase or lowercase for now
    assert normalize_column_name('"My_Col"', "duckdb") == "My_Col"
    assert normalize_column_name("my_col", "duckdb") == "my_col"


def test_maybe_use_precise_dtype_numeric():
    """Check that _maybe_use_precise_dtype uses the data_type if numeric_precision_and_scale is enabled."""
    from dbt.adapters.base.column import Column

    col = Column("col1", "DECIMAL(18,3)", None)  # data_type and dtype
    settings = YamlRefactorSettings(numeric_precision_and_scale=True)
    result = _maybe_use_precise_dtype(col, settings, node=None)
    assert result == "DECIMAL(18,3)"


def test_maybe_use_precise_dtype_string():
    """If string_length is True, we use col.data_type (like 'varchar(256)')
    instead of col.dtype (which might be 'VARCHAR').
    """
    from dbt.adapters.base.column import Column

    col = Column("col1", "VARCHAR(256)", None)
    settings = YamlRefactorSettings(string_length=True)
    result = _maybe_use_precise_dtype(col, settings, node=None)
    assert result == "VARCHAR(256)"


def test_get_setting_for_node_basic():
    """Check that _get_setting_for_node can read from node.meta, etc.
    We mock the node to have certain meta fields.
    """
    node = mock.Mock()
    node.config.extra = {}
    node.meta = {
        "dbt-osmosis-options": {
            "test-key": "test-value",
        },
    }
    # key = "test-key", which means we look for 'dbt-osmosis-options' => "test-key"
    val = _get_setting_for_node("test-key", node=node, col=None, fallback=None)
    assert val == "test-value"


class TestPropertyAccessorHasUnrenderedJinja:
    """Tests for PropertyAccessor._has_unrendered_jinja method."""

    @pytest.mark.parametrize("source", [PropertySource.DATABASE, "database"])
    def test_database_source_is_explicitly_unsupported(self, source):
        """Database source must not silently fall back to manifest values."""
        accessor = PropertyAccessor(context=mock.Mock())
        node = SimpleNamespace(description="manifest description", columns={})

        with pytest.raises(NotImplementedError, match="database.*not implemented"):
            accessor.get("description", node, source=source)

    def test_has_unrendered_jinja_doc_function(self):
        """Test detection of {{ doc() }} function calls."""
        accessor = PropertyAccessor(context=mock.Mock())
        assert accessor._has_unrendered_jinja("{{ doc('my_doc') }}") is True
        assert accessor._has_unrendered_jinja("Some text {{ doc('my_doc') }} more") is True

    def test_has_unrendered_jinja_var_function(self):
        """Test detection of {{ var() }} function calls."""
        from dbt_osmosis.core.introspection import PropertyAccessor

        accessor = PropertyAccessor(context=mock.Mock())
        assert accessor._has_unrendered_jinja("{{ var('policy_tag_user_id') }}") is True
        assert accessor._has_unrendered_jinja('{{ var("my_var") }}') is True

    def test_has_unrendered_jinja_env_var_function(self):
        """Test detection of {{ env_var() }} function calls."""
        from dbt_osmosis.core.introspection import PropertyAccessor

        accessor = PropertyAccessor(context=mock.Mock())
        assert accessor._has_unrendered_jinja('{{ env_var("MY_ENV_VAR") }}') is True

    def test_has_unrendered_jinja_generic_expression(self):
        """Test detection of generic {{ }} jinja expressions."""
        from dbt_osmosis.core.introspection import PropertyAccessor

        accessor = PropertyAccessor(context=mock.Mock())
        assert accessor._has_unrendered_jinja("{{ 'my_value' }}") is True
        assert accessor._has_unrendered_jinja("{{ some_var }}") is True

    def test_has_unrendered_jinja_docs_block(self):
        """Test detection of {% docs %} blocks."""
        from dbt_osmosis.core.introspection import PropertyAccessor

        accessor = PropertyAccessor(context=mock.Mock())
        assert accessor._has_unrendered_jinja("{% docs my_block %}{% enddocs %}") is True
        assert accessor._has_unrendered_jinja("{% docs my_block %}content{% enddocs %}") is True

    def test_has_unrendered_jinja_generic_statement(self):
        """Test detection of generic {% %} jinja statements."""
        from dbt_osmosis.core.introspection import PropertyAccessor

        accessor = PropertyAccessor(context=mock.Mock())
        assert accessor._has_unrendered_jinja("{% if condition %}") is True
        assert accessor._has_unrendered_jinja("{% for item in items %}") is True

    def test_has_unrendered_jinja_no_jinja(self):
        """Test that non-jinja strings return False."""
        from dbt_osmosis.core.introspection import PropertyAccessor

        accessor = PropertyAccessor(context=mock.Mock())
        assert accessor._has_unrendered_jinja("Plain text") is False
        assert accessor._has_unrendered_jinja("projects/my_project/locations/us") is False
        assert accessor._has_unrendered_jinja("") is False

    def test_has_unrendered_jinja_list_values(self):
        """Test detection of unrendered jinja in list values (e.g., policy_tags)."""
        from dbt_osmosis.core.introspection import PropertyAccessor

        accessor = PropertyAccessor(context=mock.Mock())
        # List with jinja
        assert accessor._has_unrendered_jinja(["{{ var('tag1') }}", "regular_value"]) is True
        assert accessor._has_unrendered_jinja(["{{ env_var('TAG') }}"]) is True
        # List without jinja
        assert accessor._has_unrendered_jinja(["value1", "value2"]) is False

    def test_has_unrendered_jinja_dict_values(self):
        """Test detection of unrendered jinja in dict values (e.g., meta fields)."""
        from dbt_osmosis.core.introspection import PropertyAccessor

        accessor = PropertyAccessor(context=mock.Mock())
        # Dict with jinja
        assert accessor._has_unrendered_jinja({"key": "{{ var('value') }}"}) is True
        assert accessor._has_unrendered_jinja({"nested": {"key": "{{ doc('x') }}"}}) is True
        # Dict without jinja
        assert accessor._has_unrendered_jinja({"key": "value"}) is False

    def test_has_unrendered_jinja_non_string_types(self):
        """Test that non-string, non-iterable types return False."""
        from dbt_osmosis.core.introspection import PropertyAccessor

        accessor = PropertyAccessor(context=mock.Mock())
        assert accessor._has_unrendered_jinja(123) is False
        assert accessor._has_unrendered_jinja(True) is False
        assert accessor._has_unrendered_jinja(None) is False
