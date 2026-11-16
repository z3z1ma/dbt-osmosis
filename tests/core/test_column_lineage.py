"""Candidate identity and real inheritance regression coverage for opt-in lineage."""

from types import SimpleNamespace

import pytest
from dbt.contracts.graph.nodes import ColumnInfo

from dbt_osmosis.core import column_lineage
from dbt_osmosis.core.column_lineage import resolve_candidates
from dbt_osmosis.core.transforms import inherit_upstream_column_knowledge


def ancestor(identifier="input", **kwargs):
    values = {
        "database": "Lake",
        "schema": "Raw",
        "identifier": identifier,
        "alias": identifier,
        "columns": {"VendorID": None, "other": None},
        "unique_id": f"source.raw.{identifier}",
    }
    values.update(kwargs)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("dialect", ["duckdb", "trino"])
@pytest.mark.parametrize(
    "sql",
    [
        'select a."VendorID" as renamed from "Lake"."Raw"."input" a',
        'with s as (select * from "Lake"."Raw"."input"), r as (select "VendorID" as renamed from s) select * from r',
    ],
)
def test_direct_and_cte_alias(dialect, sql):
    result = resolve_candidates(sql, dialect, ["renamed"], [ancestor()])["renamed"]
    assert (result.status, result.node_id, result.column) == (
        "candidate",
        "source.raw.input",
        "VendorID",
    )


@pytest.mark.parametrize(
    "sql",
    [
        'with s as (select lower("VendorID") as renamed from "Lake"."Raw"."input") select renamed from s',
        'select cast("VendorID" as varchar) as renamed from "Lake"."Raw"."input"',
        'select count("VendorID") as renamed from "Lake"."Raw"."input"',
        'select "VendorID" as renamed from "Lake"."Raw"."input" union all select "VendorID" from "Lake"."Raw"."input"',
        'select coalesce(a."VendorID", b.other) as renamed from "Lake"."Raw"."input" a join "Lake"."Raw"."other" b on true',
        'select "VendorID" as renamed from "Lake"."Raw"."input" a join "Lake"."Raw"."other" b on true',
    ],
)
@pytest.mark.parametrize("dialect", ["duckdb", "trino"])
def test_reject_changed_or_ambiguous_meaning(sql, dialect):
    assert (
        resolve_candidates(sql, dialect, ["renamed"], [ancestor(), ancestor("other")])[
            "renamed"
        ].status
        == "rejected"
    )


def test_unknown_star_and_relation_collision():
    sql = 'with s as (select * from "Lake"."Raw"."input") select "VendorID" as renamed from s'
    assert (
        resolve_candidates(sql, "duckdb", ["renamed"], [ancestor(columns={})])["renamed"].status
        == "rejected"
    )
    assert (
        resolve_candidates(sql, "trino", ["renamed"], [ancestor(), ancestor("INPUT")])[
            "renamed"
        ].status
        == "rejected"
    )


def test_escaped_identifiers():
    sql = 'select "odd""name" as renamed from "Lake"."Raw"."in""put"'
    donor = ancestor('in"put', columns={'odd"name': None})
    result = resolve_candidates(sql, "trino", ["renamed"], [donor])["renamed"]
    assert (result.status, result.column) == ("candidate", 'odd"name')


def test_unavailable_inputs_and_budgets(monkeypatch):
    assert resolve_candidates(None, "duckdb", ["x"], [])["x"].status == "unavailable"
    assert resolve_candidates("select x", "snowflake", ["x"], [])["x"].status == "unavailable"
    assert (
        resolve_candidates("x" * 65537, "duckdb", ["x"], [])["x"].reason == "input budget exceeded"
    )
    sql = 'select "VendorID" as renamed from "Lake"."Raw"."input"'
    for constant, value, reason in [
        ("MAX_AST_NODES", 1, "AST budget exceeded"),
        ("MAX_PATH_NODES", 1, "lineage budget exceeded"),
        ("MAX_RELATIONS", 0, "schema budget exceeded"),
    ]:
        with monkeypatch.context() as patch:
            patch.setattr(column_lineage, constant, value)
            assert (
                resolve_candidates(sql, "duckdb", ["renamed"], [ancestor()])["renamed"].reason
                == reason
            )


def configure(yaml_context, enabled=True, expression="first_name"):
    manifest = yaml_context.project.manifest
    node = manifest.nodes["model.jaffle_shop_duckdb.stg_customers.v1"]
    donor = manifest.nodes["seed.jaffle_shop_duckdb.raw_customers"]
    node.columns = {"renamed": ColumnInfo(name="renamed", description="")}
    node.compiled_code = (
        f'select {expression} as renamed from "{donor.database}"."{donor.schema}"."{donor.alias}"'
    )
    node.config.meta["dbt-osmosis-options"] = {"use-sqlglot-lineage": enabled}
    donor.columns["first_name"].description = "Original donor documentation"
    return node, donor


def test_real_inheritance_and_default_opt_out(yaml_context, monkeypatch):
    node, donor = configure(yaml_context, enabled=False)
    with monkeypatch.context() as patch:
        patch.setattr(
            "dbt_osmosis.core.inheritance.resolve_candidates",
            lambda *args: pytest.fail("disabled lineage parsed SQL"),
        )
        inherit_upstream_column_knowledge(yaml_context, node)
    assert node.columns["renamed"].description == ""
    node.config.meta["dbt-osmosis-options"]["use-sqlglot-lineage"] = True
    inherit_upstream_column_knowledge(yaml_context, node)
    assert node.columns["renamed"].description == donor.columns["first_name"].description
    assert node.columns["renamed"].name == "renamed"
    inherit_upstream_column_knowledge(yaml_context, node)
    assert node.columns["renamed"].description == "Original donor documentation"


def test_explicit_description_and_unrendered_donor(yaml_context):
    node, donor = configure(yaml_context)
    node.columns["renamed"].description = "Local description"
    inherit_upstream_column_knowledge(yaml_context, node)
    assert node.columns["renamed"].description == "Local description"
    node.columns["renamed"].description = ""
    node.config.meta["dbt-osmosis-options"]["use-unrendered-descriptions"] = True
    from dbt_osmosis.core.inheritance import _get_node_yaml

    raw = _get_node_yaml(yaml_context, donor)
    assert raw is not None
    next(c for c in raw["columns"] if c["name"] == "first_name")["description"] = (
        "{{ doc('first_name') }}"
    )
    inherit_upstream_column_knowledge(yaml_context, node)
    assert node.columns["renamed"].description == "{{ doc('first_name') }}"


def test_semantic_rejection_suppresses_same_name_fallback(yaml_context, monkeypatch):
    node, donor = configure(yaml_context, expression="lower(first_name)")
    donor.columns["renamed"] = ColumnInfo(name="renamed", description="Unrelated same name")
    inherit_upstream_column_knowledge(yaml_context, node)
    assert node.columns["renamed"].description == ""
    node.compiled_code = None
    monkeypatch.setattr("dbt_osmosis.core.inheritance._lineage_sql", lambda context, target: None)
    inherit_upstream_column_knowledge(yaml_context, node)
    assert node.columns["renamed"].description == "Unrelated same name"


def test_force_inherit_and_explicit_override(yaml_context):
    from dbt_osmosis.core.inheritance import _get_node_yaml

    node, donor = configure(yaml_context)
    node.columns["renamed"].description = "Local description"
    yaml_context.settings.force_inherit_descriptions = True
    inherit_upstream_column_knowledge(yaml_context, node)
    assert node.columns["renamed"].description == "Original donor documentation"
    # Explicit override retains its existing deliberate same-name semantics,
    # even when automatic lineage rejects a transformed projection.
    node.compiled_code = node.compiled_code.replace("select first_name", "select lower(first_name)")
    donor.columns["renamed"] = ColumnInfo(name="renamed", description="Explicit donor")
    raw = _get_node_yaml(yaml_context, node)
    assert raw is not None
    raw["columns"].append({
        "name": "renamed",
        "meta": {"column_default_progenitor": donor.unique_id},
    })
    inherit_upstream_column_knowledge(yaml_context, node)
    assert node.columns["renamed"].description == "Explicit donor"


def test_no_stale_candidate_after_schema_change():
    sql = 'select "VendorID" as renamed from "Lake"."Raw"."input"'
    donor = ancestor()
    assert resolve_candidates(sql, "trino", ["renamed"], [donor])["renamed"].status == "candidate"
    donor.columns = {"other": None}
    assert resolve_candidates(sql, "trino", ["renamed"], [donor])["renamed"].status == "rejected"


def test_lineage_dry_run_preserves_yaml(yaml_context):
    from pathlib import Path

    from dbt_osmosis.core.sync_operations import sync_node_to_yaml

    node, _ = configure(yaml_context)
    project = Path(yaml_context.project.runtime_cfg.project_root)
    before = {
        p: (p.read_bytes(), p.stat().st_mtime_ns) for p in project.rglob("*.yml") if p.is_file()
    }
    yaml_context.settings.dry_run = True
    inherit_upstream_column_knowledge(yaml_context, node)
    sync_node_to_yaml(yaml_context, node, commit=True)
    assert node.columns["renamed"].description == "Original donor documentation"
    after = {
        p: (p.read_bytes(), p.stat().st_mtime_ns) for p in project.rglob("*.yml") if p.is_file()
    }
    assert after == before


@pytest.mark.parametrize(
    "sql",
    [
        'select "VendorID" as renamed, other as renamed from "Lake"."Raw"."input"',
        'with c as (select "VendorID" as renamed, other as renamed from "Lake"."Raw"."input") select renamed from c',
    ],
)
def test_duplicate_output_alias_never_selects_first(sql):
    assert (
        resolve_candidates(sql, "trino", ["renamed"], [ancestor()])["renamed"].status == "rejected"
    )


def test_opt_in_compiles_actual_model_once(yaml_context, monkeypatch):
    from dbt_osmosis.core import sql_operations

    node, _ = configure(yaml_context)
    node.compiled_code = None
    node.raw_code = "select first_name as renamed, '{{ this }}' as relation_name from {{ ref('raw_customers') }}"
    calls = []
    compile_model = sql_operations.compile_model_for_lineage

    def tracked(context, model):
        calls.append(model.unique_id)
        return compile_model(context, model)

    monkeypatch.setattr(sql_operations, "compile_model_for_lineage", tracked)
    inherit_upstream_column_knowledge(yaml_context, node)
    assert node.columns["renamed"].description == "Original donor documentation"
    assert node.alias in node.compiled_code
    assert "sql_operation" not in node.compiled_code
    inherit_upstream_column_knowledge(yaml_context, node)
    assert calls == [node.unique_id]


def test_compilation_failure_logs_and_falls_back_once(yaml_context, monkeypatch, caplog):
    from dbt.exceptions import CompilationError

    node, donor = configure(yaml_context)
    node.compiled_code = None
    donor.columns["renamed"] = ColumnInfo(name="renamed", description="Legacy fallback")
    yaml_context.settings.force_inherit_descriptions = True
    calls = []

    def fail(context, model):
        calls.append(model.unique_id)
        model.compiled_code = "select first_name as renamed from invalid_partial_compilation"
        raise CompilationError("test compiler failure")

    monkeypatch.setattr("dbt_osmosis.core.sql_operations.compile_model_for_lineage", fail)
    inherit_upstream_column_knowledge(yaml_context, node)
    inherit_upstream_column_knowledge(yaml_context, node)
    assert calls == [node.unique_id]
    assert node.compiled_code is None
    assert node.columns["renamed"].description == "Legacy fallback"
    assert "test compiler failure" in caplog.text
    assert "using name matching" in caplog.text


def test_compilation_attempt_resets_for_new_manifest(yaml_context, monkeypatch):
    from copy import copy

    from dbt.exceptions import CompilationError

    node, _ = configure(yaml_context)
    node.compiled_code = None
    calls = []

    def fail(context, model):
        calls.append(model.unique_id)
        raise CompilationError("test compiler failure")

    monkeypatch.setattr("dbt_osmosis.core.sql_operations.compile_model_for_lineage", fail)
    inherit_upstream_column_knowledge(yaml_context, node)
    inherit_upstream_column_knowledge(yaml_context, node)
    assert len(calls) == 1
    yaml_context.project.manifest = copy(yaml_context.project.manifest)
    inherit_upstream_column_knowledge(yaml_context, node)
    assert len(calls) == 2
