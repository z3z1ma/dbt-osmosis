"""Final YAML sync must select the same physical catalog donor as column collection."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from dbt.adapters.base.relation import BaseRelation
from dbt.adapters.duckdb.relation import DuckDBRelation
from ruamel.yaml import YAML

from dbt_osmosis.core.catalog_operations import _load_catalog
from dbt_osmosis.core.sync_operations import _catalog_column_types

FIXTURE = Path(__file__).parent / "fixtures" / "catalog_relation_identity"


@pytest.mark.parametrize("scenario", ["stale", "new", "source-only", "reversed", "skip-types"])
def test_catalog_identity_through_cli(tmp_path, scenario):
    yaml = YAML()
    project = tmp_path / "project"
    shutil.copytree(FIXTURE / "project", project)
    profile = yaml.load((project / "profiles.yml").read_text())
    profile["alias_fixture"]["outputs"]["local"]["path"] = str(project / "fixture.duckdb")
    profile["alias_fixture"]["outputs"]["local"]["attach"][0]["path"] = str(
        project / "other.duckdb"
    )
    yaml.dump(profile, project / "profiles.yml")
    schema = project / "models" / "schema.yml"
    data = yaml.load(schema.read_text())
    for resource in [data["models"][0], data["sources"][0]["tables"][0]]:
        resource["meta"] = {"future_semantics": {"keep": True}}
        if scenario == "new":
            resource.pop("columns")
        else:
            resource["columns"][0]["data_type"] = "STALE"
            resource["columns"][0]["quote"] = True
    yaml.dump(data, schema)
    raw = (FIXTURE / "catalog.json").read_bytes()
    catalog_path = tmp_path / "catalog.json"
    catalog = json.loads(raw)
    if scenario == "source-only":
        catalog["nodes"] = {}  # Derived isolation input; raw producer is unchanged.
    if scenario == "reversed":
        catalog["nodes"] = dict(reversed(list(catalog["nodes"].items())))
        catalog["sources"] = dict(reversed(list(catalog["sources"].items())))
    catalog_path.write_text(json.dumps(catalog))
    before_catalog = catalog_path.read_bytes()
    args = [
        sys.executable,
        "-m",
        "dbt_osmosis",
        "yaml",
        "document",
        "--project-dir",
        str(project),
        "--profiles-dir",
        str(project),
        "--catalog-path",
        str(catalog_path),
        "--disable-introspection",
    ]
    if scenario == "skip-types":
        args.append("--skip-add-data-types")
    env = dict(os.environ, DBT_SEND_ANONYMOUS_USAGE_STATS="false")
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in project.rglob("*.yml")}
    dry = subprocess.run(
        [*args, "--dry-run"], capture_output=True, text=True, env=env, timeout=90, check=False
    )
    assert dry.returncode == 0, dry.stdout + dry.stderr
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before}
    prior = None
    for _ in range(2):
        run = subprocess.run(args, capture_output=True, text=True, env=env, timeout=90, check=False)
        assert run.returncode == 0, run.stdout + run.stderr
        output = yaml.load(schema.read_text())
        if prior is not None:
            assert schema.read_bytes() == prior
        prior = schema.read_bytes()
        for resource in [output["models"][0], output["sources"][0]["tables"][0]]:
            assert resource["meta"]["future_semantics"] == {"keep": True}
            assert len(resource["columns"]) == 1
            column = resource["columns"][0]
            assert column["name"] == "id"
            assert column["data_type"] == ("STALE" if scenario == "skip-types" else "INTEGER")
            if scenario != "new":
                assert column["quote"] is True
                assert column["description"] == (
                    "Correct model doc"
                    if resource["name"] == "accounts" and "identifier" not in resource
                    else "Correct source doc"
                )
                assert column.get("config", {}).get("meta", column.get("meta"))["sentinel"] in {
                    "model",
                    "source",
                }
        if scenario != "skip-types":
            other = output["sources"][1]["tables"][0]["columns"][0]
            assert other["data_type"] == "BIGINT"
            assert other["description"] == "Other database doc"
        assert not (project / "other.duckdb").exists()
        assert catalog_path.read_bytes() == before_catalog
        assert (FIXTURE / "catalog.json").read_bytes() == raw
        assert not (project / "fixture.duckdb").exists()


@pytest.mark.parametrize(
    "adapter,relation_type", [("duckdb", DuckDBRelation), ("postgres", BaseRelation)]
)
@pytest.mark.parametrize("section", ["nodes", "sources"])
@pytest.mark.parametrize("case", ["database", "quoted", "unquoted", "missing", "literal-column"])
def test_final_sync_uses_canonical_catalog_identity(
    tmp_path, adapter, relation_type, section, case
):
    """Constructed adversarial inputs, not producer/warehouse certification."""
    relation = relation_type.create(
        database="right",
        schema="main",
        identifier="Orders",
        quote_policy={"identifier": case != "unquoted"},
    )
    name = "orders" if case == "unquoted" and adapter == "postgres" else "Orders"
    column_name = "[id]" if case == "literal-column" else "id"

    def entry(database, identifier, data_type):
        return {
            "metadata": {
                "type": "TABLE",
                "database": database,
                "schema": "main",
                "name": identifier,
            },
            "columns": {
                "irrelevant_mapping_key": {"name": column_name, "type": data_type, "index": 7}
            },
            "stats": {},
        }

    wrong = entry(
        "wrong" if case in {"database", "literal-column"} else "right",
        "orders" if case in {"quoted", "missing"} else "Orders",
        "VARCHAR",
    )
    correct = entry("right", name, "INTEGER")
    data = {"metadata": {}, "nodes": {}, "sources": {}}
    data[section]["wrong"] = wrong
    if case != "missing":
        data["sources" if section == "nodes" else "nodes"]["correct"] = correct
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(data))
    catalog = _load_catalog(SimpleNamespace(catalog_path=str(path)))
    adapter_obj = SimpleNamespace(
        Relation=SimpleNamespace(create_from=Mock(return_value=relation)),
        config=object(),
        get_columns_in_relation=Mock(side_effect=AssertionError("No warehouse")),
    )
    context = SimpleNamespace(
        read_catalog=lambda: catalog,
        project=SimpleNamespace(
            adapter=adapter_obj,
            runtime_cfg=SimpleNamespace(credentials=SimpleNamespace(type=adapter)),
        ),
    )
    node = SimpleNamespace(name="Orders", schema="main")
    expected_type = "INTEGER"
    if case == "missing":
        expected_type = None if adapter == "postgres" else "VARCHAR"
    # Existing exact duplicate ordering is unchanged for DuckDB unquoted identity.
    if case == "unquoted" and adapter == "duckdb" and section == "nodes":
        expected_type = "VARCHAR"
    result = _catalog_column_types(context, node)
    expected_name = "id" if case == "literal-column" and adapter == "postgres" else column_name
    assert result == ({} if expected_type is None else {expected_name: expected_type})
    adapter_obj.get_columns_in_relation.assert_not_called()


def test_final_sync_without_catalog_does_not_construct_relation():
    adapter = Mock()
    context = SimpleNamespace(read_catalog=lambda: None, project=SimpleNamespace(adapter=adapter))
    assert _catalog_column_types(context, SimpleNamespace()) == {}
    adapter.Relation.create_from.assert_not_called()
    adapter.get_columns_in_relation.assert_not_called()


@pytest.mark.parametrize("quoted_schema", [False, True])
def test_final_sync_preserves_mixed_postgres_quote_policy(tmp_path, quoted_schema):
    relation = BaseRelation.create(
        database="db",
        schema="Sales",
        identifier="Orders",
        quote_policy={"schema": quoted_schema, "identifier": True},
    )
    data = {"metadata": {}, "nodes": {}, "sources": {}}
    for schema, data_type in [("sales", "INTEGER"), ("Sales", "BIGINT")]:
        data["nodes"][schema] = {
            "metadata": {"type": "TABLE", "database": "db", "schema": schema, "name": "Orders"},
            "columns": {"id": {"name": "id", "type": data_type, "index": 1}},
            "stats": {},
        }
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(data))
    catalog = _load_catalog(SimpleNamespace(catalog_path=str(path)))
    adapter = SimpleNamespace(
        Relation=SimpleNamespace(create_from=Mock(return_value=relation)),
        config=object(),
        get_columns_in_relation=Mock(side_effect=AssertionError("No warehouse")),
    )
    context = SimpleNamespace(
        read_catalog=lambda: catalog,
        project=SimpleNamespace(
            adapter=adapter,
            runtime_cfg=SimpleNamespace(credentials=SimpleNamespace(type="postgres")),
        ),
    )
    assert _catalog_column_types(context, SimpleNamespace(name="Orders", schema="Sales")) == {
        "id": "BIGINT" if quoted_schema else "INTEGER"
    }
    adapter.get_columns_in_relation.assert_not_called()


@pytest.mark.parametrize("catalog_present", [False, True])
def test_final_sync_without_donor_preserves_existing_type(tmp_path, catalog_present):
    from dbt.contracts.graph.nodes import ColumnInfo

    from dbt_osmosis.core.sync_operations import _sync_doc_section

    path = tmp_path / "catalog.json"
    path.write_text(json.dumps({"metadata": {}, "nodes": {}, "sources": {}}))
    catalog = _load_catalog(SimpleNamespace(catalog_path=str(path))) if catalog_present else None
    relation = DuckDBRelation.create(database="db", schema="main", identifier="missing")
    adapter = SimpleNamespace(
        Relation=SimpleNamespace(create_from=Mock(return_value=relation)),
        config=object(),
        get_columns_in_relation=Mock(side_effect=AssertionError("No warehouse")),
    )
    context = SimpleNamespace(
        settings=SimpleNamespace(
            scaffold_empty_configs=False,
            skip_add_data_types=False,
            skip_merge_meta=False,
            use_unrendered_descriptions=False,
            prefer_yaml_values=False,
            output_to_upper=False,
            output_to_lower=False,
        ),
        fusion_compat=False,
        placeholders=set(),
        read_catalog=lambda: catalog,
        project=SimpleNamespace(
            adapter=adapter,
            runtime_cfg=SimpleNamespace(credentials=SimpleNamespace(type="duckdb"), vars={}),
        ),
    )
    node = SimpleNamespace(
        unique_id="model.fixture.missing",
        name="missing",
        schema="main",
        description="",
        meta={},
        config=SimpleNamespace(extra={}, meta={}),
        unrendered_config={},
        columns={
            "id": ColumnInfo(
                name="id",
                description="Keep docs",
                data_type="STALE",
                quote=True,
                meta={"sentinel": True},
            )
        },
    )
    section = {
        "name": "missing",
        "columns": [
            {
                "name": "id",
                "data_type": "STALE",
                "description": "Keep docs",
                "quote": True,
                "meta": {"sentinel": True},
            }
        ],
    }
    _sync_doc_section(context, node, section)
    assert section["columns"] == [
        {
            "name": "id",
            "data_type": "STALE",
            "description": "Keep docs",
            "quote": True,
            "meta": {"sentinel": True},
        }
    ]
    adapter.get_columns_in_relation.assert_not_called()
