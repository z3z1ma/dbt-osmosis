"""Catalog ordinals survive metadata conversion and actual CLI transformations."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from dbt.adapters.base.column import Column
from dbt.adapters.duckdb.relation import DuckDBRelation
from dbt_common.contracts.metadata import ColumnMetadata
from ruamel.yaml import YAML

from dbt_osmosis.core.introspection import get_columns
from dbt_osmosis.core.settings import YamlRefactorSettings


@pytest.mark.parametrize("quoted", [False, True])
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("section", ["nodes", "sources"])
@pytest.mark.parametrize("indexes", [(3, 1, 2), (-3, 100, 0), (0, 0, 0)])
def test_catalog_conversion_retains_ordinals(shared, section, indexes, quoted):
    # SimpleNamespace models the distinct Core 1.8 catalog metadata shape on every
    # CI row; the CLI regression below also exercises the installed actual class.
    factory = ColumnMetadata if shared else SimpleNamespace
    raw = {
        name: factory(
            name=f"[{name}]" if quoted else name, type="INTEGER", index=index, comment=f"Doc {name}"
        )
        for name, index in zip(("last", "skip_me", "first"), indexes)
    }
    table = SimpleNamespace(
        metadata=SimpleNamespace(database="fixture", schema="main", name="accounts"),
        columns=raw,
    )
    catalog = SimpleNamespace(nodes={}, sources={})
    getattr(catalog, section)["resource"] = table
    adapter = Mock()
    context = SimpleNamespace(
        project=SimpleNamespace(
            adapter=adapter,
            config=SimpleNamespace(disable_introspection=False),
            runtime_cfg=SimpleNamespace(credentials=SimpleNamespace(type="duckdb")),
        ),
        settings=YamlRefactorSettings(),
        ignore_patterns=[r"^\[skip_" if quoted else "^skip_"],
        read_catalog=lambda: catalog,
    )
    result = get_columns(
        context, DuckDBRelation.create(database="fixture", schema="main", identifier="accounts")
    )
    assert list(result) == (["[last]", "[first]"] if quoted else ["last", "first"])
    assert [(c.name, c.type, c.comment, c.index) for c in result.values()] == [
        (f"[{name}]" if quoted else name, "INTEGER", f"Doc {name}", indexes[index])
        for name, index in [("last", 0), ("first", 2)]
    ]
    assert [c.index for c in raw.values()] == list(indexes)
    if shared:
        assert result["[last]" if quoted else "last"] is raw["last"]
    adapter.get_columns_in_relation.assert_not_called()


@pytest.mark.parametrize("producer", ["fusion_2_0_12", "dbt_oss_2_0_11"])
@pytest.mark.parametrize("alphabetical", [False, True])
def test_catalog_ordinals_through_cli(tmp_path, producer, alphabetical):
    fixture = Path(__file__).parent / "fixtures" / producer
    project = tmp_path / "project"
    shutil.copytree(fixture / "project", project)
    yaml = YAML()
    profile_path = project / "profiles.yml"
    profile = yaml.load(profile_path.read_text())
    profile["osmosis_fusion_fixture"]["outputs"]["local"]["path"] = str(project / "fixture.duckdb")
    yaml.dump(profile, profile_path)
    schema = project / "models" / "schema.yml"
    data = yaml.load(schema.read_text())
    for model in data["models"]:
        model["config"] = {"meta": {"future_semantics": {"keep": ["one", "two"]}}}
        if alphabetical:
            model["config"]["meta"]["dbt-osmosis-options"] = {"sort-by": "alphabetical"}
    if alphabetical:
        data["sources"][0]["meta"] = {"dbt-osmosis-options": {"sort-by": "alphabetical"}}
    yaml.dump(data, schema)
    raw_before = (fixture / "catalog.json").read_bytes()
    catalog = json.loads(raw_before)
    # Derived adversarial input, not an untouched producer artifact. Isolate
    # sources so they cannot pass by consuming a model for the same relation.
    catalog["nodes"] = {}
    for table in catalog["sources"].values():
        table["columns"] = dict(reversed(list(table["columns"].items())))
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(catalog))
    catalog_before = catalog_path.read_bytes()
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
    env = dict(os.environ, DBT_SEND_ANONYMOUS_USAGE_STATS="false")
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in project.rglob("*.yml")}
    dry = subprocess.run(
        [*args, "--dry-run"], capture_output=True, text=True, env=env, timeout=90, check=False
    )
    assert dry.returncode == 0, dry.stdout + dry.stderr
    assert {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in project.rglob("*.yml")} == before
    result = subprocess.run(args, capture_output=True, text=True, env=env, timeout=90, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    output = yaml.load(schema.read_text())
    expected = (
        ["account_id", "is_active", "status"]
        if alphabetical
        else ["account_id", "status", "is_active"]
    )
    # accounts matches the source catalog by physical identity; both resources
    # exercise sorting without a matching model catalog hiding the source path.
    accounts = output["models"][0]
    source = output["sources"][0]["tables"][0]
    assert [c["name"] for c in accounts["columns"]] == expected
    assert [c["name"] for c in source["columns"]] == expected
    assert accounts["config"]["meta"]["future_semantics"] == {"keep": ["one", "two"]}
    assert {c["name"]: c["description"] for c in accounts["columns"]} == {
        "account_id": "Original account identifier.",
        "status": "Original account status.",
        "is_active": "Original active marker.",
    }
    assert {c["name"]: c["data_type"] for c in source["columns"]} == {
        "account_id": "INTEGER",
        "status": "VARCHAR",
        "is_active": "BOOLEAN",
    }
    assert catalog_path.read_bytes() == catalog_before
    assert (fixture / "catalog.json").read_bytes() == raw_before
    assert not (project / "fixture.duckdb").exists()


@pytest.mark.usefixtures("fresh_caches")
def test_warehouse_flattening_keeps_enumeration():
    class NestedColumn(Column):
        # Adapter attributes called index must not become catalog ordinals.
        index = 77

        def flatten(self):
            return [Column("skip_child", "INTEGER"), Column("child", "INTEGER")]

    adapter = Mock()
    adapter.get_columns_in_relation.return_value = [
        NestedColumn("parent", "STRUCT"),
        Column("tail", "INTEGER"),
    ]
    context = SimpleNamespace(
        project=SimpleNamespace(
            adapter=adapter,
            config=SimpleNamespace(disable_introspection=False),
            runtime_cfg=SimpleNamespace(
                project_root="/tmp/ordinal-control",
                profile_name="offline",
                target_name="test",
                credentials=SimpleNamespace(type="duckdb"),
            ),
        ),
        settings=YamlRefactorSettings(),
        ignore_patterns=["^skip_"],
        read_catalog=lambda: None,
    )
    relation = DuckDBRelation.create(database="fixture", schema="main", identifier="nested")
    result = get_columns(context, relation)
    assert [(c.name, c.index) for c in result.values()] == [
        ("parent", 0),
        ("child", 1),
        ("tail", 2),
    ]
    adapter.get_columns_in_relation.assert_called_once_with(relation)
