"""Real DuckDB physical names must not collapse at the metadata boundary."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from dbt.contracts.graph.nodes import ColumnInfo
from dbt_common.contracts.metadata import ColumnMetadata
from ruamel.yaml import YAML

from dbt_osmosis.core.diff import SchemaDiff
from dbt_osmosis.core.inheritance import _get_unrendered
from dbt_osmosis.core.introspection import normalize_column_name

NAMES = ["id", "[id]", "`id`", "[[edge]]", '"id"']
TYPES = dict(zip(NAMES, ["INTEGER", "BIGINT", "SMALLINT", "INTEGER", "VARCHAR"]))
FIXTURE = Path(__file__).parent / "fixtures" / "duckdb_literal_columns"


@pytest.mark.parametrize("name", [*NAMES, "[", "]", "`", '"', "mixedCASE"])
def test_duckdb_literal_name(name):
    assert normalize_column_name(name, "duckdb") == name


@pytest.mark.parametrize("adapter", ["postgres", "bigquery", "snowflake"])
def test_other_adapter_normalization_unchanged(adapter):
    assert normalize_column_name('"id"', adapter) == "id"
    assert normalize_column_name("id", adapter) == ("ID" if adapter == "snowflake" else "id")


@pytest.mark.parametrize("case_insensitive", [False, True])
def test_diff_preserves_distinct_literal_columns(case_insensitive):
    context = SimpleNamespace(
        project=SimpleNamespace(
            runtime_cfg=SimpleNamespace(credentials=SimpleNamespace(type="duckdb"))
        )
    )
    differ = SchemaDiff(context)
    yaml_columns = {n: ColumnInfo(name=n, description="Doc " + n) for n in NAMES}
    database_columns = {
        n: ColumnMetadata(name=n, type=TYPES[n], index=i) for i, n in enumerate(NAMES)
    }
    with patch.object(differ, "_case_insensitive_output_enabled", return_value=case_insensitive):
        yaml_names, database_names = differ._comparison_columns_by_name(
            SimpleNamespace(), yaml_columns, database_columns
        )
    assert list(yaml_names) == NAMES
    assert list(database_names) == NAMES
    assert {n: c.description for n, c in yaml_names.items()} == {n: "Doc " + n for n in NAMES}


@pytest.mark.parametrize("name", NAMES)
def test_unrendered_inheritance_selects_literal_donor(name):
    context = SimpleNamespace(
        project=SimpleNamespace(
            runtime_cfg=SimpleNamespace(credentials=SimpleNamespace(type="duckdb"))
        )
    )
    raw_yaml = {"columns": [{"name": n, "description": "Doc " + n} for n in NAMES]}
    with patch("dbt_osmosis.core.inheritance._get_node_yaml", return_value=raw_yaml):
        assert _get_unrendered(context, "description", name, SimpleNamespace(), {name: [name]}) == (
            "Doc " + name
        )


@pytest.mark.parametrize(
    "scenario", ["new", "existing", "source-only", "ignore", "alphabetical", "upper"]
)
def test_literal_columns_through_cli(tmp_path, scenario):
    yaml = YAML()
    project = tmp_path / "project"
    shutil.copytree(FIXTURE / "project", project)
    profile_path = project / "profiles.yml"
    profile = yaml.load(profile_path.read_text())
    profile["literal_fixture"]["outputs"]["local"]["path"] = str(project / "fixture.duckdb")
    yaml.dump(profile, profile_path)
    schema = project / "models" / "schema.yml"
    data = yaml.load(schema.read_text())
    existing = scenario == "existing"
    for resource in [data["models"][0], data["sources"][0]["tables"][0]]:
        resource["meta"] = {"future_semantics": {"keep": True}}
        if existing:
            for c in resource["columns"]:
                c["data_type"] = TYPES[c["name"]]
                c["meta"] = {"sentinel": c["name"]}
            resource["columns"].reverse()
        else:
            resource.pop("columns")
    yaml.dump(data, schema)
    config_path = project / "dbt_project.yml"
    config = yaml.load(config_path.read_text())
    options = {}
    if scenario == "alphabetical":
        options["sort-by"] = "alphabetical"
    if scenario == "upper":
        options["output-to-upper"] = True
    if options:
        for resource in [data["models"][0], data["sources"][0]["tables"][0]]:
            resource["meta"]["dbt-osmosis-options"] = options
        yaml.dump(data, schema)
    if scenario == "ignore":
        config["vars"] = {"dbt-osmosis": {"column_ignore_patterns": [r"^\[id\]$"]}}
        yaml.dump(config, config_path)
    raw = (FIXTURE / "catalog.json").read_bytes()
    catalog_path = tmp_path / "catalog.json"
    if scenario == "source-only":
        # Explicitly derived input isolates the source map; raw producer stays unchanged.
        catalog = json.loads(raw)
        catalog["nodes"] = {}
        catalog_path.write_text(json.dumps(catalog))
    else:
        catalog_path.write_bytes(raw)
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
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before}
    expected = [n for n in NAMES if not (scenario == "ignore" and n == "[id]")]
    if scenario == "alphabetical":
        expected = sorted(expected)
    if scenario == "upper":
        expected = [n.upper() for n in expected]
    prior = None
    for _ in range(2):
        run = subprocess.run(args, capture_output=True, text=True, env=env, timeout=90, check=False)
        assert run.returncode == 0, run.stdout + run.stderr
        current = schema.read_bytes()
        if prior is not None:
            assert current == prior
        prior = current
        output = yaml.load(current)
        for resource in [output["models"][0], output["sources"][0]["tables"][0]]:
            assert [c["name"] for c in resource["columns"]] == expected
            assert resource["meta"]["future_semantics"] == {"keep": True}
            for c in resource["columns"]:
                original_name = c["name"].lower() if scenario == "upper" else c["name"]
                assert c["data_type"] == TYPES[original_name]
                if existing:
                    assert c["quote"] is True
                    assert c["description"] == "Doc " + original_name
                    assert (
                        c.get("config", {}).get("meta", c.get("meta"))["sentinel"] == original_name
                    )
        assert catalog_path.read_bytes() == catalog_before
        assert (FIXTURE / "catalog.json").read_bytes() == raw
        assert not (project / "fixture.duckdb").exists()


@pytest.mark.parametrize("absent", NAMES)
def test_remove_missing_literal_column_without_removing_neighbors(absent):
    from dbt_osmosis.core.transforms import remove_columns_not_in_database

    context = SimpleNamespace(
        settings=SimpleNamespace(output_to_upper=False, output_to_lower=False),
        project=SimpleNamespace(
            runtime_cfg=SimpleNamespace(credentials=SimpleNamespace(type="duckdb"))
        ),
    )
    node = SimpleNamespace(
        unique_id="model.literal_fixture.accounts",
        columns={n: ColumnInfo(name=n, description="Doc " + n) for n in [*NAMES, "stale"]},
    )
    incoming = {
        n: ColumnMetadata(name=n, type=TYPES[n], index=i)
        for i, n in enumerate(NAMES)
        if n != absent
    }
    with (
        patch("dbt_osmosis.core.introspection.get_columns", return_value=incoming),
        patch(
            "dbt_osmosis.core.introspection.resolve_setting",
            side_effect=lambda *args, fallback=None, **kwargs: fallback,
        ),
    ):
        remove_columns_not_in_database(context, node)
    assert list(node.columns) == [n for n in NAMES if n != absent]
    assert {n: c.description for n, c in node.columns.items()} == {
        n: "Doc " + n for n in NAMES if n != absent
    }
