"""Catalog input contracts; real producer coverage uses the built DuckDB fixture."""

import json
import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from ruamel.yaml import YAML

from dbt_osmosis.core.catalog_operations import _load_catalog
from dbt_osmosis.core.exceptions import CatalogValidationError
from dbt_osmosis.core.settings import YamlRefactorContext, YamlRefactorSettings


def _catalog():
    return {
        "metadata": {"dbt_schema_version": "https://schemas.getdbt.com/dbt/catalog/v1.json"},
        "nodes": {
            "model.analytics.example": {
                "metadata": {"type": "TABLE", "schema": "main", "name": "example"},
                "columns": {
                    "id": {"name": "id", "type": "INTEGER", "index": 1, "comment": "Identity"}
                },
                "stats": {},
            }
        },
        "sources": {},
        "errors": None,
    }


def _load(tmp_path, data):
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(data))
    return _load_catalog(SimpleNamespace(catalog_path=str(path)))


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {
            "dbt_version": "99.0",
            "future_producer": "synthetic",
            "dbt_schema_version": "https://schemas.getdbt.com/dbt/catalog/v1.json",
        },
    ],
)
def test_compatible_schema_less_or_future_metadata(tmp_path, metadata):
    data = _catalog()
    data["metadata"] = metadata
    artifact = _load(tmp_path, data)
    column = artifact.nodes["model.analytics.example"].columns["id"]
    assert (column.name, column.type, column.comment, column.index) == (
        "id",
        "INTEGER",
        "Identity",
        1,
    )
    assert json.loads((tmp_path / "catalog.json").read_text()) == data


@pytest.mark.parametrize(
    "data,reason",
    [
        ([], "JSON object"),
        ({"nodes": {}, "sources": {}}, "metadata"),
        ({"metadata": []}, "metadata"),
        ({"metadata": {}, "nodes": [], "sources": {}}, "nodes"),
        ({"metadata": {}, "nodes": {}, "sources": []}, "sources"),
        ({"metadata": {}, "nodes": {}, "sources": {}, "macros": {}}, "manifest"),
        (
            {
                "metadata": {
                    "dbt_schema_version": "https://schemas.getdbt.com/dbt/manifest/v12.json"
                },
                "nodes": {},
                "sources": {},
            },
            "catalog v1",
        ),
        (
            {
                "metadata": {
                    "dbt_schema_version": "https://schemas.getdbt.com/dbt/catalog/v2.json"
                },
                "nodes": {},
                "sources": {},
            },
            "catalog v1",
        ),
    ],
)
def test_rejects_wrong_envelope(tmp_path, data, reason):
    with pytest.raises(CatalogValidationError, match=reason) as exc:
        _load(tmp_path, data)
    assert str(tmp_path / "catalog.json") in str(exc.value)


@pytest.mark.parametrize(
    "field,value", [("name", 7), ("type", None), ("index", "1"), ("index", True), ("comment", [])]
)
def test_rejects_malformed_consumed_column(tmp_path, field, value):
    data = _catalog()
    data["nodes"]["model.analytics.example"]["columns"]["id"][field] = value
    with pytest.raises(CatalogValidationError, match=rf"columns.id.{field}"):
        _load(tmp_path, data)


@pytest.mark.parametrize("section,value", [("metadata", []), ("columns", []), ("stats", [])])
def test_rejects_malformed_table(tmp_path, section, value):
    data = _catalog()
    data["nodes"]["model.analytics.example"][section] = value
    with pytest.raises(CatalogValidationError):
        _load(tmp_path, data)


def test_invalid_json_and_unreadable_path(tmp_path):
    path = tmp_path / "catalog.json"
    path.write_text("{")
    with pytest.raises(CatalogValidationError, match="catalog.json"):
        _load_catalog(SimpleNamespace(catalog_path=str(path)))
    with pytest.raises(CatalogValidationError, match="catalog v1"):
        _load_catalog(SimpleNamespace(catalog_path=str(tmp_path)))


def test_missing_path_retains_fallback(tmp_path):
    assert _load_catalog(SimpleNamespace(catalog_path=None)) is None
    assert _load_catalog(SimpleNamespace(catalog_path=str(tmp_path / "absent.json"))) is None


def test_context_preflight_does_not_generate_or_allocate_pool(tmp_path):
    path = tmp_path / "catalog.json"
    path.write_text("[]")
    with (
        patch("dbt_osmosis.core.catalog_operations._generate_catalog") as generate,
        patch("dbt_osmosis.core.settings.ThreadPoolExecutor") as pool,
        pytest.raises(CatalogValidationError),
    ):
        YamlRefactorContext(
            project=Mock(),
            settings=YamlRefactorSettings(
                catalog_path=str(path), create_catalog_if_not_exists=True
            ),
        )
    generate.assert_not_called()
    pool.assert_not_called()


def test_actual_core_generated_catalog(built_duckdb_template):
    path = built_duckdb_template / "target" / "catalog.json"
    before = path.read_bytes()
    raw = json.loads(before)
    assert raw["metadata"]["dbt_schema_version"] == "https://schemas.getdbt.com/dbt/catalog/v1.json"
    assert raw["metadata"]["dbt_version"]
    artifact = _load_catalog(SimpleNamespace(catalog_path=str(path)))
    assert artifact.nodes
    for section in ("nodes", "sources"):
        for unique_id, record in raw[section].items():
            loaded = getattr(artifact, section)[unique_id]
            for key, column in record["columns"].items():
                assert loaded.columns[key].name == column["name"]
                assert loaded.columns[key].type == column["type"]
                assert loaded.columns[key].comment == column.get("comment")
    assert path.read_bytes() == before


@pytest.mark.parametrize("command", ["refactor", "organize", "document"])
@pytest.mark.parametrize("valid", [False, True])
def test_catalog_cli_preserves_yaml_or_fails_before_writes(tmp_path, command, valid):
    project = tmp_path / "project"
    models = project / "models"
    models.mkdir(parents=True)
    (project / "dbt_project.yml").write_text(
        "name: analytics\nversion: '1.0'\nconfig-version: 2\nprofile: analytics\nmodels:\n  analytics:\n    +dbt-osmosis: schema.yml\n"
    )
    (project / "profiles.yml").write_text(
        "analytics:\n  target: test\n  outputs:\n    test:\n      type: duckdb\n      path: ':memory:'\n      threads: 1\n"
    )
    (models / "example.sql").write_text("select 1 as id")
    (models / "original.yml").write_text(
        "version: 2\nmodels:\n  - name: example\n    config:\n      meta:\n        future_semantics: {keep: [one, two]}\n    columns:\n      - name: id\n        description: Retain me\n"
    )
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        '{"metadata":{"dbt_schema_version":"https://schemas.getdbt.com/dbt/manifest/v12.json"},"nodes":{},"sources":{}}'
    )
    if valid:
        catalog.write_text(json.dumps(_catalog()))
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in project.rglob("*.yml")}
    catalog_before = catalog.read_bytes()
    extra = ["--auto-apply"] if command in {"refactor", "organize"} else []
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "dbt_osmosis",
            "yaml",
            command,
            "--project-dir",
            str(project),
            "--profiles-dir",
            str(project),
            "--catalog-path",
            str(catalog),
            *extra,
        ],
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
        env=dict(os.environ, DBT_SEND_ANONYMOUS_USAGE_STATS="false"),
    )
    output = result.stdout + result.stderr
    if valid:
        assert result.returncode == 0, output
        nodes = [
            node
            for path in models.glob("*.yml")
            for node in YAML(typ="safe").load(path).get("models", [])
        ]
        assert len(nodes) == 1
        assert nodes[0]["config"]["meta"]["future_semantics"] == {"keep": ["one", "two"]}
        assert nodes[0]["columns"][0]["description"] == "Retain me"
    else:
        assert result.returncode != 0
        assert "Error: Cannot load catalog" in output
        assert str(catalog) in output
        assert "Traceback" not in output
        assert {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in project.rglob("*.yml")} == before
    assert catalog.read_bytes() == catalog_before
