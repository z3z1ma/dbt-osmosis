from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import duckdb
import pytest
from ruamel.yaml import YAML


@pytest.mark.parametrize(
    ("scope", "expected_source_columns"),
    [
        ("default", {"id"}),
        ("models", {"id"}),
        ("vars", set()),
        ("project_sources", set()),
        ("table_meta", set()),
        ("table_override", {"id"}),
        ("cli", set()),
    ],
)
def test_refactor_source_column_config_scope(
    tmp_path: Path, scope: str, expected_source_columns: set[str]
) -> None:
    """Source settings honor their dbt scope through parsing and YAML writeback."""
    yaml = YAML()
    options = {"dbt-osmosis-options": {"skip-add-source-columns": True}}
    project = {
        "name": "source_config",
        "version": "1.0",
        "config-version": 2,
        "profile": "source_config",
        "models": {"source_config": {"+meta": {"dbt-osmosis": "{model}.yml"}}},
    }
    table = {"name": "raw_data"}
    if scope == "models":
        project["models"]["source_config"]["+meta"].update(options)
    elif scope in {"vars", "table_override"}:
        project["vars"] = {"dbt-osmosis": {"skip-add-source-columns": True}}
    elif scope == "project_sources":
        project["sources"] = {"source_config": {"+meta": options}}
    elif scope == "table_meta":
        table["meta"] = options
    if scope == "table_override":
        table["meta"] = {"dbt-osmosis-options": {"skip-add-source-columns": False}}

    yaml.dump(project, tmp_path / "dbt_project.yml")
    database = tmp_path / "warehouse.duckdb"
    yaml.dump(
        {
            "source_config": {
                "target": "test",
                "outputs": {"test": {"type": "duckdb", "path": str(database), "threads": 1}},
            }
        },
        tmp_path / "profiles.yml",
    )
    models = tmp_path / "models"
    models.mkdir()
    source_path = models / "sources.yml"
    yaml.dump(
        {"version": 2, "sources": [{"name": "raw", "schema": "main", "tables": [table]}]},
        source_path,
    )
    (models / "my_model.sql").write_text("select * from {{ source('raw', 'raw_data') }}\n")
    with duckdb.connect(str(database)) as connection:
        connection.execute("create table raw_data as select 1 as id")
        connection.execute("create table my_model as select * from raw_data")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "dbt_osmosis",
            "yaml",
            "refactor",
            "--project-dir",
            str(tmp_path),
            "--profiles-dir",
            str(tmp_path),
            "--auto-apply",
            *(["--skip-add-source-columns"] if scope == "cli" else []),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    source = yaml.load(source_path)["sources"][0]["tables"][0]
    assert {column["name"] for column in source.get("columns", [])} == expected_source_columns
    model = yaml.load(models / "my_model.yml")["models"][0]
    assert {column["name"] for column in model["columns"]} == {"id"}
