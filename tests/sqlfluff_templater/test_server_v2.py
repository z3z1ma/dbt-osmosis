from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from dbt_osmosis.core.server_v2 import app

from tests.sqlfluff_templater.fixtures.dbt.templater import (
    DBT_FLUFF_CONFIG,
    profiles_dir,
    project_dir,
    sqlfluff_config_path,
)  # noqa: F401

client = TestClient(app)

SQL_PATH = Path(DBT_FLUFF_CONFIG["templater"]["dbt"]["project_dir"]) / "models/my_new_project/issue_1608.sql"


@pytest.mark.parametrize(
    "param_name, param_value",
    [
        ("sql_path", SQL_PATH),
        (None, SQL_PATH.read_text()),
    ],
)
def test_lint(param_name, param_value, profiles_dir, project_dir, sqlfluff_config_path, caplog):
    params = {}
    kwargs = {}
    if param_name:
        params[param_name] = param_value
    else:
        kwargs["data"] = param_value
    response = client.post(
        "/lint",
        headers={"X-dbt-Project": "dbt_project"},
        params=params,
        **kwargs,
    )
    assert response.status_code == 200
    response_json = response.json()
    assert response_json == {
        "result": [
            {
                "code": "L003",
                "description": "Expected 2 indentations, found less than 2 "
                "[compared to line 03]",
                "line_no": 4,
                "line_pos": 6,
            },
            {
                "code": "L023",
                "description": "Single whitespace expected after 'AS' in 'WITH' " "clause.",
                "line_no": 7,
                "line_pos": 7,
            },
        ]
    }


def test_lint_error_no_sql_provided(profiles_dir, project_dir, sqlfluff_config_path, caplog):
    response = client.post(
        "/lint",
        headers={"X-dbt-Project": "dbt_project"},
    )
    assert response.status_code == 400
    response_json = response.json()
    assert response_json == {
        "error": {
            "code": 6,
            "data": {},
            "message": "No SQL provided. Either provide a SQL file path or a SQL string to lint.",
        }
    }


def test_lint_parse_failure(profiles_dir, project_dir, sqlfluff_config_path, caplog):
    response = client.post(
        "/lint",
        headers={"X-dbt-Project": "dbt_project"},
        data="""select
    {{ dbt_utils.star(ref("base_cases")) }}
from {{ ref("base_cases") }}
li""",
    )
    assert response.status_code == 200
    response_json = response.json()
    assert response_json == {"result": []}
