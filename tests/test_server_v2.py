from fastapi.testclient import TestClient

from dbt_osmosis.core.server_v2 import app

from tests.sqlfluff_templater.fixtures.dbt.templater import profiles_dir, project_dir  # noqa: F401

client = TestClient(app)


def test_lint(profiles_dir, project_dir):
    response = client.post(
            "/register",
            params={
                "project_dir": project_dir,
                "profiles_dir": profiles_dir,
                "target": "dev",
            },
            headers={"X-dbt-Project": "dbt_project"},
        )
    assert response.status_code == 200
    assert response.json() == {'added': 'dbt_project', 'projects': ['dbt_project']}
    response = client.post(
        "/lint",
        headers={"X-dbt-Project": "dbt_project"},
        json={"id": "foobar", "title": "Foo Bar", "description": "The Foo Barters"},
    )
    assert response.status_code == 200
    assert response.json() == {
        "id": "foobar",
        "title": "Foo Bar",
        "description": "The Foo Barters",
    }
