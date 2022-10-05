import pytest

import dbt_osmosis.core.server_v2


@pytest.fixture(scope="session", autouse=True)
def register_dbt_project():
    dbt_osmosis.core.server_v2.app.state.dbt_project_container.add_project(
        name_override="dbt_project",
        project_dir="tests/sqlfluff_templater/fixtures/dbt/dbt_project",
        profiles_dir="tests/sqlfluff_templater/fixtures/dbt/profiles_yml",
        target="dev",
    )
