import pytest

import dbt_osmosis.core.server_v2


@pytest.fixture(scope="session", autouse=True)
def register_dbt_project():
    for name_override, project_dir in [
        ("dbt_project", "tests/sqlfluff_templater/fixtures/dbt/dbt_project"),
        # If this is enabled, some tests fail when node compilation fails down
        # inside dbt. Don't know why. The sequence matters: if this is first,
        # the failures do not occur.
        # ("dbt_project2", "tests/sqlfluff_templater/fixtures/dbt/dbt_project2"),
    ]:
        dbt_osmosis.core.server_v2.app.state.dbt_project_container.add_project(
            name_override=name_override,
            project_dir=project_dir,
            profiles_dir="tests/sqlfluff_templater/fixtures/dbt/profiles_yml",
            target="dev",
        )
