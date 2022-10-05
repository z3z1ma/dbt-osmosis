"""Fixtures for dbt templating tests."""

import pytest
from sqlfluff.core import FluffConfig


DBT_FLUFF_CONFIG = {
    "core": {
        "templater": "dbt",
        "dialect": "postgres",
    },
    "templater": {
        "dbt": {
            "profiles_dir": (
                "tests/sqlfluff_templater/fixtures/dbt/profiles_yml"
            ),
            "project_dir": (
                "tests/sqlfluff_templater/fixtures/dbt/dbt_project"
            ),
        },
    },
}


@pytest.fixture()
def profiles_dir():
    """Returns the dbt profile directory."""
    return DBT_FLUFF_CONFIG["templater"]["dbt"]["profiles_dir"]


@pytest.fixture()
def project_dir():
    """Returns the dbt project directory."""
    return DBT_FLUFF_CONFIG["templater"]["dbt"]["project_dir"]


@pytest.fixture()
def sqlfluff_config_path():
    return "tests/sqlfluff_templater/fixtures/dbt/.sqlfluff"


@pytest.fixture()
def dbt_templater():
    """Returns an instance of the DbtTemplater."""
    return FluffConfig(overrides={"dialect": "ansi"}).get_templater("dbt")
