"""Defines the hook endpoints for the dbt templater plugin."""
from sqlfluff.core.plugin import hookimpl

import dbt_osmosis.core.server_v2
from dbt_osmosis.dbt_templater.templater import OsmosisDbtTemplater


@hookimpl
def get_templaters():
    """Get templaters."""

    def create_templater(**kwargs):
        return OsmosisDbtTemplater(
            dbt_project_container=dbt_osmosis.core.server_v2.app.state.dbt_project_container,
            **kwargs
        )

    create_templater.name = OsmosisDbtTemplater.name
    return [create_templater]
