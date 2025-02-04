# pyright: reportPrivateUsage=false
import typing as t

import pytest

from dbt_osmosis.core.osmosis import (
    DbtConfiguration,
    YamlRefactorContext,
    YamlRefactorSettings,
    _build_column_knowledge_graph,
    create_dbt_project_context,
)


@pytest.fixture(scope="module")
def yaml_context() -> YamlRefactorContext:
    c = DbtConfiguration(project_dir="demo_duckdb", profiles_dir="demo_duckdb")
    c.vars = {"dbt-osmosis": {}}
    project = create_dbt_project_context(c)
    context = YamlRefactorContext(
        project, settings=YamlRefactorSettings(add_progenitor_to_meta=True, dry_run=True)
    )
    return context


class TestDbtYamlManager:
    def test_get_prior_knowledge(self, yaml_context: YamlRefactorContext):
        # Progenitor gives us an idea of where the inherited traits will come from
        knowledge: dict[str, t.Any] = {
            "customer_id": {
                "meta": {"osmosis_progenitor": "model.jaffle_shop_duckdb.stg_customers.v1"},
                "name": "customer_id",
                "data_type": "INTEGER",
                "constraints": [],
                "description": "This is a unique identifier for a customer",
            },
            "first_name": {
                "meta": {"osmosis_progenitor": "seed.jaffle_shop_duckdb.raw_customers"},
                "name": "first_name",
                "data_type": "VARCHAR",
                "constraints": [],
                "description": "Customer's first name. PII.",
            },
            "last_name": {
                "meta": {"osmosis_progenitor": "seed.jaffle_shop_duckdb.raw_customers"},
                "name": "last_name",
                "data_type": "VARCHAR",
                "constraints": [],
                "description": "Customer's last name. PII.",
            },
            "first_order": {
                "meta": {"osmosis_progenitor": "model.jaffle_shop_duckdb.customers"},
                "name": "first_order",
                "description": "Date (UTC) of a customer's first order",
                "data_type": "DATE",
                "constraints": [],
            },
            "most_recent_order": {
                "meta": {"osmosis_progenitor": "model.jaffle_shop_duckdb.customers"},
                "name": "most_recent_order",
                "description": "Date (UTC) of a customer's most recent order",
                "data_type": "DATE",
                "constraints": [],
            },
            "number_of_orders": {
                "meta": {"osmosis_progenitor": "model.jaffle_shop_duckdb.customers"},
                "name": "number_of_orders",
                "description": "Count of the number of orders a customer has placed",
                "data_type": "BIGINT",
                "constraints": [],
            },
            "customer_lifetime_value": {
                "meta": {"osmosis_progenitor": "model.jaffle_shop_duckdb.customers"},
                "name": "customer_lifetime_value",
                "data_type": "DOUBLE",
                "constraints": [],
            },
            "customer_average_value": {
                "meta": {"osmosis_progenitor": "model.jaffle_shop_duckdb.customers"},
                "name": "customer_average_value",
                "data_type": "DECIMAL(18,3)",
                "constraints": [],
            },
        }
        assert (
            _build_column_knowledge_graph(
                yaml_context,
                yaml_context.project.manifest.nodes["model.jaffle_shop_duckdb.customers"],
            )
            == knowledge
        )
