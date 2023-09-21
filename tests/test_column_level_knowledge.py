from dbt_osmosis.core.column_level_knowledge import get_prior_knowledge


class TestDbtYamlManager:
    def test_get_prior_knowledge(test):
        knowledge = {
            "myColumn": {
                "progenitor": "source.my_model.source.Order",
                "generation": "generation_0",
                "name": "my_column",
            },
            "my_column": {
                "progenitor": "model.my_model.mart.Order",
                "generation": "generation_0",
                "name": "my_column",
            },
        }
        assert (
            get_prior_knowledge(knowledge, "my_column")["progenitor"]
            == "source.my_model.source.Order"
        )

    def test_get_prior_knowledge_with_camel_case(test):
        knowledge = {
            "myColumn": {
                "progenitor": "model.my_model.dwh.Order",
                "generation": "generation_1",
                "name": "myColumn",
            },
            "my_column": {
                "progenitor": "model.my_model.mart.Order",
                "generation": "generation_0",
                "name": "my_column",
            },
        }
        assert (
            get_prior_knowledge(knowledge, "my_column")["progenitor"] == "model.my_model.dwh.Order"
        )
