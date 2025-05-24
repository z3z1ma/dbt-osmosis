import json
from dbt_osmosis.core.llm import (
    generate_model_spec_as_json,
    generate_column_doc,
    generate_table_doc,
)

# Sample SQL input for a model
sample_sql = """
SELECT id, email, created_at
FROM users
WHERE created_at > '2024-01-01'
"""


# Simplified test of each function (model, table, column)
def run_tests():
    try:
        print("✅ Testing generate_model_spec_as_json():")
        result = generate_model_spec_as_json(
            sql_content=sample_sql,
            upstream_docs=[
                "id: Unique numeric identifier for the user",
                "email: The user's primary email address",
                "created_at: The datetime when the account was created",
            ],
            existing_context="User records created after Jan 2024",
        )
        print(json.dumps(result, indent=2))

        print("\n✅ Testing generate_table_doc():")
        table_doc = generate_table_doc(
            sql_content=sample_sql,
            table_name="recent_users",
            upstream_docs=["This table tracks recently registered users."],
        )
        print(table_doc)

        print("\n✅ Testing generate_column_doc():")
        column_doc = generate_column_doc(
            column_name="email",
            existing_context="This column stores a user's main email address.",
            table_name="users",
            upstream_docs=["email: the user's registered email"],
        )
        print(column_doc)

    except Exception as e:
        print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    run_tests()
