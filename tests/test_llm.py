import pytest
from dbt_osmosis.core.llm import get_llm_client, generate_model_spec_as_json, generate_column_doc, generate_table_doc

@pytest.fixture(scope="module")
def llm_client():
    """
    Fixture to initialize the LLM client and model engine.
    """
    client, model_engine = get_llm_client()
    return client, model_engine

def test_llm_connection(llm_client):
    """
    Test the connection to the LLM client.
    """
    client, model_engine = llm_client
    assert client is not None, "LLM client initialization failed."
    assert model_engine is not None, "Model engine initialization failed."

def test_generate_model_spec_as_json(llm_client):
    """
    Test the generate_model_spec_as_json function.
    """
    client, model_engine = llm_client
    sql_content = """
    SELECT id, email, created_at
    FROM users
    WHERE created_at > '2024-01-01'
    """
    upstream_docs = [
        "id: Unique numeric identifier for the user",
        "email: The user's primary email address",
        "created_at: The datetime when the account was created",
    ]
    existing_context = "User records created after Jan 2024"
    result = generate_model_spec_as_json(sql_content, upstream_docs, existing_context)
    assert isinstance(result, dict), "Result should be a dictionary."
    assert "description" in result, "Result should contain a description."
    assert "columns" in result, "Result should contain columns."

def test_generate_table_doc(llm_client):
    """
    Test the generate_table_doc function.
    """
    client, model_engine = llm_client
    sql_content = """
    SELECT id, email, created_at
    FROM users
    WHERE created_at > '2024-01-01'
    """
    table_name = "recent_users"
    upstream_docs = ["This table tracks recently registered users."]
    result = generate_table_doc(sql_content, table_name, upstream_docs)
    assert isinstance(result, str), "Result should be a string."

def test_generate_column_doc(llm_client):
    """
    Test the generate_column_doc function.
    """
    client, model_engine = llm_client
    column_name = "email"
    existing_context = "This column stores a user's main email address."
    table_name = "users"
    upstream_docs = ["email: the user's registered email"]
    result = generate_column_doc(column_name, existing_context, table_name, upstream_docs)
    assert isinstance(result, str), "Result should be a string."
