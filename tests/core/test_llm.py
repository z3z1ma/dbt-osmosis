# pyright: reportPrivateImportUsage=false, reportPrivateUsage=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportAny=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportArgumentType=false, reportFunctionMemberAccess=false, reportUnknownVariableType=false, reportUnusedParameter=false

"""Tests for LLM functionality with mocked providers."""

import json
from unittest import mock

import pytest
from openai.types.chat import ChatCompletionMessage

from dbt_osmosis.core.llm import (
    _redact_credentials,
    generate_column_doc,
    generate_model_spec_as_json,
    generate_table_doc,
    get_llm_client,
)


class MockChoice:
    """Mock choice for OpenAI response."""

    def __init__(self, content: str) -> None:
        self.message = ChatCompletionMessage(content=content, role="assistant")


class MockResponse:
    """Mock response for OpenAI API."""

    def __init__(self, content: str) -> None:
        self.choices = [MockChoice(content)]


def test_redact_credentials_empty() -> None:
    """Test that empty string is returned as-is."""
    assert _redact_credentials("") == ""
    assert _redact_credentials(None) is None  # type: ignore[arg-type]


def test_redact_credentials_api_keys() -> None:
    """Test redaction of various API key patterns."""
    # OpenAI key pattern (needs 20+ chars)
    result = _redact_credentials("My key is sk-abc123def456ghi789jkl")
    assert "sk-REDACTED" in result

    # Bearer token (needs 20+ chars)
    result = _redact_credentials("Authorization: Bearer xyz123abc456def789ghi")
    assert "Bearer REDACTED" in result

    # Long alphanumeric strings (potential keys, needs 32+ chars)
    # Using 'x' repeated pattern to make it obviously test data
    result = _redact_credentials("key=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    assert "REDACTED_KEY" in result


def test_get_llm_client_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test OpenAI client creation with required environment variables."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-123")
    monkeypatch.setenv("LLM_PROVIDER", "openai")

    client, model = get_llm_client()

    assert client is not None
    assert model == "gpt-4o"


def test_get_llm_client_openai_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that ValueError is raised when OPENAI_API_KEY is missing."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "openai")

    with pytest.raises(ValueError, match="OPENAI_API_KEY not set"):
        get_llm_client()


def test_get_llm_client_invalid_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that ValueError is raised for invalid provider."""
    monkeypatch.setenv("LLM_PROVIDER", "invalid-provider")

    with pytest.raises(ValueError, match="Invalid LLM provider"):
        get_llm_client()


def test_get_llm_client_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test Ollama client creation with defaults."""
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("OLLAMA_API_KEY", "ollama")

    client, model = get_llm_client()

    assert client is not None
    assert model == "llama2:latest"


def test_get_llm_client_lm_studio(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test LM Studio client creation with defaults."""
    monkeypatch.setenv("LLM_PROVIDER", "lm-studio")
    monkeypatch.setenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
    monkeypatch.setenv("LM_STUDIO_API_KEY", "lm-studio")

    client, model = get_llm_client()

    assert client is not None
    assert model == "local-model"


def test_generate_model_spec_as_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test generating model specification as JSON."""
    # Set up environment
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("LLM_PROVIDER", "openai")

    # Mock the OpenAI client response
    mock_response = MockResponse(
        json.dumps({
            "description": "A test model",
            "columns": [
                {"name": "id", "description": "Unique identifier"},
                {"name": "name", "description": "User name"},
            ],
        })
    )

    with mock.patch(
        "openai.resources.chat.completions.Completions.create", return_value=mock_response
    ):
        result = generate_model_spec_as_json(
            sql_content="SELECT id, name FROM users",
            upstream_docs=["id: unique ID", "name: user name"],
            existing_context="Test context",
        )

        assert result["description"] == "A test model"
        assert len(result["columns"]) == 2
        assert result["columns"][0]["name"] == "id"
        assert result["columns"][0]["description"] == "Unique identifier"


def test_generate_model_spec_as_json_with_markdown_fences(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that JSON is correctly extracted from markdown code blocks."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("LLM_PROVIDER", "openai")

    mock_response = MockResponse(
        '```json\n{\n  "description": "A test model",\n  "columns": []\n}\n```'
    )

    with mock.patch(
        "openai.resources.chat.completions.Completions.create", return_value=mock_response
    ):
        result = generate_model_spec_as_json(sql_content="SELECT * FROM users")

        assert result["description"] == "A test model"


def test_generate_model_spec_as_json_invalid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that ValueError is raised when LLM returns invalid JSON."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("LLM_PROVIDER", "openai")

    mock_response = MockResponse("This is not valid JSON")

    with mock.patch(
        "openai.resources.chat.completions.Completions.create", return_value=mock_response
    ):
        with pytest.raises(ValueError, match="LLM returned invalid JSON"):
            generate_model_spec_as_json(sql_content="SELECT * FROM users")


def test_generate_model_spec_as_json_empty_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that ValueError is raised when LLM returns empty response."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("LLM_PROVIDER", "openai")

    # Mock response with None content to trigger empty response error
    mock_response_with_none = mock.Mock()
    mock_choice = mock.Mock()
    mock_choice.message.content = None
    mock_response_with_none.choices = [mock_choice]

    with mock.patch(
        "openai.resources.chat.completions.Completions.create", return_value=mock_response_with_none
    ):
        with pytest.raises(ValueError, match="LLM returned an empty response"):
            generate_model_spec_as_json(sql_content="SELECT * FROM users")


def test_generate_column_doc(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test generating documentation for a single column."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("LLM_PROVIDER", "openai")

    mock_response = MockResponse("The unique identifier for each user")

    with mock.patch(
        "openai.resources.chat.completions.Completions.create", return_value=mock_response
    ):
        result = generate_column_doc(
            column_name="user_id",
            existing_context="This table contains user information",
            table_name="users",
            upstream_docs=["Unique identifier for each user"],
        )

        assert result == "The unique identifier for each user"


def test_generate_column_doc_empty_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that ValueError is raised when LLM returns empty response for column doc."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("LLM_PROVIDER", "openai")

    mock_response = MockResponse("")

    with mock.patch(
        "openai.resources.chat.completions.Completions.create", return_value=mock_response
    ):
        with pytest.raises(ValueError, match="LLM returned an empty response"):
            generate_column_doc(column_name="test_col")


def test_generate_table_doc(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test generating documentation for a table."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("LLM_PROVIDER", "openai")

    mock_response = MockResponse("A table containing user account information")

    with mock.patch(
        "openai.resources.chat.completions.Completions.create", return_value=mock_response
    ):
        result = generate_table_doc(
            sql_content="SELECT * FROM users",
            table_name="users",
            upstream_docs=["User information table"],
        )

        assert result == "A table containing user account information"


def test_generate_table_doc_empty_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that ValueError is raised when LLM returns empty response for table doc."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("LLM_PROVIDER", "openai")

    mock_response = MockResponse("")

    with mock.patch(
        "openai.resources.chat.completions.Completions.create", return_value=mock_response
    ):
        with pytest.raises(ValueError, match="LLM returned an empty response"):
            generate_table_doc(sql_content="SELECT * FROM users", table_name="users")


def test_sql_truncation_with_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that SQL is truncated when OSMOSIS_LLM_MAX_SQL_CHARS is set."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OSMOSIS_LLM_MAX_SQL_CHARS", "50")

    long_sql = "SELECT * FROM users WHERE " + "x=1 " * 100

    mock_response = MockResponse('{"description": "Test", "columns": []}')

    with mock.patch("openai.resources.chat.completions.Completions.create") as mock_create:
        mock_create.return_value = mock_response
        generate_model_spec_as_json(sql_content=long_sql)

        # Check that the SQL in the prompt was truncated
        call_args = mock_create.call_args
        messages = call_args[1]["messages"]
        user_message = messages[1]["content"]
        assert "(TRUNCATED)" in user_message
