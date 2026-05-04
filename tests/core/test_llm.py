# pyright: reportPrivateImportUsage=false, reportPrivateUsage=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportAny=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportArgumentType=false, reportFunctionMemberAccess=false, reportUnknownVariableType=false, reportUnusedParameter=false

"""Tests for LLM functionality with mocked providers."""

import json
from unittest import mock

import pytest

from dbt_osmosis.core.exceptions import LLMConfigurationError, LLMResponseError
from dbt_osmosis.core.llm import (
    _call_with_retry,
    _redact_credentials,
    generate_column_doc,
    generate_model_spec_as_json,
    generate_table_doc,
    get_llm_client,
)


class FakeBaseURL:
    """Minimal stand-in for OpenAI's URL object."""

    def __init__(self, url: str) -> None:
        self.url = url if url.endswith("/") else f"{url}/"

    def __str__(self) -> str:
        return self.url


class FakeOpenAI:
    """Minimal OpenAI-compatible client used by provider-selection tests."""

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.api_key = kwargs.get("api_key")
        self.base_url = FakeBaseURL(kwargs.get("base_url", "https://api.openai.com/v1"))
        self.chat = mock.Mock()


class FakeAzureOpenAI(FakeOpenAI):
    """Minimal Azure OpenAI-compatible client used by provider-selection tests."""


@pytest.fixture(autouse=True)
def _mock_openai_sdk(monkeypatch: pytest.MonkeyPatch):
    """Keep LLM tests mock-only so the base env need not install OpenAI."""
    openai_client = mock.Mock(side_effect=FakeOpenAI)
    azure_openai_client = mock.Mock(side_effect=FakeAzureOpenAI)
    monkeypatch.setattr("dbt_osmosis.core.llm._OPENAI_AVAILABLE", True)
    monkeypatch.setattr("dbt_osmosis.core.llm.OpenAI", openai_client)
    monkeypatch.setattr("dbt_osmosis.core.llm.AzureOpenAI", azure_openai_client)
    return openai_client, azure_openai_client


def _make_mock_response(headers: dict[str, str] | None = None) -> mock.Mock:
    """Create a mock httpx.Response for use with OpenAI exceptions."""
    mock_response = mock.Mock()
    mock_response.headers = headers or {}
    mock_response.request = mock.Mock()  # Add request attribute required by RateLimitError
    return mock_response


class MockChoice:
    """Mock choice for OpenAI response."""

    def __init__(self, content: str) -> None:
        self.message = mock.Mock(content=content, role="assistant")


class MockResponse:
    """Mock response for OpenAI API."""

    def __init__(self, content: str) -> None:
        self.choices = [MockChoice(content)]


def _make_mock_llm_client(response: mock.Mock | MockResponse) -> mock.Mock:
    """Create a mocked chat-completions client without importing OpenAI."""
    mock_client = mock.Mock()
    mock_client.chat.completions.create.return_value = response
    return mock_client


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


def test_get_llm_client_defaults_to_openai_when_provider_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provider should default to OpenAI when LLM_PROVIDER is not set."""
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")

    with mock.patch("dbt_osmosis.core.llm.OpenAI") as openai_client:
        client, model = get_llm_client()

    openai_client.assert_called_once_with(api_key="test-openai-key")
    assert client is openai_client.return_value
    assert model == "gpt-4o"


def test_get_llm_client_missing_openai_dependency_is_configuration_error() -> None:
    """Missing OpenAI SDK should produce a core configuration error with a real extra hint."""
    with (
        mock.patch("dbt_osmosis.core.llm._OPENAI_AVAILABLE", False),
        pytest.raises(LLMConfigurationError, match=r"dbt-osmosis\[openai\]"),
    ):
        get_llm_client()


def test_get_llm_client_openai_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that LLMConfigurationError is raised when OPENAI_API_KEY is missing."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "openai")

    with pytest.raises(LLMConfigurationError, match="OPENAI_API_KEY not set"):
        get_llm_client()


def test_get_llm_client_invalid_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that LLMConfigurationError is raised for invalid provider."""
    monkeypatch.setenv("LLM_PROVIDER", "invalid-provider")

    with pytest.raises(LLMConfigurationError, match="Invalid LLM provider"):
        get_llm_client()


def test_get_llm_client_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test Ollama client creation with defaults."""
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)

    client, model = get_llm_client()

    assert client is not None
    assert str(client.base_url) == "http://localhost:11434/v1/"
    assert client.api_key == "ollama"
    assert model == "llama2:latest"


def test_get_llm_client_lm_studio(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test LM Studio client creation with defaults."""
    monkeypatch.setenv("LLM_PROVIDER", "lm-studio")
    monkeypatch.delenv("LM_STUDIO_BASE_URL", raising=False)
    monkeypatch.delenv("LM_STUDIO_API_KEY", raising=False)
    monkeypatch.delenv("LM_STUDIO_MODEL", raising=False)

    client, model = get_llm_client()

    assert client is not None
    assert str(client.base_url) == "http://localhost:1234/v1/"
    assert client.api_key == "lm-studio"
    assert model == "local-model"


def test_get_llm_client_google_gemini_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test Google Gemini client creation with documented optional defaults."""
    monkeypatch.setenv("LLM_PROVIDER", "google-gemini")
    monkeypatch.setenv("GOOGLE_GEMINI_API_KEY", "test-key")
    monkeypatch.delenv("GOOGLE_GEMINI_BASE_URL", raising=False)
    monkeypatch.delenv("GOOGLE_GEMINI_MODEL", raising=False)

    client, model = get_llm_client()

    assert client is not None
    assert str(client.base_url) == "https://generativelanguage.googleapis.com/v1beta/openai/"
    assert client.api_key == "test-key"
    assert model == "gemini-2.0-flash"


def test_get_llm_client_anthropic_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test Anthropic client creation with documented optional defaults."""
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)

    client, model = get_llm_client()

    assert client is not None
    assert str(client.base_url) == "https://api.anthropic.com/v1/"
    assert client.api_key == "test-key"
    assert model == "claude-3-5-haiku-latest"


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
        }),
    )

    mock_client = _make_mock_llm_client(mock_response)

    with mock.patch(
        "dbt_osmosis.core.llm.get_llm_client",
        return_value=(mock_client, "gpt-4o"),
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
        '```json\n{\n  "description": "A test model",\n  "columns": []\n}\n```',
    )

    mock_client = _make_mock_llm_client(mock_response)

    with mock.patch(
        "dbt_osmosis.core.llm.get_llm_client",
        return_value=(mock_client, "gpt-4o"),
    ):
        result = generate_model_spec_as_json(sql_content="SELECT * FROM users")

        assert result["description"] == "A test model"


def test_generate_model_spec_as_json_invalid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that LLMResponseError is raised when LLM returns invalid JSON."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("LLM_PROVIDER", "openai")

    mock_response = MockResponse("This is not valid JSON")

    with (
        mock.patch(
            "dbt_osmosis.core.llm.get_llm_client",
            return_value=(_make_mock_llm_client(mock_response), "gpt-4o"),
        ),
        pytest.raises(LLMResponseError, match="LLM returned invalid JSON"),
    ):
        generate_model_spec_as_json(sql_content="SELECT * FROM users")


def test_generate_model_spec_as_json_empty_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that LLMResponseError is raised when LLM returns empty response."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("LLM_PROVIDER", "openai")

    # Mock response with None content to trigger empty response error
    mock_response_with_none = mock.Mock()
    mock_choice = mock.Mock()
    mock_choice.message.content = None
    mock_response_with_none.choices = [mock_choice]

    with (
        mock.patch(
            "dbt_osmosis.core.llm.get_llm_client",
            return_value=(_make_mock_llm_client(mock_response_with_none), "gpt-4o"),
        ),
        pytest.raises(LLMResponseError, match="LLM returned an empty response"),
    ):
        generate_model_spec_as_json(sql_content="SELECT * FROM users")


def test_generate_column_doc(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test generating documentation for a single column."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("LLM_PROVIDER", "openai")

    mock_response = MockResponse("The unique identifier for each user")

    mock_client = _make_mock_llm_client(mock_response)

    with mock.patch(
        "dbt_osmosis.core.llm.get_llm_client",
        return_value=(mock_client, "gpt-4o"),
    ):
        result = generate_column_doc(
            column_name="user_id",
            existing_context="This table contains user information",
            table_name="users",
            upstream_docs=["Unique identifier for each user"],
        )

        assert result == "The unique identifier for each user"


def test_generate_column_doc_empty_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that LLMResponseError is raised when LLM returns empty response for column doc."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("LLM_PROVIDER", "openai")

    mock_response = MockResponse("")

    with (
        mock.patch(
            "dbt_osmosis.core.llm.get_llm_client",
            return_value=(_make_mock_llm_client(mock_response), "gpt-4o"),
        ),
        pytest.raises(LLMResponseError, match="LLM returned an empty response"),
    ):
        generate_column_doc(column_name="test_col")


def test_generate_table_doc(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test generating documentation for a table."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("LLM_PROVIDER", "openai")

    mock_response = MockResponse("A table containing user account information")

    mock_client = _make_mock_llm_client(mock_response)

    with mock.patch(
        "dbt_osmosis.core.llm.get_llm_client",
        return_value=(mock_client, "gpt-4o"),
    ):
        result = generate_table_doc(
            sql_content="SELECT * FROM users",
            table_name="users",
            upstream_docs=["User information table"],
        )

        assert result == "A table containing user account information"


def test_generate_table_doc_empty_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that LLMResponseError is raised when LLM returns empty response for table doc."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("LLM_PROVIDER", "openai")

    mock_response = MockResponse("")

    with (
        mock.patch(
            "dbt_osmosis.core.llm.get_llm_client",
            return_value=(_make_mock_llm_client(mock_response), "gpt-4o"),
        ),
        pytest.raises(LLMResponseError, match="LLM returned an empty response"),
    ):
        generate_table_doc(sql_content="SELECT * FROM users", table_name="users")


def test_sql_truncation_with_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that SQL is truncated when OSMOSIS_LLM_MAX_SQL_CHARS is set."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OSMOSIS_LLM_MAX_SQL_CHARS", "50")

    long_sql = "SELECT * FROM users WHERE " + "x=1 " * 100

    mock_response = MockResponse('{"description": "Test", "columns": []}')

    mock_client = _make_mock_llm_client(mock_response)

    with mock.patch(
        "dbt_osmosis.core.llm.get_llm_client",
        return_value=(mock_client, "gpt-4o"),
    ):
        generate_model_spec_as_json(sql_content=long_sql)

        # Check that the SQL in the prompt was truncated
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args[1]["messages"]
        user_message = messages[1]["content"]
        assert "(TRUNCATED)" in user_message


# ============================================================================
# Azure OpenAI and Retry Logic Tests
# ============================================================================


def test_get_llm_client_azure_with_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test Azure OpenAI client creation with API key authentication."""
    monkeypatch.setenv("LLM_PROVIDER", "azure-openai")
    monkeypatch.setenv("AZURE_OPENAI_BASE_URL", "https://test.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key-123")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

    client, model = get_llm_client()

    assert client is not None
    assert model == "gpt-4"
    # Verify it's an AzureOpenAI client
    assert isinstance(client, FakeAzureOpenAI)


def test_get_llm_client_azure_missing_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that LLMConfigurationError is raised when Azure endpoint is missing."""
    monkeypatch.setenv("LLM_PROVIDER", "azure-openai")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("AZURE_OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4")

    with pytest.raises(
        LLMConfigurationError,
        match="AZURE_OPENAI_BASE_URL and AZURE_OPENAI_DEPLOYMENT_NAME must be set",
    ):
        get_llm_client()


def test_get_llm_client_azure_missing_deployment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that LLMConfigurationError is raised when deployment name is missing."""
    monkeypatch.setenv("LLM_PROVIDER", "azure-openai")
    monkeypatch.setenv("AZURE_OPENAI_BASE_URL", "https://test.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("AZURE_OPENAI_DEPLOYMENT_NAME", raising=False)

    with pytest.raises(
        LLMConfigurationError,
        match="AZURE_OPENAI_BASE_URL and AZURE_OPENAI_DEPLOYMENT_NAME must be set",
    ):
        get_llm_client()


def test_get_llm_client_azure_missing_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that LLMConfigurationError is raised when API key is missing."""
    monkeypatch.setenv("LLM_PROVIDER", "azure-openai")
    monkeypatch.setenv("AZURE_OPENAI_BASE_URL", "https://test.openai.azure.com")
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4")

    with pytest.raises(
        LLMConfigurationError,
        match="AZURE_OPENAI_API_KEY must be set for azure-openai provider",
    ):
        get_llm_client()


def test_get_llm_client_azure_ad_with_service_principal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test Azure OpenAI AD with service principal authentication."""
    pytest.importorskip("azure.identity", reason="azure-identity not installed")
    monkeypatch.setenv("LLM_PROVIDER", "azure-openai-ad")
    monkeypatch.setenv("AZURE_OPENAI_BASE_URL", "https://test.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4")
    monkeypatch.setenv("AZURE_OPENAI_AD_TOKEN_SCOPE", "https://cognitiveservices.azure.com")
    monkeypatch.setenv("AZURE_TENANT_ID", "test-tenant-id")
    monkeypatch.setenv("AZURE_CLIENT_ID", "test-client-id")
    monkeypatch.setenv("AZURE_CLIENT_SECRET", "test-client-secret")

    # Mock the credential to return a token
    mock_token = mock.Mock()
    mock_token.token = "test-access-token-123"

    with mock.patch(
        "dbt_osmosis.core.llm.EnvironmentCredential.get_token",
        return_value=mock_token,
    ):
        client, model = get_llm_client()

        assert client is not None
        assert model == "gpt-4"
        assert isinstance(client, FakeAzureOpenAI)


def test_get_llm_client_azure_ad_uses_azure_client_token_auth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Azure AD should construct AzureOpenAI with token auth and an API version."""
    monkeypatch.setenv("LLM_PROVIDER", "azure-openai-ad")
    monkeypatch.setenv("AZURE_OPENAI_BASE_URL", "https://test.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    monkeypatch.setenv("AZURE_OPENAI_AD_TOKEN_SCOPE", "https://cognitiveservices.azure.com")
    monkeypatch.setenv("AZURE_TENANT_ID", "test-tenant-id")
    monkeypatch.setenv("AZURE_CLIENT_ID", "test-client-id")
    monkeypatch.setenv("AZURE_CLIENT_SECRET", "test-client-secret")

    mock_token = mock.Mock(token="test-access-token-123")
    mock_environment_credential = mock.Mock()
    mock_environment_credential.return_value.get_token.return_value = mock_token

    with (
        mock.patch("dbt_osmosis.core.llm._AZURE_IDENTITY_AVAILABLE", True),
        mock.patch("dbt_osmosis.core.llm.EnvironmentCredential", mock_environment_credential),
        mock.patch("dbt_osmosis.core.llm.DefaultAzureCredential", mock.Mock()),
        mock.patch("dbt_osmosis.core.llm.AzureOpenAI") as azure_openai,
    ):
        client, model = get_llm_client()

    mock_environment_credential.return_value.get_token.assert_called_once_with(
        "https://cognitiveservices.azure.com/.default"
    )
    azure_openai.assert_called_once_with(
        azure_endpoint="https://test.openai.azure.com",
        azure_ad_token="test-access-token-123",
        api_version="2024-02-15-preview",
    )
    assert client is azure_openai.return_value
    assert model == "gpt-4"


def test_get_llm_client_azure_ad_with_default_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test Azure OpenAI AD with DefaultAzureCredential (e.g., Azure CLI auth)."""
    pytest.importorskip("azure.identity", reason="azure-identity not installed")
    monkeypatch.setenv("LLM_PROVIDER", "azure-openai-ad")
    monkeypatch.setenv("AZURE_OPENAI_BASE_URL", "https://test.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4")
    monkeypatch.setenv("AZURE_OPENAI_AD_TOKEN_SCOPE", "https://cognitiveservices.azure.com")
    # No service principal env vars, should fall back to DefaultAzureCredential
    monkeypatch.delenv("AZURE_TENANT_ID", raising=False)
    monkeypatch.delenv("AZURE_CLIENT_ID", raising=False)
    monkeypatch.delenv("AZURE_CLIENT_SECRET", raising=False)

    # Mock the credential to return a token
    mock_token = mock.Mock()
    mock_token.token = "test-access-token-from-cli"

    captured_scope = None

    def capture_scope(scope: str):
        nonlocal captured_scope
        captured_scope = scope
        return mock_token

    with mock.patch(
        "dbt_osmosis.core.llm.DefaultAzureCredential.get_token",
        side_effect=capture_scope,
    ):
        client, model = get_llm_client()

        assert client is not None
        assert model == "gpt-4"
        assert captured_scope == "https://cognitiveservices.azure.com/.default"


def test_get_llm_client_azure_ad_without_azure_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that LLMConfigurationError is raised when azure-identity is not installed."""
    monkeypatch.setenv("LLM_PROVIDER", "azure-openai-ad")
    monkeypatch.setenv("AZURE_OPENAI_BASE_URL", "https://test.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4")
    monkeypatch.setenv("AZURE_OPENAI_AD_TOKEN_SCOPE", "https://cognitiveservices.azure.com")

    # Mock _AZURE_IDENTITY_AVAILABLE to False
    with mock.patch("dbt_osmosis.core.llm._AZURE_IDENTITY_AVAILABLE", False):
        with pytest.raises(
            LLMConfigurationError,
            match="Azure Identity library is not installed",
        ):
            get_llm_client()


def test_get_llm_client_azure_ad_token_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test error handling when Azure AD token acquisition fails."""
    pytest.importorskip("azure.identity", reason="azure-identity not installed")
    monkeypatch.setenv("LLM_PROVIDER", "azure-openai-ad")
    monkeypatch.setenv("AZURE_OPENAI_BASE_URL", "https://test.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4")
    monkeypatch.setenv("AZURE_OPENAI_AD_TOKEN_SCOPE", "https://cognitiveservices.azure.com")
    monkeypatch.delenv("AZURE_TENANT_ID", raising=False)

    # Mock credential to raise an exception
    with mock.patch(
        "dbt_osmosis.core.llm.DefaultAzureCredential.get_token",
        side_effect=Exception("Authentication failed"),
    ):
        with pytest.raises(
            LLMConfigurationError,
            match="Failed to acquire Azure AD token",
        ):
            get_llm_client()


def test_get_llm_client_azure_ad_scope_with_default_suffix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that /.default suffix is added to scope if not present."""
    try:
        pytest.importorskip("azure.identity")
    except Exception:
        pytest.skip("azure-identity not installed")

    monkeypatch.setenv("LLM_PROVIDER", "azure-openai-ad")
    monkeypatch.setenv("AZURE_OPENAI_BASE_URL", "https://test.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4")
    # Scope without /.default suffix
    monkeypatch.setenv("AZURE_OPENAI_AD_TOKEN_SCOPE", "https://cognitiveservices.azure.com")
    monkeypatch.setenv("AZURE_TENANT_ID", "test-tenant")
    monkeypatch.setenv("AZURE_CLIENT_ID", "test-client")
    monkeypatch.setenv("AZURE_CLIENT_SECRET", "test-secret")

    mock_token = mock.Mock()
    mock_token.token = "test-token"

    captured_scope = None

    def capture_scope(scope: str):
        nonlocal captured_scope
        captured_scope = scope
        return mock_token

    with mock.patch(
        "dbt_osmosis.core.llm.EnvironmentCredential.get_token",
        side_effect=capture_scope,
    ):
        get_llm_client()

    # Verify /.default was appended
    assert captured_scope == "https://cognitiveservices.azure.com/.default"


def test_get_llm_client_azure_ad_preserves_explicit_scoped_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that already-scoped gateway values are passed through unchanged."""
    try:
        pytest.importorskip("azure.identity")
    except Exception:
        pytest.skip("azure-identity not installed")

    monkeypatch.setenv("LLM_PROVIDER", "azure-openai-ad")
    monkeypatch.setenv("AZURE_OPENAI_BASE_URL", "https://test.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4")
    monkeypatch.setenv("AZURE_OPENAI_AD_TOKEN_SCOPE", "api://gateway-app/access_as_user")
    monkeypatch.delenv("AZURE_TENANT_ID", raising=False)
    monkeypatch.delenv("AZURE_CLIENT_ID", raising=False)
    monkeypatch.delenv("AZURE_CLIENT_SECRET", raising=False)

    mock_token = mock.Mock()
    mock_token.token = "test-token"

    captured_scope = None

    def capture_scope(scope: str):
        nonlocal captured_scope
        captured_scope = scope
        return mock_token

    with mock.patch(
        "dbt_osmosis.core.llm.DefaultAzureCredential.get_token",
        side_effect=capture_scope,
    ):
        get_llm_client()

    assert captured_scope == "api://gateway-app/access_as_user"


def test_get_llm_client_azure_ad_missing_token_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that LLMConfigurationError is raised when AD token scope is missing."""
    monkeypatch.setenv("LLM_PROVIDER", "azure-openai-ad")
    monkeypatch.setenv("AZURE_OPENAI_BASE_URL", "https://test.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4")
    monkeypatch.delenv("AZURE_OPENAI_AD_TOKEN_SCOPE", raising=False)

    with pytest.raises(
        LLMConfigurationError,
        match="AZURE_OPENAI_AD_TOKEN_SCOPE must be set for azure-openai-ad provider",
    ):
        get_llm_client()


def test_get_llm_client_azure_ad_missing_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that LLMConfigurationError is raised when Azure AD endpoint is missing."""
    monkeypatch.setenv("LLM_PROVIDER", "azure-openai-ad")
    monkeypatch.delenv("AZURE_OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4")
    monkeypatch.setenv("AZURE_OPENAI_AD_TOKEN_SCOPE", "https://cognitiveservices.azure.com")

    with pytest.raises(
        LLMConfigurationError,
        match="AZURE_OPENAI_BASE_URL and AZURE_OPENAI_DEPLOYMENT_NAME must be set for azure-openai-ad provider",
    ):
        get_llm_client()


def test_call_with_retry_success_on_first_attempt() -> None:
    """Test that function succeeds on first attempt without retry."""
    mock_func = mock.Mock(return_value="success")

    result = _call_with_retry(mock_func)

    assert result == "success"
    assert mock_func.call_count == 1


def test_call_with_retry_success_after_rate_limit() -> None:
    """Test successful retry after rate limit error."""
    # Import here to avoid issues if openai not installed
    try:
        import openai
    except ImportError:
        pytest.skip("openai not installed")

    mock_func = mock.Mock()
    # First call raises RateLimitError, second succeeds
    mock_func.side_effect = [
        openai.RateLimitError("Rate limit exceeded", response=_make_mock_response(), body=None),
        "success",
    ]

    # Mock time.sleep to avoid actual delays in tests
    with mock.patch("time.sleep"):
        result = _call_with_retry(mock_func, max_retries=5, initial_delay=0.1)

    assert result == "success"
    assert mock_func.call_count == 2


def test_call_with_retry_max_retries_exceeded() -> None:
    """Test that exception is raised when max retries exceeded."""
    try:
        import openai
    except ImportError:
        pytest.skip("openai not installed")

    mock_func = mock.Mock()
    # Always raise RateLimitError
    mock_func.side_effect = openai.RateLimitError(
        "Rate limit exceeded",
        response=_make_mock_response(),
        body=None,
    )

    with mock.patch("time.sleep"):
        with pytest.raises(openai.RateLimitError):
            _call_with_retry(mock_func, max_retries=2, initial_delay=0.1)

    # Should try initial + 2 retries = 3 times
    assert mock_func.call_count == 3


def test_call_with_retry_exponential_backoff() -> None:
    """Test that exponential backoff is applied correctly."""
    try:
        import openai
    except ImportError:
        pytest.skip("openai not installed")

    mock_func = mock.Mock()
    mock_func.side_effect = [
        openai.RateLimitError("Rate limit", response=_make_mock_response(), body=None),
        openai.RateLimitError("Rate limit", response=_make_mock_response(), body=None),
        "success",
    ]

    sleep_times = []

    def track_sleep(seconds: float) -> None:
        sleep_times.append(seconds)

    with mock.patch("time.sleep", side_effect=track_sleep):
        result = _call_with_retry(mock_func, max_retries=5, initial_delay=1.0)

    assert result == "success"
    # First retry: 1.0s, second retry: 2.0s (exponential backoff)
    assert len(sleep_times) == 2
    assert sleep_times[0] == 1.0
    assert sleep_times[1] == 2.0


def test_call_with_retry_respects_retry_after_header() -> None:
    """Test that retry logic respects Retry-After header if present."""
    try:
        import openai
    except ImportError:
        pytest.skip("openai not installed")

    # Create mock response with retry-after header
    mock_response = _make_mock_response(headers={"retry-after": "5.0"})

    mock_func = mock.Mock()
    error = openai.RateLimitError("Rate limit", response=mock_response, body=None)
    mock_func.side_effect = [error, "success"]

    sleep_times = []

    def track_sleep(seconds: float) -> None:
        sleep_times.append(seconds)

    with mock.patch("time.sleep", side_effect=track_sleep):
        result = _call_with_retry(mock_func, max_retries=5, initial_delay=1.0)

    assert result == "success"
    # Should use retry-after value instead of exponential backoff
    assert sleep_times[0] == 5.0


def test_call_with_retry_non_rate_limit_error() -> None:
    """Test that non-rate-limit errors are raised immediately without retry."""
    mock_func = mock.Mock()
    mock_func.side_effect = ValueError("Some other error")

    with pytest.raises(ValueError, match="Some other error"):
        _call_with_retry(mock_func, max_retries=5)

    # Should only be called once, not retried
    assert mock_func.call_count == 1
