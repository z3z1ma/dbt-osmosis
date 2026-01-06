"""Supplementary module for LLM synthesis of dbt documentation."""

from __future__ import annotations

import json
import os
import re
import typing as t
from dataclasses import dataclass
from textwrap import dedent

import openai
from openai import OpenAI

from dbt_osmosis.core.exceptions import LLMConfigurationError, LLMResponseError

__all__ = [
    "generate_model_spec_as_json",
    "generate_column_doc",
    "generate_table_doc",
    "generate_style_aware_column_doc",
    "generate_style_aware_table_doc",
    "suggest_documentation_improvements",
    "DocumentationSuggestion",
]


def _redact_credentials(text: str) -> str:
    """Redact potential API keys and credentials from text before logging.

    This function prevents sensitive credentials from being logged in error messages
    or debug output. It matches common patterns for API keys, tokens, and secrets.

    Args:
        text: The text to sanitize

    Returns:
        The text with credentials redacted
    """
    if not text:
        return text

    # Redact common API key patterns (sk-, Bearer, etc.)
    patterns = [
        (r"(sk-[a-zA-Z0-9]{20,})", "sk-REDACTED"),
        (r"(Bearer\s+[a-zA-Z0-9._-]{20,})", "Bearer REDACTED"),
        (r"([a-zA-Z0-9_-]{32,})", "REDACTED_KEY"),  # Long alphanumeric strings
    ]

    redacted = text
    for pattern, replacement in patterns:
        redacted = re.sub(pattern, replacement, redacted)

    return redacted


# Dynamic client creation function
def get_llm_client():
    """
    Creates and returns an LLM client and model engine string based on environment variables.

    Returns:
        tuple: (client, model_engine) where client is an OpenAI or openai object, and model_engine is the model name.
    Raises:
        LLMConfigurationError: If required environment variables are missing or provider is invalid.
    """
    provider = os.getenv("LLM_PROVIDER", "openai").lower()

    if provider == "openai":
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise LLMConfigurationError("OPENAI_API_KEY not set for OpenAI provider")
        client = OpenAI(api_key=openai_api_key)
        model_engine = os.getenv("OPENAI_MODEL", "gpt-4o")

    elif provider == "azure-openai":
        openai.api_type = "azure-openai"
        openai.api_base = os.getenv("AZURE_OPENAI_BASE_URL")
        openai.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
        openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        model_engine = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

        if not (openai.api_base and openai.api_key and model_engine):
            raise LLMConfigurationError(
                "Azure environment variables (AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT_NAME) not properly set for azure-openai provider"
            )
        # For Azure, the global openai object is used directly (legacy SDK structure preferred)
        return openai, model_engine

    elif provider == "lm-studio":
        client = OpenAI(
            base_url=os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1"),
            api_key=os.getenv("LM_STUDIO_API_KEY", "lm-studio"),
        )
        model_engine = os.getenv("LM_STUDIO_MODEL", "local-model")

    elif provider == "ollama":
        client = OpenAI(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            api_key=os.getenv("OLLAMA_API_KEY", "ollama"),
        )
        model_engine = os.getenv("OLLAMA_MODEL", "llama2:latest")

    elif provider == "google-gemini":
        client = OpenAI(
            base_url=os.getenv(
                "GOOGLE_GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai"
            ),
            api_key=os.getenv("GOOGLE_GEMINI_API_KEY"),
        )
        model_engine = os.getenv("GOOGLE_GEMINI_MODEL", "gemini-2.0-flash")

        if not client.api_key:
            raise LLMConfigurationError(
                "GEMINI environment variables GOOGLE_GEMINI_API_KEY not set for google-gemini provider"
            )

    elif provider == "anthropic":
        client = OpenAI(
            base_url=os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1"),
            api_key=os.getenv("ANTHROPIC_API_KEY"),
        )
        model_engine = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-latest")

        if not client.api_key:
            raise LLMConfigurationError(
                "Anthropic environment variables ANTHROPIC_API_KEY not set for anthropic provider"
            )

    else:
        raise LLMConfigurationError(
            f"Invalid LLM provider '{provider}'. Valid options: openai, azure-openai, google-gemini, anthropic, lm-studio, ollama."
        )

    # Define required environment variables for each provider
    required_env_vars = {
        "openai": ["OPENAI_API_KEY"],
        "azure-openai": [
            "AZURE_OPENAI_BASE_URL",
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_DEPLOYMENT_NAME",
        ],
        "lm-studio": ["LM_STUDIO_BASE_URL", "LM_STUDIO_API_KEY"],
        "ollama": ["OLLAMA_BASE_URL", "OLLAMA_API_KEY"],
        "google-gemini": ["GOOGLE_GEMINI_API_KEY"],
        "anthropic": ["ANTHROPIC_API_KEY"],
    }

    # Check for missing environment variables
    missing_vars = [var for var in required_env_vars[provider] if not os.getenv(var)]
    if missing_vars:
        raise LLMConfigurationError(
            f"ERROR: Missing environment variables for {provider}: {', '.join(missing_vars)}. Please refer to the documentation to set them correctly."
        )

    return client, model_engine


def _create_llm_prompt_for_model_docs_as_json(
    sql_content: str,
    existing_context: str | None = None,
    upstream_docs: list[str] | None = None,
) -> list[dict[str, t.Any]]:
    """Builds a system + user prompt instructing the model to produce a JSON structure describing the entire model (including columns)."""
    if upstream_docs is None:
        upstream_docs = []

    example_json = dedent(
        """\
    {
      "description": "A short description for the model",
      "columns": [
        {
          "name": "id",
          "description": "Unique identifier for each record",
        },
        {
          "name": "email",
          "description": "User email address",
        }
      ]
    }
    """
    )

    system_prompt = dedent(
        f"""
    You are a helpful SQL Developer and an Expert in dbt.
    You must produce a JSON object that documents a single model and its columns.
    The object must match the structure shown below.
    DO NOT WRITE EXTRA EXPLANATION OR MARKDOWN FENCES, ONLY VALID JSON.

    Example of desired JSON structure:
    {example_json}

    IMPORTANT RULES:
    1. "description" should be short and gleaned from the SQL or the provided docs if possible.
    2. "columns" is an array of objects. Each object MUST contain:
       - "name": the column name
       - "description": short explanation of what the column is
    3. If you have "upstream_docs", you may incorporate them as you see fit, but do NOT invent details.
    4. Do not output any extra text besides valid JSON.
    """
    )

    if max_sql_chars := os.getenv("OSMOSIS_LLM_MAX_SQL_CHARS"):
        if len(sql_content) > int(max_sql_chars):
            sql_content = sql_content[: int(max_sql_chars)] + "... (TRUNCATED)"

    user_message = dedent(
        f"""
    The SQL for the model is:

    >>> SQL CODE START
    {sql_content}
    >>> SQL CODE END

    The context for the model is:
    {existing_context or "(none)"}

    The upstream documentation is:
    {os.linesep.join(upstream_docs)}

    Please return only a valid JSON that matches the structure described above.
    """
    )

    return [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": user_message.strip()},
    ]


def _create_llm_prompt_for_column(
    column_name: str,
    existing_context: str | None = None,
    table_name: str | None = None,
    upstream_docs: list[str] | None = None,
) -> list[dict[str, str]]:
    """
    Builds a system + user prompt for generating a docstring for a single column.
    The final answer should be just the docstring text, not JSON or YAML.

    Args:
        column_name (str): The name of the column to describe.
        existing_context (str | None): Any relevant metadata or table definitions.
        table_name (str | None): Name of the table/model (optional).
        upstream_docs (list[str] | None): Optional docs or references you might have.

    Returns:
        list[dict[str, str]]: List of prompt messages for the LLM.
    """
    if upstream_docs is None:
        upstream_docs = []

    table_context = f"in the table '{table_name}'." if table_name else "."

    system_prompt = dedent(
        f"""
    You are a helpful SQL Developer and an Expert in dbt.
    Your job is to produce a concise documentation string
    for a single column {table_context}

    IMPORTANT RULES:
    1. DO NOT output extra commentary or Markdown fences.
    2. Provide only the column description text, nothing else.
    3. If upstream docs exist, you may incorporate them. If none exist,
       a short placeholder is acceptable.
    4. Avoid speculation. Keep it short and relevant.
    """
    )

    user_message = dedent(
        f"""
    The column name is: {column_name}

    Existing context:
    {existing_context or "(none)"}

    Upstream docs:
    {os.linesep.join(upstream_docs)}

    Return ONLY the text suitable for the "description" field.
    """
    )

    return [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": user_message.strip()},
    ]


def _create_llm_prompt_for_table(
    sql_content: str, table_name: str, upstream_docs: list[str] | None = None
) -> list[dict[str, t.Any]]:
    """
    Builds a system + user prompt instructing the model to produce a string description for a single model/table.

    Args:
        sql_content (str): The SQL code for the table.
        table_name (str): Name of the table/model.
        upstream_docs (list[str] | None): Optional docs or references you might have.

    Returns:
        list[dict[str, t.Any]]: List of prompt messages for the LLM.
    """
    if upstream_docs is None:
        upstream_docs = []

    system_prompt = dedent(
        f"""
    You are a helpful SQL Developer and an Expert in dbt.
    Your job is to produce a concise documentation string
    for a table named {table_name}.

    IMPORTANT RULES:
    1. DO NOT output extra commentary or Markdown fences.
    2. Provide only the column description text, nothing else.
    3. If upstream docs exist, you may incorporate them. If none exist,
       a short placeholder is acceptable.
    4. Avoid speculation. Keep it short and relevant.
    5. DO NOT list out the columns. Only provide a high-level description.
    """
    )

    if max_sql_chars := os.getenv("OSMOSIS_LLM_MAX_SQL_CHARS"):
        if len(sql_content) > int(max_sql_chars):
            sql_content = sql_content[: int(max_sql_chars)] + "... (TRUNCATED)"

    user_message = dedent(
        f"""
    The SQL for the model is:

    >>> SQL CODE START
    {sql_content}
    >>> SQL CODE END

    The upstream documentation is:
    {os.linesep.join(upstream_docs)}

    Please return only the text suitable for the "description" field.
    """
    )

    return [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": user_message.strip()},
    ]


def generate_model_spec_as_json(
    sql_content: str,
    upstream_docs: list[str] | None = None,
    existing_context: str | None = None,
    temperature: float = 0.3,
) -> dict[str, t.Any]:
    """Calls the LLM client to generate a JSON specification for a model's metadata and columns.

    The structure is:
      {
        "description": "...",
        "columns": [
          {"name": "...", "description": "..."},
          ...
        ]
      }

    Args:
        sql_content (str): Full SQL code of the model
        upstream_docs (list[str] | None): Optional list of strings containing context or upstream docs
        model_engine (str): Which OpenAI model to use (e.g., 'gpt-3.5-turbo', 'gpt-4')
        temperature (float): OpenAI completion temperature

    Returns:
        dict[str, t.Any]: A dictionary with keys "description", "columns".
    """
    messages = _create_llm_prompt_for_model_docs_as_json(
        sql_content, existing_context, upstream_docs
    )

    client, model_engine = get_llm_client()

    if os.getenv("LLM_PROVIDER", "openai").lower() == "azure-openai":
        # Legacy structure for Azure OpenAI Service
        response = client.ChatCompletion.create(
            engine=model_engine, messages=messages, temperature=temperature
        )
    else:
        # New SDK structure for OpenAI default, LM Studio, Ollama
        response = client.chat.completions.create(
            model=model_engine, messages=messages, temperature=temperature
        )

    content = response.choices[0].message.content
    if content is None:
        raise LLMResponseError("LLM returned an empty response")

    content = content.strip()
    if content.startswith("```") and content.endswith("```"):
        content = content[content.find("{") : content.rfind("}") + 1]
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        raise LLMResponseError("LLM returned invalid JSON:\n" + content)

    return data


def generate_column_doc(
    column_name: str,
    existing_context: str | None = None,
    table_name: str | None = None,
    upstream_docs: list[str] | None = None,
    temperature: float = 0.7,
) -> str:
    """Calls the LLM client to generate documentation for a single column in a table.

    Args:
        column_name (str): The name of the column to describe
        existing_context (str | None): Any relevant metadata or table definitions
        table_name (str | None): Name of the table/model (optional)
        upstream_docs (list[str] | None): Optional docs or references you might have
        model_engine (str): The OpenAI model to use (e.g., 'gpt-3.5-turbo')
        temperature (float): OpenAI completion temperature

    Returns:
        str: A short docstring suitable for a "description" field
    """
    messages = _create_llm_prompt_for_column(
        column_name, existing_context, table_name, upstream_docs
    )

    client, model_engine = get_llm_client()

    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    if provider == "azure-openai":
        response = client.ChatCompletion.create(
            engine=model_engine, messages=messages, temperature=temperature
        )
    else:
        response = client.chat.completions.create(
            model=model_engine, messages=messages, temperature=temperature
        )

    content = response.choices[0].message.content
    if not content:
        raise LLMResponseError("LLM returned an empty response")

    return content.strip()


def generate_table_doc(
    sql_content: str,
    table_name: str,
    upstream_docs: list[str] | None = None,
    temperature: float = 0.7,
) -> str:
    """Calls the LLM client to generate documentation for a single column in a table.

    Args:
        sql_content (str): The SQL code for the table
        table_name (str | None): Name of the table/model (optional)
        upstream_docs (list[str] | None): Optional docs or references you might have
        model_engine (str): The OpenAI model to use (e.g., 'gpt-3.5-turbo')
        temperature (float): OpenAI completion temperature

    Returns:
        str: A short docstring suitable for a "description" field
    """
    messages = _create_llm_prompt_for_table(sql_content, table_name, upstream_docs)

    client, model_engine = get_llm_client()

    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    if provider == "azure-openai":
        response = client.ChatCompletion.create(
            engine=model_engine, messages=messages, temperature=temperature
        )
    else:
        response = client.chat.completions.create(
            model=model_engine, messages=messages, temperature=temperature
        )

    content = response.choices[0].message.content
    if not content:
        raise LLMResponseError("LLM returned an empty response")

    return content.strip()


if __name__ == "__main__":
    # Kitchen sink
    sample_sql = """
        SELECT
            user_id,
            email,
            created_at,
            is_active
        FROM some_source_table
        WHERE created_at > '2021-01-01'
    """
    docs = [
        "user_id: unique integer ID for each user",
        "email: user email address",
        "created_at: record creation time",
        "is_active: boolean flag indicating active user",
    ]
    model_spec = generate_model_spec_as_json(
        sql_content=sample_sql,
        upstream_docs=docs,
        temperature=0.3,
    )

    print("\n=== Generated Model JSON Spec ===")
    print(json.dumps(model_spec, indent=2))

    col_doc = generate_column_doc(
        column_name="email",
        existing_context="This table tracks basic user information.",
        table_name="user_activity_model",
        upstream_docs=["Stores the user's primary email address."],
        temperature=0.2,
    )
    print("\n=== Single Column Documentation ===")
    print(f"Column: email => {col_doc}")


# =============================================================================
# AI Documentation Co-Pilot: Style-Aware Generation
# =============================================================================

if t.TYPE_CHECKING:
    from dbt_osmosis.core.voice_learning import ProjectStyleProfile


@dataclass
class DocumentationSuggestion:
    """A documentation suggestion with confidence score.

    Attributes:
        text: The suggested documentation text
        confidence: Confidence score from 0.0 to 1.0
        reason: Explanation for why this suggestion was made
        source: Source of the suggestion (e.g., "llm", "inheritance")
    """

    text: str
    confidence: float
    reason: str
    source: str = "llm"

    def __post_init__(self) -> None:
        """Validate confidence score is in valid range."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")


def _create_style_aware_prompt_for_column(
    column_name: str,
    existing_context: str | None = None,
    table_name: str | None = None,
    upstream_docs: list[str] | None = None,
    style_profile: ProjectStyleProfile | None = None,
    style_examples: list[str] | None = None,
    current_description: str | None = None,
) -> list[dict[str, str]]:
    """Builds a style-aware prompt for generating column documentation.

    Args:
        column_name: The name of the column to describe
        existing_context: Any relevant metadata or table definitions
        table_name: Name of the table/model (optional)
        upstream_docs: Optional docs or references
        style_profile: Project style profile for voice learning
        style_examples: Specific style examples to follow
        current_description: Current description to improve upon

    Returns:
        List of prompt messages for the LLM
    """
    if upstream_docs is None:
        upstream_docs = []

    table_context = f"in the table '{table_name}'" if table_name else ""

    # Build style guidance section
    style_guidance = ""
    if style_profile:
        style_guidance = f"\n{style_profile.to_prompt_context(max_examples=3)}"
    elif style_examples:
        style_guidance = "\n# Follow these style examples:\n" + "\n".join(style_examples[:3])

    # Build task description
    if current_description:
        task = f"IMPROVE the existing description for column '{column_name}'"
    else:
        task = f"WRITE a description for column '{column_name}'"

    system_prompt = dedent(
        f"""
    You are a helpful SQL Developer and an Expert in dbt.
    Your job is to {task}{table_context}.
    {style_guidance}

    IMPORTANT RULES:
    1. DO NOT output extra commentary or Markdown fences.
    2. Provide only the column description text, nothing else.
    3. Match the style and voice of the provided examples.
    4. Use consistent terminology with the project patterns.
    5. Keep descriptions concise but informative.
    6. If improving existing text, preserve key technical details.
    """
    )

    user_message_sections = [
        f"The column name is: {column_name}",
    ]

    if current_description:
        user_message_sections.append(f"\nCurrent description to improve:\n{current_description}")

    if existing_context:
        user_message_sections.append(f"\nExisting context:\n{existing_context}")

    if upstream_docs:
        user_message_sections.append(f"\nUpstream docs:\n{os.linesep.join(upstream_docs)}")

    user_message_sections.append("\nReturn ONLY the text suitable for the 'description' field.")

    return [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": "\n".join(user_message_sections).strip()},
    ]


def _create_style_aware_prompt_for_table(
    sql_content: str,
    table_name: str,
    upstream_docs: list[str] | None = None,
    style_profile: ProjectStyleProfile | None = None,
    style_examples: list[str] | None = None,
    current_description: str | None = None,
) -> list[dict[str, t.Any]]:
    """Builds a style-aware prompt for generating table documentation.

    Args:
        sql_content: The SQL code for the table
        table_name: Name of the table/model
        upstream_docs: Optional docs or references
        style_profile: Project style profile for voice learning
        style_examples: Specific style examples to follow
        current_description: Current description to improve upon

    Returns:
        List of prompt messages for the LLM
    """
    if upstream_docs is None:
        upstream_docs = []

    # Build style guidance section
    style_guidance = ""
    if style_profile:
        style_guidance = f"\n{style_profile.to_prompt_context(max_examples=3)}"
    elif style_examples:
        style_guidance = "\n# Follow these style examples:\n" + "\n".join(style_examples[:3])

    # Build task description
    if current_description:
        task = f"IMPROVE the existing description for table '{table_name}'"
    else:
        task = f"WRITE a description for table '{table_name}'"

    system_prompt = dedent(
        f"""
    You are a helpful SQL Developer and an Expert in dbt.
    Your job is to {task}.
    {style_guidance}

    IMPORTANT RULES:
    1. DO NOT output extra commentary or Markdown fences.
    2. Provide only the description text, nothing else.
    3. Match the style and voice of the provided examples.
    4. DO NOT list out the columns. Only provide a high-level description.
    5. Keep descriptions concise but informative.
    """
    )

    if max_sql_chars := os.getenv("OSMOSIS_LLM_MAX_SQL_CHARS"):
        if len(sql_content) > int(max_sql_chars):
            sql_content = sql_content[: int(max_sql_chars)] + "... (TRUNCATED)"

    user_message_sections = [f"The table name is: {table_name}"]

    if current_description:
        user_message_sections.append(f"\nCurrent description to improve:\n{current_description}")

    user_message_sections.append(
        f"""
The SQL for the model is:

>>> SQL CODE START
{sql_content}
>>> SQL CODE END
"""
    )

    if upstream_docs:
        user_message_sections.append(
            f"\nThe upstream documentation is:\n{os.linesep.join(upstream_docs)}"
        )

    user_message_sections.append(
        "\nPlease return only the text suitable for the 'description' field."
    )

    return [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": "\n".join(user_message_sections).strip()},
    ]


def generate_style_aware_column_doc(
    column_name: str,
    existing_context: str | None = None,
    table_name: str | None = None,
    upstream_docs: list[str] | None = None,
    temperature: float = 0.5,
    style_profile: ProjectStyleProfile | None = None,
    style_examples: list[str] | None = None,
    current_description: str | None = None,
) -> str:
    """Generate documentation for a column using style-aware prompts.

    Args:
        column_name: The name of the column to describe
        existing_context: Any relevant metadata or table definitions
        table_name: Name of the table/model (optional)
        upstream_docs: Optional docs or references
        temperature: OpenAI completion temperature
        style_profile: Project style profile for voice learning
        style_examples: Specific style examples to follow
        current_description: Current description to improve upon

    Returns:
        A short docstring suitable for a "description" field
    """
    messages = _create_style_aware_prompt_for_column(
        column_name=column_name,
        existing_context=existing_context,
        table_name=table_name,
        upstream_docs=upstream_docs,
        style_profile=style_profile,
        style_examples=style_examples,
        current_description=current_description,
    )

    client, model_engine = get_llm_client()

    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    if provider == "azure-openai":
        response = client.ChatCompletion.create(
            engine=model_engine, messages=messages, temperature=temperature
        )
    else:
        response = client.chat.completions.create(
            model=model_engine, messages=messages, temperature=temperature
        )

    content = response.choices[0].message.content
    if not content:
        raise LLMResponseError("LLM returned an empty response")

    return content.strip()


def generate_style_aware_table_doc(
    sql_content: str,
    table_name: str,
    upstream_docs: list[str] | None = None,
    temperature: float = 0.5,
    style_profile: ProjectStyleProfile | None = None,
    style_examples: list[str] | None = None,
    current_description: str | None = None,
) -> str:
    """Generate documentation for a table using style-aware prompts.

    Args:
        sql_content: The SQL code for the table
        table_name: Name of the table/model
        upstream_docs: Optional docs or references
        temperature: OpenAI completion temperature
        style_profile: Project style profile for voice learning
        style_examples: Specific style examples to follow
        current_description: Current description to improve upon

    Returns:
        A short docstring suitable for a "description" field
    """
    messages = _create_style_aware_prompt_for_table(
        sql_content=sql_content,
        table_name=table_name,
        upstream_docs=upstream_docs,
        style_profile=style_profile,
        style_examples=style_examples,
        current_description=current_description,
    )

    client, model_engine = get_llm_client()

    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    if provider == "azure-openai":
        response = client.ChatCompletion.create(
            engine=model_engine, messages=messages, temperature=temperature
        )
    else:
        response = client.chat.completions.create(
            model=model_engine, messages=messages, temperature=temperature
        )

    content = response.choices[0].message.content
    if not content:
        raise LLMResponseError("LLM returned an empty response")

    return content.strip()


def suggest_documentation_improvements(
    target: t.Literal["column", "table"],
    current_description: str | None,
    column_name: str | None = None,
    table_name: str | None = None,
    sql_content: str | None = None,
    existing_context: str | None = None,
    upstream_docs: list[str] | None = None,
    style_profile: ProjectStyleProfile | None = None,
    style_examples: list[str] | None = None,
    temperature: float = 0.5,
) -> DocumentationSuggestion:
    """Suggest an improved documentation with confidence scoring.

    This function generates an AI-powered documentation suggestion and
    provides a confidence score based on factors like:
    - Whether there's an existing description
    - Quality of style information available
    - Amount of context available

    Args:
        target: Type of documentation ("column" or "table")
        current_description: Current description to improve (or None if missing)
        column_name: Name of the column (for column targets)
        table_name: Name of the table
        sql_content: SQL code (for table targets)
        existing_context: Additional context about the target
        upstream_docs: Documentation from upstream dependencies
        style_profile: Project style profile for voice learning
        style_examples: Specific style examples to follow
        temperature: LLM temperature

    Returns:
        DocumentationSuggestion with text, confidence, and reasoning
    """
    if upstream_docs is None:
        upstream_docs = []

    # Calculate base confidence
    confidence = 0.5

    # Boost confidence if we have style information
    if style_profile or style_examples:
        confidence += 0.2

    # Boost confidence if we have upstream documentation
    if upstream_docs and any(d.strip() for d in upstream_docs):
        confidence += 0.15

    # Boost confidence if we have SQL context (for tables)
    if target == "table" and sql_content:
        confidence += 0.1

    # Adjust confidence based on current state
    has_current = bool(current_description and current_description.strip())

    # Generate the suggestion
    if target == "column":
        if not column_name:
            raise ValueError("column_name is required for column targets")

        suggestion_text = generate_style_aware_column_doc(
            column_name=column_name,
            existing_context=existing_context,
            table_name=table_name,
            upstream_docs=upstream_docs,
            temperature=temperature,
            style_profile=style_profile,
            style_examples=style_examples,
            current_description=current_description,
        )

        # Higher confidence for improvements vs new docs
        if has_current:
            confidence += 0.05
            reason = f"Improving existing description for column '{column_name}'"
        else:
            reason = f"Generating new description for undocumented column '{column_name}'"

    elif target == "table":
        if not table_name:
            raise ValueError("table_name is required for table targets")
        if not sql_content:
            raise ValueError("sql_content is required for table targets")

        suggestion_text = generate_style_aware_table_doc(
            sql_content=sql_content,
            table_name=table_name,
            upstream_docs=upstream_docs,
            temperature=temperature,
            style_profile=style_profile,
            style_examples=style_examples,
            current_description=current_description,
        )

        if has_current:
            confidence += 0.05
            reason = f"Improving existing description for table '{table_name}'"
        else:
            reason = f"Generating new description for undocumented table '{table_name}'"

    else:
        raise ValueError(f"Invalid target: {target}. Must be 'column' or 'table'")

    # Clamp confidence to [0, 1]
    confidence = max(0.0, min(1.0, confidence))

    return DocumentationSuggestion(
        text=suggestion_text,
        confidence=confidence,
        reason=reason,
        source="llm-style-aware",
    )
