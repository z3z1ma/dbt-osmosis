"""Supplementary module for LLM synthesis of dbt documentation."""

from __future__ import annotations

import json
import os
import re
import typing as t
from textwrap import dedent

import openai
from openai import OpenAI

from dbt_osmosis.core.exceptions import LLMConfigurationError, LLMResponseError

__all__ = [
    "generate_model_spec_as_json",
    "generate_column_doc",
    "generate_table_doc",
    "generate_dbt_model_from_nl",
    "generate_sql_from_nl",
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


def _create_llm_prompt_for_nl_to_sql(
    query: str,
    available_sources: list[dict[str, t.Any]] | None = None,
    schema_context: str | None = None,
) -> list[dict[str, str]]:
    """Builds a system + user prompt for generating SQL from natural language.

    Args:
        query: The natural language query from the user
        available_sources: List of available sources/models with their columns
        schema_context: Additional schema context information

    Returns:
        list[dict[str, str]]: List of prompt messages for the LLM
    """
    if available_sources is None:
        available_sources = []

    sources_info = ""
    if available_sources:
        sources_info = "\nAvailable sources and models:\n"
        for source in available_sources[:20]:  # Limit to prevent token overflow
            name = source.get("name", "unknown")
            source_type = source.get("type", "model")
            columns = source.get("columns", [])
            sources_info += f"  - {name} ({source_type}): {', '.join(columns[:10])}\n"

    system_prompt = dedent(
        f"""
    You are a helpful SQL Developer and an Expert in dbt.
    Your job is to translate natural language queries into valid SQL.

    IMPORTANT RULES:
    1. Use dbt ref() syntax to reference models: {{{{ ref('model_name') }}}}
    2. Use dbt source() syntax to reference sources: {{{{ source('source_name', 'table_name') }}}}
    3. Return ONLY the SQL, no extra commentary or Markdown fences
    4. Use proper SQL syntax compatible with modern data warehouses (Snowflake, BigQuery, Databricks, Postgres, etc.)
    5. Include helpful comments in the SQL to explain the logic
    6. Use CTEs (WITH clauses) for complex queries to improve readability
    7. Handle NULL values appropriately
    8. Use standard date functions (CURRENT_DATE, DATE_TRUNC, etc.)

    {sources_info}

    {schema_context or ""}
    """
    )

    user_message = dedent(
        f"""
    Natural language query:
    {query}

    Return ONLY the SQL code that answers this query.
    """
    )

    return [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": user_message.strip()},
    ]


def _create_llm_prompt_for_nl_to_dbt_model(
    query: str,
    available_sources: list[dict[str, t.Any]] | None = None,
    schema_context: str | None = None,
) -> list[dict[str, str]]:
    """Builds a system + user prompt for generating a complete dbt model from natural language.

    Args:
        query: The natural language query describing the desired model
        available_sources: List of available sources/models with their columns
        schema_context: Additional schema context information

    Returns:
        list[dict[str, str]]: List of prompt messages for the LLM
    """
    if available_sources is None:
        available_sources = []

    sources_info = ""
    if available_sources:
        sources_info = "Available sources and models:\n"
        for source in available_sources[:20]:  # Limit to prevent token overflow
            name = source.get("name", "unknown")
            source_type = source.get("type", "model")
            columns = source.get("columns", [])
            description = source.get("description", "")
            if description:
                sources_info += f"  - {name} ({source_type}): {description}\n"
                sources_info += f"    Columns: {', '.join(columns[:10])}\n"
            else:
                sources_info += f"  - {name} ({source_type}): {', '.join(columns[:10])}\n"

    example_output = dedent("""
    {{
      "model_name": "customer_churn_last_30_days",
      "description": "Identifies customers who have churned in the last 30 days based on inactivity period",
      "sql": "WITH customer_activity AS (\\n    SELECT\\n        customer_id,\\n        MAX(order_date) as last_order_date\\n    FROM {{{{ ref('orders') }}}}\\n    GROUP BY customer_id\\n)\\nSELECT\\n    c.customer_id,\\n    c.email,\\n    c.created_at,\\n    COALESCE(ca.last_order_date, c.created_at) as last_activity\\nFROM {{{{ ref('customers') }}}} c\\nLEFT JOIN customer_activity ca USING (customer_id)\\nWHERE ac.last_activity < CURRENT_DATE - INTERVAL '30 days'",
      "materialized": "table",
      "columns": [
        {{"name": "customer_id", "description": "Unique customer identifier"}},
        {{"name": "email", "description": "Customer email address"}},
        {{"name": "created_at", "description": "Customer account creation date"}},
        {{"name": "last_activity", "description": "Date of last customer activity"}}
      ]
    }}
    """)

    system_prompt = dedent(
        f"""
    You are a helpful SQL Developer and an Expert in dbt.
    Your job is to understand a natural language request and generate a complete dbt model specification.

    IMPORTANT RULES:
    1. Return a valid JSON object with keys: model_name, description, sql, materialized, columns
    2. "model_name" should be snake_case and descriptive
    3. "description" should briefly explain what the model does
    4. "sql" should use dbt ref() and source() syntax appropriately
    5. "materialized" should be one of: table, view, incremental, ephemeral
    6. "columns" is an array with name and description for each column
    7. Use CTEs for complex logic
    8. Include helpful comments in the SQL
    9. DO NOT output extra text, ONLY valid JSON

    {sources_info}

    {schema_context or ""}

    Example of desired JSON structure:
    {example_output}
    """
    )

    user_message = dedent(
        f"""
    Natural language request:
    {query}

    Return ONLY a valid JSON object that matches the structure described above.
    """
    )

    return [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": user_message.strip()},
    ]


def generate_sql_from_nl(
    query: str,
    available_sources: list[dict[str, t.Any]] | None = None,
    schema_context: str | None = None,
    temperature: float = 0.3,
) -> str:
    """Generates SQL (with dbt refs) from a natural language query.

    Args:
        query: The natural language query from the user
        available_sources: Optional list of available sources/models with their columns
        schema_context: Additional schema context information
        temperature: LLM temperature (lower = more deterministic)

    Returns:
        str: The generated SQL code

    Raises:
        LLMResponseError: If the LLM returns an invalid response
    """
    messages = _create_llm_prompt_for_nl_to_sql(query, available_sources, schema_context)

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

    # Clean up markdown fences if present
    content = content.strip()
    if content.startswith("```"):
        # Extract SQL from markdown code blocks
        lines = content.split("\n")
        sql_lines = []
        in_sql = False
        for line in lines:
            if line.startswith("```sql") or line.startswith("```SQL"):
                in_sql = True
                continue
            elif line.startswith("```") and in_sql:
                break
            elif in_sql or not line.startswith("```"):
                sql_lines.append(line)
        content = "\n".join(sql_lines).strip()

    return content


def generate_dbt_model_from_nl(
    query: str,
    available_sources: list[dict[str, t.Any]] | None = None,
    schema_context: str | None = None,
    temperature: float = 0.3,
) -> dict[str, t.Any]:
    """Generates a complete dbt model specification from a natural language query.

    The structure returned is:
      {
        "model_name": "...",
        "description": "...",
        "sql": "...",  # SQL with dbt refs/sources
        "materialized": "table|view|incremental|ephemeral",
        "columns": [
          {"name": "...", "description": "..."},
          ...
        ]
      }

    Args:
        query: The natural language query describing the desired model
        available_sources: Optional list of available sources/models with their columns
        schema_context: Additional schema context information
        temperature: LLM temperature (lower = more deterministic)

    Returns:
        dict[str, t.Any]: A dictionary with the complete model specification

    Raises:
        LLMResponseError: If the LLM returns an invalid response
    """
    messages = _create_llm_prompt_for_nl_to_dbt_model(query, available_sources, schema_context)

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
    if content is None:
        raise LLMResponseError("LLM returned an empty response")

    # Clean up markdown fences if present
    content = content.strip()
    if content.startswith("```"):
        content = content[content.find("{") : content.rfind("}") + 1]

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        raise LLMResponseError("LLM returned invalid JSON:\n" + content)

    # Validate required keys
    required_keys = {"model_name", "description", "sql", "materialized", "columns"}
    missing_keys = required_keys - set(data.keys())
    if missing_keys:
        raise LLMResponseError(
            f"LLM response missing required keys: {missing_keys}\nGot keys: {list(data.keys())}"
        )

    return data


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
