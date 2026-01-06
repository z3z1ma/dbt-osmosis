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
    "generate_column_doc",
    "generate_model_spec_as_json",
    "generate_table_doc",
    "analyze_column_semantics",
    "generate_semantic_description",
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
    """Creates and returns an LLM client and model engine string based on environment variables.

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
                "Azure environment variables (AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT_NAME) not properly set for azure-openai provider",
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
                "GOOGLE_GEMINI_BASE_URL",
                "https://generativelanguage.googleapis.com/v1beta/openai",
            ),
            api_key=os.getenv("GOOGLE_GEMINI_API_KEY"),
        )
        model_engine = os.getenv("GOOGLE_GEMINI_MODEL", "gemini-2.0-flash")

        if not client.api_key:
            raise LLMConfigurationError(
                "GEMINI environment variables GOOGLE_GEMINI_API_KEY not set for google-gemini provider",
            )

    elif provider == "anthropic":
        client = OpenAI(
            base_url=os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1"),
            api_key=os.getenv("ANTHROPIC_API_KEY"),
        )
        model_engine = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-latest")

        if not client.api_key:
            raise LLMConfigurationError(
                "Anthropic environment variables ANTHROPIC_API_KEY not set for anthropic provider",
            )

    else:
        raise LLMConfigurationError(
            f"Invalid LLM provider '{provider}'. Valid options: openai, azure-openai, google-gemini, anthropic, lm-studio, ollama.",
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
            f"ERROR: Missing environment variables for {provider}: {', '.join(missing_vars)}. Please refer to the documentation to set them correctly.",
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
    """,
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
    """,
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
    """,
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
    """Builds a system + user prompt for generating a docstring for a single column.
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
    """,
    )

    user_message = dedent(
        f"""
    The column name is: {column_name}

    Existing context:
    {existing_context or "(none)"}

    Upstream docs:
    {os.linesep.join(upstream_docs)}

    Return ONLY the text suitable for the "description" field.
    """,
    )

    return [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": user_message.strip()},
    ]


def _create_llm_prompt_for_table(
    sql_content: str,
    table_name: str,
    upstream_docs: list[str] | None = None,
) -> list[dict[str, t.Any]]:
    """Builds a system + user prompt instructing the model to produce a string description for a single model/table.

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
    """,
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
    """,
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
        sql_content,
        existing_context,
        upstream_docs,
    )

    client, model_engine = get_llm_client()

    if os.getenv("LLM_PROVIDER", "openai").lower() == "azure-openai":
        # Legacy structure for Azure OpenAI Service
        response = client.ChatCompletion.create(
            engine=model_engine,
            messages=messages,
            temperature=temperature,
        )
    else:
        # New SDK structure for OpenAI default, LM Studio, Ollama
        response = client.chat.completions.create(
            model=model_engine,
            messages=messages,
            temperature=temperature,
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
        column_name,
        existing_context,
        table_name,
        upstream_docs,
    )

    client, model_engine = get_llm_client()

    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    if provider == "azure-openai":
        response = client.ChatCompletion.create(
            engine=model_engine,
            messages=messages,
            temperature=temperature,
        )
    else:
        response = client.chat.completions.create(
            model=model_engine,
            messages=messages,
            temperature=temperature,
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
            engine=model_engine,
            messages=messages,
            temperature=temperature,
        )
    else:
        response = client.chat.completions.create(
            model=model_engine,
            messages=messages,
            temperature=temperature,
        )

    content = response.choices[0].message.content
    if not content:
        raise LLMResponseError("LLM returned an empty response")

    return content.strip()


def _create_llm_prompt_for_semantic_analysis(
    column_name: str,
    data_type: str | None = None,
    table_name: str | None = None,
    model_context: str | None = None,
    upstream_columns: list[dict[str, str]] | None = None,
) -> list[dict[str, t.Any]]:
    """Builds a system + user prompt for semantic analysis of a column.

    The LLM analyzes the column name, data type, and context to infer:
    - Business meaning (e.g., PK, FK, metric, dimension, timestamp)
    - Data relationships (e.g., foreign key to another table)
    - Semantic category (e.g., PII, currency, status, identifier)

    Args:
        column_name: The column name to analyze
        data_type: The column's data type (optional)
        table_name: The table/model name (optional)
        model_context: The model's description or SQL context (optional)
        upstream_columns: List of upstream columns with names and descriptions (optional)

    Returns:
        list[dict[str, t.Any]]: List of prompt messages for the LLM
    """
    if upstream_columns is None:
        upstream_columns = []

    example_json = dedent(
        """\
    {
      "semantic_type": "foreign_key",
      "business_meaning": "References the customer entity",
      "inferred_relationship": "customers.customer_id",
      "description": "Foreign key reference to the customers table",
      "tags": ["pk", "fk", "relationship"],
      "meta": {
        "foreign_key": "customers.customer_id",
        "domain": "customer"
      }
    }
    """
    )

    # Build upstream columns context
    upstream_context = ""
    if upstream_columns:
        upstream_context = "\n    ".join(
            f"- {col['name']}: {col.get('description', 'no description')}"
            for col in upstream_columns[:20]  # Limit to 20 columns
        )

    system_prompt = dedent(
        f"""
    You are an expert data modeler and SQL analyst. Your task is to perform semantic analysis
    on a database column to infer its business meaning and relationships.

    Return ONLY a valid JSON object with this structure:
    {example_json}

    SEMANTIC TYPES to detect:
    - primary_key: Unique identifier for the entity (e.g., id, pk, uuid)
    - foreign_key: Reference to another entity (e.g., customer_id, user_id)
    - metric: Numeric measure or aggregation (e.g., total_amount, count, sum)
    - dimension: Categorical attribute (e.g., status, type, category)
    - timestamp: Date/time column (e.g., created_at, updated_date)
    - pii: Personally identifiable information (e.g., email, ssn, phone)
    - currency: Money/currency values (e.g., price, amount, cost)
    - boolean: True/false flag (e.g., is_active, has_flag, enabled)
    - text: Free-form text (e.g., description, notes, comments)
    - json: Structured data (e.g., metadata, properties, attributes)

    INFERRED RELATIONSHIPS:
    - For columns ending in _id, _key, _fk: infer the referenced table
    - For columns with common prefixes (e.g., customer_): relate to customer entity
    - For timestamp columns: indicate if it's a created/updated/deleted time

    BUSINESS MEANING:
    - Infer what business concept this column represents
    - Consider the column name, data type, and table context
    - Be concise but descriptive

    IMPORTANT RULES:
    1. Return ONLY valid JSON, no markdown fences or extra text
    2. If relationships are unclear, use null for inferred_relationship
    3. Include relevant tags based on semantic analysis
    4. meta should contain structured inferences (foreign_key, domain, format, etc.)
    """
    )

    user_message = dedent(
        f"""
    Column to analyze:
    - Name: {column_name}
    - Data Type: {data_type or "unknown"}
    - Table: {table_name or "unknown"}

    Model context:
    {model_context or "No context provided"}

    Upstream columns (for relationship inference):
    {upstream_context or "No upstream columns provided"}

    Return ONLY a valid JSON object matching the structure above.
    """
    )

    return [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": user_message.strip()},
    ]


def analyze_column_semantics(
    column_name: str,
    data_type: str | None = None,
    table_name: str | None = None,
    model_context: str | None = None,
    upstream_columns: list[dict[str, str]] | None = None,
    temperature: float = 0.3,
) -> dict[str, t.Any]:
    """Analyzes a column semantically to infer business meaning and relationships.

    Uses LLM to analyze the column name, data type, and context to produce:
    - semantic_type: The type of data (pk, fk, metric, dimension, etc.)
    - business_meaning: Business interpretation of the column
    - inferred_relationship: Detected foreign key or entity relationship
    - description: Generated description based on semantic analysis
    - tags: Suggested tags for the column
    - meta: Structured metadata (foreign_key, domain, etc.)

    Args:
        column_name: The column name to analyze
        data_type: The column's data type (optional)
        table_name: The table/model name (optional)
        model_context: The model's description or SQL context (optional)
        upstream_columns: List of upstream columns with names and descriptions (optional)
        temperature: LLM temperature (default 0.3 for more deterministic output)

    Returns:
        dict[str, t.Any]: JSON object with semantic analysis results

    Example:
        >>> result = analyze_column_semantics("customer_id", data_type="INTEGER", table_name="orders")
        >>> print(result["semantic_type"])
        'foreign_key'
        >>> print(result["inferred_relationship"])
        'customers.customer_id'
    """
    messages = _create_llm_prompt_for_semantic_analysis(
        column_name=column_name,
        data_type=data_type,
        table_name=table_name,
        model_context=model_context,
        upstream_columns=upstream_columns,
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

    content = content.strip()
    # Remove markdown fences if present
    if content.startswith("```") and content.endswith("```"):
        content = content[content.find("{") : content.rfind("}") + 1]

    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise LLMResponseError(f"LLM returned invalid JSON: {content}") from e

    return data


def generate_semantic_description(
    column_name: str,
    semantic_analysis: dict[str, t.Any] | None = None,
    table_name: str | None = None,
    upstream_description: str | None = None,
    temperature: float = 0.5,
) -> str:
    """Generates a contextual description based on semantic analysis.

    Creates a natural language description that incorporates the semantic meaning
    of the column, inferred relationships, and business context.

    Args:
        column_name: The column name to describe
        semantic_analysis: Pre-computed semantic analysis (optional, will compute if None)
        table_name: The table/model name (optional)
        upstream_description: Existing upstream description to incorporate (optional)
        temperature: LLM temperature (default 0.5)

    Returns:
        str: A natural language description for the column

    Example:
        >>> desc = generate_semantic_description("customer_id", table_name="orders")
        >>> print(desc)
        'Foreign key reference to the customers table. Identifies the customer associated with this order.'
    """
    # If no semantic analysis provided, compute it
    if semantic_analysis is None:
        semantic_analysis = analyze_column_semantics(
            column_name=column_name,
            table_name=table_name,
            temperature=temperature,
        )

    # Build context from semantic analysis
    context_parts = []
    if semantic_analysis.get("business_meaning"):
        context_parts.append(f"Business meaning: {semantic_analysis['business_meaning']}")
    if semantic_analysis.get("inferred_relationship"):
        context_parts.append(f"Relationship: {semantic_analysis['inferred_relationship']}")
    if semantic_analysis.get("semantic_type"):
        context_parts.append(f"Type: {semantic_analysis['semantic_type']}")

    context_str = ". ".join(context_parts) if context_parts else "No additional context"

    system_prompt = dedent(
        """
    You are a helpful SQL Developer and an Expert in dbt.
    Your job is to produce a concise documentation string for a database column.

    IMPORTANT RULES:
    1. DO NOT output extra commentary or Markdown fences.
    2. Provide only the column description text, nothing else.
    3. Incorporate the semantic analysis and upstream description if provided.
    4. Keep it concise (1-2 sentences).
    5. Focus on business meaning and relationships.
    """
    )

    user_message = dedent(
        f"""
    Column: {column_name}
    Table: {table_name or "unknown"}

    Semantic analysis:
    {context_str}

    Upstream description:
    {upstream_description or "None provided"}

    Generate a concise description that incorporates the semantic meaning and relationships.
    Return ONLY the description text.
    """
    )

    messages = [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": user_message.strip()},
    ]

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
