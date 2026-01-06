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
    "analyze_column_semantics",
    "generate_column_doc",
    "generate_dbt_model_from_nl",
    "generate_model_spec_as_json",
    "generate_semantic_description",
    "generate_sql_from_nl",
    "generate_table_doc",
    "generate_staging_model_spec",
    "generate_staging_sql",
    "ColumnTransformation",
    "StagingModelSpec",
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
# AI-Powered Staging Model Generation


@dataclass
class ColumnTransformation:
    """Represents a column transformation for staging models.

    Attributes:
        original_name: The original column name in the source
        new_name: The new column name after transformation (e.g., id -> customer_id)
        expression: SQL expression for the transformation (e.g., "amount / 100")
        description: Documentation for the transformed column
    """

    original_name: str
    new_name: str
    expression: str | None = None
    description: str = ""

    def to_sql_select(self) -> str:
        """Generate the SELECT clause for this transformation.

        Returns:
            SQL SELECT expression for this column
        """
        if self.expression:
            return f"    {self.expression} as {self.new_name}"
        elif self.original_name != self.new_name:
            return f"    {self.original_name} as {self.new_name}"
        else:
            return f"    {self.original_name}"


@dataclass
class StagingModelSpec:
    """Specification for an AI-generated staging model.

    Attributes:
        source_name: Name of the source table or model
        staging_name: Name for the staging model (e.g., stg_customers)
        description: Description of what the staging model does
        columns: List of column transformations
        materialization: Suggested materialization (view or table)
    """

    source_name: str
    staging_name: str
    description: str
    columns: list[ColumnTransformation]
    materialization: str = "view"
    source_type: str = "source"  # 'source', 'seed', or 'model'

    def to_sql(self) -> str:
        """Generate the complete staging SQL file content.

        Returns:
            Complete SQL content for the staging model
        """
        column_selects = ",\n".join(col.to_sql_select() for col in self.columns)

        source_ref = (
            f"{{{{ source('{self.source_name.split('.')[0]}', '{self.source_name.split('.', 1)[1] if '.' in self.source_name else self.source_name}') }}}}"
            if self.source_type == "source"
            else f"{{{{ ref('{self.source_name}') }}}}"
        )

        return f"""{{{{ config(materialized='{self.materialization}') }}}}

with source as (

    select * from {source_ref}

),

renamed as (

    select
{column_selects}

    from source

)

select * from renamed
"""


def _create_staging_spec_prompt(
    source_name: str,
    columns: list[dict[str, t.Any]],
    table_description: str = "",
    source_type: str = "source",
) -> list[dict[str, str]]:
    """Builds a system + user prompt for generating staging model specifications.

    Args:
        source_name: Name of the source table
        columns: List of column definitions with name, data_type, and optional description
        table_description: Optional description of the source table
        source_type: Type of source ('source', 'seed', or 'model')

    Returns:
        List of prompt messages for the LLM
    """
    columns_text = "\n".join(
        f"      - {col.get('name')}: {col.get('data_type', 'unknown')}"
        + (f" - {col.get('description', '')}" if col.get("description") else "")
        for col in columns
    )

    system_prompt = """You are a helpful SQL Developer and an Expert in dbt.

Your task is to generate a specification for a staging model that transforms a source table into a clean, documented staging layer.

KEY PRINCIPLES FOR STAGING MODELS:
1. **Renaming**: Rename columns to be more descriptive (e.g., `id` -> `customer_id`, `user_id` -> `customer_id`)
2. **Type Casting**: Apply appropriate type casting for data integrity (e.g., string to date, numeric precision)
3. **Data Cleaning**: Basic cleaning like trimming strings, handling null values, removing duplicates
4. **Standardization**: Consistent naming conventions (lowercase with underscores)
5. **Documentation**: Clear descriptions for each transformation

COMMON TRANSFORMATION PATTERNS:
- `id` -> `<entity>_id` (prefix with entity name for clarity)
- `user_id` -> `customer_id` (rename to match business domain)
- `amount` stored in cents -> `amount / 100.0 as amount` (convert units)
- String dates -> `cast(date_col as date)` (type casting)
- `created_at` / `updated_at` -> keep as-is (standard timestamp columns)
- Phone numbers -> `trim(phone_number) as phone_number` (clean whitespace)

OUTPUT FORMAT:
Return ONLY valid JSON matching this exact structure:
{
    "staging_name": "stg_<table_name>",
    "description": "Brief description of what this staging model does",
    "columns": [
        {
            "original_name": "original_column_name",
            "new_name": "transformed_column_name",
            "expression": "sql_expression or null",
            "description": "What this column represents and why the transformation"
        }
    ],
    "materialization": "view"
}

RULES:
1. DO NOT write extra explanation or Markdown fences
2. `expression` should be null if just renaming (use original_name as new_name pattern)
3. Only include SQL expressions for actual transformations (type casting, calculations, etc.)
4. Provide descriptions that explain WHY the transformation is needed
5. Use "view" for materialization unless table is explicitly needed
"""

    user_message = f"""Generate a staging model specification for the following source:

**Source Name:** {source_name}
**Source Type:** {source_type}
**Description:** {table_description or "(no description provided)"}

**Columns:**
    {columns_text}

Analyze the column names and data types to infer appropriate transformations. Return ONLY valid JSON matching the specified structure.
"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]


def _create_staging_sql_refinement_prompt(
    initial_spec: StagingModelSpec,
    feedback: str = "",
) -> list[dict[str, str]]:
    """Builds a prompt for refining a staging model specification.

    Args:
        initial_spec: The initial staging model specification
        feedback: Optional feedback for improvements

    Returns:
        List of prompt messages for the LLM
    """
    system_prompt = """You are a helpful SQL Developer and an Expert in dbt.

Your task is to refine a staging model specification based on feedback. Return ONLY valid JSON matching the same structure as the input."""

    columns_text = "\n".join(
        f"  - {col.original_name} -> {col.new_name}"
        + (f" ({col.expression})" if col.expression else "")
        for col in initial_spec.columns
    )

    user_message = f"""Refine the following staging model specification:

**Current Spec:**
- Source: {initial_spec.source_name}
- Staging: {initial_spec.staging_name}
- Description: {initial_spec.description}
- Columns:
{columns_text}

**Feedback:**
{feedback or "Make any improvements you see fit for better data quality and clarity."}

Return ONLY valid JSON with the refined specification.
"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]


def generate_staging_model_spec(
    source_name: str,
    columns: list[dict[str, t.Any]],
    table_description: str = "",
    source_type: str = "source",
    temperature: float = 0.3,
) -> StagingModelSpec:
    """Generate a staging model specification using AI.

    Analyzes the source table schema and infers appropriate transformations
    for column renaming, type casting, and basic data cleaning.

    Args:
        source_name: Name of the source table
        columns: List of column definitions with name, data_type, and optional description
        table_description: Optional description of the source table
        source_type: Type of source ('source', 'seed', or 'model')
        temperature: LLM temperature for generation

    Returns:
        StagingModelSpec with AI-generated transformations

    Raises:
        LLMConfigurationError: If LLM client configuration is invalid
        LLMResponseError: If LLM returns invalid or empty response
    """
    messages = _create_staging_spec_prompt(
        source_name=source_name,
        columns=columns,
        table_description=table_description,
        source_type=source_type,
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
    if content.startswith("```"):
        # Extract JSON from markdown code block
        content = content[content.find("{") : content.rfind("}") + 1]

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        raise LLMResponseError("LLM returned invalid JSON:\n" + content)

    # Convert JSON to StagingModelSpec
    column_transforms = []
    for col_data in data.get("columns", []):
        column_transforms.append(
            ColumnTransformation(
                original_name=col_data.get("original_name", col_data.get("name", "")),
                new_name=col_data.get("new_name", col_data.get("name", "")),
                expression=col_data.get("expression"),
                description=col_data.get("description", ""),
            )
        )

    return StagingModelSpec(
        source_name=source_name,
        staging_name=data.get("staging_name", f"stg_{source_name}"),
        description=data.get("description", f"Staging model for {source_name}"),
        columns=column_transforms,
        materialization=data.get("materialization", "view"),
        source_type=source_type,
    )


def generate_staging_sql(
    source_name: str,
    columns: list[dict[str, t.Any]],
    table_description: str = "",
    source_type: str = "source",
    temperature: float = 0.3,
) -> tuple[str, StagingModelSpec]:
    """Generate complete staging SQL file content using AI.

    This is a convenience function that generates both the specification
    and the complete SQL file content in one call.

    Args:
        source_name: Name of the source table
        columns: List of column definitions with name, data_type, and optional description
        table_description: Optional description of the source table
        source_type: Type of source ('source', 'seed', or 'model')
        temperature: LLM temperature for generation

    Returns:
        Tuple of (sql_content, staging_spec) where sql_content is the complete
        SQL file content and staging_spec contains the transformation metadata
    """
    spec = generate_staging_model_spec(
        source_name=source_name,
        columns=columns,
        table_description=table_description,
        source_type=source_type,
        temperature=temperature,
    )
    return spec.to_sql(), spec


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


# AI Documentation Co-Pilot: Style-Aware Generation

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
