"""Supplementary module for LLM synthesis of dbt documentation."""

import json
import os
import typing as t
from textwrap import dedent
from openai import OpenAI
import openai

__all__ = [
    "generate_model_spec_as_json",
    "generate_column_doc",
    "generate_table_doc",
]


# Dynamic client creation function
def get_llm_client():
    """
    Creates and returns an LLM client and model engine string based on environment variables.

    Returns:
        tuple: (client, model_engine) where client is an OpenAI or openai object, and model_engine is the model name.
    Raises:
        ValueError: If required environment variables are missing or provider is invalid.
    """
    provider = os.getenv("LLM_PROVIDER", "openai").lower()

    if provider == "openai":
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not set for OpenAI provider")
        client = OpenAI(api_key=openai_api_key)
        model_engine = os.getenv("OPENAI_MODEL", "gpt-4o")

    elif provider == "azure-openai":
        openai.api_type = "azure-openai"
        openai.api_base = os.getenv("AZURE_OPENAI_BASE_URL")
        openai.api_version = os.getenv(
            "AZURE_OPENAI_API_VERSION",
            "2025-01-01-preview"
        )
        openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        model_engine = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

        if not (openai.api_base and openai.api_key and model_engine):
            raise ValueError(
                "Azure environment variables (AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT_NAME) not properly set for azure-openai provider"
            )
        # For Azure, the global openai object is used directly (legacy SDK structure preferred)
        return openai, model_engine

    elif provider == "lm-studio":
        client = OpenAI(
            base_url=os.getenv(
                "LM_STUDIO_BASE_URL",
                "http://localhost:1234/v1"
            ),
            api_key=os.getenv(
                "LM_STUDIO_API_KEY",
                "lm-studio"
            ),
        )
        model_engine = os.getenv("LM_STUDIO_MODEL", "local-model")

    elif provider == "ollama":
        client = OpenAI(
            base_url=os.getenv(
                "OLLAMA_BASE_URL",
                "http://localhost:11434/v1"
            ),
            api_key=os.getenv(
                "OLLAMA_API_KEY",
                "ollama"
            ),
        )
        model_engine = os.getenv("OLLAMA_MODEL", "llama2:latest")

    elif provider == "google-gemini":
        client = OpenAI(
            base_url=os.getenv(
                "GOOGLE_GEMINI_BASE_URL",
                "https://generativelanguage.googleapis.com/v1beta/openai"
            ),
            api_key=os.getenv("GOOGLE_GEMINI_API_KEY")
        )
        model_engine = os.getenv("GOOGLE_GEMINI_MODEL", "gemini-2.0-flash")

        if not client.api_key:
            raise ValueError("GEMINI environment variables GOOGLE_GEMINI_API_KEY not set for google-gemini provider")

    elif provider == "anthropic":
        client = OpenAI(
            base_url=os.getenv(
                "ANTHROPIC_BASE_URL",
                "https://api.anthropic.com/v1"
            ),
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )
        model_engine = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-latest")

        if not client.api_key:
            raise ValueError("Anthropic environment variables ANTHROPIC_API_KEY not set for anthropic provider")
        
    else:
        raise ValueError(
            f"Invalid LLM provider '{provider}'. Valid options: openai, azure-openai, google-gemini, anthropic lm-studio, ollama "
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
            engine=model_engine,
            messages=messages,
            temperature=temperature
        )
    else:
        # New SDK structure for OpenAI default, LM Studio, Ollama
        response = client.chat.completions.create(
            model=model_engine,
            messages=messages,
            temperature=temperature
        )

    content = response.choices[0].message.content
    if content is None:
        raise ValueError("LLM returned an empty response")

    content = content.strip()
    if content.startswith("```") and content.endswith("```"):
        content = content[content.find("{"):content.rfind("}") + 1]
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        raise ValueError("LLM returned invalid JSON:\n" + content)

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
            engine=model_engine,
            messages=messages,
            temperature=temperature
        )
    else:
        response = client.chat.completions.create(
            model=model_engine,
            messages=messages,
            temperature=temperature
        )

    content = response.choices[0].message.content
    if not content:
        raise ValueError("LLM returned an empty response")

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
    messages = _create_llm_prompt_for_table(
        sql_content, table_name, upstream_docs
    )

    client, model_engine = get_llm_client()

    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    if provider == "azure-openai":
        response = client.ChatCompletion.create(
            engine=model_engine,
            messages=messages,
            temperature=temperature
        )
    else:
        response = client.chat.completions.create(
            model=model_engine,
            messages=messages,
            temperature=temperature
        )

    content = response.choices[0].message.content
    if not content:
        raise ValueError("LLM returned an empty response")

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
