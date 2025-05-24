---
sidebar_position: 6
---
# Synthesize Functionality (Experimental)

The `--synthesize` functionality in dbt-osmosis leverages Large Language Models (LLMs) to automatically generate missing documentation for dbt models and columns. This feature is experimental and requires careful validation of the generated content.

## Overview

To be able to use a LLM client you need to define the right information using environmental variables, such as API key, base url, and other specific information depending on the LLM client.
The `--synthesize` flag can be used with the following commands:
- `dbt-osmosis yaml document`
- `dbt-osmosis yaml refactor`

When enabled, dbt-osmosis attempts to generate descriptions for models and columns that lack documentation. The generated content is based on the SQL structure, existing context, and upstream documentation.

## Supported LLM Clients

The following LLM clients are supported:
1. **OpenAI**
   - Environment Variables:
     - `OPENAI_API_KEY` (required)
     - `OPENAI_MODEL` (default: `gpt-4o`)
2. **Azure OpenAI**
   - Environment Variables:
     - `AZURE_OPENAI_ENDPOINT` (required)
     - `AZURE_OPENAI_API_KEY` (required)
     - `AZURE_OPENAI_DEPLOYMENT_NAME` (required)
     - `AZURE_OPENAI_API_VERSION` (default: `2024-02-15-preview`)
3. **LM Studio**
   - Environment Variables:
     - `LM_STUDIO_BASE_URL` (default: `http://localhost:1234/v1`)
     - `LM_STUDIO_API_KEY` (default: `lm-studio`)
     - `LM_STUDIO_MODEL` (default: `local-model`)
4. **Ollama**
   - Environment Variables:
     - `OLLAMA_BASE_URL` (default: `http://localhost:11434/v1`)
     - `OLLAMA_API_KEY` (default: `ollama`)
     - `OLLAMA_MODEL` (default: `llama3`)

## Setting Up Environment Variables

To configure the required environment variables, you can use a `.env` or `.envrc` file. Tools like [direnv](https://direnv.net/) can help manage these variables efficiently.

Example `.env` file:
```
LLM_PROVIDER=openai
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4o
```

## Testing the Connection

To test the connection to the configured LLM client, use the following command:
```bash
dbt-osmosis --test-llm
```
- If the connection is successful, you will see: `LLM client connection successful.`
- If the connection fails, you will see: `LLM client connection failed.`

## Important Notes

- **Experimental Feature**: The `--synthesize` functionality is experimental and requires a human in the loop to validate the generated descriptions.
- **Context Limitations**: LLMs may not always generate accurate descriptions, especially if the specific context is outside the training data of the LLM.
- **Validation Required**: Always review and refine the auto-generated content to ensure it aligns with your use case.

## Installation

To use the `--synthesize` functionality, install dbt-osmosis with the `[openai]` extra:
```bash
pip install "dbt-osmosis[openai]"
```

## Example Usage

```bash
dbt-osmosis yaml refactor --synthesize
```

This command will:
1. Organize your YAML files.
2. Generate missing documentation using the configured LLM client.
