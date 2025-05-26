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
     - `AZURE_OPENAI_BASE_URL` (required)
     - `AZURE_OPENAI_API_KEY` (required)
     - `AZURE_OPENAI_DEPLOYMENT_NAME` (required)
     - `AZURE_OPENAI_API_VERSION` (default: `2025-01-01-preview`)
   - Available deploymnets
     -  To check your current deployments and the values needed to config this environmental variables visit your [Open Ai Azure portal](https://oai.azure.com/resource/deployments){:target="_blank"}
3. **LM Studio**
   - Environment Variables:
     - `LM_STUDIO_BASE_URL` (default: `http://localhost:1234/v1`)
     - `LM_STUDIO_API_KEY` (default: `lm-studio`)
     - `LM_STUDIO_MODEL` (default: `local-model`)
4. **Ollama**
   - Environment Variables:
     - `OLLAMA_BASE_URL` (default: `http://localhost:11434/v1`)
     - `OLLAMA_API_KEY` (default: `ollama`)
     - `OLLAMA_MODEL` (default: `llama2:latest`)
   - Available models:
      - For a list of available models and instructions on how to install and run Ollama locally visit: [Ollama](https://ollama.com){:target="_blank"}
5. **Google Gemini**
   - Environment Variables:
     - `GOOGLE_GEMINI_BASE_URL` (default: `https://generativelanguage.googleapis.com/v1beta/openai`)
     - `GOOGLE_GEMINI_API_KEY` (required)
     - `GOOGLE_GEMINI_MODEL` (default: `gemini-2.0-flash`)
6. **Anthropic**
   - Environment Variables:
     - `ANTHROPIC_BASE_URL` (default: `https://api.anthropic.com/v1`)
     - `ANTHROPIC_API_KEY` (required)
     - `ANTHROPIC_MODEL` (default: `claude-3-5-haiku-latest`)
   - Available models:
     - For a full list of models available visit [Anthropic](https://docs.anthropic.com/en/docs/about-claude/models/overview#model-names){:target="_blank"}

## Setting Up Environment Variables

To configure the required environment variables, you can use a `.env` or `.envrc` file. Tools like [direnv](https://direnv.net/) can help manage these variables efficiently.

Example `.env` file:
```
# OpenAI
export LLM_PROVIDER="openai"
export OPENAI_API_KEY="your_openai_api_key"
export OPENAI_MODEL="gpt-4o"

# Azure OpenAI
export LLM_PROVIDER="azure-openai"
export AZURE_OPENAI_BASE_URL="https://your-azure-openai-instance.openai.azure.com"
export AZURE_OPENAI_API_KEY="your_azure_api_key"
export AZURE_OPENAI_DEPLOYMENT_NAME="your_deployment_name"

# LM Studio
export LLM_PROVIDER="lm-studio"
export LM_STUDIO_BASE_URL="http://localhost:1234/v1"
export LM_STUDIO_API_KEY="lm-studio"
export LM_STUDIO_MODEL="local-model"

# Ollama
export LLM_PROVIDER="ollama"
export OLLAMA_BASE_URL="http://localhost:11434/v1"
export OLLAMA_API_KEY="ollama"
export OLLAMA_MODEL="llama3.1"

# Google Gemini
export LLM_PROVIDER="google-gemini"
export GOOGLE_GEMINI_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai"
export GOOGLE_GEMINI_API_KEY="your_google_gemini_api_key"
export GOOGLE_GEMINI_MODEL="gemini-2.0-flash"

# Anthropic
export LLM_PROVIDER="anthropic"
export ANTHROPIC_BASE_URL="https://api.anthropic.com/v1"
export ANTHROPIC_API_KEY="your_anthropic_api_key"
export ANTHROPIC_MODEL="claude-3-5-haiku-latest"
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
2. Generate missing documentation inheriting the descriptions from parent to child tables and synthetizing description for empty fields using the configured LLM client.
