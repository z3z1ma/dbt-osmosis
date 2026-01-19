---
sidebar_position: 6
---

# Synthesize (experimental)

`--synthesize` uses an LLM to generate missing descriptions for models and columns. Treat the output as a starting point and review everything before committing.

## Supported providers

Set `LLM_PROVIDER` to one of:

- `openai`
- `azure-openai`
- `google-gemini`
- `anthropic`
- `lm-studio`
- `ollama`

## Required environment variables

| Provider | Required variables | Optional variables |
| --- | --- | --- |
| `openai` | `OPENAI_API_KEY` | `OPENAI_MODEL` (default `gpt-4o`) |
| `azure-openai` | `AZURE_OPENAI_BASE_URL`, `AZURE_OPENAI_DEPLOYMENT_NAME`, and either `AZURE_OPENAI_API_KEY` or `AZURE_OPENAI_AD_TOKEN_SCOPE` | `AZURE_OPENAI_API_VERSION` (default `2025-01-01-preview`) |
| `google-gemini` | `GOOGLE_GEMINI_API_KEY` | `GOOGLE_GEMINI_BASE_URL`, `GOOGLE_GEMINI_MODEL` |
| `anthropic` | `ANTHROPIC_API_KEY` | `ANTHROPIC_BASE_URL`, `ANTHROPIC_MODEL` |
| `lm-studio` | `LM_STUDIO_BASE_URL`, `LM_STUDIO_API_KEY` | `LM_STUDIO_MODEL` |
| `ollama` | `OLLAMA_BASE_URL`, `OLLAMA_API_KEY` | `OLLAMA_MODEL` |

## Install dependencies

```bash
pip install "dbt-osmosis[openai]"
```

## Test your configuration

```bash
dbt-osmosis test-llm
```

If the command succeeds, it prints the provider and model engine.

## Run synthesis

```bash
dbt-osmosis yaml refactor --synthesize
```

You can also use `--synthesize` with `dbt-osmosis yaml document`.

## Tips

- Start with `--dry-run` to see what will change.
- Use selection flags or positional selectors to limit scope while you build trust.

## Azure OpenAI Authentication

Azure OpenAI supports two authentication methods:

### API Key (Traditional)
```bash
export LLM_PROVIDER=azure-openai
export AZURE_OPENAI_BASE_URL="https://your-resource.openai.azure.com"
export AZURE_OPENAI_API_KEY="your-api-key"
export AZURE_OPENAI_DEPLOYMENT_NAME="gpt-4"
```

### Azure AD Token (Enterprise)
```bash
# Install azure-identity
pip install "dbt-osmosis[openai]" azure-identity

# Authenticate with Azure CLI
az login

export LLM_PROVIDER=azure-openai
export AZURE_OPENAI_BASE_URL="https://your-resource.openai.azure.com"
export AZURE_OPENAI_DEPLOYMENT_NAME="gpt-4"
export AZURE_OPENAI_AD_TOKEN_SCOPE="https://cognitiveservices.azure.com"
```

For service principal authentication, also set:
```bash
export AZURE_TENANT_ID="your-tenant-id"
export AZURE_CLIENT_ID="your-client-id"
export AZURE_CLIENT_SECRET="your-client-secret"
```

**Note:** Azure AD tokens expire after ~1 hour. For long-running processes, restart periodically or use API key authentication.
