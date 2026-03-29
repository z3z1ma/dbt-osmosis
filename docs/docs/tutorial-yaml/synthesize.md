---
sidebar_position: 6
---

# Synthesize (experimental)

`--synthesize` uses an LLM to generate missing descriptions for models and columns. Treat the output as a starting point and review everything before committing.

## Supported providers

Set `LLM_PROVIDER` to one of:

- `openai`
- `azure-openai`
- `azure-openai-ad`
- `google-gemini`
- `anthropic`
- `lm-studio`
- `ollama`

## Required environment variables

| Provider | Required variables | Optional variables |
| --- | --- | --- |
| `openai` | `OPENAI_API_KEY` | `OPENAI_MODEL` (default `gpt-4o`) |
| `azure-openai` | `AZURE_OPENAI_BASE_URL`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_DEPLOYMENT_NAME` | `AZURE_OPENAI_API_VERSION` (default `2025-01-01-preview`) |
| `azure-openai-ad` | `AZURE_OPENAI_BASE_URL`, `AZURE_OPENAI_AD_TOKEN_SCOPE`, `AZURE_OPENAI_DEPLOYMENT_NAME` | `AZURE_TENANT_ID`, `AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET` (for service principal auth). `AZURE_OPENAI_AD_TOKEN_SCOPE` may be either the full `/.default` scope or the bare resource URI that dbt-osmosis normalizes for you. |
| `google-gemini` | `GOOGLE_GEMINI_API_KEY` | `GOOGLE_GEMINI_BASE_URL` (default `https://generativelanguage.googleapis.com/v1beta/openai`), `GOOGLE_GEMINI_MODEL` (default `gemini-2.0-flash`) |
| `anthropic` | `ANTHROPIC_API_KEY` | `ANTHROPIC_BASE_URL` (default `https://api.anthropic.com/v1`), `ANTHROPIC_MODEL` (default `claude-3-5-haiku-latest`) |
| `lm-studio` | none | `LM_STUDIO_BASE_URL` (default `http://localhost:1234/v1`), `LM_STUDIO_API_KEY` (default `lm-studio`), `LM_STUDIO_MODEL` (default `local-model`) |
| `ollama` | none | `OLLAMA_BASE_URL` (default `http://localhost:11434/v1`), `OLLAMA_API_KEY` (default `ollama`), `OLLAMA_MODEL` (default `llama2:latest`) |

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

export LLM_PROVIDER=azure-openai-ad
export AZURE_OPENAI_BASE_URL="https://your-resource.openai.azure.com"
export AZURE_OPENAI_DEPLOYMENT_NAME="gpt-4"
export AZURE_OPENAI_AD_TOKEN_SCOPE="https://cognitiveservices.azure.com/.default"
```

**For custom gateways/proxies**, use the custom API scope provided by your gateway admin:
```bash
export AZURE_OPENAI_AD_TOKEN_SCOPE="api://your-gateway-app-id/.default"
```

dbt-osmosis also accepts the bare resource URI form (for example `https://cognitiveservices.azure.com` or `api://your-gateway-app-id`) and normalizes it to `/.default` before requesting a token. If your gateway admin gives you a more specific delegated scope such as `api://your-gateway-app-id/access_as_user`, keep that exact value.

For service principal authentication, also set:
```bash
export AZURE_TENANT_ID="your-tenant-id"
export AZURE_CLIENT_ID="your-client-id"
export AZURE_CLIENT_SECRET="your-client-secret"
```

**Important Notes:**
- Azure AD tokens expire after ~1 hour. For long-running processes, restart periodically or use API key authentication.
- `azure-openai-ad` uses the OpenAI SDK client (not Azure OpenAI SDK), so `AZURE_OPENAI_BASE_URL` should be the base URL without `/openai/deployments/<name>`. The SDK will construct the full path.
- If using a custom gateway/proxy, ensure it handles standard OpenAI API paths (`/chat/completions`) and routes to your deployment.
