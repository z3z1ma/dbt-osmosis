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
| `azure-openai` | `AZURE_OPENAI_BASE_URL`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_DEPLOYMENT_NAME` | `AZURE_OPENAI_API_VERSION` (default `2025-01-01-preview`) |
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
