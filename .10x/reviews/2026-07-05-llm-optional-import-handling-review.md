Status: recorded
Created: 2026-07-05
Updated: 2026-07-05
Target: .10x/tickets/done/2026-07-05-tighten-llm-optional-import-handling.md
Verdict: pass

# LLM optional import handling review

## Target

Review of the LLM optional OpenAI import cleanup in `src/dbt_osmosis/core/llm.py` for `.10x/tickets/done/2026-07-05-tighten-llm-optional-import-handling.md`.

## Assumptions Tested

- Removing `openai = None` is safe because no source or tests reference `dbt_osmosis.core.llm.openai`.
- Binding `RateLimitError` directly preserves retry behavior when the OpenAI SDK is installed.
- The fallback `_FallbackOpenAIRateLimitError` still exists for environments without the optional OpenAI SDK.
- Malformed `Retry-After` handling remains a fallback to exponential delay, not a behavior change.

## Findings

No blocking findings.

## Verdict

Pass. The change removes one unused global, documents intentional malformed-header fallback behavior, keeps optional dependency behavior intact, and clears all CodeQL findings for `src/dbt_osmosis/core/llm.py`.

## Residual Risk

Focused tests in this local environment skip optional OpenAI and Azure Identity integration paths because those extras are not installed. Existing mock-based provider tests still cover the base behavior in this repository environment.
