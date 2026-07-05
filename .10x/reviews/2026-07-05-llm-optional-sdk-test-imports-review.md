Status: recorded
Created: 2026-07-05
Updated: 2026-07-05
Target: .10x/tickets/done/2026-07-05-stabilize-llm-optional-sdk-test-imports.md
Verdict: pass

# LLM optional SDK test imports review

## Target

Review of the optional OpenAI SDK test import cleanup in `tests/core/test_llm.py`.

## Assumptions Tested

- `pytest.importorskip("openai", reason="openai not installed")` preserves the prior skip behavior when the SDK is unavailable.
- When the SDK is available, assigning the return value to `openai` gives the same module object the tests previously imported.
- The cleanup should not change retry behavior or production LLM code.

## Findings

No blocking findings.

## Verdict

Pass. The patch replaces four conditional import/skip blocks with pytest's native initialized binding, keeps the local skip behavior unchanged, and clears all CodeQL `py/uninitialized-local-variable` findings.

## Residual Risk

The optional OpenAI retry paths are still skipped locally because the SDK is not installed. That matches the pre-existing local test behavior and optional dependency design.
