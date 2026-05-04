---
id: "wiki:model-validation-timeouts"
kind: wiki
page_type: reference
status: accepted
created_at: 2026-05-04T16:26:17Z
updated_at: 2026-05-04T16:26:17Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  tickets:
    - ticket:c10val27
  evidence:
    - evidence:c10val27-validation-timeout-verification
  critique:
    - critique:c10val27-validation-timeout-review
---

# Summary

Model validation timeout support is a best-effort local guard against dbt-osmosis waiting indefinitely. It is not a cross-adapter guarantee that the warehouse query was cancelled.

# Contract

- `validate_models(..., timeout_seconds=N)` asks dbt-osmosis to stop waiting after roughly `N` seconds for each model execution.
- Positive timeouts use a main-thread `SIGALRM`/`setitimer` boundary in `src/dbt_osmosis/core/validation.py`.
- On timeout, validation returns `ModelValidationStatus.TIMEOUT` with the compiled SQL and a message that warehouse-side query cancellation is adapter-specific and not guaranteed.
- Non-positive or absent timeouts preserve the historical synchronous execution path.
- Fast success, compile errors, and fast execution errors keep their existing classifications even when a positive timeout is configured.

# Examples

Use timeout evidence to prove dbt-osmosis returned promptly, not to prove warehouse cancellation:

```text
status: timeout
meaning: dbt-osmosis stopped waiting for validation execution
not proven: the remote query was cancelled by the adapter or warehouse
```

# Notes

- The timeout mechanism is POSIX/main-thread/process-global. If future work needs stronger cancellation, it must be adapter-specific or move execution into an isolation boundary that can be terminated safely.
- Do not describe this behavior as a guaranteed query cancellation feature in tickets, docs, or release notes.
- If a future adapter provides explicit cancellation or statement timeout configuration, record that as a separate behavior claim with adapter-specific tests.

# Sources

- `ticket:c10val27`
- `evidence:c10val27-validation-timeout-verification`
- `critique:c10val27-validation-timeout-review`
- `packet:ralph-ticket-c10val27-20260504T152817Z`

# Related Pages

- `wiki:ci-compatibility-matrix`
