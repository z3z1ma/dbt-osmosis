Status: recorded
Created: 2026-07-05
Updated: 2026-07-05
Target: .10x/tickets/done/2026-07-05-reduce-vulture-boilerplate-noise.md
Verdict: pass

# Vulture boilerplate cleanup review

## Target

Behavior-preserving cleanup of high-confidence Vulture findings in protocol stubs, context managers, and the optional SQL proxy query callback.

## Findings

None.

## Assumptions tested

- Protocol keyword names were preserved for dbt and logging shapes; only stub bodies were changed.
- `DbtProjectContext.__exit__` and `YamlRefactorContext.__exit__` still call `close()` and ignore exception objects.
- `DbtSession.query()` still accepts `attrs` and preserves the original SQL string, as covered by `tests/core/test_sql_proxy.py`.

## Verdict

Pass. The cleanup reduces high-confidence Vulture noise without removing public names or changing runtime behavior.

## Residual risk

Lower-confidence Vulture output remains intentionally unacted-on because it includes Click command callbacks, protocols, and compatibility/public API. Any future dead-code deletion should be scoped separately and require import/call-site proof.
