Status: recorded
Created: 2026-07-05
Updated: 2026-07-05
Target: .10x/tickets/done/2026-07-05-make-optional-return-paths-explicit.md
Verdict: pass

# Optional return paths review

## Target

Review of explicit return-path cleanup in `src/dbt_osmosis/sql/proxy.py` and `tests/core/test_test_suggestions.py`.

## Assumptions Tested

- `_regex_parse_to_complete_dict` already advertised `dict[str, str] | None`; adding explicit `None` preserves the intended no-match behavior.
- `pytest.skip` raises a skip exception in normal pytest execution; the added assertion is only a defensive unreachable marker.
- Focused SQL proxy and test suggestion tests still pass.

## Findings

No blocking findings.

## Verdict

Pass. The patch makes existing optional paths explicit, clears the targeted CodeQL rule, and does not change parser or fixture selection semantics.

## Residual Risk

No residual risk specific to this ticket. Broader CodeQL architecture and contract-file findings remain outside scope.
