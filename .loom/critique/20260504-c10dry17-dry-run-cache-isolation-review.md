---
id: critique:c10dry17-dry-run-cache-isolation-review
kind: critique
status: final
created_at: 2026-05-04T12:24:01Z
updated_at: 2026-05-04T12:24:01Z
scope:
  kind: repository
  repositories:
    - repo:root
review_target: "ticket:c10dry17 dry-run cache isolation diff through d9eee85"
links:
  tickets:
    - ticket:c10dry17
  evidence:
    - evidence:c10dry17-dry-run-cache-isolation-validation
  packets:
    - packet:ralph-ticket-c10dry17-20260504T115035Z
  wiki:
    - wiki:yaml-sync-safety
external_refs: {}
---

# Summary

Reviewed the `ticket:c10dry17` dry-run YAML cache isolation implementation after Ralph output and parent follow-up. The review focused on cache correctness, mutation tracking, preserved-section safety, fixture fidelity, test coverage, and acceptance criteria alignment.

# Review Target

Target: implementation commit `d9eee85a212485c5d6f2944e52eafc2fad3c345e` on branch `loom/dbt-110-111-hardening`, plus associated ticket/packet/evidence/wiki reconciliation records.

Reviewed changed surfaces:

- `src/dbt_osmosis/core/schema/writer.py`
- `src/dbt_osmosis/core/sync_operations.py`
- `tests/conftest.py`
- `tests/core/test_schema.py`
- `tests/core/test_sync_operations.py`
- cache-fixture cleanup in related tests
- `packet:ralph-ticket-c10dry17-20260504T115035Z`
- `evidence:c10dry17-dry-run-cache-isolation-validation`

# Verdict

`pass_with_findings`.

No medium or high findings were found. The implementation is acceptable for ticket acceptance under local-validation-only ticket policy with one low residual risk documented below.

# Findings

## FIND-001: Preview-then-apply callers must reread after dry-run cleanup

Severity: low
Confidence: medium
State: open

Observation: `src/dbt_osmosis/core/schema/writer.py` now discards `_YAML_ORIGINAL_CACHE` after dry-run processing. Preserved-section restoration on a later real `_write_yaml` depends on `_YAML_ORIGINAL_CACHE` when the caller supplies already-filtered YAML data. A long-lived caller that reads mixed YAML, runs `_write_yaml(..., dry_run=True)`, then writes the same filtered `data` object with `dry_run=False` without a fresh `_read_yaml` could drop preserved top-level sections.

Why it matters: The new behavior is correct for the ticket's same-process dry-run isolation claim, but preview-then-apply helper flows should refresh disk-backed YAML state before the real write if they need preserved sections.

Follow-up: No required ticket-blocking change. Record the safe operating pattern in `wiki:yaml-sync-safety`: after a dry-run preview, reread YAML before a real write instead of reusing filtered data that depended on discarded original-cache state.

Challenges: None - not claim-specific; this is a low residual usage risk adjacent to the accepted cache-isolation behavior.

# Evidence Reviewed

- `evidence:c10dry17-dry-run-cache-isolation-validation`
- Ralph packet child red/green output in `packet:ralph-ticket-c10dry17-20260504T115035Z`
- Implementation commit `d9eee85a212485c5d6f2944e52eafc2fad3c345e`
- Focused dry-run cache pytest output: `4 passed in 0.13s`
- Changed/related pytest output: `186 passed, 3 skipped`
- Post-commit changed/related pytest output: `186 passed, 3 skipped in 96.52s`
- Ruff format/check output
- Targeted pre-commit output
- `git diff --check`
- Read-only oracle review result reporting accept/no medium-high blockers

# Residual Risks

- Dry-run sync still mutates the shared cached document before finalization discards it; concurrent same-process readers could observe transient dry-run state.
- `--check` coverage is helper-level through mutation tracking rather than a Click integration test.
- Full repository suite and GitHub Actions matrix were not run locally; final initiative-level CI remains the broader compatibility gate.

# Required Follow-up

No critique-required implementation follow-up remains before ticket closure. The low preview-then-apply risk is documented in `wiki:yaml-sync-safety` and accepted as non-blocking for this ticket.

# Acceptance Recommendation

`no-critique-blockers`
