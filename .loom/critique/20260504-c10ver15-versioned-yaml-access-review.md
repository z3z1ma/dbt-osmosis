---
id: critique:c10ver15-versioned-yaml-access-review
kind: critique
status: final
created_at: 2026-05-04T11:20:10Z
updated_at: 2026-05-04T11:20:10Z
scope:
  kind: repository
  repositories:
    - repo:root
review_target: "ticket:c10ver15 versioned model YAML access diff through ef1d540"
links:
  tickets:
    - ticket:c10ver15
  evidence:
    - evidence:c10ver15-versioned-yaml-access-validation
  packets:
    - packet:ralph-ticket-c10ver15-20260504T095044Z
  wiki:
    - wiki:versioned-model-yaml
external_refs: {}
---

# Summary

Reviewed the `ticket:c10ver15` versioned model YAML access, validation, and selector-preserving sync/refactor implementation after Ralph output and parent critique-driven fixes. The review focused on dbt version semantics, version-level column access, unrendered inheritance, selector validation/preservation, sync data preservation, validation coverage, and regression tests.

# Review Target

Target: implementation commit `ef1d5409bcceee40c3403c06d75bf5cbe4cc4bb1` on branch `loom/dbt-110-111-hardening`, plus the associated Loom evidence/ticket/wiki reconciliation records.

Reviewed changed surfaces:

- `src/dbt_osmosis/core/inheritance.py`
- `src/dbt_osmosis/core/schema/validation.py`
- `src/dbt_osmosis/core/sync_operations.py`
- `tests/core/test_inheritance_behavior.py`
- `tests/core/test_property_accessor.py`
- `tests/core/test_validation.py`
- `tests/core/test_sync_operations.py`
- `packet:ralph-ticket-c10ver15-20260504T095044Z`
- `evidence:c10ver15-versioned-yaml-access-validation`

# Verdict

`pass_with_low_residual_risks`.

No medium or high findings remain after parent-side fixes for selector entries, dbt-compatible selector validation, exact string version matching, selector-preserving sync/refactor, blank version description fallback, duplicate/latest-version validation, int/string sync lookup skew, and over-broad numeric string fallback. The implementation is acceptable for ticket acceptance under local-validation-only ticket policy.

# Findings

No open medium/high findings.

Low residual risks and testing gaps:

- Non-finite numeric version values are not explicitly modeled; malformed exotic YAML remains primarily dbt-parse territory.
- `_get_node_yaml()` returns a shallow read-only version view; nested YAML lists/dicts remain mutable, matching existing behavior.
- `ModelValidator` remains a lightweight dbt-osmosis validator and does not attempt to validate every dbt version-level field such as `config`, `docs`, and `constraints`.
- Full repository suite and GitHub Actions matrix were not run locally; final initiative-level CI remains the broader compatibility gate.

# Evidence Reviewed

Reviewed the Ralph packet child output, current implementation diff, local red/green outputs, focused and broader pytest outputs, post-commit acceptance pytest output, targeted pre-commit output, Ruff output, `git diff --check`, and repeated no-edit oracle review results.

Key evidence record:

- evidence:c10ver15-versioned-yaml-access-validation

# Residual Risks

- The version identity helper is deliberately conservative for string versions; future dbt behavior changes around `NodeVersion` coercion should recheck this area.
- Final CI matrix evidence is deferred to the initiative-level validation pass.

# Required Follow-up

No critique-required implementation follow-up remains before ticket closure. Related future work such as `ticket:c10loss16` should reuse the accepted version identity and selector-preservation rules rather than re-deriving them.

# Acceptance Recommendation

`accept`.
