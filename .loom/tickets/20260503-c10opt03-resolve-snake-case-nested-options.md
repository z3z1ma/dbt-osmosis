---
id: ticket:c10opt03
kind: ticket
status: active
change_class: code-behavior
risk_class: medium
created_at: 2026-05-03T21:10:43Z
updated_at: 2026-05-04T16:54:20Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  initiative:
    - initiative:dbt-110-111-hardening
  evidence:
    - evidence:oracle-backlog-scan
  packets:
    - packet:ralph-ticket-c10opt03-20260504T165420Z
depends_on: []
---

# Summary

Make nested `dbt_osmosis_options` and `dbt-osmosis-options` blocks resolve both kebab-case and snake_case setting keys across config meta, unrendered config, and supplementary config sources.

# Context

The audit found that `ConfigMetaSource`, `UnrenderedConfigSource`, and `SupplementaryFileSource` check `kebab_key` inside nested option dicts but can miss `snake_key`. This affects valid Python-friendly config such as `dbt_osmosis_options={"output_to_lower": true}` and YAML-friendly options that mix namespace/key styles.

# Why Now

dbt 1.10/1.11 deprecations push custom configuration under `config.meta`. If the nested option readers fail to support snake_case keys, users can migrate to the right dbt location and still have osmosis settings ignored.

# Scope

- Audit `src/dbt_osmosis/core/introspection.py` option-source classes around `ConfigMetaSource`, `UnrenderedConfigSource`, and `SupplementaryFileSource`.
- Normalize setting key lookup so both `output-to-lower` and `output_to_lower` work inside both `dbt-osmosis-options` and `dbt_osmosis_options` containers.
- Add tests for all affected source classes and at least one end-to-end resolver call.

# Out Of Scope

- Resolving project-level supplementary/vars precedence if the resolver cannot currently reach those sources; ticket:c10res14 owns that broader integration.
- Introducing new option names or aliases beyond kebab/snake normalization.

# Acceptance Criteria

- ACC-001: `ConfigMetaSource.get("output-to-lower")` resolves from `{"dbt_osmosis_options": {"output_to_lower": true}}`.
- ACC-002: `UnrenderedConfigSource.get("output-to-lower")` resolves from the same snake_case nested shape.
- ACC-003: `SupplementaryFileSource` resolves snake_case keys inside both kebab and snake namespace containers when that source is invoked.
- ACC-004: Existing kebab-case behavior continues to pass.
- ACC-005: Tests document precedence if both snake_case and kebab-case variants are present.

# Coverage

Covers:

- ticket:c10opt03#ACC-001
- ticket:c10opt03#ACC-002
- ticket:c10opt03#ACC-003
- ticket:c10opt03#ACC-004
- ticket:c10opt03#ACC-005
- initiative:dbt-110-111-hardening#OBJ-002

# Claim Matrix

| Claim | Evidence | Critique | Status |
| --- | --- | --- | --- |
| ticket:c10opt03#ACC-001 | evidence:oracle-backlog-scan | None | open |
| ticket:c10opt03#ACC-003 | None - tests not written yet | None | open |

# Execution Notes

Prefer a tiny shared lookup helper over copy/pasted checks in every source class. Include a test for `False` values so the resolver does not mistake falsey but explicit settings for missing values.

# Blockers

None.

# Evidence

Existing evidence: evidence:oracle-backlog-scan. Missing evidence: focused tests after implementation.

# Critique Disposition

Risk class: medium

Critique policy: recommended

Policy rationale: Config precedence changes can be subtle but this is a bounded normalization fix.

Required critique profiles: code-change, test-coverage

Findings: None - no critique yet.

Disposition status: pending

Deferral / not-required rationale: None.

# Retrospective / Promotion Disposition

Disposition status: pending

Promoted: None - implementation not complete.

Deferred / not-required rationale: Likely not wiki-worthy unless combined with ticket:c10res14.

# Wiki Disposition

N/A - no wiki promotion selected yet.

# Acceptance Decision

Accepted by: Not accepted yet.
Accepted at: N/A.
Basis: Pending tests.
Residual risks: None known beyond adjacent resolver precedence work.

# Dependencies

Coordinate with ticket:c10res14 if implementing in shared resolver infrastructure.

# Journal

- 2026-05-03T21:10:43Z: Created from dbt compatibility oracle finding.
- 2026-05-04T16:54:20Z: Activated ticket and compiled Ralph packet `packet:ralph-ticket-c10opt03-20260504T165420Z` for focused nested options snake/kebab coverage and any required narrow resolver fix.
