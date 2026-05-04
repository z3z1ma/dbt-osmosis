---
id: ticket:c10opt03
kind: ticket
status: closed
change_class: code-behavior
risk_class: medium
created_at: 2026-05-03T21:10:43Z
updated_at: 2026-05-04T17:10:34Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  initiative:
    - initiative:dbt-110-111-hardening
  evidence:
    - evidence:oracle-backlog-scan
    - evidence:c10opt03-nested-options-validation
  critique:
    - critique:c10opt03-nested-options-review
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
| ticket:c10opt03#ACC-001 | evidence:c10opt03-nested-options-validation | critique:c10opt03-nested-options-review | supported |
| ticket:c10opt03#ACC-002 | evidence:c10opt03-nested-options-validation | critique:c10opt03-nested-options-review | supported |
| ticket:c10opt03#ACC-003 | evidence:c10opt03-nested-options-validation | critique:c10opt03-nested-options-review | supported |
| ticket:c10opt03#ACC-004 | evidence:c10opt03-nested-options-validation | critique:c10opt03-nested-options-review | supported |
| ticket:c10opt03#ACC-005 | evidence:c10opt03-nested-options-validation | critique:c10opt03-nested-options-review | supported |

# Execution Notes

Prefer a tiny shared lookup helper over copy/pasted checks in every source class. Include a test for `False` values so the resolver does not mistake falsey but explicit settings for missing values.

# Blockers

None.

# Evidence

Evidence `evidence:c10opt03-nested-options-validation` records focused pytest, Ruff, whitespace, and optional-SDK basedpyright observations from final implementation/coverage commit `3c644601c5812fd1333e3fb1627e252baddfd40a`.

Evidence disposition: sufficient for ticket-local acceptance. The ticket resolved a coverage gap; no resolver source change was needed because the current implementation already satisfied the scoped lookup behavior.

# Critique Disposition

Risk class: medium

Critique policy: recommended

Policy rationale: Config precedence changes can be subtle but this is a bounded normalization fix.

Required critique profiles: code-change, test-coverage

Findings: `critique:c10opt03-nested-options-review` records no open findings. Initial critique findings `C10OPT03-F001` and `C10OPT03-F002` were resolved during review by commit `3c644601c5812fd1333e3fb1627e252baddfd40a` and this ticket/evidence reconciliation.

Disposition status: completed

Deferral / not-required rationale: None.

# Retrospective / Promotion Disposition

Disposition status: not_required

Promoted: None.

Deferred / not-required rationale: No new accepted behavior or workflow needed promotion. Existing `wiki:config-resolution` already states that supported options objects accept kebab-case and snake_case setting names, including nested `dbt-osmosis-options` and `dbt_osmosis_options` inner keys.

# Wiki Disposition

Disposition status: not_required

Rationale: `wiki:config-resolution` already covers the accepted nested-key behavior; this ticket only added focused regression coverage and did not change the accepted explanation.

# Acceptance Decision

Accepted by: OpenCode parent acceptance gate.
Accepted at: 2026-05-04T17:10:34Z.
Basis: `evidence:c10opt03-nested-options-validation` and `critique:c10opt03-nested-options-review` support all ticket-local acceptance criteria. Final focused validation reported `56 passed`, Ruff and whitespace checks passed, and optional-SDK basedpyright reported `errorCount=0`.
Residual risks: Validation is focused unit-level coverage rather than a full dbt-version matrix. Broader resolver precedence integration remains out of scope and owned by adjacent config-resolution work.

# Dependencies

Coordinate with ticket:c10res14 if implementing in shared resolver infrastructure.

# Journal

- 2026-05-03T21:10:43Z: Created from dbt compatibility oracle finding.
- 2026-05-04T16:54:20Z: Activated ticket and compiled Ralph packet `packet:ralph-ticket-c10opt03-20260504T165420Z` for focused nested options snake/kebab coverage and any required narrow resolver fix.
- 2026-05-04T17:10:34Z: Accepted Ralph output after focused tests showed current resolver behavior already satisfied the scoped nested option lookup contract. Added exact ACC-002 unrendered coverage after critique, recorded evidence/critique, consumed packet, and closed ticket.
