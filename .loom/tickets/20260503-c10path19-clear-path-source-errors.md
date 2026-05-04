---
id: ticket:c10path19
kind: ticket
status: closed
change_class: code-behavior
risk_class: medium
created_at: 2026-05-03T21:10:43Z
updated_at: 2026-05-04T12:55:31Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  initiative:
    - initiative:dbt-110-111-hardening
  evidence:
    - evidence:oracle-backlog-scan
    - evidence:c10path19-path-source-error-validation
  packets:
    - packet:ralph-ticket-c10path19-20260504T124520Z
  critique:
    - critique:c10path19-path-source-error-review
depends_on: []
---

# Summary

Replace raw YAML path-template/source-table crashes with clear user-facing errors and safe defaults.

# Context

`src/dbt_osmosis/core/path_management.py:138-166` directly formats user-configured YAML path templates. Invalid placeholders can surface raw `AttributeError` or `KeyError`, and tests currently expect raw exceptions. `src/dbt_osmosis/core/sync_operations.py:437-445` assumes `doc_source["tables"]` exists after `_get_or_create_source()` can return an existing source that lacks `tables`.

# Why Now

Compatibility work should make command failures clearer, not more cryptic. Bad path templates or incomplete source YAML are user-input problems and should fail closed with actionable messages or be repaired safely when unambiguous.

# Scope

- Catch path template formatting errors and raise `PathResolutionError` with node ID, template, and missing attribute/key.
- Keep existing project-root path traversal validation.
- Make source table creation use `setdefault("tables", [])` with type validation.
- Add tests for invalid path template fields and source YAML with missing `tables`.

# Out Of Scope

- Designing a new path template language.
- Auto-repairing malformed `tables` values that are not lists without surfacing validation.

# Acceptance Criteria

- ACC-001: Invalid YAML path template placeholders raise `PathResolutionError`, not raw `AttributeError`/`KeyError`.
- ACC-002: Error messages include the node unique ID and configured template.
- ACC-003: Existing project-root safety checks still reject traversal outside the dbt project.
- ACC-004: Existing source YAML with `sources: [{name: raw}]` can sync a source table by creating `tables: []` when safe.
- ACC-005: Existing source YAML with non-list `tables` fails with a clear validation/error message.

# Coverage

Covers:

- ticket:c10path19#ACC-001
- ticket:c10path19#ACC-002
- ticket:c10path19#ACC-003
- ticket:c10path19#ACC-004
- ticket:c10path19#ACC-005
- initiative:dbt-110-111-hardening#OBJ-003
- initiative:dbt-110-111-hardening#OBJ-006

# Claim Matrix

| Claim | Evidence | Critique | Status |
| --- | --- | --- | --- |
| ticket:c10path19#ACC-001 | evidence:c10path19-path-source-error-validation | critique:c10path19-path-source-error-review | accepted |
| ticket:c10path19#ACC-002 | evidence:c10path19-path-source-error-validation | critique:c10path19-path-source-error-review | accepted |
| ticket:c10path19#ACC-003 | evidence:c10path19-path-source-error-validation | critique:c10path19-path-source-error-review | accepted |
| ticket:c10path19#ACC-004 | evidence:c10path19-path-source-error-validation | critique:c10path19-path-source-error-review | accepted |
| ticket:c10path19#ACC-005 | evidence:c10path19-path-source-error-validation | critique:c10path19-path-source-error-review | accepted |

# Execution Notes

Keep path error wrapping narrow so real programmer errors are not swallowed. Tests should assert exception type and useful message, not exact full wording.

# Blockers

None.

# Evidence

Evidence recorded:

- evidence:oracle-backlog-scan
- evidence:c10path19-path-source-error-validation

# Critique Disposition

Risk class: medium

Critique policy: recommended

Policy rationale: User-facing error handling and source YAML mutation should be reviewed for clarity and data preservation.

Required critique profiles: code-change, operator-clarity, test-coverage

Findings: critique:c10path19-path-source-error-review#FIND-001 is low severity and accepted as a non-blocking adjacent risk. No medium/high findings.

Disposition status: completed

Deferral / not-required rationale: N/A - critique completed with no blockers.

# Retrospective / Promotion Disposition

Disposition status: not_required

Promoted: None.

Deferred / not-required rationale: The fix is localized error handling and source list validation; no reusable wiki concept was promoted.

# Wiki Disposition

N/A - no wiki promotion selected; behavior is covered by the ticket, evidence, critique, and tests.

# Acceptance Decision

Accepted by: OpenCode parent agent.
Accepted at: 2026-05-04T12:55:31Z.
Basis: implementation commit `b324adc0f1a7cca571504aa168b2e58010e5a81f`, evidence:c10path19-path-source-error-validation, and critique:c10path19-path-source-error-review.
Residual risks: Full repository suite and GitHub Actions matrix are deferred to final initiative validation; malformed entries inside a list-valued `tables` field can still leak raw errors and belong to broader source YAML structural validation; malformed format syntax or positional placeholders remain outside the narrow template missing-key/attribute wrapping scope.

# Dependencies

None.

# Journal

- 2026-05-03T21:10:43Z: Created from core architecture oracle finding.
- 2026-05-04T12:45:20Z: Activated ticket and compiled Ralph packet `packet:ralph-ticket-c10path19-20260504T124520Z` for test-first path-template error wrapping and safe source-table list handling with local-only validation.
- 2026-05-04T12:55:31Z: Ralph iteration consumed. Implementation commit `b324adc0f1a7cca571504aa168b2e58010e5a81f` wrapped invalid path template placeholders in `PathResolutionError`, preserved traversal validation, initialized missing source `tables`, and rejected non-list `tables` with `YamlValidationError`. Local validation passed with `72 passed, 1 skipped`; final critique found no medium/high blockers. Accepted and closed with final initiative-level CI deferred.
