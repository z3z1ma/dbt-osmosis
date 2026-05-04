---
id: ticket:c10gen20
kind: ticket
status: closed
change_class: code-behavior
risk_class: high
created_at: 2026-05-03T21:10:43Z
updated_at: 2026-05-04T13:41:04Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  initiative:
    - initiative:dbt-110-111-hardening
  evidence:
    - evidence:oracle-backlog-scan
    - evidence:c10gen20-safe-generate-yaml-writes-validation
  packets:
    - packet:ralph-ticket-c10gen20-20260504T125850Z
  critique:
    - critique:c10gen20-safe-generate-yaml-writes-review
depends_on: []
---

# Summary

Route generate/NL schema YAML writes through the ruamel/schema helper pipeline, add overwrite safety, and validate output paths relative to the project root.

# Context

`src/dbt_osmosis/cli/main.py:719-751`, `843-845`, and `931-933` write SQL/YAML directly with `Path.write_text()` and PyYAML/raw strings. `src/dbt_osmosis/core/generators.py:251-264` hand-builds AI staging YAML. This bypasses schema reader/writer preservation, ruamel formatting, atomic writes, and path safety conventions.

# Why Now

Generated YAML can overwrite user files or produce invalid YAML for descriptions containing colons, quotes, newlines, or Jinja. This violates the repository's YAML preservation contract.

# Scope

- Use structured data and ruamel/schema helpers for schema YAML generation where existing YAML files may be touched.
- Add explicit overwrite behavior such as `--overwrite` or fail-closed refusal when output files exist.
- Validate generated output paths relative to project root unless the user explicitly opts into external paths.
- Ensure dry-run lists every file that would be written.
- Add tests for descriptions with colon, multiline text, quotes, and Jinja `{{ doc(...) }}`.

# Out Of Scope

- Redesigning AI generation prompts.
- Changing deterministic SQL generation behavior unrelated to file writing.

# Acceptance Criteria

- ACC-001: Existing schema YAML files are not overwritten without explicit opt-in.
- ACC-002: Generated schema YAML is serialized with the same ruamel/schema machinery used by core YAML workflows.
- ACC-003: Existing non-osmosis sections are preserved when updating an existing YAML file.
- ACC-004: Output paths outside project root fail or require explicit opt-in.
- ACC-005: Dry-run prints all planned writes and writes no files.
- ACC-006: Generated descriptions with colon, quotes, multiline text, and Jinja parse cleanly after write.

# Coverage

Covers:

- ticket:c10gen20#ACC-001
- ticket:c10gen20#ACC-002
- ticket:c10gen20#ACC-003
- ticket:c10gen20#ACC-004
- ticket:c10gen20#ACC-005
- ticket:c10gen20#ACC-006
- initiative:dbt-110-111-hardening#OBJ-003
- initiative:dbt-110-111-hardening#OBJ-006

# Claim Matrix

| Claim | Evidence | Critique | Status |
| --- | --- | --- | --- |
| ticket:c10gen20#ACC-001 | evidence:c10gen20-safe-generate-yaml-writes-validation | critique:c10gen20-safe-generate-yaml-writes-review | accepted |
| ticket:c10gen20#ACC-002 | evidence:c10gen20-safe-generate-yaml-writes-validation | critique:c10gen20-safe-generate-yaml-writes-review | accepted |
| ticket:c10gen20#ACC-003 | evidence:c10gen20-safe-generate-yaml-writes-validation | critique:c10gen20-safe-generate-yaml-writes-review | accepted |
| ticket:c10gen20#ACC-004 | evidence:c10gen20-safe-generate-yaml-writes-validation | critique:c10gen20-safe-generate-yaml-writes-review | accepted |
| ticket:c10gen20#ACC-005 | evidence:c10gen20-safe-generate-yaml-writes-validation | critique:c10gen20-safe-generate-yaml-writes-review | accepted |
| ticket:c10gen20#ACC-006 | evidence:c10gen20-safe-generate-yaml-writes-validation | critique:c10gen20-safe-generate-yaml-writes-review | accepted |

# Execution Notes

Use the smallest safe writer abstraction. If SQL model file writes remain direct, make sure this ticket's YAML-focused acceptance is not diluted by unrelated SQL file behavior.

# Blockers

None.

# Evidence

Evidence recorded:

- evidence:oracle-backlog-scan
- evidence:c10gen20-safe-generate-yaml-writes-validation

# Critique Disposition

Risk class: high

Critique policy: mandatory

Policy rationale: Direct writes can lose user YAML content and break generated files.

Required critique profiles: code-change, data-preservation, operator-clarity, test-coverage

Findings: None in final critique. Earlier critique blockers for SQL path validation and writer no-clobber enforcement were resolved before acceptance.

Disposition status: completed

Deferral / not-required rationale: N/A - mandatory critique completed with no blockers.

# Retrospective / Promotion Disposition

Disposition status: completed

Promoted: Updated wiki:yaml-sync-safety with generated/NL YAML write-safety rules.

Deferred / not-required rationale: Final initiative-level CI remains deferred to initiative closure.

# Wiki Disposition

Updated wiki:yaml-sync-safety with accepted guidance for generated YAML writes: use structured ruamel/schema helpers, fail closed on existing YAML without `--overwrite`, validate generated output paths under the project root, and make dry-run planned writes explicit.

# Acceptance Decision

Accepted by: OpenCode parent agent.
Accepted at: 2026-05-04T13:41:04Z.
Basis: implementation commit `750bff70765850751ea157e21ee4bedc0540d138`, evidence:c10gen20-safe-generate-yaml-writes-validation, and critique:c10gen20-safe-generate-yaml-writes-review.
Residual risks: SQL file overwrite semantics remain unchanged; legacy docs may not mention the new generated YAML `--overwrite` option yet; full repository suite and GitHub Actions matrix are deferred to final initiative validation.

# Dependencies

Coordinate with ticket:c10loss16 for data-preservation policy.

# Journal

- 2026-05-03T21:10:43Z: Created from CLI/SQL/workbench and core architecture oracle findings.
- 2026-05-04T12:58:50Z: Activated ticket and compiled Ralph packet `packet:ralph-ticket-c10gen20-20260504T125850Z` for test-first safe generate/NL schema YAML writes with explicit overwrite safety, project-root path validation, dry-run planned-write reporting, ruamel/schema helper serialization, and local-only validation.
- 2026-05-04T13:41:04Z: Ralph iteration consumed. Implementation commit `750bff70765850751ea157e21ee4bedc0540d138` routed generated YAML through ruamel/schema helpers, added explicit `--overwrite`, enforced writer no-clobber, validated generated output paths under the project root, listed dry-run planned writes, and serialized AI staging YAML from structured data. Local validation passed with `119 passed, 1 skipped`; final mandatory critique found no blockers. Accepted and closed with final initiative-level CI deferred.
