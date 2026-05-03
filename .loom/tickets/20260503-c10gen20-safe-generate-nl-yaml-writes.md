---
id: ticket:c10gen20
kind: ticket
status: ready
change_class: code-behavior
risk_class: high
created_at: 2026-05-03T21:10:43Z
updated_at: 2026-05-03T21:10:43Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  initiative:
    - initiative:dbt-110-111-hardening
  evidence:
    - evidence:oracle-backlog-scan
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
| ticket:c10gen20#ACC-002 | evidence:oracle-backlog-scan | None | open |
| ticket:c10gen20#ACC-006 | None - output parsing tests not written yet | None | open |

# Execution Notes

Use the smallest safe writer abstraction. If SQL model file writes remain direct, make sure this ticket's YAML-focused acceptance is not diluted by unrelated SQL file behavior.

# Blockers

Potential behavior decision if existing commands historically overwrite files silently; may need human approval or a deprecation path.

# Evidence

Existing evidence: evidence:oracle-backlog-scan. Missing evidence: generated YAML parse tests and dry-run output.

# Critique Disposition

Risk class: high

Critique policy: mandatory

Policy rationale: Direct writes can lose user YAML content and break generated files.

Required critique profiles: code-change, data-preservation, operator-clarity, test-coverage

Findings: None - no critique yet.

Disposition status: pending

Deferral / not-required rationale: None.

# Retrospective / Promotion Disposition

Disposition status: pending

Promoted: None - implementation not complete.

Deferred / not-required rationale: Consider wiki/YAML writer guidance after acceptance.

# Wiki Disposition

N/A - no wiki promotion selected yet.

# Acceptance Decision

Accepted by: Not accepted yet.
Accepted at: N/A.
Basis: Pending tests and critique.
Residual risks: Behavior change for users relying on overwrite behavior.

# Dependencies

Coordinate with ticket:c10loss16 for data-preservation policy.

# Journal

- 2026-05-03T21:10:43Z: Created from CLI/SQL/workbench and core architecture oracle findings.
