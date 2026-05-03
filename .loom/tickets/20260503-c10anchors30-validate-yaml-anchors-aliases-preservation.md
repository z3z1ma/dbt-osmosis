---
id: ticket:c10anchors30
kind: ticket
status: proposed
change_class: code-behavior
risk_class: medium
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

Decide and test whether schema reader/writer preservation supports YAML anchors and aliases that cross managed and unmanaged sections.

# Context

`src/dbt_osmosis/core/schema/parser.py`, `reader.py`, and `writer.py` parse managed and preserved sections separately, then merge preserved keys back. The audit flagged that anchors defined in unmanaged sections and referenced in managed model/source sections may duplicate anchors, move definitions, or produce undefined aliases.

# Why Now

YAML preservation is a core promise. Anchors/aliases are less common, but if users rely on them for tests or metadata, schema mutation must either preserve them or fail/document unsupported patterns.

# Scope

- Determine whether cross-section anchors/aliases are intended to be supported.
- Add tests with anchors defined in unmanaged sections and aliases used in managed sections.
- If supported, fix parser/reader/writer handling to round-trip without duplicate/undefined anchors.
- If unsupported, document and validate/fail clearly before destructive writes.

# Out Of Scope

- Supporting every advanced YAML feature if the chosen policy is explicit non-support.
- Replacing ruamel.yaml.

# Acceptance Criteria

- ACC-001: The project has an explicit supported/unsupported policy for cross-section YAML anchors and aliases.
- ACC-002: A schema file using `x-common-tests: &common` and `data_tests: *common` is covered by a regression test.
- ACC-003: If supported, write/read round-trip keeps aliases valid and parseable by ruamel/dbt.
- ACC-004: If unsupported, commands fail clearly before writing and docs/errors tell users what to change.
- ACC-005: Preserved unmanaged sections remain preserved under the chosen policy.

# Coverage

Covers:

- ticket:c10anchors30#ACC-001
- ticket:c10anchors30#ACC-002
- ticket:c10anchors30#ACC-003
- ticket:c10anchors30#ACC-004
- ticket:c10anchors30#ACC-005
- initiative:dbt-110-111-hardening#OBJ-003

# Claim Matrix

| Claim | Evidence | Critique | Status |
| --- | --- | --- | --- |
| ticket:c10anchors30#ACC-001 | evidence:oracle-backlog-scan | None | open |

# Execution Notes

Start with characterization tests. Do not assume support until a real ruamel/dbt round trip proves it.

# Blockers

Potential behavior decision: support anchors or explicitly reject them.

# Evidence

Existing evidence: evidence:oracle-backlog-scan. Missing evidence: characterization tests.

# Critique Disposition

Risk class: medium

Critique policy: recommended

Policy rationale: YAML preservation edge case can cause data loss or invalid files.

Required critique profiles: code-change, data-preservation, test-coverage

Findings: None - no critique yet.

Disposition status: pending

Deferral / not-required rationale: None.

# Retrospective / Promotion Disposition

Disposition status: pending

Promoted: None - implementation not complete.

Deferred / not-required rationale: Documentation/wiki promotion likely if anchors are unsupported.

# Wiki Disposition

N/A - no wiki promotion selected yet.

# Acceptance Decision

Accepted by: Not accepted yet.
Accepted at: N/A.
Basis: Pending characterization evidence.
Residual risks: YAML anchor behavior can be subtle across ruamel versions.

# Dependencies

Coordinate with ticket:c10loss16 and ticket:c10gen20 if writer behavior changes.

# Journal

- 2026-05-03T21:10:43Z: Created from core architecture oracle finding.
