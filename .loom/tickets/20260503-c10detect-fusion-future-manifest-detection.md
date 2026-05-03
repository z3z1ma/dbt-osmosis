---
id: ticket:c10detect
kind: ticket
status: ready
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

Make `_detect_fusion_manifest()` distinguish Fusion evidence from generic future dbt-core manifest schema versions, or rename/document the behavior as a future-manifest compatibility hint.

# Context

`src/dbt_osmosis/core/config.py:78-123` treats any manifest schema version greater than v12 as Fusion. The comment says dbt-core currently produces v12 and Fusion produces v20+, but a future dbt-core manifest schema bump could be misclassified. `YamlRefactorContext.fusion_compat` then affects YAML output behavior.

# Why Now

The initiative is specifically about dbt 1.10/1.11 compatibility, but this brittle detection can turn the next dbt-core schema version into an accidental Fusion mode and create confusing YAML output changes.

# Scope

- Inspect manifest metadata available in dbt-core and Fusion manifests for producer-specific evidence.
- Replace schema-version-only detection if reliable producer evidence exists.
- If producer evidence is unavailable, rename/document the behavior so it does not overclaim Fusion detection.
- Update tests that currently assert v13 means Fusion.

# Out Of Scope

- Full Fusion support.
- Changing YAML output mode defaults beyond making detection honest.

# Acceptance Criteria

- ACC-001: dbt-core v12 manifests are not treated as Fusion.
- ACC-002: A synthetic future dbt-core v13 manifest is not automatically treated as Fusion unless explicit producer evidence justifies it.
- ACC-003: A synthetic known Fusion manifest is detected if producer/schema metadata supports detection.
- ACC-004: Tests and log messages accurately describe the detection as Fusion-specific or future-manifest-specific.
- ACC-005: Users can override or understand the behavior when stale `target/manifest.json` exists.

# Coverage

Covers:

- ticket:c10detect#ACC-001
- ticket:c10detect#ACC-002
- ticket:c10detect#ACC-003
- ticket:c10detect#ACC-004
- ticket:c10detect#ACC-005
- initiative:dbt-110-111-hardening#OBJ-002

# Claim Matrix

| Claim | Evidence | Critique | Status |
| --- | --- | --- | --- |
| ticket:c10detect#ACC-002 | evidence:oracle-backlog-scan | None | open |

# Execution Notes

Search current tests around `_detect_fusion_manifest()` before changing behavior. If a compatibility override already exists in settings, prefer documenting that over adding another flag.

# Blockers

None.

# Evidence

Existing evidence: evidence:oracle-backlog-scan. Missing evidence: updated synthetic manifest tests.

# Critique Disposition

Risk class: medium

Critique policy: recommended

Policy rationale: Output-mode detection affects YAML shape and future dbt compatibility.

Required critique profiles: code-change, test-coverage

Findings: None - no critique yet.

Disposition status: pending

Deferral / not-required rationale: None.

# Retrospective / Promotion Disposition

Disposition status: pending

Promoted: None - implementation not complete.

Deferred / not-required rationale: Consider wiki promotion only if Fusion/future manifest behavior remains a recurring operator concern.

# Wiki Disposition

N/A - no wiki promotion selected yet.

# Acceptance Decision

Accepted by: Not accepted yet.
Accepted at: N/A.
Basis: Pending tests.
Residual risks: Fusion metadata may be unavailable or unstable.

# Dependencies

None.

# Journal

- 2026-05-03T21:10:43Z: Created from compatibility and core architecture oracle findings.
