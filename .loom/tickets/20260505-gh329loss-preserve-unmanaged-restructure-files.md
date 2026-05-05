---
id: ticket:gh329loss
kind: ticket
status: ready
change_class: code-behavior
risk_class: high
created_at: 2026-05-05T06:02:19Z
updated_at: 2026-05-05T06:02:19Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  initiative:
    - initiative:issue-pr-zero
external_refs:
  github_issue: https://github.com/z3z1ma/dbt-osmosis/issues/329
  related_github_issue: https://github.com/z3z1ma/dbt-osmosis/issues/306
depends_on: []
---

# Summary

Prevent `yaml refactor` restructure cleanup from deleting schema YAML files that still contain unmanaged top-level content such as `databricks-tags` or `semantic_models` after the last managed model/source/seed is moved out.

# Context

Issue #329 reports loss of Databricks/custom metadata, and issue #306 reports `semantic_model` definitions being removed. Current main preserves unmanaged top-level sections during same-path read/write cycles and has tests for `semantic_models` and arbitrary unknown top-level keys. Oracle triage found a remaining restructure edge: `apply_restructure_plan()` reads superseded files through filtered YAML, removes managed entries, then may unlink the old file based only on filtered managed content. That can lose root-level unmanaged sections when refactor moves the last managed entry out of a file.

# Why Now

Silent YAML data loss is more harmful than visible command failure. The previous hardening initiative improved same-path preservation; this ticket closes the remaining file-deletion path before closing user-reported preservation issues.

# Scope

- Make superseded-file cleanup consider original unmanaged top-level sections before deleting a schema YAML file.
- Preserve root-level unmanaged sections such as `semantic_models`, `macros`, `databricks-tags`, and unknown future dbt sections when managed entries are restructured away.
- Keep safe deletion behavior for truly empty superseded files with no unmanaged content.
- Add regression tests for restructure/refactor movement, not only same-path writer merges.

# Out Of Scope

- Databricks-specific tag generation or validation.
- Configurable model-level preserve lists for keys inside a managed model entry.
- Metadata inheritance filtering, which is owned by ticket:gh333meta.
- Rewriting the parser/writer preservation model beyond the superseded-file cleanup edge.

# Acceptance Criteria

- ACC-001: When refactor moves the last managed model/source/seed out of a schema file that contains root `semantic_models`, the old file is not deleted or the unmanaged section is otherwise preserved without loss.
- ACC-002: The same preservation holds for an arbitrary root key such as `databricks-tags`.
- ACC-003: Superseded schema files with no managed entries and no unmanaged top-level content are still removed when appropriate.
- ACC-004: Existing same-path read/write preservation tests for unmanaged top-level sections still pass.
- ACC-005: Dry-run/check behavior reports truthful mutations without deleting or stale-caching unmanaged sections.

# Coverage

Covers:

- ticket:gh329loss#ACC-001
- ticket:gh329loss#ACC-002
- ticket:gh329loss#ACC-003
- ticket:gh329loss#ACC-004
- ticket:gh329loss#ACC-005
- initiative:issue-pr-zero#OBJ-001
- initiative:issue-pr-zero#OBJ-002
- initiative:issue-pr-zero#OBJ-005

# Claim Matrix

| Claim | Evidence | Critique | Status |
| --- | --- | --- | --- |
| ticket:gh329loss#ACC-001 | None yet | None yet | open |
| ticket:gh329loss#ACC-002 | None yet | None yet | open |
| ticket:gh329loss#ACC-003 | None yet | None yet | open |
| ticket:gh329loss#ACC-004 | None yet | None yet | open |
| ticket:gh329loss#ACC-005 | None yet | None yet | open |

# Execution Notes

Current preservation evidence is not enough to close issues #329 or #306 because it covers same-path read/write behavior, not restructure deletion of superseded source files. The likely implementation path is in `src/dbt_osmosis/core/restructuring.py` around superseded path cleanup and should continue using schema reader/writer cache discipline.

# Blockers

None.

# Evidence

Expected evidence: a failing restructure regression that loses `semantic_models` or `databricks-tags`, a green focused test after the fix, existing schema preservation tests, and remote CI before closure.

# Critique Disposition

Risk class: high

Critique policy: mandatory

Policy rationale: This ticket addresses a silent data-loss path in schema YAML rewriting.

Required critique profiles:

- yaml-data-loss
- cache-and-dry-run

Findings:

None - no critique yet.

Disposition status: pending

Deferral / not-required rationale: N/A.

# Retrospective / Promotion Disposition

Disposition status: pending

Promoted:

None yet.

Deferred / not-required rationale: N/A.

# Wiki Disposition

N/A - no wiki promotion selected yet.

# Acceptance Decision

Accepted by: Pending implementation, mandatory critique, and evidence.
Accepted at: N/A.
Basis: N/A.
Residual risks: N/A.

# Dependencies

None.

# Journal

- 2026-05-05T06:02:19Z: Created from GitHub issues #329 and #306 after Oracle identified a remaining restructure deletion edge despite current same-path preservation fixes.
