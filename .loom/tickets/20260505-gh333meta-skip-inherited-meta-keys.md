---
id: ticket:gh333meta
kind: ticket
status: ready
change_class: code-behavior
risk_class: medium
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
  github_issue: https://github.com/z3z1ma/dbt-osmosis/issues/333
depends_on: []
---

# Summary

Add a granular setting that prevents selected column `meta` keys from being inherited across dbt lineage while preserving normal inheritance for all other metadata.

# Context

Issue #333 reports that dbt-external-tables and dbt 1.11/Fusion-compatible project structures place package-specific data under `config.meta`. dbt-osmosis currently exposes `config.meta` as effective `meta` and inherits `meta` wholesale unless `skip-merge-meta` disables all meta inheritance. That makes keys such as `expression` and `doc_blocks` flow down the DAG even though they only make sense on the external table, while users still need other metadata such as GDPR and security-classification keys to inherit.

# Why Now

The recent dbt 1.10/1.11 hardening work made `config.meta` a first-class compatibility surface. That increases the impact of package-specific metadata being inherited too broadly.

# Scope

- Add a config setting and CLI option for skipped inherited meta keys, using SettingsResolver-compatible kebab/snake lookup.
- Filter inherited column `meta` for classic top-level `meta` and Fusion/dbt 1.10+ `config.meta` paths.
- Preserve all meta keys not listed in the skip setting.
- Preserve local child meta keys when inheritance runs.
- Keep `skip-merge-meta` as the existing all-meta opt-out.

# Out Of Scope

- Hardcoding dbt-external-tables or other package-specific keys as defaults.
- Skipping arbitrary non-meta fields.
- Changing default metadata inheritance behavior when the new setting is absent.
- Removing or redefining `skip-merge-meta`.

# Acceptance Criteria

- ACC-001: With `skip-inheritance-for-meta-keys: [expression, doc_blocks]`, upstream `expression` and `doc_blocks` meta keys are not inherited into child columns.
- ACC-002: Other upstream meta keys, such as `gdpr` or `security_classification`, still inherit when meta inheritance is otherwise enabled.
- ACC-003: Existing child/local meta keys survive the inheritance pass.
- ACC-004: The skip behavior works for both classic column `meta` and dbt 1.10+/Fusion `config.meta` inputs.
- ACC-005: `skip-merge-meta` still disables all meta inheritance and is not weakened by the new granular skip setting.

# Coverage

Covers:

- ticket:gh333meta#ACC-001
- ticket:gh333meta#ACC-002
- ticket:gh333meta#ACC-003
- ticket:gh333meta#ACC-004
- ticket:gh333meta#ACC-005
- initiative:issue-pr-zero#OBJ-001
- initiative:issue-pr-zero#OBJ-002
- initiative:issue-pr-zero#OBJ-005

# Claim Matrix

| Claim | Evidence | Critique | Status |
| --- | --- | --- | --- |
| ticket:gh333meta#ACC-001 | None yet | None yet | open |
| ticket:gh333meta#ACC-002 | None yet | None yet | open |
| ticket:gh333meta#ACC-003 | None yet | None yet | open |
| ticket:gh333meta#ACC-004 | None yet | None yet | open |
| ticket:gh333meta#ACC-005 | None yet | None yet | open |

# Execution Notes

Oracle triage found `transforms.py` inheriting `meta` wholesale unless `skip-merge-meta` disables all meta, and `inheritance.py` exposing and merging `config.meta` through effective inherited `meta`. A Ralph implementation should keep the setting narrow and thread it through existing config-resolution paths rather than adding ad hoc lookups.

# Blockers

None.

# Evidence

Expected evidence: a failing regression test for skipped meta keys in classic and Fusion/config meta shapes, then passing focused inheritance/config tests. Remote CI evidence is needed before closing.

# Critique Disposition

Risk class: medium

Critique policy: recommended

Policy rationale: The change affects inheritance behavior and config precedence, but it is opt-in and bounded.

Required critique profiles:

- inheritance-behavior
- config-resolution

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

Accepted by: Pending implementation and evidence.
Accepted at: N/A.
Basis: N/A.
Residual risks: N/A.

# Dependencies

None.

# Journal

- 2026-05-05T06:02:19Z: Created from GitHub issue #333 and Oracle triage as a validated medium-risk metadata-inheritance feature ticket.
