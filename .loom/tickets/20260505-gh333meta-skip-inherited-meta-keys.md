---
id: ticket:gh333meta
kind: ticket
status: closed
change_class: code-behavior
risk_class: medium
created_at: 2026-05-05T06:02:19Z
updated_at: 2026-05-05T08:07:56Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  initiative:
    - initiative:issue-pr-zero
  evidence:
    - evidence:gh333meta-skip-inherited-meta-validation
  critique:
    - critique:gh333meta-skip-inherited-meta-review
  packets:
    - packet:ralph-ticket-gh333meta-20260505T070046Z
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
| ticket:gh333meta#ACC-001 | evidence:gh333meta-skip-inherited-meta-validation | critique:gh333meta-skip-inherited-meta-review | accepted |
| ticket:gh333meta#ACC-002 | evidence:gh333meta-skip-inherited-meta-validation | critique:gh333meta-skip-inherited-meta-review | accepted |
| ticket:gh333meta#ACC-003 | evidence:gh333meta-skip-inherited-meta-validation | critique:gh333meta-skip-inherited-meta-review | accepted |
| ticket:gh333meta#ACC-004 | evidence:gh333meta-skip-inherited-meta-validation | critique:gh333meta-skip-inherited-meta-review | accepted |
| ticket:gh333meta#ACC-005 | evidence:gh333meta-skip-inherited-meta-validation | critique:gh333meta-skip-inherited-meta-review | accepted |

# Execution Notes

Ralph added `skip-inheritance-for-meta-keys`, wired it into CLI/settings, and filters configured keys from ancestor graph edges before merging so local child meta is preserved.

# Blockers

None.

# Evidence

Evidence status: local red/green Ralph evidence, parent inheritance/settings pytest, Ruff, format check, CLI help observation, and whitespace check support ACC-001 through ACC-005 for the uncommitted implementation diff. Remote CI will be checked at the initiative level after the full batch push per operator direction.

# Critique Disposition

Risk class: medium

Critique policy: recommended

Policy rationale: The change affects inheritance behavior and config precedence, but it is opt-in and bounded.

Required critique profiles:

- inheritance-behavior
- config-resolution

Findings:

None - critique:gh333meta-skip-inherited-meta-review returned `pass` with no findings.

Disposition status: completed

Deferral / not-required rationale: N/A.

# Retrospective / Promotion Disposition

Disposition status: completed

Promoted:

None - retrospective found no durable explanation needing wiki/research/spec promotion beyond this ticket, evidence, and critique.

Deferred / not-required rationale: New option behavior is captured in the ticket and tests; broader docs can be handled during release documentation if desired.

# Wiki Disposition

N/A - no wiki promotion selected yet.

# Acceptance Decision

Accepted by: OpenCode parent agent.
Accepted at: 2026-05-05T08:07:56Z.
Basis: Local implementation evidence, focused validation, and Oracle critique support ACC-001 through ACC-005. The ticket is ready for issue closure with remote CI deferred to the issue-backlog initiative gate per operator direction.
Residual risks: Remote CI not yet checked; no dedicated CLI invocation test was added beyond parent help observation.

# Dependencies

None.

# Journal

- 2026-05-05T06:02:19Z: Created from GitHub issue #333 and Oracle triage as a validated medium-risk metadata-inheritance feature ticket.
- 2026-05-05T07:19:30Z: Ralph implemented the inherited meta-key skip option. Parent validation passed and Oracle critique accepted with no findings. Retrospective completed with no promotion needed beyond ticket/evidence/critique records. Moved to complete_pending_acceptance pending final implementation commit packaging.
- 2026-05-05T08:07:56Z: Accepted and closed locally for combined inheritance-options packaging with ticket:gh326skip. GitHub issue #333 is ready for commit, push, comment, and closure.
