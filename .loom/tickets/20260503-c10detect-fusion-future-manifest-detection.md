---
id: ticket:c10detect
kind: ticket
status: closed
change_class: code-behavior
risk_class: medium
created_at: 2026-05-03T21:10:43Z
updated_at: 2026-05-04T20:38:29Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  initiative:
    - initiative:dbt-110-111-hardening
  evidence:
    - evidence:oracle-backlog-scan
    - evidence:c10detect-fusion-manifest-detection-validation
    - evidence:c10detect-main-ci-success
  critique:
    - critique:c10detect-fusion-manifest-detection-review
  packets:
    - packet:ralph-ticket-c10detect-20260504T200316Z
external_refs:
  github_issue: https://github.com/z3z1ma/dbt-osmosis/issues/356
  github_issue_comment: https://github.com/z3z1ma/dbt-osmosis/issues/356#issuecomment-4374318745
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
| ticket:c10detect#ACC-001 | evidence:c10detect-fusion-manifest-detection-validation; evidence:c10detect-main-ci-success | critique:c10detect-fusion-manifest-detection-review | accepted |
| ticket:c10detect#ACC-002 | evidence:oracle-backlog-scan; evidence:c10detect-fusion-manifest-detection-validation; evidence:c10detect-main-ci-success | critique:c10detect-fusion-manifest-detection-review | accepted |
| ticket:c10detect#ACC-003 | evidence:c10detect-fusion-manifest-detection-validation; evidence:c10detect-main-ci-success | critique:c10detect-fusion-manifest-detection-review | accepted |
| ticket:c10detect#ACC-004 | evidence:c10detect-fusion-manifest-detection-validation; evidence:c10detect-main-ci-success | critique:c10detect-fusion-manifest-detection-review | accepted |
| ticket:c10detect#ACC-005 | evidence:c10detect-fusion-manifest-detection-validation; evidence:c10detect-main-ci-success | critique:c10detect-fusion-manifest-detection-review | accepted |

# Execution Notes

Search current tests around `_detect_fusion_manifest()` before changing behavior. If a compatibility override already exists in settings, prefer documenting that over adding another flag.

# Blockers

None.

# Evidence

Existing evidence: evidence:oracle-backlog-scan. Validation evidence: evidence:c10detect-fusion-manifest-detection-validation and evidence:c10detect-main-ci-success. Missing evidence: none for this ticket's closure gate.

# Critique Disposition

Risk class: medium

Critique policy: recommended

Policy rationale: Output-mode detection affects YAML shape and future dbt compatibility.

Required critique profiles: code-change, test-coverage

Findings: critique:c10detect-fusion-manifest-detection-review reported pass with no findings.

Disposition status: completed

Deferral / not-required rationale: N/A - critique completed.

# Retrospective / Promotion Disposition

Disposition status: completed

Promoted: User-facing CLI help and docs now state known Fusion manifest evidence, not generic `> v12` schema versions or PATH binaries.

Deferred / not-required rationale: No separate wiki/research/spec promotion is needed; the durable operator-facing explanation belongs in the CLI help and docs pages updated by this ticket.

# Wiki Disposition

Not required - the accepted explanation for this user-facing behavior is in CLI help plus `docs/docs/reference/cli.md`, `docs/docs/reference/settings.md`, and `docs/docs/tutorial-yaml/configuration.md`.

# Acceptance Decision

Accepted by: OpenCode
Accepted at: 2026-05-04T20:38:29Z.
Basis: `evidence:c10detect-fusion-manifest-detection-validation` records the red/green test-first pass, parent focused and broader local validation, Ruff, basedpyright `errorCount: 0`, and full local pre-commit. `critique:c10detect-fusion-manifest-detection-review` records a final pass verdict with no findings. `evidence:c10detect-main-ci-success` records successful main-branch Labeler, lint, Tests, and Release validation for commit `2df9ba5f4353716a6051760affc2499044e6b54d`. GitHub issue #356 was commented and closed.
Residual risks: Detection is intentionally allowlist-based, so future Fusion schema versions require a code update or explicit `--fusion-compat`; detection still depends on `dbt_schema_version` appearing near the start of `target/manifest.json`; this ticket does not claim full Fusion support.

# Dependencies

None.

# Journal

- 2026-05-03T21:10:43Z: Created from compatibility and core architecture oracle findings.
- 2026-05-04T20:03:16Z: Started Ralph iteration 01 to make manifest detection fail closed for generic future dbt-core schema versions while preserving known Fusion evidence and user override clarity.
- 2026-05-04T20:08:59Z: Ralph iteration 01 returned stop. Parent recorded evidence:c10detect-fusion-manifest-detection-validation and moved ticket to review_required for recommended critique.
- 2026-05-04T20:13:20Z: Critique passed with no findings in critique:c10detect-fusion-manifest-detection-review. Local pre-commit passed; ticket moved to complete_pending_acceptance pending remote CI and final acceptance.
- 2026-05-04T20:38:29Z: Pushed commit `2df9ba5f4353716a6051760affc2499044e6b54d` to `origin/main`, observed successful Labeler `25341191692`, lint `25341191710`, Tests `25341191679`, and Release `25341712789` workflows, recorded `evidence:c10detect-main-ci-success`, commented on GitHub issue #356, closed the issue, and closed ticket.
