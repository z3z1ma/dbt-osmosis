---
id: ticket:c10loom04
kind: ticket
status: closed
change_class: code-behavior
risk_class: medium
created_at: 2026-05-03T21:10:43Z
updated_at: 2026-05-04T21:47:54Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  initiative:
    - initiative:dbt-110-111-hardening
  research:
    - research:dbt-110-111-api-surfaces
  evidence:
    - evidence:oracle-backlog-scan
    - evidence:c10loom04-dbt-loom-parser-validation
    - evidence:c10loom04-main-ci-success
  critique:
    - critique:c10loom04-dbt-loom-parser-review
external_refs:
  github_issue: https://github.com/z3z1ma/dbt-osmosis/issues/355
  dbt_core_110_model_parser: https://raw.githubusercontent.com/dbt-labs/dbt-core/v1.10.0/core/dbt/parser/models.py
  dbt_core_111_model_parser: https://raw.githubusercontent.com/dbt-labs/dbt-core/v1.11.0/core/dbt/parser/models.py
depends_on: []
---

# Summary

Replace the invalid dbt-loom cross-project node parsing path that calls `ModelParser.parse_from_dict(None, node)` and avoid relying on read-only/private manifest setters.

# Context

`src/dbt_osmosis/core/config.py:498-541` loads dbt-loom manifests and calls `ModelParser.parse_from_dict(None, node)` at line 529. dbt-core v1.10.0 and v1.11.0 define `parse_from_dict(self, dct, validate=True)`, an instance method. `src/dbt_osmosis/core/config.py:424-432` also mutates an internal project manifest attribute through `object.__setattr__`.

# Why Now

The current implementation is private API coupling in a compatibility-sensitive path. Cross-project references can silently fail or warn, and the code hides whether dbt-loom integration is truly working under dbt 1.10/1.11.

# Scope

- Replace the parser hack with a valid construction path such as `ModelNode.from_dict(...)` or a properly constructed parser if validation requires parser context.
- Make manifest mutation explicit and avoid assigning through read-only/private project internals unless there is no safe alternative.
- Add a regression test with a minimal dbt-loom manifest containing public/protected models.
- Ensure failures clearly log which dbt-loom node could not be imported without breaking projects that do not use dbt-loom.

# Out Of Scope

- Full dbt-loom feature redesign.
- Adding dbt-loom as a required dependency.
- Changing access semantics beyond preserving the intended public/protected filtering.

# Acceptance Criteria

- ACC-001: Cross-project model nodes are parsed through a valid dbt 1.10/1.11-compatible path.
- ACC-002: The code no longer calls `ModelParser.parse_from_dict(None, ...)`.
- ACC-003: Manifest mutation is done through a clear supported boundary or a narrowly documented compatibility shim.
- ACC-004: A test mocks or fixtures dbt-loom manifests and verifies eligible nodes are added to the manifest.
- ACC-005: dbt-loom load failure remains non-fatal with clear warnings.

# Coverage

Covers:

- ticket:c10loom04#ACC-001
- ticket:c10loom04#ACC-002
- ticket:c10loom04#ACC-003
- ticket:c10loom04#ACC-004
- ticket:c10loom04#ACC-005
- initiative:dbt-110-111-hardening#OBJ-001
- initiative:dbt-110-111-hardening#OBJ-007

# Claim Matrix

| Claim | Evidence | Critique | Status |
| --- | --- | --- | --- |
| ticket:c10loom04#ACC-001 | evidence:c10loom04-dbt-loom-parser-validation | critique:c10loom04-dbt-loom-parser-review | supported |
| ticket:c10loom04#ACC-002 | evidence:c10loom04-dbt-loom-parser-validation; research:dbt-110-111-api-surfaces | critique:c10loom04-dbt-loom-parser-review | supported |
| ticket:c10loom04#ACC-003 | evidence:c10loom04-dbt-loom-parser-validation | critique:c10loom04-dbt-loom-parser-review | supported |
| ticket:c10loom04#ACC-004 | evidence:c10loom04-dbt-loom-parser-validation | critique:c10loom04-dbt-loom-parser-review | supported |
| ticket:c10loom04#ACC-005 | evidence:c10loom04-dbt-loom-parser-validation | critique:c10loom04-dbt-loom-parser-review | supported |

# Execution Notes

Start by checking the exact node dict shape provided by dbt-loom and whether `ModelNode.from_dict` validates it. Keep the optional dependency boundary intact so users without dbt-loom are unaffected.

# Blockers

None.

# Evidence

Existing evidence: research:dbt-110-111-api-surfaces, evidence:oracle-backlog-scan, and evidence:c10loom04-dbt-loom-parser-validation.

Evidence status: local red/green, parent validation, full pre-commit, final critique, and green remote CI support ACC-001 through ACC-005 for commit `8d47587d5485dadab67bc39008aea8f73c159241`.

# Critique Disposition

Risk class: medium

Critique policy: recommended

Policy rationale: Optional integration path but it touches manifest construction and private dbt APIs.

Required critique profiles: code-change, dbt-compatibility

Findings: None.

Disposition status: completed.

Review: critique:c10loom04-dbt-loom-parser-review

Acceptance recommendation: no-critique-blockers.

Deferral / not-required rationale: None.

# Retrospective / Promotion Disposition

Disposition status: not_required

Promoted: None.

Deferred / not-required rationale: The durable lesson is local to the c10loom04 compatibility boundary and is documented directly in `_set_project_manifest()` plus the ticket, evidence, and critique records. No reusable wiki, research, spec, plan, initiative, or constitution update is needed.

# Wiki Disposition

N/A - no wiki promotion selected yet.

# Acceptance Decision

Accepted by: OpenCode parent acceptance gate.
Accepted at: 2026-05-04T21:47:54Z.
Basis: Implementation commit `8d47587d5485dadab67bc39008aea8f73c159241`, evidence:c10loom04-dbt-loom-parser-validation, critique:c10loom04-dbt-loom-parser-review, and evidence:c10loom04-main-ci-success.
Residual risks: Real dbt-loom package integration was not run; dbt-loom may change manifest shape; `_set_project_manifest()` remains a narrow private compatibility shim when no public setter is available.

# Dependencies

None.

# Journal

- 2026-05-03T21:10:43Z: Created from dbt compatibility oracle finding.
- 2026-05-04T21:08:14Z: Started Ralph iteration 01 to replace the dbt-loom parser hack with a valid node construction path and focused regression coverage.
- 2026-05-04T21:14:22Z: Ralph iteration 01 returned stop. Parent accepted the implementation iteration after focused tests, Ruff, and basedpyright zero-error validation; moved ticket to review_required for recommended critique.
- 2026-05-04T21:20:35Z: Recommended critique completed with pass/no findings. Moved ticket to complete_pending_acceptance pending final validation, commit, push, and remote CI evidence.
- 2026-05-04T21:22:06Z: Full pre-commit and focused config tests passed after formatting; ticket remains complete_pending_acceptance pending commit, push, and remote CI evidence.
- 2026-05-04T21:47:54Z: Commit `8d47587d5485dadab67bc39008aea8f73c159241` reached green Labeler, lint, Tests, and Release workflows on `origin/main`; accepted and closed ticket.
