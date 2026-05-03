---
id: ticket:c10bounds29
kind: ticket
status: ready
change_class: release-packaging
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
  research:
    - research:dbt-110-111-api-surfaces
  evidence:
    - evidence:oracle-backlog-scan
depends_on: []
---

# Summary

Define the actual dbt support range, package constraints, canary policy, and deprecation-warning policy so unsupported dbt minors do not resolve silently while warnings that predict breakage are hidden.

# Context

`pyproject.toml:27` allows `dbt-core>=1.8` with no upper bound even though the code imports private dbt internals. `pyproject.toml:114-116` ignores all `DeprecationWarning`, which can hide dbt migration warnings. CI should prove dbt 1.10.x and 1.11.x, but future dbt versions beyond the audited range should be explicit canaries or constrained.

# Why Now

The project must be compatible with dbt 1.10.x and 1.11.x for sure. That support claim should not accidentally imply support for dbt 1.12+ or hide warnings that signal upcoming breakage.

# Scope

- Decide whether published package metadata should constrain dbt to `<1.12` until future versions are audited.
- Add latest-dbt canary jobs if keeping an open upper bound.
- Narrow warning filters so dbt-osmosis deprecations and relevant dbt deprecations remain visible.
- Add tests or CI flags that surface dbt deprecations in compatibility jobs.
- Update README/docs support statements to match package metadata and CI.

# Out Of Scope

- Dropping support for currently supported dbt versions without a separate decision.
- Fixing every deprecation warning surfaced by this ticket unless needed to make the policy pass.

# Acceptance Criteria

- ACC-001: Package dbt dependency constraints match the dbt versions the project actually supports or clearly intentionally canaries.
- ACC-002: Documentation states the supported dbt-core and adapter version policy.
- ACC-003: CI has a latest-dbt canary if the package allows unreviewed future dbt minors.
- ACC-004: Blanket `ignore::DeprecationWarning` is replaced with targeted ignores or a focused warning policy.
- ACC-005: Compatibility jobs surface dbt deprecations relevant to dbt 1.10/1.11 and future migration risk.

# Coverage

Covers:

- ticket:c10bounds29#ACC-001
- ticket:c10bounds29#ACC-002
- ticket:c10bounds29#ACC-003
- ticket:c10bounds29#ACC-004
- ticket:c10bounds29#ACC-005
- initiative:dbt-110-111-hardening#OBJ-001
- initiative:dbt-110-111-hardening#OBJ-005

# Claim Matrix

| Claim | Evidence | Critique | Status |
| --- | --- | --- | --- |
| ticket:c10bounds29#ACC-001 | evidence:oracle-backlog-scan, research:dbt-110-111-api-surfaces | None | open |
| ticket:c10bounds29#ACC-004 | evidence:oracle-backlog-scan | None | open |

# Execution Notes

This is partly a policy decision. If adding an upper bound is considered breaking or too restrictive, route to the user before changing package metadata.

# Blockers

Potential human decision on whether to constrain `dbt-core` upper bound or keep open with canaries.

# Evidence

Existing evidence: evidence:oracle-backlog-scan and research:dbt-110-111-api-surfaces. Missing evidence: CI/docs/package diff and warning output.

# Critique Disposition

Risk class: high

Critique policy: mandatory

Policy rationale: Package constraints and warning policy directly shape user compatibility expectations.

Required critique profiles: release-packaging, operator-clarity, dbt-compatibility

Findings: None - no critique yet.

Disposition status: pending

Deferral / not-required rationale: None.

# Retrospective / Promotion Disposition

Disposition status: pending

Promoted: None - implementation not complete.

Deferred / not-required rationale: Docs/wiki update likely needed after support policy is accepted.

# Wiki Disposition

N/A - no wiki promotion selected yet.

# Acceptance Decision

Accepted by: Not accepted yet.
Accepted at: N/A.
Basis: Pending support decision and CI evidence.
Residual risks: Constraining too tightly can block users; constraining too loosely can admit broken future dbt versions.

# Dependencies

Coordinate with ticket:c10ci06 and ticket:c10lock07.

# Journal

- 2026-05-03T21:10:43Z: Created from dbt compatibility and CI/build oracle findings.
