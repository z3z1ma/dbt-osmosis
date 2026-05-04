---
id: ticket:c10bounds29
kind: ticket
status: complete_pending_acceptance
change_class: release-packaging
risk_class: high
created_at: 2026-05-03T21:10:43Z
updated_at: 2026-05-04T23:45:07Z
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
    - evidence:c10bounds29-support-policy-validation
  critique:
    - critique:c10bounds29-support-policy-review
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
| ticket:c10bounds29#ACC-001 | evidence:c10bounds29-support-policy-validation; evidence:oracle-backlog-scan; research:dbt-110-111-api-surfaces | critique:c10bounds29-support-policy-review | supported |
| ticket:c10bounds29#ACC-002 | evidence:c10bounds29-support-policy-validation | critique:c10bounds29-support-policy-review#FIND-002 accepted_risk | supported with accepted low risk |
| ticket:c10bounds29#ACC-003 | evidence:c10bounds29-support-policy-validation | critique:c10bounds29-support-policy-review#FIND-001 accepted_risk | supported with accepted low risk |
| ticket:c10bounds29#ACC-004 | evidence:c10bounds29-support-policy-validation; evidence:oracle-backlog-scan | critique:c10bounds29-support-policy-review | supported |
| ticket:c10bounds29#ACC-005 | evidence:c10bounds29-support-policy-validation | critique:c10bounds29-support-policy-review | supported |

# Execution Notes

This is partly a policy decision. The operator selected the keep-open-plus-canary policy on 2026-05-04: keep `dbt-core>=1.8` package metadata open, add explicit future-dbt canary coverage, and document future minors as canary-only until audited.

# Blockers

None. The support-range decision was resolved as keep open plus canary.

# Evidence

Existing evidence: evidence:oracle-backlog-scan, research:dbt-110-111-api-surfaces, and evidence:c10bounds29-support-policy-validation.

Evidence status: local test-first validation, full pre-commit, focused package metadata tests, uv lock check, and mandatory critique support ACC-001 through ACC-005 for the current source state. Missing evidence: remote CI after commit/push.

# Critique Disposition

Risk class: high

Critique policy: mandatory

Policy rationale: Package constraints and warning policy directly shape user compatibility expectations.

Required critique profiles: release-packaging, operator-clarity, dbt-compatibility

Findings:

- critique:c10bounds29-support-policy-review#FIND-001: accepted_risk. The canary is an unpinned resolver visibility signal, not proof that every future dbt minor is exercised once released. Adapter constraints may keep the resolver on an audited line until a compatible adapter exists; the docs state adapter compatibility remains the user's responsibility. No pre-acceptance code change required.
- critique:c10bounds29-support-policy-review#FIND-002: accepted_risk. Docs state audited support across the dbt Core support range, but do not enumerate every Python/dbt matrix exclusion. The current CI explicitly excludes Python 3.13 for dbt 1.8/1.9, and the residual ambiguity is low enough to accept for this ticket.

Disposition status: completed

Review: critique:c10bounds29-support-policy-review

Acceptance recommendation: risk-disposition-needed; low findings dispositioned as accepted risk.

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
- 2026-05-04T23:32:10Z: Operator chose the keep-open-plus-canary policy for package metadata; started Ralph iteration 01 to add future-dbt canary coverage, warning visibility, docs, and structural tests without adding a `dbt-core` upper bound.
- 2026-05-04T23:38:36Z: Ralph iteration 01 returned stop. Parent accepted the implementation after diff review, package metadata tests, whitespace validation, and touched-file pre-commit; moved ticket to review_required for mandatory release-packaging/operator-clarity/dbt-compatibility critique.
- 2026-05-04T23:43:46Z: Mandatory critique returned pass_with_findings with two low findings; parent accepted both low risks in ticket-owned critique disposition and moved ticket to complete_pending_acceptance pending final validation, commit, push, and remote CI evidence.
- 2026-05-04T23:45:07Z: Full pre-commit, package metadata tests, and uv lock check passed; ticket remains complete_pending_acceptance pending commit, push, and remote CI evidence.
