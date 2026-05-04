---
id: ticket:c10ord18
kind: ticket
status: closed
change_class: code-behavior
risk_class: medium
created_at: 2026-05-03T21:10:43Z
updated_at: 2026-05-04T12:42:05Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  initiative:
    - initiative:dbt-110-111-hardening
  evidence:
    - evidence:oracle-backlog-scan
    - evidence:c10ord18-inheritance-ordering-validation
  packets:
    - packet:ralph-ticket-c10ord18-20260504T122819Z
  critique:
    - critique:c10ord18-inheritance-ordering-review
depends_on: []
---

# Summary

Fix brittle inheritance graph cycle/depth ordering and deterministic tag merging so documentation inheritance produces stable, correct YAML.

# Context

`src/dbt_osmosis/core/inheritance.py:141` initializes `visited = set(node.unique_id)`, which creates a set of characters rather than `{node.unique_id}`. Generation keys are sorted lexicographically, so `generation_10` can sort before `generation_2`. Tag merging in `inheritance.py:390-393`, `424-427`, and `transforms.py:900-906` uses sets, producing nondeterministic output ordering.

# Why Now

Inheritance is a core feature and dbt version matrix runs can make nondeterministic YAML diffs more visible. Ordering bugs can change which upstream docs win in deep DAGs.

# Scope

- Initialize cycle-detection visited sets with full node IDs.
- Sort generation keys numerically.
- Preserve existing tag order and append unseen inherited tags deterministically.
- Add tests for cycles, depth greater than 10, and repeated tag output stability.

# Out Of Scope

- Redesigning the inheritance algorithm.
- Changing semantic tag rules beyond deterministic ordering.

# Acceptance Criteria

- ACC-001: The root node is marked visited by full unique ID before recursion.
- ACC-002: A cycle such as A -> B -> A does not re-add A as a later generation.
- ACC-003: Generation depth ordering is numeric, so generation 2 precedes generation 10.
- ACC-004: Tag merging preserves local order and appends inherited unseen tags in deterministic upstream order.
- ACC-005: Repeated runs over the same inputs produce byte-identical tag order.

# Coverage

Covers:

- ticket:c10ord18#ACC-001
- ticket:c10ord18#ACC-002
- ticket:c10ord18#ACC-003
- ticket:c10ord18#ACC-004
- ticket:c10ord18#ACC-005
- initiative:dbt-110-111-hardening#OBJ-003

# Claim Matrix

| Claim | Evidence | Critique | Status |
| --- | --- | --- | --- |
| ticket:c10ord18#ACC-001 | evidence:c10ord18-inheritance-ordering-validation | critique:c10ord18-inheritance-ordering-review | accepted |
| ticket:c10ord18#ACC-002 | evidence:c10ord18-inheritance-ordering-validation | critique:c10ord18-inheritance-ordering-review | accepted |
| ticket:c10ord18#ACC-003 | evidence:c10ord18-inheritance-ordering-validation | critique:c10ord18-inheritance-ordering-review | accepted |
| ticket:c10ord18#ACC-004 | evidence:c10ord18-inheritance-ordering-validation | critique:c10ord18-inheritance-ordering-review | accepted |
| ticket:c10ord18#ACC-005 | evidence:c10ord18-inheritance-ordering-validation | critique:c10ord18-inheritance-ordering-review | accepted |

# Execution Notes

This should be a small fix. Avoid broad graph abstractions unless tests show the current structure cannot support the correct order.

# Blockers

None.

# Evidence

Evidence recorded:

- evidence:oracle-backlog-scan
- evidence:c10ord18-inheritance-ordering-validation

# Critique Disposition

Risk class: medium

Critique policy: recommended

Policy rationale: Small code change, but inheritance precedence can affect user docs.

Required critique profiles: code-change, test-coverage

Findings: critique:c10ord18-inheritance-ordering-review reported no findings.

Disposition status: completed

Deferral / not-required rationale: N/A - critique completed with no blockers.

# Retrospective / Promotion Disposition

Disposition status: not_required

Promoted: None.

Deferred / not-required rationale: The fix was a localized deterministic ordering correction and did not reveal a broader inheritance concept needing wiki promotion.

# Wiki Disposition

N/A - no wiki promotion selected; localized behavior is covered by the ticket, evidence, critique, and tests.

# Acceptance Decision

Accepted by: OpenCode parent agent.
Accepted at: 2026-05-04T12:42:05Z.
Basis: implementation commit `96a43cbac9947e4a189163329bfd0febadf77ea1`, evidence:c10ord18-inheritance-ordering-validation, and critique:c10ord18-inheritance-ordering-review.
Residual risks: Full repository suite and GitHub Actions matrix are deferred to final initiative validation; ACC-005 is supported by ordered in-memory assertions and removal of set-based merges rather than a repeated byte-for-byte YAML write fixture; deep DAG duplicate-ancestor precedence remains governed by existing global visited behavior and is out of scope.

# Dependencies

None.

# Journal

- 2026-05-03T21:10:43Z: Created from core architecture oracle finding.
- 2026-05-04T12:28:19Z: Activated ticket and compiled Ralph packet `packet:ralph-ticket-c10ord18-20260504T122819Z` for test-first inheritance ordering, cycle detection, and deterministic tag merge fixes with local-only validation.
- 2026-05-04T12:42:05Z: Ralph iteration consumed. Implementation commit `96a43cbac9947e4a189163329bfd0febadf77ea1` fixed full-ID cycle detection, numeric generation processing, and order-preserving tag merges. Local validation passed with `70 passed`; final critique found no blockers. Accepted and closed with final initiative-level CI deferred.
