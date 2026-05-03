---
id: ticket:c10ord18
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
| ticket:c10ord18#ACC-001 | evidence:oracle-backlog-scan | None | open |
| ticket:c10ord18#ACC-005 | None - deterministic output test not written yet | None | open |

# Execution Notes

This should be a small fix. Avoid broad graph abstractions unless tests show the current structure cannot support the correct order.

# Blockers

None.

# Evidence

Existing evidence: evidence:oracle-backlog-scan. Missing evidence: targeted tests.

# Critique Disposition

Risk class: medium

Critique policy: recommended

Policy rationale: Small code change, but inheritance precedence can affect user docs.

Required critique profiles: code-change, test-coverage

Findings: None - no critique yet.

Disposition status: pending

Deferral / not-required rationale: None.

# Retrospective / Promotion Disposition

Disposition status: pending

Promoted: None - implementation not complete.

Deferred / not-required rationale: Likely not required unless the fix reveals a broader inheritance concept.

# Wiki Disposition

N/A - no wiki promotion selected yet.

# Acceptance Decision

Accepted by: Not accepted yet.
Accepted at: N/A.
Basis: Pending tests.
Residual risks: Deep DAG precedence may need additional fixture coverage.

# Dependencies

None.

# Journal

- 2026-05-03T21:10:43Z: Created from core architecture oracle finding.
