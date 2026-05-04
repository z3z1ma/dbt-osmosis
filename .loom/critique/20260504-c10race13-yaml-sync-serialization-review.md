---
id: critique:c10race13-yaml-sync-serialization-review
kind: critique
status: final
created_at: 2026-05-04T07:51:02Z
updated_at: 2026-05-04T07:51:02Z
scope:
  kind: repository
  repositories:
    - repo:root
review_target: "ticket:c10race13 path-grouped YAML sync serialization diff from a30825f working tree"
links:
  tickets:
    - ticket:c10race13
  evidence:
    - evidence:c10race13-yaml-sync-serialization-verification
  packets:
    - packet:ralph-ticket-c10race13-20260504T071043Z
external_refs: {}
---

# Summary

Reviewed the `ticket:c10race13` implementation after Ralph and parent review-driven fixes. The review focused on YAML data preservation, same-target concurrency scheduling, path grouping, versioned model behavior, temp-file safety, permission preservation, and regression coverage.

# Review Target

Target: `ticket:c10race13` uncommitted working-tree diff on branch `loom/dbt-110-111-hardening`, based on commit `a30825fde582ba1886938a228958eeb68a221414`.

Reviewed changed surfaces:

- `src/dbt_osmosis/core/sync_operations.py`
- `src/dbt_osmosis/core/schema/writer.py`
- `tests/core/test_sync_operations.py`
- `packet:ralph-ticket-c10race13-20260504T071043Z`
- `evidence:c10race13-yaml-sync-serialization-verification`

# Verdict

`pass_with_low_residual_risks`.

No medium or high findings remain after the parent-side fixes for file mode preservation, process-global umask avoidance, deterministic same-path node ordering, and repeated threaded same-target sync survival coverage. The implementation is acceptable to commit, with post-commit CI still required before ticket closure.

# Findings

No open medium/high findings.

Low residual risks and testing gaps:

- The repeated same-target survival test uses `ThreadPoolExecutor(max_workers=2)`, but after path grouping there is only one same-target work item. This is acceptable because grouping is the fix, but the test proves grouped-section survival rather than active same-file write contention.
- Source same-path coverage uses source-shaped mocks because the current demo fixture has no parsed sources; it proves grouping but not a full real-source `_sync_source_node()` path.
- `commit_yamls()` relies on the same unique temp helper but lacks direct new temp-file coverage.
- Version ordering is deterministic but string-based, matching prior version string behavior rather than enforcing numeric semantic ordering.
- Cross-process final-write locking remains out of scope; separate processes can still race with last-writer-wins semantics.

# Evidence Reviewed

Reviewed the current diff, Ralph child output, parent review-driven patches, focused red/green output, full sync test output, related schema/path/sync test output, targeted pre-commit output, Ruff output, `git diff --check`, and final review output.

Key evidence record:

- evidence:c10race13-yaml-sync-serialization-verification

# Residual Risks

- Direct concurrent single-node calls from separate callers are not serialized by this grouping change.
- Separate processes can still race at final atomic replace if they target the same YAML file at the same time.
- Source sync grouping has grouping-level coverage but not real parsed source fixture coverage.

# Required Follow-up

Before closure, the ticket should record post-commit CI evidence for the implementation commit and consume this critique as completed in the ticket-owned acceptance dossier. Retrospective / promotion disposition should decide whether the YAML sync serialization rule belongs in wiki documentation.

# Acceptance Recommendation

`ticket-acceptance-review-needed`.
