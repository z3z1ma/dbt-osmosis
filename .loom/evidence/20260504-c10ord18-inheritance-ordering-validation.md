---
id: evidence:c10ord18-inheritance-ordering-validation
kind: evidence
status: recorded
created_at: 2026-05-04T12:42:05Z
updated_at: 2026-05-04T12:42:05Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  tickets:
    - ticket:c10ord18
  packets:
    - packet:ralph-ticket-c10ord18-20260504T122819Z
  critique:
    - critique:c10ord18-inheritance-ordering-review
external_refs: {}
---

# Summary

Observed test-first local verification for `ticket:c10ord18` inheritance cycle detection, numeric generation ordering, and deterministic tag merging. The evidence records red/green behavior, parent validation, hooks, post-commit checks, and critique; it does not decide ticket closure by itself.

# Procedure

Observed at: 2026-05-04T12:42:05Z
Source state: branch `loom/dbt-110-111-hardening`, implementation commit `96a43cbac9947e4a189163329bfd0febadf77ea1`, based on prior main commit `4c4178b68d0724ce38cc23c263c18cb9613cc861`.
Procedure: Reviewed Ralph child output, inspected implementation diff, reran focused c10ord18 regressions, reran related inheritance/transform/pipeline tests, ran Ruff format/check, ran `git diff --check`, ran targeted pre-commit hooks, ran a read-only adversarial critique, committed the implementation, then reran related tests and Ruff/diff checks against the committed source state.
Expected result when applicable: root cycle detection should use the full node unique ID; cycles should not re-add the root in later generations; generation depth processing should sort numerically; tag merges should preserve local/current order and append unseen inherited or suggested tags deterministically; repeated same-input runs should no longer depend on set ordering.
Actual observed result: Initial red tests failed before implementation for root cycle re-entry, lexicographic generation ordering, set-ordered top-level/config tag merges, and set-ordered semantic tag merges. Final focused, related, hook, lint, and post-commit checks passed locally. Final critique reported no findings.
Procedure verdict / exit code: mixed red/green sequence. Final green checks observed exit 0 for focused pytest, related pytest, Ruff, targeted pre-commit, and `git diff --check`.

# Artifacts

Red observations recorded in `packet:ralph-ticket-c10ord18-20260504T122819Z`:

- Focused red run failed 4 selected tests before implementation.
- `test_ancestor_tree_detects_cycle_back_to_root_unique_id` showed `generation_2: ['model.pkg.a']`, proving `A -> B -> A` re-entered the root.
- `test_column_knowledge_processes_numeric_generations_farthest_to_closest` got `Raw source description` instead of `Closer staging description`, proving lexicographic `generation_2`/`generation_10` ordering let farther metadata win.
- `test_graph_node_tag_merges_preserve_local_then_inherited_order` returned set-ordered tags instead of local/current order followed by inherited unseen tags.
- `test_semantic_analysis_tag_merge_preserves_existing_then_suggested_order` returned set-ordered semantic tags instead of existing order followed by suggested unseen tags.

Final green observations:

- Focused c10ord18 regressions returned `4 passed, 18 deselected in 5.43s`.
- Related inheritance/transform/pipeline suites returned `70 passed in 53.81s` before commit.
- Ruff format/check over changed source/test files passed.
- `git diff --check` passed.
- Targeted `pre-commit run --files ...` over changed source/test/packet/ticket files passed all applicable hooks.
- Implementation commit: `96a43cbac9947e4a189163329bfd0febadf77ea1`.
- Post-implementation-commit acceptance validation: `uv run pytest tests/core/test_inheritance_behavior.py tests/test_yaml_inheritance.py tests/core/test_transforms.py tests/core/test_pipeline_integration.py -q` returned `70 passed in 53.29s`.
- Post-implementation-commit `uv run ruff check src/dbt_osmosis/core/inheritance.py src/dbt_osmosis/core/transforms.py tests/core/test_inheritance_behavior.py && git diff --check` passed.
- Final critique `critique:c10ord18-inheritance-ordering-review` reported no findings.

# Supports Claims

- ticket:c10ord18#ACC-001: the root-cycle regression proves `_build_node_ancestor_tree()` now treats the root as the full unique ID in `visited`.
- ticket:c10ord18#ACC-002: the same regression proves `A -> B -> A` no longer re-adds `A` as a later generation.
- ticket:c10ord18#ACC-003: the generation-depth regression proves numeric sorting makes generation 10 farther than generation 2 so closer metadata wins.
- ticket:c10ord18#ACC-004: top-level/config tag and semantic tag regressions prove local/existing tags stay ordered and unseen inherited/suggested tags append in input order.
- ticket:c10ord18#ACC-005: deterministic ordered assertions and removal of set-based merging support byte-stable tag ordering for the covered paths.
- initiative:dbt-110-111-hardening#OBJ-003: local validation supports stable schema YAML output and inheritance correctness.

# Challenges Claims

None - final observed checks matched the expected post-fix results for the cited claims.

# Environment

Commit: implementation commit `96a43cbac9947e4a189163329bfd0febadf77ea1`.
Branch: `loom/dbt-110-111-hardening`
Runtime: local `uv run` environment.
OS: macOS Darwin.
Relevant config: `src/dbt_osmosis/core/inheritance.py`, `src/dbt_osmosis/core/transforms.py`, and inheritance/transform tests from the reviewed source state.
External service / harness / data source when applicable: no production service exercised; tests used local mocks and existing demo fixture where applicable.

# Validity

Valid for: local inheritance cycle detection, numeric generation processing, and deterministic tag merge behavior in the observed implementation commit.
Fresh enough for: critique and local ticket acceptance under the current directive to defer per-ticket GitHub Actions waiting to final initiative validation.
Recheck when: inheritance graph traversal changes, tag merge semantics change, semantic analysis transforms change, or final initiative-level CI fails for the same claims.
Invalidated by: source changes after this evidence that alter `_build_node_ancestor_tree()`, `_build_column_knowledge_graph()`, `_merge_graph_node_data()`, or semantic tag merging.
Supersedes / superseded by: Supplements `evidence:oracle-backlog-scan` with local implementation verification; final initiative-level CI should supplement this evidence later.

# Limitations

- Full repository test suite and GitHub Actions matrix were not run for this ticket; per-ticket CI waiting is intentionally deferred to final initiative validation.
- ACC-005 is supported by ordered in-memory assertions and removal of set-based merges, not by a repeated byte-for-byte YAML file write fixture.
- Deep DAG duplicate-ancestor precedence remains governed by existing global visited behavior and is outside this ticket.

# Result

The observed checks showed root cycle detection, numeric generation ordering, and deterministic tag merging now satisfy the ticket acceptance criteria with no critique findings.

# Interpretation

The evidence supports ticket acceptance with final initiative-level CI still pending outside this ticket. It does not replace broader initiative validation.

# Related Records

- ticket:c10ord18
- packet:ralph-ticket-c10ord18-20260504T122819Z
- critique:c10ord18-inheritance-ordering-review
- evidence:oracle-backlog-scan
