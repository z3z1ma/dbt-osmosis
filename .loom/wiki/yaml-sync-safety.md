---
id: wiki:yaml-sync-safety
kind: wiki
page_type: reference
status: active
created_at: 2026-05-04T08:19:55Z
updated_at: 2026-05-04T11:48:07Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  tickets:
    - ticket:c10race13
    - ticket:c10loss16
  evidence:
    - evidence:c10race13-yaml-sync-serialization-verification
    - evidence:c10race13-main-ci-success
    - evidence:c10loss16-duplicate-yaml-fail-closed-validation
  critique:
    - critique:c10race13-yaml-sync-serialization-review
    - critique:c10loss16-duplicate-yaml-fail-closed-review
---

# Summary

YAML sync must serialize work by resolved target YAML path. Within one `sync_node_to_yaml(context, node=None)` run, every model, seed, version, or source that resolves to the same schema file is one work item: one worker prepares one ruamel document, mutates all grouped sections sequentially, and finalizes that document once.

# Rules

- Resolve each candidate's effective target path the same way single-node sync does: use the current YAML path when it exists, otherwise use `get_target_yaml_path()`.
- Group all candidates by that resolved target path before submitting work to `context.pool.map()`.
- Preserve concurrency only across independent target YAML files.
- Process same-path groups in deterministic order so versioned model YAML does not churn based on manifest iteration.
- Keep package-node skip behavior intact; package nodes are not writable project YAML.
- Fail closed on duplicate writable YAML entries before syncing. Duplicate `models[]`, `seeds[]`, or model `versions[]` entries must raise actionable validation errors instead of choosing one and deleting the others.
- Preflight all grouped target YAML documents before all-node sync dispatches workers, so a duplicate in one target aborts the command before unrelated YAML files are finalized or written.
- Use unique temporary files in the target directory for YAML writes, then atomically replace the target after write validation.
- Preserve existing YAML file mode when replacing an existing target file.

# Boundaries

- This rule serializes same-process all-node sync scheduling. It does not add cross-process locks.
- Direct concurrent calls to `sync_node_to_yaml(context, node=<single node>)` remain outside the grouped scheduler.
- Source grouping coverage currently relies on source-shaped nodes because the demo fixture does not parse sources.
- Generated/NL writer migration is outside this page's current sync path and is tracked separately by ticket:c10gen20.

# Sources

- ticket:c10race13
- evidence:c10race13-yaml-sync-serialization-verification
- evidence:c10race13-main-ci-success
- critique:c10race13-yaml-sync-serialization-review
- ticket:c10loss16
- evidence:c10loss16-duplicate-yaml-fail-closed-validation
- critique:c10loss16-duplicate-yaml-fail-closed-review
- `src/dbt_osmosis/core/sync_operations.py`
- `src/dbt_osmosis/core/schema/writer.py`
- `tests/core/test_sync_operations.py`

# Related Pages

- wiki:repository-atlas
