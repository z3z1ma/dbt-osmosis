---
id: wiki:yaml-sync-safety
kind: wiki
page_type: reference
status: active
created_at: 2026-05-04T08:19:55Z
updated_at: 2026-05-04T13:41:04Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  tickets:
    - ticket:c10race13
    - ticket:c10loss16
    - ticket:c10dry17
    - ticket:c10gen20
  evidence:
    - evidence:c10race13-yaml-sync-serialization-verification
    - evidence:c10race13-main-ci-success
    - evidence:c10loss16-duplicate-yaml-fail-closed-validation
    - evidence:c10dry17-dry-run-cache-isolation-validation
    - evidence:c10gen20-safe-generate-yaml-writes-validation
  critique:
    - critique:c10race13-yaml-sync-serialization-review
    - critique:c10loss16-duplicate-yaml-fail-closed-review
    - critique:c10dry17-dry-run-cache-isolation-review
    - critique:c10gen20-safe-generate-yaml-writes-review
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
- Dry-run and `--check` paths may compare mutated YAML and count mutations, but they must discard processed `_YAML_BUFFER_CACHE` and `_YAML_ORIGINAL_CACHE` entries before later reads in the same process.
- Keep non-dry-run `sync_node_to_yaml(..., commit=False)` as the intentional in-memory inspection path; tests that rely on that helper behavior should set `dry_run = False` explicitly.
- Test cache reset fixtures should clear `_COLUMN_LIST_CACHE`, `_YAML_BUFFER_CACHE`, and `_YAML_ORIGINAL_CACHE` in place under their production locks instead of monkeypatching the cache globals to replacement dicts.
- Generate/NL commands that write schema/source/staging YAML should construct structured mappings and serialize/write through the shared ruamel/schema helpers, not PyYAML or raw string templates.
- Generated YAML defaults to fail closed when the target YAML file exists; callers must require explicit `--overwrite`, and no-clobber behavior should be enforced at write time rather than only by CLI preflight.
- Generated YAML and SQL output paths should resolve under the dbt project root by default, including auto-derived paths from generated model names.
- Generate/NL dry-run paths should validate/serialize planned YAML and list every planned SQL/YAML write while leaving no files behind.

# Boundaries

- This rule serializes same-process all-node sync scheduling. It does not add cross-process locks.
- Direct concurrent calls to `sync_node_to_yaml(context, node=<single node>)` remain outside the grouped scheduler.
- Source grouping coverage currently relies on source-shaped nodes because the demo fixture does not parse sources.
- Generated/NL commands still keep SQL file overwrite semantics separate from YAML overwrite policy; ticket:c10gen20 added project-root SQL path validation but did not make SQL writes no-clobber.
- Dry-run cleanup intentionally discards original-cache state. A preview-then-apply helper flow should reread YAML before the real write rather than reusing filtered data from before the dry-run preview.
- Dry-run sync can still mutate a shared cached document transiently before finalization discards it; the accepted guarantee is post-call same-process read isolation, not concurrent-reader isolation during the call.

# Sources

- ticket:c10race13
- evidence:c10race13-yaml-sync-serialization-verification
- evidence:c10race13-main-ci-success
- critique:c10race13-yaml-sync-serialization-review
- ticket:c10loss16
- evidence:c10loss16-duplicate-yaml-fail-closed-validation
- critique:c10loss16-duplicate-yaml-fail-closed-review
- ticket:c10dry17
- evidence:c10dry17-dry-run-cache-isolation-validation
- critique:c10dry17-dry-run-cache-isolation-review
- ticket:c10gen20
- evidence:c10gen20-safe-generate-yaml-writes-validation
- critique:c10gen20-safe-generate-yaml-writes-review
- `src/dbt_osmosis/core/sync_operations.py`
- `src/dbt_osmosis/core/schema/writer.py`
- `src/dbt_osmosis/cli/main.py`
- `src/dbt_osmosis/core/generators.py`
- `tests/core/test_sync_operations.py`
- `tests/core/test_schema.py`
- `tests/core/test_cli_generate_group.py`
- `tests/core/test_generators.py`
- `tests/conftest.py`

# Related Pages

- wiki:repository-atlas
