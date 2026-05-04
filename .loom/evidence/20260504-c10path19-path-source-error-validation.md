---
id: evidence:c10path19-path-source-error-validation
kind: evidence
status: recorded
created_at: 2026-05-04T12:55:31Z
updated_at: 2026-05-04T12:55:31Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  tickets:
    - ticket:c10path19
  packets:
    - packet:ralph-ticket-c10path19-20260504T124520Z
  critique:
    - critique:c10path19-path-source-error-review
external_refs: {}
---

# Summary

Observed test-first local verification for `ticket:c10path19` path-template error wrapping and source `tables` handling. The evidence records red/green behavior, parent validation, hooks, post-commit checks, and critique; it does not decide ticket closure by itself.

# Procedure

Observed at: 2026-05-04T12:55:31Z
Source state: branch `loom/dbt-110-111-hardening`, implementation commit `b324adc0f1a7cca571504aa168b2e58010e5a81f`, based on prior main commit `e0c5d7a07b03da3f6fd06a78518129ae8cae5724`.
Procedure: Reviewed Ralph child output, inspected implementation diff, reran focused c10path19 regressions, reran related path/source/error test suites, ran Ruff format/check, ran `git diff --check`, ran targeted pre-commit hooks, ran a read-only adversarial critique, committed the implementation, then reran related tests and Ruff/diff checks against the committed source state.
Expected result when applicable: invalid path template fields should raise `PathResolutionError` with node ID and template context; existing traversal safety should still reject paths outside the project; missing source `tables` should initialize safely; non-list `tables` should fail clearly without auto-repair.
Actual observed result: Initial red tests failed before implementation because invalid path templates leaked raw `AttributeError`/`KeyError`, missing `tables` leaked raw `KeyError`, and non-list `tables` leaked raw `AttributeError`. Final focused, related, hook, lint, and post-commit checks passed locally. Final critique reported no medium/high findings.
Procedure verdict / exit code: mixed red/green sequence. Final green checks observed exit 0 for focused pytest, related pytest, Ruff, targeted pre-commit, and `git diff --check`.

# Artifacts

Red observations recorded in `packet:ralph-ticket-c10path19-20260504T124520Z`:

- Focused red run returned 5 failures before implementation.
- Path template tests leaked raw `AttributeError` for `{node.source_name}` and raw `KeyError` for `{missing_placeholder}`.
- Source table tests leaked raw `KeyError: 'tables'` for missing `tables` and raw `AttributeError: 'str' object has no attribute 'get'` for non-list `tables` during source matching/table sync.

Final green observations:

- Focused c10path19 regressions returned `5 passed in 5.35s`.
- Related path/source/error suites returned `72 passed, 1 skipped in 23.14s` before commit.
- Ruff format/check over changed source/test files passed.
- `git diff --check` passed.
- Targeted `pre-commit run --files ...` over changed source/test/packet/ticket files passed all applicable hooks.
- Implementation commit: `b324adc0f1a7cca571504aa168b2e58010e5a81f`.
- Post-implementation-commit acceptance validation: `uv run pytest tests/core/test_path_management.py tests/core/test_security.py tests/core/test_sync_operations.py tests/core/test_error_handling.py -q` returned `72 passed, 1 skipped in 22.84s`.
- Post-implementation-commit `uv run ruff check src/dbt_osmosis/core/path_management.py src/dbt_osmosis/core/sync_operations.py tests/core/test_path_management.py tests/core/test_sync_operations.py tests/core/test_security.py && git diff --check` passed.
- Final critique `critique:c10path19-path-source-error-review` reported no medium/high blockers.

# Supports Claims

- ticket:c10path19#ACC-001: invalid template placeholder/attribute regressions now raise `PathResolutionError` instead of raw `AttributeError`/`KeyError`.
- ticket:c10path19#ACC-002: path-template error tests assert the message includes the node unique ID and configured template.
- ticket:c10path19#ACC-003: related security/path tests passed, preserving existing traversal checks.
- ticket:c10path19#ACC-004: missing `tables` regression proves a source entry like `sources: [{name: raw}]` initializes `tables: []` and creates the requested table.
- ticket:c10path19#ACC-005: non-list `tables` regressions prove malformed `tables` values fail with `YamlValidationError` and are not auto-repaired.
- initiative:dbt-110-111-hardening#OBJ-003: local validation supports safe schema YAML mutation behavior.
- initiative:dbt-110-111-hardening#OBJ-006: local validation supports clearer user-facing errors for bad user input.

# Challenges Claims

None - final observed checks matched the expected post-fix results for the cited claims.

# Environment

Commit: implementation commit `b324adc0f1a7cca571504aa168b2e58010e5a81f`.
Branch: `loom/dbt-110-111-hardening`
Runtime: local `uv run` environment.
OS: macOS Darwin.
Relevant config: `src/dbt_osmosis/core/path_management.py`, `src/dbt_osmosis/core/sync_operations.py`, and path/source/error tests from the reviewed source state.
External service / harness / data source when applicable: no production service exercised; tests used local mocks and existing demo fixture where applicable.

# Validity

Valid for: local path-template rendering errors, project-root traversal regression coverage, and source `tables` missing/non-list behavior in the observed implementation commit.
Fresh enough for: critique and local ticket acceptance under the current directive to defer per-ticket GitHub Actions waiting to final initiative validation.
Recheck when: path template rendering changes, source YAML matching changes, source table validation changes, or final initiative-level CI fails for the same claims.
Invalidated by: source changes after this evidence that alter `get_target_yaml_path()`, `_get_or_create_source()`, `_get_or_create_source_table()`, or related path/source tests.
Supersedes / superseded by: Supplements `evidence:oracle-backlog-scan` with local implementation verification; final initiative-level CI should supplement this evidence later.

# Limitations

- Full repository test suite and GitHub Actions matrix were not run for this ticket; per-ticket CI waiting is intentionally deferred to final initiative validation.
- Broader source YAML structural validation remains outside this ticket.
- List entries inside a valid `tables` list are still assumed to be table-like mappings; malformed entries such as strings remain a low adjacent error-clarity risk.
- Malformed format syntax or positional placeholders remain outside the narrow `KeyError`/`AttributeError` wrapping scope.

# Result

The observed checks showed invalid path template placeholders now fail through `PathResolutionError`, source entries missing `tables` can safely create source tables, and non-list `tables` values fail clearly without data repair.

# Interpretation

The evidence supports ticket acceptance with low residual risks and final initiative-level CI still pending outside this ticket. It does not replace broader initiative validation.

# Related Records

- ticket:c10path19
- packet:ralph-ticket-c10path19-20260504T124520Z
- critique:c10path19-path-source-error-review
- evidence:oracle-backlog-scan
