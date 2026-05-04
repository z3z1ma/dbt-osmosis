---
id: evidence:c10wb22-workbench-target-validation
kind: evidence
status: recorded
created_at: 2026-05-04T14:43:11Z
updated_at: 2026-05-04T14:43:11Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  tickets:
    - ticket:c10wb22
  packets:
    - packet:ralph-ticket-c10wb22-20260504T141941Z
  critique:
    - critique:c10wb22-workbench-target-review
external_refs: {}
---

# Summary

Observed test-first local verification for `ticket:c10wb22` workbench server invocation, optional dependency errors, pass-through args, and target switching. The evidence records red/green behavior, parent hardening after critique, post-commit checks, and final critique; it does not decide ticket closure by itself.

# Procedure

Observed at: 2026-05-04T14:43:11Z
Source state: branch `loom/dbt-110-111-hardening`, implementation commit `ab138546acf0e32d7ad9339382d291bac8365fdc`, based on prior main commit `a39da58b0820266bf2a8b46f1fd2f768b8f034e4`.
Procedure: Reviewed Ralph child output, inspected implementation diff, reran focused CLI/workbench checks, added parent-side fixes after mandatory critique found stale widget-state target switching and incomplete dependency/pass-through coverage, reran related CLI/workbench/SQL/legacy suites, ran Ruff format/check, ran `git diff --check`, ran targeted pre-commit hooks, reran mandatory read-only critique to acceptance, committed the implementation, then reran post-commit validation against the committed source state.
Expected result when applicable: workbench `--host`/`--port` should bind Streamlit server address/port; missing Streamlit and missing normal-launch workbench extras should fail with a clear install hint; unknown and literal `--` Streamlit args should be preserved before the app script; target switching should build a fresh dbt context for the selected target, update state only after success, close/release old connections, and roll back with a visible error on failure.
Actual observed result: Initial red tests failed before implementation because CLI used browser flags and raw missing-Streamlit errors, and workbench target switching mutated the existing context instead of rebuilding. Final focused, related, hook, lint, diff, and post-commit checks passed locally. Final mandatory critique reported no findings.
Procedure verdict / exit code: mixed red/green sequence. Final green checks observed exit 0 for focused/related pytest, Ruff, targeted pre-commit, and `git diff --check`.

# Artifacts

Red observations recorded in `packet:ralph-ticket-c10wb22-20260504T141941Z`:

- Workbench target-switch tests failed because `change_target()` mutated the existing context and called `_reload_manifest(ctx)` instead of rebuilding a fresh selected-target context.
- CLI workbench tests failed because host/port were passed as `--browser.serverAddress`/`--browser.serverPort`.
- Missing Streamlit surfaced as raw `FileNotFoundError` without the workbench extra install hint for normal, `--options`, and `--config` paths.

Final green observations:

- Child focused workbench validation returned `25 passed` across CLI/workbench tests.
- Parent focused validation after critique fixes returned `11 passed, 16 deselected` for workbench-focused CLI/workbench tests.
- Parent related validation returned `36 passed` before commit.
- Ruff format/check over changed source/test files passed.
- `git diff --check` passed.
- Targeted `pre-commit run --files ...` over changed source/test/packet/ticket files passed all applicable hooks.
- Final mandatory read-only critique accepted the diff with no findings.
- Implementation commit: `ab138546acf0e32d7ad9339382d291bac8365fdc`.
- Post-implementation-commit acceptance validation: `uv run pytest tests/core/test_cli.py tests/core/test_workbench_app.py tests/core/test_sql_operations.py tests/core/test_legacy.py -q` returned `36 passed in 12.27s`.
- Post-implementation-commit `uv run ruff check src/dbt_osmosis/cli/main.py src/dbt_osmosis/workbench/app.py tests/core/test_cli.py tests/core/test_workbench_app.py && git diff --check` passed.
- Post-implementation-commit targeted pre-commit over changed source/test/packet/ticket files passed all applicable hooks.

# Supports Claims

- ticket:c10wb22#ACC-001: CLI regression proves `--host 0.0.0.0 --port 8502` passes `--server.address=0.0.0.0` and `--server.port=8502`, not browser flags.
- ticket:c10wb22#ACC-002: missing Streamlit regressions prove normal workbench launch emits a clear install hint.
- ticket:c10wb22#ACC-003: missing Streamlit regressions prove `--options` and `--config` paths fail clearly when Streamlit is unavailable.
- ticket:c10wb22#ACC-004: target-switch regressions prove `change_target()` reads the radio widget state, creates a fresh `DbtProjectContext` with the selected target, swaps state to that context, refreshes model nodes, and recompiles the query.
- ticket:c10wb22#ACC-005: target-switch regression proves the old context close hook is called after successful context replacement.
- ticket:c10wb22#ACC-006: target-switch failure regression proves old context/target/compiled query remain active and a clear Streamlit error is displayed.
- initiative:dbt-110-111-hardening#OBJ-006: local validation supports clearer user-facing workbench CLI and target-switch behavior.

# Challenges Claims

None - final observed checks matched the expected post-fix results for the cited claims.

# Environment

Commit: implementation commit `ab138546acf0e32d7ad9339382d291bac8365fdc`.
Branch: `loom/dbt-110-111-hardening`
Runtime: local `uv run` environment.
OS: macOS Darwin.
Relevant config: workbench CLI command, workbench app target switching, CLI tests, and workbench app tests from the reviewed source state.
External service / harness / data source when applicable: no live Streamlit server/browser exercised; tests used subprocess mocks and workbench import/context stubs.

# Validity

Valid for: local workbench CLI invocation construction, missing dependency errors, pass-through argument ordering, target-switch state handling, old context close hook behavior, and failure rollback in the observed implementation commit.
Fresh enough for: critique and local ticket acceptance under the current directive to defer per-ticket GitHub Actions waiting to final initiative validation.
Recheck when: Streamlit CLI flags change, workbench launch behavior changes, workbench target-switch state changes, or dbt context construction changes.
Invalidated by: source changes after this evidence that alter `workbench()`, `_run_streamlit_command()`, `change_target()`, or workbench app state initialization.
Supersedes / superseded by: Supplements `evidence:oracle-backlog-scan`; final initiative-level CI should supplement this evidence later.

# Limitations

- No live Streamlit server launch or browser behavior was exercised locally.
- No real multi-target dbt adapter switch was exercised; target switching used mocked/stubbed contexts.
- Streamlit CLI flag compatibility may vary in future Streamlit versions.

# Result

The observed checks showed workbench CLI now uses server bind flags, fails clearly for missing Streamlit/workbench extras, preserves pass-through args, and target switching rebuilds context safely with old-context cleanup and rollback on failure.

# Interpretation

The evidence supports ticket acceptance with residual risks documented and final initiative-level CI still pending outside this ticket. It does not replace live workbench/browser validation.

# Related Records

- ticket:c10wb22
- packet:ralph-ticket-c10wb22-20260504T141941Z
- critique:c10wb22-workbench-target-review
- evidence:oracle-backlog-scan
