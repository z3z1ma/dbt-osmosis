---
id: evidence:c10gen20-safe-generate-yaml-writes-validation
kind: evidence
status: recorded
created_at: 2026-05-04T13:41:04Z
updated_at: 2026-05-04T13:41:04Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  tickets:
    - ticket:c10gen20
  packets:
    - packet:ralph-ticket-c10gen20-20260504T125850Z
  critique:
    - critique:c10gen20-safe-generate-yaml-writes-review
  wiki:
    - wiki:yaml-sync-safety
external_refs: {}
---

# Summary

Observed test-first local verification for `ticket:c10gen20` safe generate/NL YAML writes. The evidence records red/green behavior, parent hardening after mandatory critique findings, post-commit validation, hooks, and final critique; it does not decide ticket closure by itself.

# Procedure

Observed at: 2026-05-04T13:41:04Z
Source state: branch `loom/dbt-110-111-hardening`, implementation commit `750bff70765850751ea157e21ee4bedc0540d138`, based on prior main commit `bcfc8330dd9eee053f0f195c84cea555e48295b5`.
Procedure: Reviewed Ralph child output, inspected implementation diff, reran focused generate/generator checks, added parent-side writer and path-safety hardening after mandatory critique blockers, reran expanded generate/generator/schema/sync/restructuring checks, ran Ruff format/check, ran `git diff --check`, ran targeted pre-commit hooks, reran mandatory read-only critique to acceptance, committed the implementation, then reran post-commit validation against the committed source state.
Expected result when applicable: generated schema/source/staging YAML should not overwrite existing YAML without explicit `--overwrite`; generated YAML should use shared ruamel/schema machinery and preserve unmanaged top-level sections when overwriting; generated output paths should remain under the project root by default; dry-run should list planned writes and write no files; generated descriptions containing colons, quotes, multiline text, and Jinja should parse cleanly.
Actual observed result: Initial red tests failed before implementation because generated YAML overwrote existing files, `--overwrite` did not exist, outside-project schema paths were accepted, dry-run omitted planned writes, and AI staging descriptions could produce invalid YAML. Final expanded tests, hook, lint, diff, and post-commit checks passed locally. Final mandatory critique reported no findings.
Procedure verdict / exit code: mixed red/green sequence. Final green checks observed exit 0 for focused/expanded pytest, Ruff, targeted pre-commit, and `git diff --check`.

# Artifacts

Red observations recorded in `packet:ralph-ticket-c10gen20-20260504T125850Z`:

- Focused red run returned `6 failed, 30 passed` before implementation.
- Existing generated model/source YAML was overwritten silently instead of failing closed.
- `generate model --overwrite` was unavailable.
- Generated schema paths outside the project root were accepted.
- `generate staging --dry-run` omitted `Planned writes:`.
- AI staging YAML with colon/quotes/multiline/Jinja descriptions raised `ruamel.yaml.scanner.ScannerError`.

Final green observations:

- Child focused generate/generator checks returned `36 passed` after the initial implementation.
- Parent expanded generate/generator/schema/sync/restructuring validation returned `119 passed, 1 skipped in 60.94s` before commit.
- Ruff format/check over changed source/test files passed.
- `git diff --check` passed.
- Targeted `pre-commit run --files ...` over changed source/test/packet/ticket files passed all applicable hooks.
- Final mandatory read-only critique accepted the diff with no findings.
- Implementation commit: `750bff70765850751ea157e21ee4bedc0540d138`.
- Post-implementation-commit acceptance validation: `uv run pytest tests/core/test_cli_generate_group.py tests/core/test_generators.py tests/core/test_schema.py tests/core/test_sync_operations.py tests/core/test_restructuring.py -q` returned `119 passed, 1 skipped in 61.26s`.
- Post-implementation-commit `uv run ruff check src/dbt_osmosis/cli/main.py src/dbt_osmosis/core/generators.py src/dbt_osmosis/core/schema/writer.py tests/core/test_cli_generate_group.py tests/core/test_generators.py tests/core/test_schema.py && git diff --check` passed.
- Post-implementation-commit targeted pre-commit over changed source/test/packet/ticket files passed all applicable hooks.

# Supports Claims

- ticket:c10gen20#ACC-001: generated model/source/staging YAML tests prove existing YAML is refused without `--overwrite`, and `_write_yaml(..., allow_overwrite=False)` enforces no-clobber at writer time.
- ticket:c10gen20#ACC-002: generate/NL YAML writes and AI staging YAML generation use shared ruamel/schema machinery instead of PyYAML/raw f-string YAML.
- ticket:c10gen20#ACC-003: explicit overwrite regression proves unmanaged top-level sections such as `semantic_models` are preserved through the schema reader/writer cache preservation path.
- ticket:c10gen20#ACC-004: generated YAML and SQL path regressions prove explicit paths, default model-name-derived paths, deprecated `nl generate`, and staging SQL-only paths fail when they resolve outside the dbt project root.
- ticket:c10gen20#ACC-005: dry-run regressions prove planned writes are listed, existing YAML refusal matches real-run behavior, and files are not written.
- ticket:c10gen20#ACC-006: AI staging YAML parse regression proves descriptions with colon, quotes, multiline text, and Jinja parse cleanly after generation.
- initiative:dbt-110-111-hardening#OBJ-003: local validation supports safe schema YAML mutation behavior.
- initiative:dbt-110-111-hardening#OBJ-006: local validation supports clearer fail-closed operator behavior for generated output paths and overwrites.

# Challenges Claims

None - final observed checks matched the expected post-fix results for the cited claims.

# Environment

Commit: implementation commit `750bff70765850751ea157e21ee4bedc0540d138`.
Branch: `loom/dbt-110-111-hardening`
Runtime: local `uv run` environment.
OS: macOS Darwin.
Relevant config: generate/NL CLI helpers, schema writer, generator YAML serialization, and related core tests from the reviewed source state.
External service / harness / data source when applicable: no production service exercised; tests used local mocks and existing test fixtures where applicable.

# Validity

Valid for: generated YAML overwrite refusal, writer no-clobber enforcement, generated output project-root validation, dry-run planned writes, unmanaged top-level YAML preservation on overwrite, and parse-safe generated descriptions in the observed implementation commit.
Fresh enough for: critique and local ticket acceptance under the current directive to defer per-ticket GitHub Actions waiting to final initiative validation.
Recheck when: generate/NL command writing changes, schema writer overwrite behavior changes, generated YAML serialization changes, or final initiative-level CI fails for the same claims.
Invalidated by: source changes after this evidence that alter `generate model`, `generate sources`, `generate staging`, deprecated `nl generate`, `_write_yaml()`, or AI staging YAML generation.
Supersedes / superseded by: Supplements `evidence:oracle-backlog-scan`; final initiative-level CI should supplement this evidence later.

# Limitations

- Full repository test suite and GitHub Actions matrix were not run for this ticket; per-ticket CI waiting is intentionally deferred to final initiative validation.
- SQL files are now project-root validated but keep their prior overwrite semantics; this ticket changed generated YAML overwrite policy.
- Legacy docs may not yet mention the new generated YAML `--overwrite` option.
- Cross-process file locking was not added; the no-clobber writer path enforces refusal at write time within the existing atomic-write model.

# Result

The observed checks showed generated YAML writes now fail closed by default, preserve unmanaged YAML sections on explicit overwrite, validate generated output paths under the project root, report planned dry-run writes, and serialize special descriptions with ruamel-safe structured data.

# Interpretation

The evidence supports ticket acceptance with residual risks documented and final initiative-level CI still pending outside this ticket. It does not replace broader initiative validation.

# Related Records

- ticket:c10gen20
- packet:ralph-ticket-c10gen20-20260504T125850Z
- critique:c10gen20-safe-generate-yaml-writes-review
- wiki:yaml-sync-safety
- evidence:oracle-backlog-scan
