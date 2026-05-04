---
id: evidence:c10detect-fusion-manifest-detection-validation
kind: evidence
status: recorded
created_at: 2026-05-04T20:08:22Z
updated_at: 2026-05-04T20:13:20Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  initiative:
    - initiative:dbt-110-111-hardening
  ticket:
    - ticket:c10detect
  packet:
    - packet:ralph-ticket-c10detect-20260504T200316Z
external_refs:
  dbt_manifest_versions_snippet: https://raw.githubusercontent.com/dbt-labs/docs.getdbt.com/main/website/snippets/_manifest-versions.md
---

# Summary

Observed a test-first implementation pass for `ticket:c10detect`: a synthetic future dbt-core manifest schema v13 now fails closed instead of enabling Fusion detection, while known Fusion schema v20 remains detected and user-facing docs/help describe known manifest evidence and explicit overrides.

# Procedure

Observed at: 2026-05-04T20:08:22Z

Source state: worktree `loom/dbt-110-111-hardening` at base commit `87aefa8071dbe14057c3a33e557b0b507f012c3a`, with uncommitted ticket, packet, source, docs, and test changes for `ticket:c10detect`.

Procedure: Ralph child updated the focused v13 regression test first, ran the red command, implemented the code/docs change, and ran focused green validation. Parent inspected the packet-scoped diff and reran focused validation, full `tests/core/test_config.py`, Ruff on changed Python files, and changed-source basedpyright.

Expected result when applicable: The updated focused test should fail before implementation because current `_detect_fusion_manifest()` returned `True` for any manifest schema version greater than 12. After implementation, v13 should return `False`, v20 should still return `True`, user override semantics should remain covered by existing settings/help tests, Ruff should pass, and basedpyright should report zero errors.

Actual observed result: Red test failed for the expected reason before implementation. After implementation, focused tests and full `tests/core/test_config.py` passed; Ruff passed; basedpyright reported `errorCount: 0` with existing warnings; full pre-commit passed.

Procedure verdict / exit code: pass for the observed focused and structural checks after an expected red failure.

# Artifacts

- Red command: `uv run pytest tests/core/test_config.py::TestDetectFusionManifest -q`.
- Red result: `test_future_manifest_v13` failed because `_detect_fusion_manifest()` returned `True` for v13; `1 failed, 7 passed`.
- Child green command: `uv run pytest tests/core/test_config.py::TestDetectFusionManifest -q` -> `8 passed`.
- Child green command: `uv run pytest tests/core/test_config.py::TestDetectFusionManifest tests/core/test_settings.py::TestFusionCompat tests/core/test_cli.py::test_fusion_compat_flag_in_yaml_commands -q` -> `20 passed`.
- Child structural command: `uv run ruff check src/dbt_osmosis/core/config.py src/dbt_osmosis/core/settings.py src/dbt_osmosis/cli/main.py tests/core/test_config.py` -> passed.
- Parent green command: `uv run pytest tests/core/test_config.py::TestDetectFusionManifest tests/core/test_settings.py::TestFusionCompat tests/core/test_cli.py::test_fusion_compat_flag_in_yaml_commands -q` -> `20 passed in 0.19s`, rerun after parent debug-message tweak -> `20 passed in 0.20s`.
- Parent broader command: `uv run pytest tests/core/test_config.py -q` -> `20 passed in 5.30s`.
- Parent structural command: `uv run ruff check src/dbt_osmosis/core/config.py src/dbt_osmosis/core/settings.py src/dbt_osmosis/cli/main.py tests/core/test_config.py` -> passed, rerun after parent tweak -> passed.
- Parent type command: `uv run basedpyright src/dbt_osmosis/core/config.py src/dbt_osmosis/core/settings.py src/dbt_osmosis/cli/main.py --outputjson` -> `errorCount: 0`, `warningCount: 412` after parent tweak.
- Parent hygiene command: `uv run pre-commit run --all-files` -> all hooks passed, including basedpyright, Detect hardcoded secrets, and GitHub Actions workflow lint.
- Diff inspection: `git diff -- src/dbt_osmosis/core/config.py src/dbt_osmosis/core/settings.py src/dbt_osmosis/cli/main.py tests/core/test_config.py docs/docs/reference/cli.md docs/docs/reference/settings.md docs/docs/tutorial-yaml/configuration.md` showed only packet-scoped code, docs, and test changes.

# Supports Claims

- ticket:c10detect#ACC-001 — dbt-core v12 remains non-Fusion through existing focused test coverage.
- ticket:c10detect#ACC-002 — synthetic future dbt-core v13 now returns `False` and no longer automatically enables Fusion detection.
- ticket:c10detect#ACC-003 — known Fusion schema v20 remains detected by focused test coverage.
- ticket:c10detect#ACC-004 — tests, docstrings, CLI help, and docs now describe known Fusion manifest evidence rather than `> v12` detection; debug logging for unrecognized future schema versions does not call them Fusion.
- ticket:c10detect#ACC-005 — docs and CLI/settings text preserve the explicit `--fusion-compat/--no-fusion-compat` override path and explain stale manifest evidence is known Fusion evidence, not generic future schemas.
- initiative:dbt-110-111-hardening#OBJ-002 — partially supports honest dbt 1.10+ YAML output-mode handling without overclaiming Fusion for future dbt-core manifest schema versions.

# Challenges Claims

None - the observed post-implementation results matched the ticket and packet expectations for the tested scenarios.

# Environment

Commit: base `87aefa8071dbe14057c3a33e557b0b507f012c3a`; changes uncommitted at observation time.

Branch: `loom/dbt-110-111-hardening`

Runtime: `uv run` pytest, Ruff, basedpyright in local macOS worktree.

OS: darwin

Relevant config: `pyproject.toml`, `.pre-commit-config.yaml`, `tests/core/test_config.py`, `tests/core/test_settings.py`, `tests/core/test_cli.py`.

External service / harness / data source when applicable: dbt docs raw snippet `_manifest-versions.md` observed through web fetch; child execution via Ralph fixer subagent.

# Validity

Valid for: the edited source/docs/tests in this worktree at the observed source state and the cited focused validation commands.

Fresh enough for: critique of `ticket:c10detect` implementation shape and ticket acceptance coverage after parent reconciliation.

Recheck when: `_detect_fusion_manifest()`, `fusion_compat` resolution/docs, dbt artifact version docs, or package dependency versions change; before final release packaging; or if critique requests broader coverage.

Invalidated by: source edits that change Fusion detection, settings resolution, CLI help, docs text, or tests after this observation without rerunning validation.

Supersedes / superseded by: none.

# Limitations

- This evidence uses synthetic manifests; it does not prove every real future Fusion artifact shape will be detected.
- Full `uv run pytest` was not run for this evidence record.
- basedpyright warnings remain in the inspected files, but `errorCount` was zero; the warning count is not evidence of zero-warning type hygiene.
- The allowlist intentionally makes future Fusion schema versions require a code update or explicit user override.

# Result

The observed red/green pass showed the prior schema-version-only behavior was covered by a failing test and that the implementation now treats v13 as non-Fusion while preserving v20 detection and explicit override clarity.

# Interpretation

This evidence supports the ticket's focused acceptance criteria for synthetic future dbt-core and known Fusion schema detection. It does not by itself decide critique verdict, ticket acceptance, or future Fusion support policy.

# Related Records

- ticket:c10detect
- packet:ralph-ticket-c10detect-20260504T200316Z
- initiative:dbt-110-111-hardening
- evidence:oracle-backlog-scan
