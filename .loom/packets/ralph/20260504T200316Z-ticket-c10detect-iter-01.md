---
id: packet:ralph-ticket-c10detect-20260504T200316Z
kind: packet
packet_kind: ralph
status: consumed
target: ticket:c10detect
mode: execution
change_class: code-behavior
risk_class: medium
style: snapshot-first
verification_posture: test-first
iteration: 1
created_at: 2026-05-04T20:03:16Z
updated_at: 2026-05-04T20:08:59Z
scope:
  kind: repository
  repositories:
    - repo:root
child_write_scope:
  records:
    - "None - child returns output only; parent reconciles Loom records"
  paths:
    - src/dbt_osmosis/core/config.py
    - src/dbt_osmosis/core/settings.py
    - src/dbt_osmosis/cli/main.py
    - tests/core/test_config.py
    - tests/core/test_settings.py
    - docs/docs/reference/cli.md
    - docs/docs/reference/settings.md
    - docs/docs/tutorial-yaml/configuration.md
parent_merge_scope:
  records:
    - ticket:c10detect
    - packet:ralph-ticket-c10detect-20260504T200316Z
  paths:
    - .loom/tickets/20260503-c10detect-fusion-future-manifest-detection.md
    - .loom/packets/ralph/20260504T200316Z-ticket-c10detect-iter-01.md
    - .loom/evidence/*c10detect*.md
    - .loom/critique/*c10detect*.md
source_fingerprint:
  git_commit: 87aefa8071dbe14057c3a33e557b0b507f012c3a
  integration_remote: origin
  integration_ref: origin/main
  integration_commit: 87aefa8071dbe14057c3a33e557b0b507f012c3a
  git_status_summary: dirty_mixed
  git_status_detail: "parent-owned active ticket update plus untracked compiled packet; no source/test/doc write-scope changes before launch"
  compiled_from:
    - ticket:c10detect
    - initiative:dbt-110-111-hardening
    - evidence:oracle-backlog-scan
    - research:dbt-110-111-api-surfaces
execution_context:
  branch: loom/dbt-110-111-hardening
  push_remote: origin
  worktree: /Users/alexanderbutler/code_projects/personal/dbt-osmosis-dbt-110-111-hardening
  isolation: worktree
  git_shared_metadata_mutations: forbidden
  destructive_commands: forbidden
  network: allowed
context_budget:
  posture: normal
  max_source_files: 8
  max_excerpt_lines_per_file: 100
  avoid_full_file_reads: true
sources:
  constitution:
    - constitution:main
  initiative:
    - initiative:dbt-110-111-hardening
  research:
    - research:dbt-110-111-api-surfaces
  spec: []
  plan: []
  ticket:
    - ticket:c10detect
  evidence:
    - evidence:oracle-backlog-scan
links:
  docs_manifest_versions: https://raw.githubusercontent.com/dbt-labs/docs.getdbt.com/main/website/snippets/_manifest-versions.md
---

# Mission

Make `_detect_fusion_manifest()` honest: a synthetic future dbt-core manifest schema such as v13 must not enable `fusion_compat`, while a known Fusion manifest shape remains detectable and users can understand/override stale manifest behavior.

# Bound Context

This packet advances `ticket:c10detect` under `initiative:dbt-110-111-hardening`. The initiative explicitly excludes claiming broad Fusion compatibility; this iteration should only make the existing auto-detection fail closed and truthful.

The constitution requires preserving public CLI behavior unless a change has explicit justification. Keep `--fusion-compat/--no-fusion-compat` behavior intact as the user override.

# Source Snapshot

Current implementation reality:

- `src/dbt_osmosis/core/config.py:78-123` reads the first 4KB of `target/manifest.json`, extracts `metadata.dbt_schema_version`, and returns `True` for any manifest schema version greater than v12.
- `src/dbt_osmosis/core/settings.py:293-313` says auto detection order is explicit setting, Fusion manifest detected by `schema version > v12`, then dbt version >= 1.9.6.
- `tests/core/test_config.py:317-327` currently asserts v13 returns `True`, which is the behavior this ticket rejects.
- `docs/docs/tutorial-yaml/configuration.md:70-72` says schema version > v12 enables Fusion compatibility and also mentions PATH binary detection, but current tests assert PATH binaries are ignored.
- `docs/docs/reference/cli.md` and `docs/docs/reference/settings.md` describe auto-detection from Fusion manifest evidence and dbt Core >= 1.9.6.

External source snapshot:

- Official dbt docs snippet `_manifest-versions.md` says Fusion engine v2.0 uses manifest v20, identical to v12; Core v1.11, v1.10, v1.9, and v1.8 all use v12.
- Official dbt artifact docs say artifact versions can change by dbt minor, so a future Core v13 manifest is plausible and should not be treated as Fusion without known Fusion evidence.

# Change Class

`code-behavior`, medium risk. This changes automatic YAML output-mode selection for stale `target/manifest.json` evidence. Critique is recommended after implementation with code-change and test-coverage profiles.

# Verification Targets

- ticket:c10detect#ACC-001
- ticket:c10detect#ACC-002
- ticket:c10detect#ACC-003
- ticket:c10detect#ACC-004
- ticket:c10detect#ACC-005
- initiative:dbt-110-111-hardening#OBJ-002

# Task For This Iteration

1. Start with a failing test update in `tests/core/test_config.py` proving a synthetic future Core v13 manifest does not enable Fusion detection.
2. Preserve or add a positive test for known Fusion evidence, preferably manifest schema v20 per current official dbt docs.
3. Update `_detect_fusion_manifest()` to detect only known Fusion evidence rather than any schema version greater than v12. Fail closed for unrecognized future schema versions and log/debug with wording that does not call them Fusion.
4. Update nearby docstrings/help/docs so they do not claim `> v12` or PATH binary detection. Keep the existing explicit `--fusion-compat` override clear.
5. Do not add a new config flag unless the existing explicit override is insufficient.

# Verification Posture

`test-first`.

Expected red:

- Command: `uv run pytest tests/core/test_config.py::TestDetectFusionManifest -q`
- Before implementation, changing the v13 test to expect `False` should fail because current code returns `True` for any schema version greater than 12.

Expected green:

- Same focused command passes after implementation.
- Also run any directly affected settings/help tests if changed, such as `uv run pytest tests/core/test_config.py::TestDetectFusionManifest tests/core/test_settings.py::TestFusionCompat tests/core/test_cli.py::test_fusion_compat_flag_in_yaml_commands -q`.

# Stop Conditions

Stop and return `blocked` or `escalate` if:

- The worktree or target files have material changes outside this packet that conflict with the write scope.
- Real Fusion manifests lack stable schema/producer metadata and detection cannot be made honest without a project-level support decision.
- Satisfying the ticket requires removing `fusion_compat` auto-detection or changing the public override flag behavior.
- The failing check cannot be made to fail for the expected reason before implementation.
- The solution requires edits outside `child_write_scope.paths`.

# Output Contract

Return:

- outcome (`continue|stop|blocked|escalate`)
- files changed
- records changed
- red command and failure summary
- green command(s) and result summary
- implementation summary
- self-review findings or concerns
- blockers or residual risks
- ticket recommendation

# Working Notes

Parent refreshed `origin/main` before packet compilation. `HEAD` and `origin/main` both resolved to `87aefa8071dbe14057c3a33e557b0b507f012c3a`; `git merge-base --is-ancestor origin/main HEAD` succeeded. The active branch is `loom/dbt-110-111-hardening` in worktree `/Users/alexanderbutler/code_projects/personal/dbt-osmosis-dbt-110-111-hardening`.

# Child Output

- outcome: stop
- files changed:
  - `src/dbt_osmosis/core/config.py`
  - `src/dbt_osmosis/core/settings.py`
  - `src/dbt_osmosis/cli/main.py`
  - `tests/core/test_config.py`
  - `docs/docs/reference/cli.md`
  - `docs/docs/reference/settings.md`
  - `docs/docs/tutorial-yaml/configuration.md`
- records changed: none
- red command and failure summary:
  - `uv run pytest tests/core/test_config.py::TestDetectFusionManifest -q`
  - Expected red: `test_future_manifest_v13` failed because `_detect_fusion_manifest()` returned `True` for v13; `1 failed, 7 passed`.
- green command(s) and result summary:
  - `uv run pytest tests/core/test_config.py::TestDetectFusionManifest -q` -> `8 passed`
  - `uv run pytest tests/core/test_config.py::TestDetectFusionManifest tests/core/test_settings.py::TestFusionCompat tests/core/test_cli.py::test_fusion_compat_flag_in_yaml_commands -q` -> `20 passed`
  - `uv run ruff check src/dbt_osmosis/core/config.py src/dbt_osmosis/core/settings.py src/dbt_osmosis/cli/main.py tests/core/test_config.py` -> passed
- implementation summary:
  - Changed Fusion manifest detection to recognize only known Fusion schema evidence, currently schema v20.
  - Future unrecognized schema versions like v13 now fail closed and log debug instead of enabling Fusion compatibility.
  - Preserved existing `--fusion-compat/--no-fusion-compat` override behavior.
  - Updated tests, docstrings, CLI help, and docs to remove `> v12` and PATH binary detection claims.
- self-review findings or concerns:
  - Detection is intentionally conservative; future Fusion schema versions will require an explicit allowlist update.
- blockers or residual risks:
  - Full test suite not run.
- ticket recommendation:
  - Parent can reconcile this iteration and route to recommended critique/acceptance review.

# Parent Merge Notes

Parent accepted the child output as inside packet scope after inspecting `git diff` for the allowed source, test, and docs paths. Parent made one narrow follow-up tweak in `src/dbt_osmosis/core/config.py` to avoid a new basedpyright implicit string-concatenation warning in the debug message.

Parent validation recorded in `evidence:c10detect-fusion-manifest-detection-validation`:

- `uv run pytest tests/core/test_config.py::TestDetectFusionManifest tests/core/test_settings.py::TestFusionCompat tests/core/test_cli.py::test_fusion_compat_flag_in_yaml_commands -q` -> `20 passed` after parent tweak.
- `uv run pytest tests/core/test_config.py -q` -> `20 passed`.
- `uv run ruff check src/dbt_osmosis/core/config.py src/dbt_osmosis/core/settings.py src/dbt_osmosis/cli/main.py tests/core/test_config.py` -> passed.
- `uv run basedpyright src/dbt_osmosis/core/config.py src/dbt_osmosis/core/settings.py src/dbt_osmosis/cli/main.py --outputjson` -> `errorCount: 0`, `warningCount: 412`.

Next route: recommended critique for medium-risk code-behavior change.
