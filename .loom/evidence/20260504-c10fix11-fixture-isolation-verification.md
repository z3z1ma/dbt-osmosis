---
id: evidence:c10fix11-fixture-isolation-verification
kind: evidence
status: recorded
created_at: 2026-05-04T05:19:49Z
updated_at: 2026-05-04T05:27:07Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  tickets:
    - ticket:c10fix11
  packets:
    - packet:ralph-ticket-c10fix11-20260504T050248Z
external_refs: {}
---

# Summary

Observed red/green fixture-isolation regression coverage, isolated demo parse behavior, safe integration smoke behavior, artifact guards, targeted hooks, and whitespace checks for `ticket:c10fix11`. This evidence supports the ticket's fixture isolation claims; it does not decide acceptance or closure.

# Procedure

Observed at: 2026-05-04T05:27:07Z
Source state: branch `loom/dbt-110-111-hardening`, base commit `b7bdc45db5b97e41ffdbf2cd8d7174585a9769bf`, with the `ticket:c10fix11` working-tree diff applied.
Procedure: Reviewed the Ralph packet child output and current diff, checked for repo-root/source-tree artifacts, ran focused pytest, ran the safe integration smoke, ran `task parse-demo`, rechecked artifacts, ran targeted `pre-commit`, ran `git diff --check`, addressed critique findings, and reran the same green checks.
Expected result when applicable: Red tests should fail before implementation for the documented fixture-isolation gaps; after implementation, focused tests, integration smoke, isolated parse helper, artifact guards, hooks, and whitespace checks should pass without creating repo-root `test.db` or source `demo_duckdb/target/manifest.json`.
Actual observed result: The packet records expected red failures before implementation. The current working tree passed focused pytest, integration smoke, `task parse-demo`, post-run artifact guards, targeted hooks, and whitespace checks.
Procedure verdict / exit code: mixed red/green sequence. Final green checks observed exit 0 for focused pytest, integration smoke, `task parse-demo`, targeted `pre-commit`, and `git diff --check`.

# Artifacts

Red observations recorded in `packet:ralph-ticket-c10fix11-20260504T050248Z`:

- `uv run pytest tests/test_fixture_isolation.py tests/core/test_demo_fixture_support.py -q` returned `6 failed, 5 passed in 0.25s` before implementation.
- The red failures covered relative DuckDB profile paths in temp copies, generated/local artifact leakage, source-tree demo manifest dependence, destructive integration cleanup, CI's inline destructive cleanup, and missing isolated manifest helper behavior.

Final green observations:

- Pre-check artifact guards returned `No files found` for repo-root `test.db` and `demo_duckdb/target/manifest.json`.
- Critique-driven patch expanded default fixture-copy exclusions for common ignored/local artifacts (`.cache`, `.env`, `.DS_Store`, `*.log`, sqlite files, cache dirs, and bytecode) and added CI source-artifact guards before and after matrix validation.
- `uv run pytest tests/test_fixture_isolation.py tests/core/test_demo_fixture_support.py tests/core/test_legacy.py -q` returned `14 passed in 2.17s` after the critique-driven patch.
- `uv run demo_duckdb/integration_tests.sh` passed after the critique-driven patch and ended with `All dbt-osmosis yaml integration tests passed against /var/folders/1b/6mg4g2fs2zx99h46b9j5r7mh0000gp/T/dbt-osmosis-demo.NehWcD/demo_duckdb!`.
- `task parse-demo` passed after the critique-driven patch, running dbt `1.11.8` with `dbt-duckdb` `1.10.1` against an isolated temp copy at `/var/folders/1b/6mg4g2fs2zx99h46b9j5r7mh0000gp/T/dbt-osmosis-parse-mxsvry17/demo_duckdb` and writing parse artifacts under that temp copy.
- Post-run artifact guards again returned `No files found` for repo-root `test.db` and `demo_duckdb/target/manifest.json`.
- `pre-commit run --files .github/workflows/tests.yml Taskfile.yml demo_duckdb/integration_tests.sh tests/conftest.py tests/core/conftest.py tests/core/test_demo_fixture_support.py tests/core/test_legacy.py tests/support.py tests/test_fixture_isolation.py` passed check-ast, YAML checks, end-of-file, trailing whitespace, private-key detection, debug-statements, ruff-format, ruff, gitleaks, and actionlint.
- A final targeted `pre-commit run --files` including the ticket, packet, evidence, and critique records also passed the same applicable hooks.
- `git diff --check` produced no output.

# Supports Claims

- ticket:c10fix11#ACC-001: focused tests assert copied DuckDB profile paths are absolute and resolve under the temp project; focused pytest passed.
- ticket:c10fix11#ACC-002: `tests/core/test_legacy.py` now consumes a session-scoped temp manifest path, and focused pytest including the real manifest test passed without a source-tree manifest.
- ticket:c10fix11#ACC-003: core manifest tests always parse an isolated temp demo copy, and CI/Taskfile parse smokes parse temp copies with `--target test` instead of relying on source `demo_duckdb/target`.
- ticket:c10fix11#ACC-004: integration smoke uses a temp copy and targeted static tests/hooks verify no normal `git checkout` or `git clean` cleanup remains in the script or CI workflow.
- ticket:c10fix11#ACC-005: focused tests assert generated/local artifacts and representative ignored local files are excluded from default fixture copies; focused pytest passed after the critique-driven patch.
- ticket:c10fix11#ACC-006: artifact guards before and after focused/integration/parse validation found no repo-root `test.db`; CI now has source-artifact guard steps before and after matrix validation, and focused tests assert copied database paths avoid repo root.
- initiative:dbt-110-111-hardening#OBJ-004: local test harness and integration smoke validation now avoid stale source manifests and repo-root DuckDB files for the observed source state.

# Challenges Claims

None - the final observed checks matched the expected post-fix results for the cited claims.

# Environment

Commit: base `b7bdc45db5b97e41ffdbf2cd8d7174585a9769bf` plus uncommitted `ticket:c10fix11` diff.
Branch: `loom/dbt-110-111-hardening`
Runtime: local `uv run` environment for pytest and integration smoke; `task parse-demo` reported dbt `1.11.8` and `dbt-duckdb` `1.10.1`.
OS: macOS Darwin.
Relevant config: `tests/support.py`, `tests/conftest.py`, `tests/core/conftest.py`, `tests/core/test_legacy.py`, `tests/core/test_demo_fixture_support.py`, `tests/test_fixture_isolation.py`, `demo_duckdb/integration_tests.sh`, `.github/workflows/tests.yml`, and `Taskfile.yml` from the reviewed diff.
External service / harness / data source when applicable: no production services exercised; package/runtime dependencies came from the local uv environment.

# Validity

Valid for: the fixture-copy, profile-rewrite, isolated core manifest, integration smoke, CI parse, and Taskfile parse behavior in the observed source state.
Fresh enough for: local acceptance review and mandatory critique of `ticket:c10fix11` before post-commit CI.
Recheck when: fixture-copy support, demo profiles, core manifest fixtures, CI/Taskfile parse commands, integration smoke script, dbt version matrix, DuckDB adapter version, or source artifact policy changes.
Invalidated by: source changes after this evidence that alter fixture isolation behavior, failed post-commit CI for the same claims, or dbt/DuckDB behavior changes that make temp parse/smoke behavior diverge.
Supersedes / superseded by: Supersedes `evidence:oracle-backlog-scan` for the implemented fixture-isolation claims; should be supplemented by post-commit CI evidence before final closure.

# Limitations

- This evidence was gathered before the implementation commit existed, so final closure should also cite post-commit CI or confirm the committed diff matches the observed source state.
- This evidence ran the local default dbt environment and an isolated `task parse-demo`; it does not execute the full GitHub Actions matrix locally.
- `task parse-demo` currently leaves its temporary parse copy to the operating-system temp cleanup policy; the observed claim is that artifacts stay out of the source tree, not that every temp directory is removed immediately.
- Existing repository guidance and wiki pages that describe the old unsafe integration script need reconciliation before final acceptance.

# Result

The observed checks showed that the working-tree implementation has red/green regression coverage and local green validation for temp project copying, absolute DuckDB profile rewrites, isolated manifest parsing, safe integration smoke execution, source-artifact guards, targeted hooks, and whitespace cleanliness.

# Interpretation

The evidence supports the ticket's local fixture isolation claims. It does not by itself accept or close the ticket, and it does not replace mandatory critique or post-commit CI evidence.

# Related Records

- ticket:c10fix11
- packet:ralph-ticket-c10fix11-20260504T050248Z
- evidence:oracle-backlog-scan
