---
id: ticket:c10fix11
kind: ticket
status: complete_pending_acceptance
change_class: code-behavior
risk_class: high
created_at: 2026-05-03T21:10:43Z
updated_at: 2026-05-04T05:29:21Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  initiative:
    - initiative:dbt-110-111-hardening
  evidence:
    - evidence:oracle-backlog-scan
    - evidence:c10fix11-fixture-isolation-verification
  critique:
    - critique:c10fix11-fixture-isolation-review
  packets:
    - packet:ralph-ticket-c10fix11-20260504T050248Z
depends_on: []
---

# Summary

Make test fixtures and integration smoke run from isolated temp projects with absolute DuckDB paths, version-aware manifests, clean artifact copying, and no destructive source-tree mutation.

# Context

`demo_duckdb/profiles.yml` uses `path: "test.db"`. `tests/conftest.py:134-141` changes cwd to build a template, but later `yaml_context` creates contexts from outside that temp project. `tests/core/conftest.py` writes or reuses `demo_duckdb/target/manifest.json` in the source tree. `demo_duckdb/integration_tests.sh:15-23` mutates tracked fixture YAML and restores with `git checkout`/`git clean`. `_create_temp_project_copy()` in `tests/conftest.py:65-93` excludes only `target/` when requested.

# Why Now

dbt 1.10/1.11 matrix failures will be hard to trust if tests depend on cwd, stale source-tree manifests, local ignored DuckDB files, or destructive fixture cleanup.

# Scope

- Rewrite temp fixture `profiles.yml` to use an absolute DuckDB path under each copied project.
- Remove source-tree `demo_duckdb/target/manifest.json` dependence from tests.
- Make manifest freshness account for dbt version/target/adapter or always parse in temp.
- Run integration smoke on a temp copy of `demo_duckdb` rather than mutating source fixtures.
- Exclude generated/ignored artifacts such as logs, DuckDB files, target artifacts, and local user files from fixture copies unless intentionally included.
- Standardize `--target test` across parse/build commands.

# Out Of Scope

- Rewriting all tests to be fully hermetic if a smaller fixture isolation boundary works.
- Changing demo project source semantics unless required for isolation.

# Acceptance Criteria

- ACC-001: Test DuckDB database paths resolve inside temp project directories, not repo root.
- ACC-002: Core tests pass from a clean checkout without pre-existing `demo_duckdb/target` artifacts.
- ACC-003: Running tests across dbt versions cannot reuse a stale manifest from a different dbt/core/adapter/target combination.
- ACC-004: Integration smoke no longer uses destructive `git checkout` or `git clean` as normal cleanup.
- ACC-005: Fixture copies exclude generated and ignored artifacts unless a test explicitly opts in.
- ACC-006: Tests assert that no repo-root `test.db` is created or required.

# Coverage

Covers:

- ticket:c10fix11#ACC-001
- ticket:c10fix11#ACC-002
- ticket:c10fix11#ACC-003
- ticket:c10fix11#ACC-004
- ticket:c10fix11#ACC-005
- ticket:c10fix11#ACC-006
- initiative:dbt-110-111-hardening#OBJ-004

# Claim Matrix

| Claim | Evidence | Critique | Status |
| --- | --- | --- | --- |
| ticket:c10fix11#ACC-001 | evidence:c10fix11-fixture-isolation-verification | critique:c10fix11-fixture-isolation-review | locally supported; post-commit CI pending |
| ticket:c10fix11#ACC-002 | evidence:c10fix11-fixture-isolation-verification | critique:c10fix11-fixture-isolation-review#FIND-002 resolved | locally supported; post-commit CI pending |
| ticket:c10fix11#ACC-003 | evidence:c10fix11-fixture-isolation-verification | critique:c10fix11-fixture-isolation-review | locally supported; post-commit CI pending |
| ticket:c10fix11#ACC-004 | evidence:c10fix11-fixture-isolation-verification | critique:c10fix11-fixture-isolation-review | locally supported; post-commit CI pending |
| ticket:c10fix11#ACC-005 | evidence:c10fix11-fixture-isolation-verification | critique:c10fix11-fixture-isolation-review#FIND-001 resolved | locally supported; post-commit CI pending |
| ticket:c10fix11#ACC-006 | evidence:c10fix11-fixture-isolation-verification | critique:c10fix11-fixture-isolation-review#FIND-002 resolved | locally supported; post-commit CI pending |

# Execution Notes

Be careful not to delete user edits in `demo_duckdb`. The pre-implementation integration script used destructive Git cleanup; the current implementation runs against a temp copy and should stay free of normal `git checkout` / `git clean` cleanup.

# Blockers

None.

# Evidence

Existing evidence: evidence:oracle-backlog-scan.

Local implementation evidence: evidence:c10fix11-fixture-isolation-verification records red/green fixture isolation tests, focused pytest, safe integration smoke, `task parse-demo`, source artifact guards, targeted hooks, and `git diff --check`.

Missing evidence: post-commit/main CI evidence after implementation push.

# Critique Disposition

Risk class: high

Critique policy: mandatory

Policy rationale: Fixture isolation underpins the reliability of every compatibility claim.

Required critique profiles: test-coverage, code-change, release-packaging

Findings:

- critique:c10fix11-fixture-isolation-review#FIND-001 - resolved. Fixture-copy exclusions now cover representative ignored/local artifacts and focused tests assert the exclusions.
- critique:c10fix11-fixture-isolation-review#FIND-002 - resolved. CI now has source-artifact guards before and after matrix/latest-core validation.

Disposition status: completed

Deferral / not-required rationale: None.

# Retrospective / Promotion Disposition

Disposition status: pending

Promoted: None yet - retrospective pending.

Deferred / not-required rationale: A wiki/testing note may be warranted after acceptance.

# Wiki Disposition

Pending retrospective review. Existing wiki/operator guidance still contains stale descriptions of the old unsafe integration script and source-tree manifest workflow.

# Acceptance Decision

Accepted by: Not accepted yet.
Accepted at: N/A.
Basis: Pending post-commit CI and retrospective / promotion follow-through. Local evidence and mandatory critique are complete.
Residual risks: Fixture performance cost may increase; fixture exclusions are denylist-based rather than fully `.gitignore`-aware; `task parse-demo` temp copies are left to operating-system temp cleanup.

# Dependencies

Coordinate with ticket:c10ci06 and ticket:c10cfg12.

# Journal

- 2026-05-03T21:10:43Z: Created from tests/fixtures oracle findings.
- 2026-05-04T05:02:48Z: Activated ticket and compiled Ralph packet `packet:ralph-ticket-c10fix11-20260504T050248Z` for test-first fixture isolation implementation.
- 2026-05-04T05:29:21Z: Consumed Ralph output, recorded `evidence:c10fix11-fixture-isolation-verification`, completed mandatory critique `critique:c10fix11-fixture-isolation-review`, resolved critique findings, and moved ticket to `complete_pending_acceptance` pending implementation commit, post-commit CI, and retrospective / promotion follow-through.
