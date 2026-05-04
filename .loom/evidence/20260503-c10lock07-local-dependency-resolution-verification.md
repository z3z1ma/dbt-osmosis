---
id: evidence:c10lock07-local-dependency-resolution-verification
kind: evidence
status: recorded
created_at: 2026-05-03T23:50:14Z
updated_at: 2026-05-04T00:19:53Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  tickets:
    - ticket:c10lock07
  packets:
    - packet:ralph-ticket-c10lock07-20260503T234103Z
  wiki:
    - wiki:ci-compatibility-matrix
  critique:
    - critique:c10lock07-dependency-resolution
    - critique:c10lock07-integration-path-follow-up
external_refs:
  github_issue: https://github.com/z3z1ma/dbt-osmosis/issues/358
---

# Summary

Observed the local result of the `ticket:c10lock07` Ralph implementation and follow-up critique fix that harden CI dependency resolution. The changes add a uv lock freshness gate, clean per-matrix environments, `uv --no-config` matrix installs, dependency consistency checks, exact version logging, a direct matrix integration path, and a plain pip install smoke outside uv resolution.

# Procedure

Observed at: 2026-05-03T23:50:14Z

Source state: local worktree at `9f743e4eb8da7b47cc0fed82e28e7b650666da6e` plus uncommitted c10lock07 changes to `.github/workflows/tests.yml`, `Taskfile.yml`, packet, ticket, and this evidence record.

Procedure:

- Ralph child inspected the pre-change workflow and Taskfile and reported missing or weak lock freshness, clean matrix isolation, dependency consistency checks, and pip install smoke coverage.
- Ralph child implemented the workflow and Taskfile changes within the packet write scope.
- Ralph child ran local lock, workflow lint, pip smoke, and uv matrix smoke commands.
- Parent inspected the diff and reran local lock freshness, Taskfile syntax, workflow hooks, and pip smoke.
- Mandatory critique identified that the matrix integration step still called `demo_duckdb/integration_tests.sh`, whose `uv run` calls could resync away from the matrix runtime.
- Parent updated the workflow integration step to call `dbt-osmosis` directly from the matrix environment and reran workflow hooks; follow-up critique passed.

Expected result when applicable: local structural and smoke checks should pass before mandatory critique, and the diff should satisfy the ticket's acceptance shape without broad package metadata cleanup.

Actual observed result: local checks passed. The workflow now has a `lockfile` job using `uv lock --check`, matrix jobs create clean virtual environments and install with `uv --no-config pip install`, matrix/latest jobs run dependency consistency checks, version-reporting steps print uv and package versions, the matrix integration step invokes `dbt-osmosis` directly without `uv run`, and a pip resolver smoke installs `.[openai]` with dbt 1.11 and dbt-duckdb 1.10.

Procedure verdict / exit code: pass for local verification; final GitHub Actions evidence remains pending after commit/push.

# Artifacts

Ralph child verification reported:

- `uv lock --check` - passed, exit 0.
- `uvx --from uv==0.5.13 uv lock --check` - passed, exit 0.
- `task --list` - passed, exit 0.
- `task lock` - passed, exit 0.
- `uvx pre-commit run check-yaml --files .github/workflows/tests.yml Taskfile.yml` - passed, exit 0.
- `uvx pre-commit run actionlint --files .github/workflows/tests.yml` - passed, exit 0.
- Local plain pip smoke with `.[openai]`, `dbt-core~=1.11.0`, and `dbt-duckdb~=1.10.1` - passed, exit 0; `pip check` passed; installed dbt-core 1.11.8 and dbt-duckdb 1.10.1.
- `task pip-install-smoke` - passed, exit 0.
- Final uv matrix smoke with `uv --no-config pip install` and `uv --no-config pip check` - passed, exit 0.
- Same uv matrix smoke using constrained `uv==0.5.13` - passed, exit 0.

Parent verification reran:

- `uv lock --check` - passed, exit 0; output included `Resolved 156 packages in 14ms`.
- `task --list` - passed, exit 0 and listed the new `lock` and `pip-install-smoke` tasks.
- `pre-commit run --files .github/workflows/tests.yml Taskfile.yml` - passed, exit 0, including YAML checks and actionlint.
- `task pip-install-smoke` - passed, exit 0; `pip check` reported `No broken requirements found`, `dbt-core: 1.11.8`, `dbt-duckdb: 1.10.1`, `dbt-osmosis: 1.3.0`, `dbt-osmosis --help` succeeded, and `import dbt_osmosis.cli.main` succeeded.
- Follow-up workflow hook after replacing the integration script call with direct `dbt-osmosis` commands: `pre-commit run --files .github/workflows/tests.yml Taskfile.yml` - passed, exit 0, including YAML checks and actionlint.
- Follow-up critique verdict: `critique:c10lock07-integration-path-follow-up` returned `pass` with no open findings.

# Supports Claims

- `ticket:c10lock07#ACC-001` - local `uv lock --check` passes and CI now has a `lockfile` job that should fail stale locks.
- `ticket:c10lock07#ACC-002` - workflow and Taskfile matrix jobs now create clean virtual environments and install matrix dependencies with `uv --no-config pip install`; CI integration coverage uses direct `dbt-osmosis` commands from the matrix environment; local uv matrix smoke passed.
- `ticket:c10lock07#ACC-003` - workflow and Taskfile matrix/latest jobs now run `uv --no-config pip check`; local uv and pip smoke checks passed.
- `ticket:c10lock07#ACC-004` - workflow steps continue to assert Python, dbt-core, dbt-duckdb, and dbt-osmosis versions and now also print `uv --version`; local smoke output printed exact installed versions.
- `ticket:c10lock07#ACC-005` - workflow and Taskfile now include a plain pip install smoke; local `task pip-install-smoke` passed outside uv resolution.

# Challenges Claims

- `ticket:c10lock07#ACC-001` through `ticket:c10lock07#ACC-005` remain pending final GitHub Actions evidence from `main` because this evidence is local plus structural.

# Environment

Commit: `9f743e4eb8da7b47cc0fed82e28e7b650666da6e` plus uncommitted c10lock07 changes.

Branch: local worktree branch `loom/dbt-110-111-hardening`; delivery target remains `main`.

Runtime: local macOS/darwin for parent checks and Ralph child checks.

OS: macOS/darwin.

Relevant config: `.github/workflows/tests.yml`, `Taskfile.yml`, `pyproject.toml`, `uv.lock`.

External service / harness / data source when applicable: none for this local evidence; GitHub Actions evidence is still pending.

# Validity

Valid for: supporting mandatory critique and a commit/push trial for `ticket:c10lock07`.

Fresh enough for: review of the current uncommitted c10lock07 diff.

Recheck when: workflow dependency installation steps, Taskfile compatibility tasks, uv lock behavior, package extras, dbt support policy, or adapter mappings change.

Invalidated by: failing GitHub Actions runs for these changes, removal of `uv --no-config`, removal of dependency consistency checks, or broad package metadata changes without new evidence.

Supersedes / superseded by: should be superseded by final `main` CI evidence after this change is pushed.

# Limitations

- Full matrix pytest/integration was not run locally because of cost.
- Parent pip smoke used local Python 3.12; CI pip smoke is configured for Python 3.13.
- Existing `[tool.uv] override-dependencies = ["protobuf>=5.0,<6.0"]` remains in `pyproject.toml`; matrix installs intentionally bypass project uv config with `uv --no-config` to avoid hiding dbt 1.11 resolver behavior. Broader metadata cleanup remains with `ticket:c10pkg10`.

# Result

The local c10lock07 implementation verification and follow-up critique fix passed and support committing/pushing for final GitHub Actions evidence.

# Interpretation

The diff appears to satisfy the ticket's dependency-resolution shape locally: matrix jobs are clean rather than lock-inherited, lock freshness is checked, dependency consistency is validated, versions are logged, integration coverage runs from the matrix environment directly, and pip metadata is exercised outside uv resolution. The ticket still needs `main` CI evidence before acceptance.

# Related Records

- `ticket:c10lock07`
- `packet:ralph-ticket-c10lock07-20260503T234103Z`
- `wiki:ci-compatibility-matrix`
- `ticket:c10ci06`
- `critique:c10lock07-dependency-resolution`
- `critique:c10lock07-integration-path-follow-up`
