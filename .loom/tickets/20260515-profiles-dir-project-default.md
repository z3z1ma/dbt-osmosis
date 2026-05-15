# Make Project-Local Profiles Discovery Honor Explicit Project Dir

ID: ticket:20260515-profiles-dir-project-default
Type: Ticket
Status: closed
Created: 2026-05-15
Updated: 2026-05-15
Risk: medium - the fix touches shared CLI/bootstrap option resolution used by multiple command families.
Priority: high - the bug breaks common project-local fixture and user workflows unless callers redundantly pass `--profiles-dir`.

## Summary

When a user passes `--project-dir` without `--profiles-dir`, dbt-osmosis still resolves profiles from the current working directory or `~/.dbt` instead of checking the explicit project directory. The bounded result is that commands using shared dbt options discover a project-local `profiles.yml` from the effective project directory when the user did not explicitly supply a profiles directory.

This matters because the CLI help says `--profiles-dir` defaults through the discovered project root, and the demo fixture contains `demo_duckdb/profiles.yml`. From the repository root, `uv run dbt-osmosis sql compile --project-dir demo_duckdb "select 1"` fails today because it tries `/Users/alexanderbutler/.dbt`; adding `--profiles-dir demo_duckdb` succeeds.

## Related Records

- `AGENTS.md` - defines the shared dbt bootstrap spine and canonical CLI examples.
- `src/dbt_osmosis/core/AGENTS.md` - constrains core config changes if the fix lands in `src/dbt_osmosis/core/config.py`.

## Scope

May change CLI option resolution in `src/dbt_osmosis/cli/main.py`, configuration discovery in `src/dbt_osmosis/core/config.py`, and focused tests under `tests/core/`. The fix should cover command families using `dbt_opts` and the `workbench` launcher, because both currently use a callable `discover_profiles_dir` default that cannot see the resolved `project_dir`.

Do not change dbt profile semantics when the user explicitly passes `--profiles-dir`, and do not introduce ad hoc environment management. Keep the resolution behavior aligned with `discover_profiles_dir(project_dir)`.

## Acceptance

- ACC-001: From the repository root, `uv run dbt-osmosis sql compile --project-dir demo_duckdb "select 1"` succeeds by using `demo_duckdb/profiles.yml` when `--profiles-dir` is omitted.
  - Evidence: Add or update a focused CLI/config test and run the command or an equivalent `CliRunner` test against the demo fixture.
  - Audit: Review that the change applies to every shared `dbt_opts` command without requiring per-command copy/paste.

- ACC-002: Explicit `--profiles-dir` values still win unchanged.
  - Evidence: Add a test covering explicit profiles-dir pass-through and run targeted CLI tests.
  - Audit: Review for regressions in `discover_profiles_dir()` search order and environment-variable precedence.

- ACC-003: The workbench launcher either passes no profiles-dir when omitted or derives it from the effective project directory before invoking Streamlit.
  - Evidence: Add or update a workbench CLI test that inspects the Streamlit command arguments.
  - Audit: Review for compatibility with `--options`, `--config`, and extra Streamlit args.

## Current State

Closed. The CLI-only fix in `src/dbt_osmosis/cli/main.py` now resolves an omitted `profiles_dir` after Click has resolved `project_dir`, using `discover_profiles_dir(project_dir)`. Explicit `--profiles-dir` values pass through unchanged. The workbench launcher uses the same resolver before building app arguments.

Observed before the fix:

- `uv run dbt-osmosis sql compile --project-dir demo_duckdb "select 1"` failed with `Could not find profile named 'jaffle_shop'` after resolving `profiles_dir='/Users/alexanderbutler/.dbt'`.
- `uv run dbt-osmosis sql compile --project-dir demo_duckdb --profiles-dir demo_duckdb "select 1"` succeeded and printed `select 1`.

Verification after the fix:

- `uv run ruff check src/dbt_osmosis/cli/main.py tests/core/test_cli.py` passed.
- `uv run pytest tests/core/test_cli.py` passed with 39 passed and 3 deprecation warnings.
- `uv run dbt-osmosis sql compile --project-dir demo_duckdb "select 1"` passed and printed `select 1`.
- Final repo verification passed: `uv run pytest` with 943 passed and 11 skipped, `uv run ruff check`, and `uv run dbt-osmosis sql compile --project-dir demo_duckdb "select 1"`.
- Final type-check observation: `uv run basedpyright` reported 0 errors and existing warning-only output.

## Journal

- 2026-05-15: Created ticket with Status `open` from repo bug scan evidence. Automated baseline at creation time: `uv lock --check` passed, `uv run ruff check` passed, `uv run dbt parse --project-dir demo_duckdb --profiles-dir demo_duckdb -t test` passed, `uv run pytest` passed with 933 passed and 11 skipped, and `uv run basedpyright` reported 0 errors with existing warnings.
- 2026-05-15: Activated for Ralph execution via `packet:20260515T085606Z-profiles-dir-resolution`.
- 2026-05-15: Implemented the profiles-dir resolution fix in the shared `dbt_opts` wrapper and mirrored it in the `workbench` launcher. Added focused CLI tests for omitted project-local profiles discovery, explicit profiles-dir pass-through, and workbench script argument construction.
- 2026-05-15: Verification passed: `uv run ruff check src/dbt_osmosis/cli/main.py tests/core/test_cli.py`, `uv run pytest tests/core/test_cli.py`, and `uv run dbt-osmosis sql compile --project-dir demo_duckdb "select 1"`.
- 2026-05-15: Parent review closed the ticket after final verification: `uv run pytest` passed with 943 passed and 11 skipped, `uv run ruff check` passed, the original demo compile command passed, and `uv run basedpyright` reported 0 errors with existing warnings.
