# Ralph Packet: Project-Local Profiles Discovery

ID: packet:20260515T085606Z-profiles-dir-resolution
Type: Ralph Packet
Status: consumed
Created: 2026-05-15
Updated: 2026-05-15
Branch: main
Context Style: hybrid
Mode: execution
Worker: subagent
Verification Posture: test-first, then focused command proof

## Ticket

- `ticket:20260515-profiles-dir-project-default`

## Objective

Fix shared CLI option resolution so commands that accept `--project-dir` and omit `--profiles-dir` discover a project-local `profiles.yml` from the effective project directory. Preserve explicit `--profiles-dir` behavior.

## Read Scope

- `AGENTS.md`
- `src/dbt_osmosis/cli/main.py`
- `src/dbt_osmosis/core/config.py`
- `tests/core/test_cli.py`
- `tests/core/conftest.py`
- `demo_duckdb/profiles.yml`

## Write Scope

- `src/dbt_osmosis/cli/main.py`
- `src/dbt_osmosis/core/config.py` only if a CLI-only fix is insufficient
- `tests/core/test_cli.py`
- `.loom/tickets/20260515-profiles-dir-project-default.md`
- this packet

## Constraints

- Workers are not alone in the codebase. Do not revert edits by others; adjust your implementation to fit any concurrent changes.
- Do not change explicit `--profiles-dir` semantics.
- Do not add ad hoc environment management.
- Keep the behavior aligned with `discover_profiles_dir(project_dir)`.
- Cover `dbt_opts` commands and the `workbench` launcher.

## Acceptance

- `uv run dbt-osmosis sql compile --project-dir demo_duckdb "select 1"` succeeds from repo root without explicit `--profiles-dir`.
- Explicit `--profiles-dir` values still pass through unchanged.
- Workbench command construction uses the effective project-local profiles dir when omitted.

## Verification Commands

- `uv run pytest tests/core/test_cli.py`
- `uv run dbt-osmosis sql compile --project-dir demo_duckdb "select 1"`
- `uv run ruff check src/dbt_osmosis/cli/main.py tests/core/test_cli.py`

## Stop Conditions

- Stop and report if resolving omitted `profiles_dir` requires changing dbt configuration semantics outside CLI bootstrap.
- Stop and report if existing tests reveal incompatible assumptions around Click defaults.

## Output Contract

Update this packet with a `## Worker Report` section containing changed files, commands run, results, and unresolved risks. Update the ticket journal with concise implementation and verification evidence.

## Worker Report

Outcome: stop

Changed files:

- `src/dbt_osmosis/cli/main.py`
- `tests/core/test_cli.py`
- `.loom/tickets/20260515-profiles-dir-project-default.md`
- `.loom/packets/ralph/20260515T085606Z-profiles-dir-resolution.md`

Implementation:

- Changed shared `dbt_opts` so an omitted `--profiles-dir` is resolved inside the wrapper after Click has resolved the effective `--project-dir`.
- Added `_resolve_profiles_dir(project_dir, profiles_dir)` to preserve explicit `--profiles-dir` values unchanged and call `discover_profiles_dir(project_dir)` only for omitted values.
- Updated `workbench` to use the same resolver before passing app args to Streamlit.
- Added focused tests for omitted project-local profiles discovery, explicit profiles-dir pass-through, and workbench script argument construction.

Commands run:

- `uv run dbt-osmosis sql compile --project-dir demo_duckdb "select 1"` before implementation: failed, resolving `profiles_dir='/Users/alexanderbutler/.dbt'` and reporting `Could not find profile named 'jaffle_shop'`.
- `uv run ruff check src/dbt_osmosis/cli/main.py tests/core/test_cli.py`: passed.
- `uv run pytest tests/core/test_cli.py`: passed, 39 passed and 3 deprecation warnings.
- `uv run dbt-osmosis sql compile --project-dir demo_duckdb "select 1"` after implementation: passed and printed `select 1`.

Unresolved risks:

- No separate audit pass was run in this worker packet. The ticket is left in `review` with targeted command evidence recorded.

## Parent Review

Outcome: accepted

Final verification from the main workspace:

- `uv run pytest` passed with 943 passed and 11 skipped.
- `uv run ruff check` passed.
- `uv run dbt-osmosis sql compile --project-dir demo_duckdb "select 1"` passed and printed `select 1`.
- `uv run basedpyright` reported 0 errors with existing warning-only output.

The referenced ticket was closed after parent review.
