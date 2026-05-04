---
id: evidence:c10bounds29-support-policy-validation
kind: evidence
status: recorded
created_at: 2026-05-04T23:38:36Z
updated_at: 2026-05-04T23:45:07Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  ticket:
    - ticket:c10bounds29
  packet:
    - packet:ralph-ticket-c10bounds29-20260504T233210Z
external_refs: {}
---

# Summary

Observed test-first validation for the keep-open-plus-canary dbt support policy: package metadata remains open at `dbt-core>=1.8`, docs distinguish audited blocking support from future-minor canaries, CI declares a scheduled/manual latest-dbt canary, and pytest/CI warning filters surface dbt/dbt-osmosis deprecations instead of ignoring all deprecations.

# Procedure

Observed at: 2026-05-04T23:38:36Z

Source state: commit `0af38b2b7c2db6b448418525d0dbfdc3068e37bb` on branch `loom/dbt-110-111-hardening` with uncommitted c10bounds29 changes in `pyproject.toml`, `.github/workflows/tests.yml`, `tests/test_package_metadata.py`, `README.md`, `docs/docs/intro.md`, and `docs/docs/tutorial-basics/installation.md`.

Procedure: Ralph child added structural support-policy tests before implementation and ran `uv run pytest tests/test_package_metadata.py -q` to capture the expected red result. Child then implemented the support policy changes and ran the same test plus touched-file pre-commit. Parent reviewed the implementation diff and reran package metadata tests, whitespace validation, and touched-file pre-commit. After mandatory critique and ticket-owned low-risk dispositions, parent ran full pre-commit, package metadata tests, and `uv lock --check`.

Expected result when applicable: red structural tests fail on the pre-fix state because docs do not state the future-minor canary policy, `.github/workflows/tests.yml` lacks `future-dbt-canary`, and pytest config still contains blanket `ignore::DeprecationWarning`. Post-fix tests pass while showing relevant dbt deprecation warnings instead of suppressing them.

Actual observed result: child observed the expected red with `3 failed, 6 passed`; parent observed green package metadata tests with `9 passed, 2 warnings` and visible dbt deprecation warnings from `dbt/cli/options.py`; parent observed touched-file pre-commit pass. Parent final validation observed full pre-commit pass, package metadata tests still green with the same visible dbt deprecation warnings, and `uv lock --check` exit successfully.

Procedure verdict / exit code: mixed expected red then green. Parent-observed green commands exited successfully.

# Artifacts

Child-reported red command:

```bash
uv run pytest tests/test_package_metadata.py -q
```

Child-reported red result:

```text
3 failed, 6 passed
```

Expected red failure summary:

- Docs did not state audited dbt Core 1.8.x-1.11.x support and future-minor canary-only policy.
- Workflow lacked `future-dbt-canary`.
- Pytest config still contained blanket `ignore::DeprecationWarning`.

Parent-observed green package metadata test:

```bash
uv run pytest tests/test_package_metadata.py -q
```

```text
.........                                                                [100%]
=============================== warnings summary ===============================
.venv/lib/python3.10/site-packages/dbt/cli/options.py:6
  /Users/alexanderbutler/code_projects/personal/dbt-osmosis-dbt-110-111-hardening/.venv/lib/python3.10/site-packages/dbt/cli/options.py:6: DeprecationWarning: 'parser.OptionParser' is deprecated and will be removed in Click 9.0. The old parser is available in 'optparse'.
    from click.parser import OptionParser, ParsingState

.venv/lib/python3.10/site-packages/dbt/cli/options.py:6
  /Users/alexanderbutler/code_projects/personal/dbt-osmosis-dbt-110-111-hardening/.venv/lib/python3.10/site-packages/dbt/cli/options.py:6: DeprecationWarning: 'parser.ParsingState' is deprecated and will be removed in Click 9.0. The old parser is available in 'optparse'.
    from click.parser import OptionParser, ParsingState

9 passed, 2 warnings in 0.11s
```

Parent-observed whitespace check:

```bash
git diff --check -- pyproject.toml .github/workflows/tests.yml tests/test_package_metadata.py README.md docs/docs/intro.md docs/docs/tutorial-basics/installation.md
```

```text
no output
```

Parent-observed touched-file pre-commit:

```bash
uv run pre-commit run --files pyproject.toml .github/workflows/tests.yml tests/test_package_metadata.py README.md docs/docs/intro.md docs/docs/tutorial-basics/installation.md
```

```text
check python ast.........................................................Passed
check json...........................................(no files to check)Skipped
check yaml...............................................................Passed
check toml...............................................................Passed
fix end of files.........................................................Passed
trim trailing whitespace.................................................Passed
detect private key.......................................................Passed
debug statements (python)................................................Passed
ruff-format..............................................................Passed
ruff.....................................................................Passed
basedpyright.............................................................Passed
Detect hardcoded secrets.................................................Passed
Lint GitHub Actions workflow files.......................................Passed
```

Parent-observed final validation after mandatory critique:

```bash
uv run pre-commit run --all-files && uv run pytest tests/test_package_metadata.py -q && uv lock --check
```

```text
check python ast.........................................................Passed
check json...............................................................Passed
check yaml...............................................................Passed
check toml...............................................................Passed
fix end of files.........................................................Passed
trim trailing whitespace.................................................Passed
detect private key.......................................................Passed
debug statements (python)................................................Passed
ruff-format..............................................................Passed
ruff.....................................................................Passed
basedpyright.............................................................Passed
Detect hardcoded secrets.................................................Passed
Lint GitHub Actions workflow files.......................................Passed
.........                                                                [100%]
9 passed, 2 warnings in 0.09s
Resolved 165 packages in 5ms
```

Parent diff inspection observed:

- `pyproject.toml` keeps `dbt-core>=1.8` and replaces blanket `ignore::DeprecationWarning` with targeted `default::DeprecationWarning` filters for `dbt` and `dbt_osmosis`.
- `.github/workflows/tests.yml` adds `workflow_dispatch`, workflow-level `PYTHONWARNINGS`, and a scheduled/manual non-blocking `future-dbt-canary` job that installs unpinned latest `dbt-core` and `dbt-duckdb`, checks versions and dependency consistency, runs basedpyright, parses the demo project, imports CLI modules, runs `pytest -q`, and checks fixture cleanliness.
- README and docs now distinguish audited blocking support for dbt Core 1.8.x through 1.11.x from canary-only future dbt Core minors.
- `tests/test_package_metadata.py` structurally guards the open dependency, docs policy, canary workflow, and warning filters.

# Supports Claims

- ticket:c10bounds29#ACC-001: package metadata remains open as `dbt-core>=1.8`, and structural tests/doc wording describe future minors as intentional canaries rather than audited support.
- ticket:c10bounds29#ACC-002: README, intro docs, and installation docs state audited dbt Core support, adapter compatibility responsibility, and future-minor canary policy.
- ticket:c10bounds29#ACC-003: `.github/workflows/tests.yml` now declares scheduled/manual `future-dbt-canary` using unpinned latest `dbt-core` and `dbt-duckdb` with non-blocking canary semantics.
- ticket:c10bounds29#ACC-004: blanket `ignore::DeprecationWarning` was removed from pytest config and replaced by targeted dbt/dbt-osmosis visibility filters.
- ticket:c10bounds29#ACC-005: local green pytest output visibly reports dbt deprecation warnings, and CI has workflow-level `PYTHONWARNINGS` for dbt/dbt-osmosis deprecation visibility.
- initiative:dbt-110-111-hardening#OBJ-001: support policy now distinguishes audited dbt matrix support from future-minor canaries.
- initiative:dbt-110-111-hardening#OBJ-005: CI policy and package metadata are aligned with the open dependency strategy.

# Challenges Claims

None observed for the post-fix source state.

# Environment

Commit: `0af38b2b7c2db6b448418525d0dbfdc3068e37bb` plus uncommitted c10bounds29 implementation and Loom record changes.

Branch: `loom/dbt-110-111-hardening`

Runtime: `uv run` project environment; `VIRTUAL_ENV` mismatch warning observed and ignored by `uv`.

OS: macOS 15.7.5 build 24G624

Relevant config: base `uv` environment; repository pre-commit hooks; local dbt package set in `.venv`.

External service / harness / data source when applicable: none for local validation.

# Validity

Valid for: local source state containing the c10bounds29 support-policy diff and the installed local dependency set.

Fresh enough for: mandatory release-packaging/operator-clarity/dbt-compatibility critique and pre-commit/commit readiness.

Recheck when: `pyproject.toml`, `.github/workflows/tests.yml`, support-policy docs, package metadata tests, dependency constraints, or warning policy changes.

Invalidated by: failing package metadata tests, YAML/TOML/pre-commit failures, a change that adds a `dbt-core` upper bound without a new decision, or evidence that future dbt minors are still described as audited support instead of canary-only.

Supersedes / superseded by: none.

# Limitations

This evidence does not execute the new GitHub Actions `future-dbt-canary` job, does not install the latest upstream dbt packages in CI, and does not prove future dbt minors pass. The canary is intentionally non-blocking and scheduled/manual, so future upstream breakage remains a visible signal to triage rather than current audited support evidence.

# Result

The observed post-fix source state passes structural package metadata policy tests and touched-file pre-commit while surfacing dbt deprecation warnings locally.

# Interpretation

This evidence supports commit readiness when combined with `critique:c10bounds29-support-policy-review`. It does not by itself satisfy remote CI expectations or final acceptance decision.

# Related Records

- ticket:c10bounds29
- packet:ralph-ticket-c10bounds29-20260504T233210Z
- initiative:dbt-110-111-hardening
- research:dbt-110-111-api-surfaces
