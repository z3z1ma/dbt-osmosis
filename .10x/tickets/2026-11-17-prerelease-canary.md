Status: active
Created: 2026-11-17
Updated: 2026-11-17
Parent: ../knowledge/maintenance-roadmap.md

# Python Core prerelease canary

## Scope
Add a distinct scheduled/manual non-blocking Python Core 1.x prerelease canary, with explicit candidate selection and real compatibility execution. Existing stable canary and supported matrix remain unchanged. Execute against current non-yanked candidate and record findings; do not promote audited support.

## Non-goals
No package bounds/lock edits, legacy support removal, Rust v2 runtime audit, warehouse credentials, release or deployment. No roadmap edits (another worker owns it).

## Acceptance Criteria
1. Candidate selection excludes non-1.x, yanked/empty releases and prereleases not newer than the latest usable stable 1.x release; no candidate is an explicit skip, metadata failures are failures.
2. Only the selected Core prerelease is opted in; do not globally permit prereleases for unrelated dependencies. Installed exact version is asserted and actual adapter versions logged, pip check passes.
3. Dedicated schedule/dispatch-only continue-on-error job exercises type checks, temporary-project parse, CLI imports/help and pytest; failures are visible, never reclassified as successful validation. Existing blocking jobs and stable canary remain unchanged.
4. Meaningful selector policy tests cover actual traps: 2.0rc, yanked1.13rc, 1.13beta, old prerelease and no candidate. Documentation distinguishes prerelease detection from audited support.
5. Local current-candidate audit and independent review recorded; draft PR and exact-head remote CI including manual canary before merge; coordinate merge window with root.

## References and assumptions
- User authorized implementation of next roadmap priorities. Existing roadmap P3 calls for bounded prerelease detection; current supported versions stay unchanged.
- https://pypi.org/project/dbt-core/ checked2026-11-17: latest stable1.12.7;1.13.0b1 current non-yanked preview;1.13.0rc1 yanked;2.0.0rc10 also occupies namespace. A global --pre is not a Python-runtime-only audit.
- https://github.com/dbt-labs/dbt-core/releases/tag/v1.13.0b1 redirects to dbt-labs/dbt and documents14Oct2026 beta plus flag-default changes.
- https://docs.astral.sh/uv/concepts/resolution/ describes explicit prerequisite version handling; install exact selected version without global--pre.
- Record-backed existing workflow stablecanary runs Python3.13, dbt-duckdb unconstrained within project extra and DBT_ADAPTERS_CONSTRAINT. Reuse it for prerelease validation in isolated environment.

## Journal
- Fresh worktree ci/prerelease-canary at f1d8947f. Read10x/AGENTS/workflow and existing canary policy tests. No existing prerelease owner found.

## Blockers
None for implementation. Remote publication/gates pending.

## Evidence
Pending executor commands and results.

## Review
Pending independent review.

## Retrospective
Pending.

## Execution journal (append-only)
- Implemented `.github/scripts/select_dbt_prerelease.py`, policy tests, separate `prerelease-dbt-canary` job and support-policy documentation. The selector validates release/file structure, excludes empty/all-yanked releases, non-1.x/epoch/local candidates, and obsolete previews; no upcoming preview writes an empty output and explicit skip message. Metadata/network failures propagate. Exact selected Core version is installed without global prerelease flags.
- Verified the old workflow is byte-identical after removing the new job from the working file. Existing matrix and stable canary preserved, including original supported versions. `ruamel.yaml` safely parsed the resulting workflow and confirmed the separate seven-step job. `git diff --check` passed; no dependency manifest or lockfile changed.
- `uv venv --python 3.13 /tmp/osmosis-prerelease-audit-env` then `uv --no-config pip install --python /tmp/osmosis-prerelease-audit-env/bin/python -e '.[dev,openai,duckdb]' 'dbt-core==1.13.0b1' dbt-duckdb 'dbt-adapters>=1.16.3,<2.0'` succeeded without global prerelease permission. Installed Python3.13.14, Core1.13.0b1, adapter1.11.1, dbt-adapters1.24.5, SQLGlot30.32.0, DuckDB2.0.0. `uv --no-config pip check --python /tmp/osmosis-prerelease-audit-env/bin/python` passed (115 packages).
- Live selector command `/tmp/osmosis-prerelease-audit-env/bin/python .github/scripts/select_dbt_prerelease.py` returned `1.13.0b1`, skipping the yanked1.13rc and2.0rc namespace traps.
- `/tmp/osmosis-prerelease-audit-env/bin/pytest -q tests/test_prerelease_canary.py tests/test_package_metadata.py`:22 passed in0.22s after adding the workflow-gating assertion. Ruff check and format check passed for all changed Python files.
- Initial local basedpyright invocation lacked exported candidate PATH/VIRTUAL_ENV and reported107 unresolved-import errors; that observation is invalid as a runtime audit. Re-running with `VIRTUAL_ENV=/tmp/osmosis-prerelease-audit-env PATH=/tmp/osmosis-prerelease-audit-env/bin:$PATH basedpyright --outputjson` produced0 errors/1990 warnings (39 files). Workflow already exports both before checking, matching the existing stable canary. Raw corrected summary `/tmp/osmosis-prerelease-typecheck-correct.json`.
- Actual candidate audit script `/tmp/run-osmosis-prerelease-audit.sh` uses isolated temporary DuckDB project copy, runs `dbt parse`, CLI help/imports and `pytest -q`. Parse/help/imports passed; full suite still executing. Raw log `/tmp/osmosis-prerelease-full-audit.log`. No external warehouse credentials or model execution outside fixture projects.

## Independent review (prerelease_audit/review, 2026-11-17)
Verdict: **pass**, conditional on the planned exact-head blocking CI and an executed manual prerelease canary before merge. No critical or significant findings.

Reviewed the working diff, selector, policy assertions and CI command flow without repeating executor tests. The selector requires a valid releases object/file-yanked metadata, ignores empty/all-yanked releases, excludes non-1.x/epochs/local versions, and selects a prerelease only when newer than the latest non-yanked stable 1.x release. Missing stable metadata raises rather than reporting no candidate. The canonical `Version` string prevents arbitrary release-key text reaching GitHub output or shell interpolation; the workflow supplies it through an environment variable and quotes the exact Core requirement.

The new schedule/dispatch-only job preserves the existing blocking jobs and stable canary in the inspected diff, and explicitly opts in only the selected Core version. Metadata/network failure fails its selection step; GitHub's default success gating prevents later no-candidate reporting from converting that failure into a skip. A genuine absent candidate is reported in the job summary. The validation environment sets both VIRTUAL_ENV and PATH before type/import/CLI/tests, asserts the installed exact candidate and Python generation, and retains pip/type/parse/pytest failure checks. Documentation does not expand audited support.

Residual risk: PyPI availability and adapter resolution remain external inputs; a future candidate may be incompatible with Python 3.13 or the adapter constraints, intentionally producing a visible non-blocking canary failure. The stable canary remains untouched, including its existing behavior. Static review does not establish GitHub Actions execution or full prerelease runtime compatibility; the executor's full-suite and remote manual-run evidence must close those gates. No test assertions were weakened.

### Completed local audit and bounded compatibility finding
- Full actual Core1.13.0b1 suite completed: **1104 passed, 11 skipped, 1 failed in369.75s**. The sole failure was `tests/core/test_real_config_shapes.py::test_real_dbt_parse_exposes_dbt_110_config_shapes`: raw `node.unrendered_config["dbt_osmosis_skip_add_columns"]` changed from Boolean `False` to string `"False"`. The immediately preceding rendered `node.config.extra` Boolean assertion passed. Preserve this initial finding; it was not a full-suite pass.
- Direct real-parser probes using the same temporary project fixture and `SettingsResolver.resolve("skip-add-columns", node, fallback=True)` recorded:
  - Core1.13.0-b1: raw `'False'`, rendered `False`, resolved `False` (`/tmp/osmosis-beta-config-probe.log`).
  - Core1.12.7: raw `False`, rendered `False`, resolved `False` (`/tmp/osmosis-stable-config-probe.log`). Stable comparator installed in separate `/tmp/osmosis-prerelease-stable-probe` with the same Python3.13 and adapter1.11.1/adapters1.24.5; no shared environment mutated.
- Root/parent authorized the bounded compatibility-test expansion following these probes. The real config-shape test now expects exact string `"False"` on Core>=1.13.0b1, retains exact Boolean `False` on earlier supported Core, and explicitly asserts resolver output `is False` despite a True fallback. No runtime source changed, no arbitrary truthy string accepted, and no broad skip introduced. The rendered configuration source already precedes raw unrendered configuration.
- Final focused command in each isolated environment: `pytest -q tests/core/test_real_config_shapes.py tests/test_prerelease_canary.py tests/test_package_metadata.py`: **24 passed in3.46s** on1.13.0b1 and **24 passed in4.15s** on1.12.7. Logs `/tmp/osmosis-beta-focused.log` and `/tmp/osmosis-stable-focused.log`. Ruff check/format check for all four modified/new Python files and `git diff --check` passed. The full local suite was not repeated solely for this assertion change; exact-head remote full-suite/canary execution remains a required gate.
- Audit recommendation: retain the rendered-config precedence as the semantic contract; continue checking upstream raw shapes precisely so future parser changes remain visible. The beta emitted `MissingArgumentsPropertyInGenericTestDeprecation` for fixture generic-test arguments; this is a separate follow-up fixture migration, not suppressed here. Current evidence does not expand advertised support or imply Fusion compatibility.
- Execution retrospective: explicit candidate selection avoided both the yanked1.13rc and Rust2.x prereleases; a real beta audit caught upstream representation drift that version inventory alone could not. Remaining gates belong to the integration owner: independent review of the final test delta, refresh against current main, new draft PR, exact-head blocking CI and actual manual canary execution, then coordinated validated merge/post-merge verification. No commit/publication performed by executor.

## Independent review addendum: real config-shape assertion (2026-11-17)
Verdict: **pass** for the bounded test-only delta in `tests/core/test_real_config_shapes.py`; no critical or significant findings. Reviewed the actual assertions without repeating executor runs. The version branch uses the existing installed-version helper and `packaging.version` ordering, with an explicit `1.13.0b1` boundary matching the observed beta. It retains strict boolean identity for older runtimes and requires the exact observed `"False"` raw representation for the beta and later versions; it does not accept either shape indiscriminately. A future parser change will still fail visibly rather than silently pass.

The existing rendered `config.extra` assertion remains `is False`, and the added `SettingsResolver.resolve(..., fallback=True) is False` proves the behavior contract separately from parser representation. A truthy string or accidental fallback cannot satisfy that assertion. No production path, dependency bound, supported-version declaration, or skip rule changed. Focused execution on both observed runtimes and the corrected full prerelease run remain executor/CI evidence gates; review does not claim those results.
