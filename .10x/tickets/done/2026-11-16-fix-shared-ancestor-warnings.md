Status: done
Created: 2026-11-16
Updated: 2026-11-16

# Fix shared-ancestor cycle warnings (#452)

## Scope
Reproduce https://github.com/z3z1ma/dbt-osmosis/issues/452 on main and distinguish false warnings from inheritance loss. Separate recursion-path cycle detection from global deduplication. Preserve existing depth-first first-discovery generation placement and sorted per-generation precedence.

## Non-goals
Changing ancestor precedence, shortest-path semantics, unrelated CI assertions, cross-package YAML (#443), merge or release.

## Acceptance Criteria
- Diamond and unequal-depth shared ancestors cause no cycle warning and occur once.
- Real cycles terminate and warn, including cycles not involving the root.
- Knowledge inheritance and same-generation precedence remain intact.
- Focused and aggregate checks are recorded; unrelated failures disclosed.

## References
- Issue #452 (canonical report, open as observed 2026-11-16).
- `src/dbt_osmosis/core/inheritance.py`, `tests/core/test_inheritance_behavior.py`.
- `.10x/tickets/done/2026-07-05-reduce-inheritance-complexipy-hotspots.md` (historical preservation scope).

## Assumptions
- User-authorized: isolated fix, regression tests, commit, push and draft PR.
- Source-backed: global visited deduplicates ancestors at first DFS discovery; generations are sorted and merged farthest to closest. Retain that behavior.

## Journal
- Fetched origin/main: 3cc00a57. Isolated clone/worktree under task-3 avoids modifying other workers' checkout or shared git metadata.
- Source inspection confirms visited is conflated with active recursion path. A shared ancestor is already included before the false warning; warning alone cannot explain blanket loss of inheritance.

## Blockers
None for implementation.

## Evidence
All local acceptance criteria verified below. Remote CI is observed separately and remains pending at handoff.

## Review
Independent reviewer verdict: pass, no findings. See verification journal.

## Retrospective
A global visited set answers whether a node was already processed, not whether it is on the active path. Keep these responsibilities separate. Compare inherited values before and after a warning fix; the report’s symptom attribution was not supported by reproduction.

### Verification journal (2026-11-16)
- Baseline `.venv/bin/pytest -q tests/core/test_inheritance_shared_ancestors.py`: 9 failed, 3 passed (28.13s). All DAG failures were `warning.assert_not_called()`; tree and inherited-description assertions preceding them passed. Root, non-root and self cycles passed. This reproduces false warnings, not missing documentation, on main 3cc00a57.
- Implementation adds branch-local `active_path`, checked before global visited. Keeps all original global visited updates, traversal order, generation sorting and depth guards.
- `.venv/bin/ruff check`: pass. `.venv/bin/ruff format --check --preview`: 366 files formatted. `.venv/bin/basedpyright --outputjson`: 0 errors, 1980 warnings; warnings remain non-blocking under repository policy.
- `uv lock --check`: pass (178 packages). Isolated locked environment: Python 3.13.14, dbt-core 1.10.20, dbt-duckdb 1.10.0, dbt-adapters 1.22.4.
- Independent review `/root/review_452`: pass, no findings. Confirmed active-path handling, deduplication, first-discovery placement, cycle coverage and inheritance precedence. No tests rerun by reviewer.
- Limitations accepted for this bounded fix: unequal-depth ancestors retain first-DFS placement rather than shortest-path semantics; reporter's broader inheritance-loss claim remains unconfirmed. No new precedence policy is introduced.
- Focused `.venv/bin/pytest -q tests/core/test_inheritance_shared_ancestors.py tests/core/test_inheritance_complex.py tests/core/test_inheritance_behavior.py tests/test_yaml_inheritance.py`: 59 passed, 2 dependency deprecation warnings (115.15s).
- `pre-commit run --all-files`: pass, including basedpyright, gitleaks, actionlint and all hygiene hooks. New staged files also checked separately after initial hook file discovery.
- Full `.venv/bin/pytest -q` started; result pending at initial fix commit. Draft PR publication is authorized; no merge or release.

- Full `.venv/bin/pytest -q`: 983 passed, 15 skipped, 2 dependency deprecation warnings (461.03s). No unrelated failures in this locked environment. Skips: nine optional OpenAI/Azure cases and six source/demo-fixture-dependent cases.
- Reporter-series check: Python 3.11.15 / dbt-core 1.9.11 / dbt-duckdb 1.9.6 / dbt-adapters 1.24.5; new regression file: 12 passed, 2 warnings (20.97s).
- Fix commit `e4f39744346f59728a6a80568e59e135d6b2061e` pushed and remote SHA verified. Draft PR: https://github.com/z3z1ma/dbt-osmosis/pull/454 . Initial work-account push was denied by GitHub; operation succeeded with existing personal-account credentials, without changing the global login.
- CI observed: Tests https://github.com/z3z1ma/dbt-osmosis/actions/runs/43874268182 has a successful lockfile job; matrix/docs/smoke jobs queued. Lint https://github.com/z3z1ma/dbt-osmosis/actions/runs/43874267952 queued. This is not a claim of passing remote CI.
- Follow-up owner: #452 retains the unconfirmed inheritance-loss symptom; no separate issue comment posted. PR #454 owns remote CI completion. No merge/release performed.

## Acceptance mapping
- Diamond/unequal-path warning classification and unchanged trees: five new graph cases pass.
- Actual root/non-root/self cycles: three cases pass and assert the exact warning edge.
- Shared description propagation and sibling precedence: four transform cases pass in both parent orders.
- Local aggregate validation: full pytest, pre-commit, Ruff, format, lock check and type-check passed; independent review passed. Remote matrix remains pending; full Taskfile multi-interpreter matrix was not additionally run locally.
