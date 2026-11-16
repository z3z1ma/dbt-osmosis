Status: done
Created: 2026-11-16
Updated: 2026-11-16

# Residual issue reproductions: #443 and #452

## Scope
Extend existing evidence on main 8b86daef with actual CLI/default introspection paths: cross-package dry-run with absent/empty/populated phantom YAML, and a parsed/seeded/built DuckDB shared-ancestor diamond through UNION. The user authorized responsible issue resolution and further investigation; no external publication or closure is part of this ticket.

## Non-goals
No dependency changes, external-path semantics changes, lineage feature changes, release, or claim about unprovided SQL Server/Trino projects.

## Acceptance criteria
1. Snapshot project/package YAML bytes and mtimes around dry-run, including a populated duplicate; distinguish parser rejection from execution.
2. Verify actual downstream YAML description after CLI refactor of a real shared-ancestor graph, with introspection enabled; distinguish rendered and docs-block behavior.
3. Record versions, commands, outcomes and limits; independent review of assertions and disposition before handoff.

## References
- #443 and #452 on z3z1ma/dbt-osmosis, read 2026-11-16; no new reporter reproductions after maintainer comments.
- .10x/tickets/2026-11-16-cross-package-patch-443.md
- .10x/tickets/2026-11-16-reconcile-443-452.md
- tests/test_cross_package_patch.py and tests/core/test_inheritance_shared_ancestors.py

## Assumptions
Existing CLI dry-run means no schema YAML writes; graph is the reporter's shared lookup feeding two branches feeding UNION. DuckDB is the repository-supported fixture adapter. Earlier null results do not prove absent bugs in other environments.

## Journal
- Inspected issue bodies/comments and original regression assertions. Existing cross-package cases disable introspection; diamond tests use real node objects with synthetic edges. This ticket adds distinct end-to-end evidence, not duplicate suite runs.

## Blockers
None for bounded reproduction; exact reporter projects and warehouse adapters remain unavailable.

## Evidence
All three acceptance criteria map to the execution evidence and observation sections below.

## Review
Independent review passed; see the appended review below.

## Retrospective
See retrospective findings below.

## Execution evidence
- Added five subprocess integration cases in `tests/test_residual_issue_reproductions.py`; no production edits. Uses local declared package, file-backed temporary DuckDB, real dbt seed/run/parse, and CLI refactor with introspection enabled.
- Initial import was blocked by sandbox protection of standard `~/.dbt-osmosis/logs`; allowed test execution with standard logging. Initial diamond fixture used unsupported scalar models routing, then lacked confirmation for a nonempty dry-run plan. Corrected fixture to established `+dbt-osmosis: schema.yml` and `--dry-run --auto-apply`; neither was an inheritance failure.
- Python 3.13.14/Core 1.10.20/adapter 1.10.0/pytest 9.0.3: `PYTHONPATH=src /Users/alexanderbut/Documents/Codex/2026-11-16/task-3/issue-452/.venv/bin/python -m pytest tests/test_residual_issue_reproductions.py -q` => 5 passed, 2 dbt Click deprecation warnings in 48.27s. Log `/tmp/residual-issues.log`.
- Python 3.11.15/Core 1.9.11/adapter 1.9.6: `PYTHONPATH=src /private/tmp/dbt-osmosis-452-dbt19/bin/python -m pytest tests/test_residual_issue_reproductions.py -k diamond -q` => 2 passed, 3 deselected, 2 dbt Click deprecation warnings in 19.82s. Log `/tmp/residual-issues-dbt19.log`.
- Ruff check and format check passed on final test file; `git diff --check` passed.

- Python 3.13.14/Core 1.12.7/adapter 1.11.1: `PYTHONPATH=src /private/tmp/osmosis-canary/bin/python -m pytest tests/test_residual_issue_reproductions.py -k cross_package -q` => 3 passed, 2 deselected in 29.97s. Log `/tmp/residual-issues-dbt112.log`. This matches the #443 reporter's Core minor with the repository-supported DuckDB adapter; Trino is not claimed.

## Observation and proposed disposition
#443: Bare `yaml refactor --dry-run` with introspection enabled preserves every project/package YAML byte and mtime when the phantom file is absent or an empty scaffold. A populated duplicate is rejected by dbt before YAML refactor (asserted duplicate-schema diagnostic); bytes/mtimes still unchanged. Package patch ownership is asserted from the real seed manifest. No routed duplicate is created; valid cases reparse. This is a null reproduction of the reported dry-run symptom, not a general guarantee that dry-run writes no artifacts: dbt logs/target artifacts are outside the YAML contract.

#452: The real built graph matches the report's lookup JOINed into two historical branches, UNIONed downstream. CLI live introspection discovers undocumented child columns and refactor writes the expected description into both branches and final model, for rendered prose and raw docs blocks. Both dry-run snapshots and warning absence are asserted, and final full parse succeeds. This covers the described topology on the reporter's Core 1.9 series, but not their unprovided SQL Server adapter/model/configuration.

Recommended maintainer disposition: the primary reported defects are repaired on main by #455/#454; these additional CLI cases strengthen evidence without another speculative product fix. Keep the existing concrete reproduction requests as the remaining action. Do not assert that the residual claims were fixed, and do not rewrite precedence/path behavior to chase an unprovided case. If closing as fixed for the supplied topology/default ownership bug, explicitly invite reopening with a remaining symptom reproducer; whether to close now is the integration owner's communication decision, not a claim this investigation proves every configuration.

## Retrospective findings
A synthetic graph test cannot establish the CLI introspection/write path; the added real database fixture does. Conversely a CLI abort at confirmation or dbt duplicate-schema validation does not establish a mutation bug. Record the boundary at which execution stops. Preserve raw docs-block identity in assertions because rendered equality alone misses source-of-truth regressions.

## Independent review (2026-11-16)
Reviewer: `/root/issue_reproduction/review_residual_repro`.
Verdict: pass for the bounded reproduction evidence; no critical or significant findings.

Reviewed the actual five test cases and recorded log summaries without repeating verification. #443 asserts real package patch ownership, successful or specifically duplicate-schema-rejected invocation, unchanged YAML bytes/mtimes and absent routed duplicate. #452 uses an actual file-backed temporary database and JOIN/UNION graph, asserts discovered downstream documentation in all three descendants, both rendered prose and raw docs references, unchanged dry-run YAML, no cycle warning and successful final parse. Temporary paths isolate database/project state; no production code or old protective tests changed.

Minor record-maintenance finding: top-level Evidence/Review/Retrospective still say Pending although completed sections follow; reconcile these before marking the ticket done. The Core 1.12 result currently follows Retrospective findings and belongs with Execution evidence. This does not affect assertion validity.

Residual risk: only DuckDB fixtures and the listed Core versions are observed, not unprovided SQL Server/Trino projects or every option combination. `_snapshot` covers `.yml` files (all schema/config files in these fixtures), not arbitrary `.yaml` files or dbt log/target artifacts. Introspection is enabled for the #443 invocation, but the package-owned seed is intentionally excluded from mutation, so this is not evidence of a live metadata query on that seed. #452's initially absent downstream columns and resulting populated YAML substantiate the live introspection path. Null reproductions support the stated topology and ownership behavior; they do not prove residual allegations globally fixed or require issue closure.
