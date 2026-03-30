---
id: incremental-basedpyright-hardening-across-supported-dbt-core-versions
title: "Incremental basedpyright hardening across supported dbt-core versions"
status: active
created-at: 2026-03-30T02:47:25.993Z
tags:
  - compatibility
  - static-analysis
  - typing
source-refs: []
---

## Question
Which basedpyright suppressions and disabled diagnostics can be peeled back safely while preserving compatibility across supported dbt-core versions from 1.11 backward to 1.8?

## Objective
Identify the next tranche of strictness that can be enabled truthfully, fix surfaced issues in repository-managed code, and validate latest and oldest supported dbt-core environments with static analysis and targeted runtime coverage.

## Status Summary
First stricter tranche completed: reportAttributeAccessIssue is now enabled globally, repository-managed fixes landed, and the supported dbt-core matrix plus latest 1.11 line all pass the basedpyright error gate, dbt parse, and full pytest.

## Scope
- .github/workflows/tests.yml
- demo_duckdb fixtures
- pyproject.toml
- src/dbt_osmosis/cli
- src/dbt_osmosis/core
- Taskfile.yml

## Non-Goals
- Claiming support for adapter versions that do not exist upstream
- Eliminating every Any in one change
- Typing the excluded workbench/sql modules in this pass

## Methodology
- Enable stricter diagnostics incrementally and fix repository-managed code until checks pass
- Inspect current pyright configuration and in-file suppressions
- Run basedpyright in isolated latest and oldest supported dbt-core environments
- Validate with pytest slices and the full supported dbt-core matrix

## Keywords
- basedpyright
- compatibility
- dbt-core
- pyright
- typing

## Conclusions
- dbt-core version drift shows up in type surfaces as well as runtime behavior: catalog artifact typing in 1.8.x and generic-test YAML syntax in the demo fixture both needed compatibility-aware handling.
- Matrix validation needs basedpyright as a first-class step, not just the latest-line compat job, otherwise stricter rules can regress on older supported dbt-core lines unnoticed.
- reportAttributeAccessIssue has a good signal-to-fix ratio for the currently included source tree and can remain enabled.

## Recommendations
- Keep the demo fixture on legacy top-level generic-test arguments until the minimum supported dbt-core line is raised above 1.8/1.9.
- Next strictness candidates should be evaluated one at a time with the same matrix workflow; reportOptionalMemberAccess or targeted removal of file-level reportPrivateImportUsage ignores are better next bets than enabling the whole unknown-type family at once.

## Open Questions
- Which next diagnostic beyond reportAttributeAccessIssue offers the best signal-to-fix ratio across 1.8-1.11?

## Linked Work
_Generated summary. Reconcile ignores edits in this section so canonical hypotheses, artifacts, and linked work are preserved._

(none)

## Hypotheses
_Generated summary. Reconcile ignores edits in this section so canonical hypotheses, artifacts, and linked work are preserved._

(none)

## Artifacts
_Generated summary. Reconcile ignores edits in this section so canonical hypotheses, artifacts, and linked work are preserved._

(none)
