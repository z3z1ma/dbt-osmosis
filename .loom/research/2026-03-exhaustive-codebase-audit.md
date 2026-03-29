---
id: 2026-03-exhaustive-codebase-audit
title: "2026-03 exhaustive codebase audit"
status: active
created-at: 2026-03-29T03:46:01.891Z
tags:
  - audit
  - backlog
  - quality
source-refs:
  - /Users/alexanderbutler/code_projects/personal/dbt-osmosis/AGENTS.md
  - artifact://2
  - jobs://bg_14a53a76b760becc
  - src/dbt_osmosis/core/AGENTS.md
---

## Question
What correctness bugs, architectural issues, code smells, TODOs, hacks, missing abstractions, and compatibility risks exist across the dbt-osmosis codebase when evaluated from latest supported dbt versions backward?

## Objective
Produce a durable, evidence-backed audit across the full repository, prioritize latest supported dbt compatibility surfaces first, and convert every actionable finding into sequenced tickets that can be executed safely in parallel.

## Status Summary
Repository-wide audit complete. The delayed pytest artifact confirmed the broad suite passes under the drifted latest uv environment, while its warnings/deprecations map to already-filed tickets for fixture support drift and test-suggestion naming/shape issues; no additional ticket was required.

## Scope
- .pre-commit-config.yaml
- demo_duckdb
- docs
- pyproject.toml
- src/dbt_osmosis
- Taskfile.yml
- tests

## Non-Goals
- Audit vendored _deps code unless directly implicated by repository usage
- Create speculative tickets without repository evidence
- Implement fixes in this audit pass

## Methodology
- Inspect repository guidance and memory context
- Map package surfaces and supported dependency matrix from project config
- Review subsystem code and tests for correctness and architecture risks
- Run static analysis and targeted diagnostics starting from latest supported dbt environment
- Search for TODO/FIXME/HACK/deprecation markers and high-complexity hotspots
- Synthesize findings into de-duplicated tickets with explicit dependencies and safe parallelization guidance

## Keywords
- architecture
- audit
- basedpyright
- bugs
- dbt
- tickets

## Conclusions
- Several of the highest-value fixes are foundational: support-matrix alignment, compatibility-facade cleanup, and live introspection contract repair all unblock safer parallel remediation downstream.
- The delayed full basedpyright warning artifact reinforced existing grouped tickets rather than revealing a distinct new design defect; most remaining warnings collapse into the already-filed tickets around deprecated settings access, private-boundary violations, and `Any`-driven blind spots.
- The delayed pytest artifact likewise did not reveal a new ticket cluster: dbt generic-test deprecations belong to the fixture/demo support-line ticket, and PytestCollectionWarnings for production `Test*` classes belong to the test-suggestions ticket.
- The latest default uv environment currently drifts beyond the explicitly tested dbt support matrix, so version-sensitive behavior is ambiguous until the support contract is tightened.
- The test suite is strong on happy-path behavior, but static analysis and subsystem review uncovered multiple hidden correctness bugs in YAML sync, schema buffering/validation, SQL compilation/linting, optional AI/workbench surfaces, and bootstrap heuristics.

## Recommendations
- Keep the linked plan order as the execution queue of record so later workers do not create overlapping edits in shared files.
- Start with do-0004, do-0002, do-0001, do-0007, and do-0003 before parallelizing the rest of the backlog.
- Treat do-0005, do-0019, do-0016, do-0009, do-0010, do-0015, and do-0020 as explicitly dependency-constrained follow-ons.

## Open Questions
- Should PostgreSQL fixture support be repaired and exercised, or explicitly retired as unsupported test surface?
- Should the project continue supporting Fusion auto-detection at all, or should that move behind explicit opt-in after support-matrix alignment?

## Linked Work
_Generated summary. Reconcile ignores edits in this section so canonical hypotheses, artifacts, and linked work are preserved._

- ticket:do-0001
- ticket:do-0002
- ticket:do-0003
- ticket:do-0004
- ticket:do-0005
- ticket:do-0006
- ticket:do-0007
- ticket:do-0008
- ticket:do-0009
- ticket:do-0010
- ticket:do-0011
- ticket:do-0012
- ticket:do-0013
- ticket:do-0014
- ticket:do-0015
- ticket:do-0016
- ticket:do-0017
- ticket:do-0018
- ticket:do-0019
- ticket:do-0020

## Hypotheses
_Generated summary. Reconcile ignores edits in this section so canonical hypotheses, artifacts, and linked work are preserved._

(none)

## Artifacts
_Generated summary. Reconcile ignores edits in this section so canonical hypotheses, artifacts, and linked work are preserved._

(none)
