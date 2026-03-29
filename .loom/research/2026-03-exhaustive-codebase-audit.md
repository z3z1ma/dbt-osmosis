---
id: 2026-03-exhaustive-codebase-audit
title: "2026-03 exhaustive codebase audit"
status: synthesized
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
Audit fully synthesized and executed. All 20 remediation tickets were completed, merged to main, and pushed to origin.

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
- Execute the resulting ticket backlog through Ralph worktrees, merging each completed ticket to main and pushing to origin
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
- Repository docs and AGENTS guidance were updated to reflect the post-fix system shape, including the narrowed public facade, DuckDB-only fixture support, and current CLI/runtime surfaces.
- The major hidden correctness issues identified in YAML buffering, sync/version handling, inheritance overrides, SQL compile/lint flows, optional provider defaults, workbench AI truthfulness, and fixture drift were all addressed and merged.
- The support contract is now truthful: packaging, lockfile, docs, and CI are aligned to the exercised dbt 1.8-1.10 window.

## Recommendations
- Consider following up on governed docs ownership so docs-memory closeout can use linked docs records without waivers.
- If future work expands the dbt support window, treat it as one coherent change across dependency bounds, CI, docs, and verification evidence.
- Monitor Ralph's projected-plan/ticket lookup gaps in worktree mode; they did not block delivery here, but they add avoidable operator friction.

## Open Questions
(none)

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
