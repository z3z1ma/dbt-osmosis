# Repository-wide defect audit and remediation backlog

## Purpose / Big Picture

Establish one durable execution-strategy artifact for the exhaustive repository audit so findings, ticket sequencing, and validation rationale do not fragment across chat or ad hoc notes.

## Progress

_Generated snapshot. Reconcile ignores edits in this section so live ticket truth and append-only plan history remain canonical._

- [x] (2026-03-29T04:25:00Z) Synthesized subsystem findings into 20 durable tickets and linked them to the plan in execution order.

Linked ticket snapshot from the live execution ledger:
- [ ] Ticket do-0004 [ready] — Align the installable dbt contract with the exercised support matrix (audit-backlog)
- [ ] Ticket do-0002 [ready] — Collapse compatibility facades into a truthful public API (audit-backlog)
- [ ] Ticket do-0001 [ready] — Repair the live introspection contract for settings resolution and column caching (audit-backlog)
- [ ] Ticket do-0007 [ready] — Stop evicting dirty YAML buffers before commit (audit-backlog)
- [ ] Ticket do-0003 [ready] — Make source bootstrap and restructure write tracking truthful (audit-backlog)
- [ ] Ticket do-0005 [ready] — Replace adapter factory monkey-patches with a stable bootstrap contract (audit-backlog)
- [ ] Ticket do-0006 [ready] — Make bootstrap environment detection evidence-based (audit-backlog)
- [ ] Ticket do-0008 [ready] — Make schema parsing and validation tell the truth about supported YAML (audit-backlog)
- [ ] Ticket do-0009 [ready] — Repair sync operations for versioned models and source matching (audit-backlog)
- [ ] Ticket do-0010 [ready] — Make progenitor overrides use the same inheritance contract as normal lineage (audit-backlog)
- [ ] Ticket do-0011 [ready] — Make external-node selection come from one truthful contract (audit-backlog)
- [ ] Ticket do-0012 [ready] — Fix rename detection crashing in schema diff (audit-backlog)
- [ ] Ticket do-0013 [ready] — Ensure plain-SQL compilation cleans up temporary manifest nodes (audit-backlog)
- [ ] Ticket do-0014 [ready] — Make staging generation dry-run truly side-effect free (audit-backlog)
- [ ] Ticket do-0015 [ready] — Lint compiled model SQL instead of raw manifest nodes (audit-backlog)
- [ ] Ticket do-0016 [ready] — Make test suggestions work against real manifest nodes without OpenAI (audit-backlog)
- [ ] Ticket do-0017 [ready] — Make optional LLM provider configuration match documented defaults (audit-backlog)
- [ ] Ticket do-0018 [ready] — Stop the workbench AI panel from crashing or faking success (audit-backlog)
- [ ] Ticket do-0019 [ready] — Harden test fixtures and demo assets for the declared dbt support line (audit-backlog)
- [ ] Ticket do-0020 [ready] — Make top-level documentation truthful after the runtime fixes land (audit-backlog)

## Surprises & Discoveries

_Generated snapshot. Reconcile ignores edits in this section so live ticket truth and append-only plan history remain canonical._

- Observation: The delayed pytest artifact confirmed that broad runtime coverage passes even in the drifted latest uv environment, but the residual warnings still map to existing backlog items rather than a new workstream.
  Evidence: `bg_14a53a76b760becc` reported 713 passed / 10 skipped with dbt=1.11.2, plus generic-test deprecation warnings from the demo fixture and PytestCollectionWarnings for production `TestPatternExtractor` / `TestSuggestion` classes; these already map to do-0019 and do-0016.

## Decision Log

_Generated snapshot. Reconcile ignores edits in this section so live ticket truth and append-only plan history remain canonical._

- Decision: Use a dedicated research record plus workspace plan before scanning the repository.
  Rationale: The request is exploratory, broad, and explicitly requires sequenced tickets; durable evidence and execution strategy should not live only in chat.
  Date/Author: 2026-03-28 / assistant

## Outcomes & Retrospective

Pending audit completion.

## Context and Orientation

This workspace has no prior active plans, tickets, research, or constitutional artifacts. The repository is a Python CLI/package for dbt workflows with heavy logic in src/dbt_osmosis/core and integration fixtures under demo_duckdb. The audit must be exhaustive, evidence-backed, and oriented around the latest supported dbt version before reasoning backward across compatibility-sensitive surfaces. No fixes are being applied in this pass; the output is a trustworthy backlog and execution sequence.

## Projection Context

_Generated snapshot. Reconcile ignores edits in this section so live ticket truth and append-only plan history remain canonical._

- Status: active
- Source target: workspace:repo_fe3425a740fe6165
- Scope paths: https-github-com-vdfaller-dbt-osmosis-git:src/dbt_osmosis, https-github-com-vdfaller-dbt-osmosis-git:tests, https-github-com-vdfaller-dbt-osmosis-git:docs, https-github-com-vdfaller-dbt-osmosis-git:demo_duckdb, https-github-com-vdfaller-dbt-osmosis-git:pyproject.toml, https-github-com-vdfaller-dbt-osmosis-git:Taskfile.yml, https-github-com-vdfaller-dbt-osmosis-git:.pre-commit-config.yaml
- Research: 2026-03-exhaustive-codebase-audit

## Milestones

1. Establish audit container and dependency/runtime context.
2. Map repository surfaces, diagnostics entrypoints, and version-support boundaries.
3. Run broad static and structural discovery, then parallel subsystem reviews.
4. Synthesize de-duplicated findings into fully detailed tickets.
5. Sequence tickets with dependency edges to maximize safe parallel execution.

## Plan of Work

Use repository config and docs to establish supported Python/dbt surface and likely risk hotspots. Run static analysis in a uv-managed environment emphasizing the latest supported dbt package set. Search the repository for TODO/HACK markers, large/complex modules, duplicate patterns, and stale docs/config inconsistencies. Fan out parallel reviews by subsystem where contracts do not overlap: CLI surface, core config/introspection, schema round-tripping, inheritance/transforms/sql operations, workbench/AI extras, tests/demo/docs/tooling. Consolidate evidence into a ticket set grouped by coherent design decisions rather than by individual lines. Encode ticket dependencies where shared abstractions or interface changes must land first; otherwise keep tickets independent to enable parallel execution.

## Concrete Steps

- Read project configuration and key docs to confirm supported Python/dbt bounds and audit surfaces.
- Ensure a latest-supported dbt environment exists under uv and run basedpyright/LSP diagnostics.
- Perform structural and textual searches for TODO/FIXME/HACK/deprecation markers and high-complexity hotspots.
- Launch parallel subsystem investigations with explicit file scopes and acceptance criteria.
- Aggregate findings into durable research artifacts and create tickets with acceptance, risk, and dependency data.
- Update this plan with ticket linkage and recommended execution ordering.

## Validation and Acceptance

Audit quality is acceptable only if: repository-wide diagnostics were actually executed in the chosen environment; every reported finding cites concrete repository evidence; every actionable issue becomes a durable ticket; and the final sequencing makes dependency relationships explicit enough that a follow-on worker can execute tickets in parallel without hidden coupling.

## Idempotence and Recovery

The audit is read-only with respect to repository files. If diagnostics setup fails, record the exact dependency/environment blocker in research and continue with source-based inspection rather than fabricating analysis. Ticket creation is idempotent through de-duplication against the empty current ledger and subsequent updates rather than duplicate creates.

## Artifacts and Notes

Primary durable artifacts will be the research record 2026-03-exhaustive-codebase-audit, this plan, linked tickets created from findings, and any critique or docs updates if later needed. Command outputs and key observations should be summarized in durable progress/discovery notes rather than left only in shell output.

## Interfaces and Dependencies

Key dependencies include uv-managed Python environments, latest supported dbt packages from pyproject constraints, basedpyright/pyright compatibility, pytest fixtures built around demo_duckdb manifests, and repo guidance in AGENTS.md files. Ticket dependencies will likely concentrate around shared core abstractions such as config resolution, schema IO, inheritance/transforms, and workbench optional extras.

## Linked Tickets

_Generated snapshot. Reconcile ignores edits in this section so live ticket truth and append-only plan history remain canonical._

- do-0004 [ready] Align the installable dbt contract with the exercised support matrix — audit-backlog
- do-0002 [ready] Collapse compatibility facades into a truthful public API — audit-backlog
- do-0001 [ready] Repair the live introspection contract for settings resolution and column caching — audit-backlog
- do-0007 [ready] Stop evicting dirty YAML buffers before commit — audit-backlog
- do-0003 [ready] Make source bootstrap and restructure write tracking truthful — audit-backlog
- do-0005 [ready] Replace adapter factory monkey-patches with a stable bootstrap contract — audit-backlog
- do-0006 [ready] Make bootstrap environment detection evidence-based — audit-backlog
- do-0008 [ready] Make schema parsing and validation tell the truth about supported YAML — audit-backlog
- do-0009 [ready] Repair sync operations for versioned models and source matching — audit-backlog
- do-0010 [ready] Make progenitor overrides use the same inheritance contract as normal lineage — audit-backlog
- do-0011 [ready] Make external-node selection come from one truthful contract — audit-backlog
- do-0012 [ready] Fix rename detection crashing in schema diff — audit-backlog
- do-0013 [ready] Ensure plain-SQL compilation cleans up temporary manifest nodes — audit-backlog
- do-0014 [ready] Make staging generation dry-run truly side-effect free — audit-backlog
- do-0015 [ready] Lint compiled model SQL instead of raw manifest nodes — audit-backlog
- do-0016 [ready] Make test suggestions work against real manifest nodes without OpenAI — audit-backlog
- do-0017 [ready] Make optional LLM provider configuration match documented defaults — audit-backlog
- do-0018 [ready] Stop the workbench AI panel from crashing or faking success — audit-backlog
- do-0019 [ready] Harden test fixtures and demo assets for the declared dbt support line — audit-backlog
- do-0020 [ready] Make top-level documentation truthful after the runtime fixes land — audit-backlog

## Risks and Open Questions

Main risks: static diagnostics may be noisy or incomplete without the right extras; some optional AI/workbench paths may require unavailable dependencies; dbt-version support may be implied rather than exhaustively codified; broad ticket grouping may accidentally merge unrelated issues unless evidence is preserved. Open questions include how much debt belongs in docs/tooling versus runtime code and which findings are latest-dbt regressions versus general maintainability issues.

## Revision Notes

_Generated snapshot. Reconcile ignores edits in this section so live ticket truth and append-only plan history remain canonical._

- 2026-03-28T00:00:00Z — Created initial audit plan.
  Reason: User requested an exhaustive scan and sequenced ticket backlog.

- 2026-03-29T03:46:35.816Z — Created durable workplan scaffold from workspace:repo_fe3425a740fe6165.
  Reason: Establish a self-contained execution-strategy artifact that can be resumed without prior chat context.

- 2026-03-28T03:56:00Z — Added diagnostics and support-surface findings.
  Reason: Preserve evidence before subsystem ticket synthesis.

- 2026-03-29T03:51:48.507Z — Updated progress, surprises and discoveries, revision notes.
  Reason: Keep the workplan aligned with the current execution strategy and observable validation story.

- 2026-03-29T04:20:26.389Z — Linked ticket do-0016 as audit-backlog.
  Reason: Keep the Loom workplan coordinated with the ticket ledger without copying live execution state into plan.md.

- 2026-03-29T04:20:27.305Z — Linked ticket do-0015 as audit-backlog.
  Reason: Keep the Loom workplan coordinated with the ticket ledger without copying live execution state into plan.md.

- 2026-03-29T04:20:28.130Z — Linked ticket do-0003 as audit-backlog.
  Reason: Keep the Loom workplan coordinated with the ticket ledger without copying live execution state into plan.md.

- 2026-03-29T04:20:28.775Z — Linked ticket do-0017 as audit-backlog.
  Reason: Keep the Loom workplan coordinated with the ticket ledger without copying live execution state into plan.md.

- 2026-03-29T04:20:32.130Z — Linked ticket do-0005 as audit-backlog.
  Reason: Keep the Loom workplan coordinated with the ticket ledger without copying live execution state into plan.md.

- 2026-03-29T04:20:33.282Z — Linked ticket do-0018 as audit-backlog.
  Reason: Keep the Loom workplan coordinated with the ticket ledger without copying live execution state into plan.md.

- 2026-03-29T04:20:35.926Z — Linked ticket do-0006 as audit-backlog.
  Reason: Keep the Loom workplan coordinated with the ticket ledger without copying live execution state into plan.md.

- 2026-03-29T04:20:37.418Z — Linked ticket do-0019 as audit-backlog.
  Reason: Keep the Loom workplan coordinated with the ticket ledger without copying live execution state into plan.md.

- 2026-03-29T04:20:38.535Z — Linked ticket do-0008 as audit-backlog.
  Reason: Keep the Loom workplan coordinated with the ticket ledger without copying live execution state into plan.md.

- 2026-03-29T04:20:40.057Z — Linked ticket do-0020 as audit-backlog.
  Reason: Keep the Loom workplan coordinated with the ticket ledger without copying live execution state into plan.md.

- 2026-03-29T04:20:42.345Z — Linked ticket do-0009 as audit-backlog.
  Reason: Keep the Loom workplan coordinated with the ticket ledger without copying live execution state into plan.md.

- 2026-03-29T04:20:44.429Z — Linked ticket do-0010 as audit-backlog.
  Reason: Keep the Loom workplan coordinated with the ticket ledger without copying live execution state into plan.md.

- 2026-03-29T04:21:42.766Z — Linked ticket do-0004 as audit-backlog.
  Reason: Keep the Loom workplan coordinated with the ticket ledger without copying live execution state into plan.md.

- 2026-03-29T04:21:53.013Z — Linked ticket do-0002 as audit-backlog.
  Reason: Keep the Loom workplan coordinated with the ticket ledger without copying live execution state into plan.md.

- 2026-03-29T04:21:55.689Z — Linked ticket do-0001 as audit-backlog.
  Reason: Keep the Loom workplan coordinated with the ticket ledger without copying live execution state into plan.md.

- 2026-03-29T04:21:58.260Z — Linked ticket do-0007 as audit-backlog.
  Reason: Keep the Loom workplan coordinated with the ticket ledger without copying live execution state into plan.md.

- 2026-03-29T04:22:01.058Z — Linked ticket do-0003 as audit-backlog.
  Reason: Keep the Loom workplan coordinated with the ticket ledger without copying live execution state into plan.md.

- 2026-03-29T04:22:52.988Z — Linked ticket do-0014 as audit-backlog.
  Reason: Keep the Loom workplan coordinated with the ticket ledger without copying live execution state into plan.md.

- 2026-03-29T04:23:07.062Z — Linked ticket do-0011 as audit-backlog.
  Reason: Keep the Loom workplan coordinated with the ticket ledger without copying live execution state into plan.md.

- 2026-03-29T04:23:12.586Z — Linked ticket do-0012 as audit-backlog.
  Reason: Keep the Loom workplan coordinated with the ticket ledger without copying live execution state into plan.md.

- 2026-03-29T04:23:20.070Z — Linked ticket do-0013 as audit-backlog.
  Reason: Keep the Loom workplan coordinated with the ticket ledger without copying live execution state into plan.md.

- 2026-03-29T04:25:00Z — Recorded final ticket synthesis and sequencing.
  Reason: Durably capture the audit backlog and safe parallel execution order.

- 2026-03-29T04:23:48.209Z — Updated progress, surprises and discoveries, revision notes.
  Reason: Keep the workplan aligned with the current execution strategy and observable validation story.

- 2026-03-29T04:30:00Z — Triaged the delayed full diagnostics artifact.
  Reason: Confirm whether any additional ticket creation was required before closing the audit.

- 2026-03-29T04:25:29.218Z — Updated surprises and discoveries, revision notes.
  Reason: Keep the workplan aligned with the current execution strategy and observable validation story.

- 2026-03-29T04:35:00Z — Triaged the delayed pytest artifact against the backlog.
  Reason: Confirm whether passing runtime coverage and residual warnings required any additional tickets before closing the audit.

- 2026-03-29T04:26:07.219Z — Updated surprises and discoveries, revision notes.
  Reason: Keep the workplan aligned with the current execution strategy and observable validation story.
