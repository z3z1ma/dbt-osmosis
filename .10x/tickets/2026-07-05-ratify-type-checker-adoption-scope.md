Status: open
Created: 2026-07-05
Updated: 2026-07-05
Parent: None
Depends-On: None

# Ratify type checker adoption scope

## Scope

Decide how this project should adopt `ty` and mypy as quality gates, if at all, after the quality optimizer run exposed broad non-clean output from both tools.

In scope:

- Classify current `ty` and mypy diagnostics by source: optional extras missing from the base environment, tests, dbt/private stubs, current protocol strictness, and true production issues.
- Decide whether adoption should be whole-repo, source-only, configured to installed extras, or limited to existing basedpyright.
- Decide whether new tool configuration, dependency extras, stubs, or CI gates are in scope.
- Open bounded child tickets for approved implementation slices.

Out of scope:

- Adding broad suppressions or baselines without ratification.
- Treating current `ty`/mypy output as a release blocker before an adoption scope is approved.
- Changing optional extra dependency policy implicitly.

## Acceptance Criteria

- ACC-001: Current `ty` and mypy diagnostics are summarized by category with representative examples.
- ACC-002: The user or an active decision record ratifies whether `ty`, mypy, both, or neither become enforced gates.
- ACC-003: Any approved implementation work is split into bounded executable tickets.

## Evidence Expectations

- Capture the current whole-repo `ty check` and `mypy .` outputs or summarized artifacts.
- Reference `.10x/evidence/2026-07-05-quality-optimizer-final-vector.md`.

## Progress and Notes

- 2026-07-05: Final quality pass ran `uv run --no-sync --with ty ty check`; it reported broad diagnostics across optional extras, tests, CLI, and protocol strictness.
- 2026-07-05: Final quality pass ran `uv run --no-sync --with mypy mypy .`; it reported broad diagnostics across optional extras, tests, and current typing contracts.
- 2026-07-05: A bounded cleanup in `.10x/tickets/done/2026-07-05-tighten-final-type-tool-feedback.md` fixed diagnostics tied to recently touched optimizer surfaces and left broader adoption decisions for this ticket.

## Blockers

The adoption scope is a project policy decision because it may require configuration, optional extra installation assumptions, stubs, baselines, CI changes, or many test typing edits.

## References

- `.10x/evidence/2026-07-05-final-type-tool-feedback.md`
- `.10x/evidence/2026-07-05-quality-optimizer-final-vector.md`
- `.10x/tickets/done/2026-07-05-tighten-final-type-tool-feedback.md`
