---
id: critique:c10lint24-lint-diff-cli-review
kind: critique
status: final
created_at: 2026-05-04T19:05:24Z
updated_at: 2026-05-04T19:05:24Z
scope:
  kind: repository
  repositories:
    - repo:root
review_target: "ticket:c10lint24 uncommitted lint/diff CLI implementation diff"
links:
  initiative:
    - initiative:dbt-110-111-hardening
  ticket:
    - ticket:c10lint24
  evidence:
    - evidence:c10lint24-lint-diff-cli-validation
  packets:
    - packet:ralph-ticket-c10lint24-20260504T184608Z
external_refs: {}
---

# Summary

Reviewed the `ticket:c10lint24` uncommitted implementation diff after the Ralph iteration and parent validation. Profiles used: `code-change`, `operator-clarity`, and `test-coverage`.

# Review Target

Target: the current working-tree diff for `src/dbt_osmosis/cli/main.py`, `src/dbt_osmosis/core/sql_lint.py`, `tests/core/test_cli.py`, `tests/core/test_sql_lint.py`, `docs/docs/reference/cli.md`, the ticket activation, and `packet:ralph-ticket-c10lint24-20260504T184608Z`.

The review inspected whether the diff satisfies `ticket:c10lint24#ACC-001` through `ticket:c10lint24#ACC-006`, stays inside the Ralph write scope, and carries enough evidence for acceptance review.

# Verdict

`pass_with_findings` — no blocking code or operator-clarity finding remains after parent refinement. One low-severity evidence-discipline finding remains for the missing distinct red evidence on rule-precedence behavior.

# Findings

## FIND-001: Rule-precedence red evidence was not distinctly recorded

Severity: low
Confidence: high
State: open

Observation: The Ralph packet red evidence at `.loom/packets/ralph/20260504T184608Z-ticket-c10lint24-iter-01.md:192-200` records failures for selector propagation, disabled-rule forwarding, and duplicate output grouping, but it does not record a distinct failing test for `--rules` and `--disable-rules` overlap precedence. Green tests now cover precedence in `tests/core/test_sql_lint.py:592` and `tests/core/test_sql_lint.py:598`, and the pre-change implementation at `src/dbt_osmosis/core/sql_lint.py:512` would have returned early on `enabled_rules`, but the packet does not provide strict red-before-green evidence for this acceptance criterion.

Why it matters: The behavior is covered green, but the packet's `test-first` evidence is weaker than claimed for `ticket:c10lint24#ACC-004`. Future ticket acceptance should avoid overstating that every acceptance criterion had distinct red evidence.

Follow-up: Before closure, record this as a low residual evidence limitation in the ticket acceptance basis or critique disposition. No implementation follow-up is required unless the project wants strict red evidence for every sub-claim.

Challenges:

- `ticket:c10lint24#ACC-004` — challenges only the strict red-evidence strength, not the implemented green behavior.

# Evidence Reviewed

- `ticket:c10lint24` scope, acceptance criteria, claim matrix, critique disposition, and journal.
- `packet:ralph-ticket-c10lint24-20260504T184608Z` mission, output contract, child output, and red/green evidence.
- Current source/test/docs diff for `src/dbt_osmosis/cli/main.py`, `src/dbt_osmosis/core/sql_lint.py`, `tests/core/test_cli.py`, `tests/core/test_sql_lint.py`, and `docs/docs/reference/cli.md`.
- Parent validation: `uv run pytest tests/core/test_cli.py tests/core/test_sql_lint.py tests/core/test_diff.py -q` -> `113 passed`; Ruff check plus `git diff --check` -> passed; basedpyright changed-source JSON -> `errorCount: 0`.
- Oracle review pass `ses_20ba56f18ffe37B8pJ21JRmOfL`, which reported three low findings. Parent resolved the docs clarity finding by documenting `lint model` and `lint project` together, and resolved the indirect diff-selector test finding by changing the CLI tests to exercise real `SchemaDiff.compare_all()` node selection.

# Residual Risks

- `diff schema` tests validate command flow with fixture/mocked comparison rather than a live warehouse schema diff.
- `lint model` and `lint project` now intentionally differ from prior substring/external/ephemeral behavior by adopting shared project-owned, non-ephemeral segment matching semantics.
- Existing basedpyright warnings remain outside this ticket's zero-error gate.

# Required Follow-up

No implementation follow-up required before acceptance. Ticket acceptance should explicitly consume `critique:c10lint24-lint-diff-cli-review#FIND-001` as a low residual evidence limitation.

# Acceptance Recommendation

`no-critique-blockers` — critique sees no required implementation follow-up before the ticket's acceptance gate makes its own decision.
