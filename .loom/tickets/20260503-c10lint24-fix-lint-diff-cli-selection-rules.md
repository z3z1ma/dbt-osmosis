---
id: ticket:c10lint24
kind: ticket
status: closed
change_class: code-behavior
risk_class: medium
created_at: 2026-05-03T21:10:43Z
updated_at: 2026-05-04T19:33:25Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  initiative:
    - initiative:dbt-110-111-hardening
  evidence:
    - evidence:oracle-backlog-scan
    - evidence:c10lint24-lint-diff-cli-validation
    - evidence:c10lint24-main-ci-success
  critique:
    - critique:c10lint24-lint-diff-cli-review
  packets:
    - packet:ralph-ticket-c10lint24-20260504T184608Z
depends_on: []
---

# Summary

Fix lint and diff CLI mismatches: positional selectors ignored by `diff schema`, `lint file --disable-rules` ignored, duplicate output categories, and lint node selection divergence.

# Context

`src/dbt_osmosis/cli/main.py:1417-1528` accepts positional `models` via `yaml_opts` but `diff schema` only filters by `fqn`. `src/dbt_osmosis/cli/main.py:1967-2028` accepts `--disable-rules` for `lint file` but never passes it into `lint_sql_code()`. Output grouping compares enum values to strings at `cli/main.py:2033-2059` and `2131-2157`, risking duplicate categories. `src/dbt_osmosis/core/sql_lint.py:603-618` has custom node selection rather than reusing shared node filters.

# Why Now

Selector and rule flags that are accepted but ignored can make users lint/diff the wrong scope. That becomes more important when CI uses these commands for compatibility checks.

# Scope

- Make `diff schema` positional selectors work or remove them from the command contract.
- Pass disabled rules through `lint file` and define precedence with `--rules`.
- Fix output grouping so errors/warnings are not reprinted as "Other".
- Align lint project/model selection with shared node selection semantics or document the difference.
- Add CLI tests for selection, disabled rules, unknown selectors, and output grouping.

# Out Of Scope

- Rewriting the SQL linter rule engine.
- Promoting `diff` or `lint` to release gates unless another ticket does that.

# Acceptance Criteria

- ACC-001: `dbt-osmosis diff schema my_model` scopes to `my_model` or exits clearly if positional selectors are unsupported.
- ACC-002: Unknown selectors do not silently run against the whole project.
- ACC-003: `lint file ... --disable-rules select-star` suppresses the disabled rule.
- ACC-004: `--rules` and `--disable-rules` precedence is deterministic and documented in tests/docs.
- ACC-005: CLI output does not duplicate errors/warnings under "Other".
- ACC-006: Lint project/model selection matches documented package/ephemeral inclusion behavior.

# Coverage

Covers:

- ticket:c10lint24#ACC-001
- ticket:c10lint24#ACC-002
- ticket:c10lint24#ACC-003
- ticket:c10lint24#ACC-004
- ticket:c10lint24#ACC-005
- ticket:c10lint24#ACC-006
- initiative:dbt-110-111-hardening#OBJ-006

# Claim Matrix

| Claim | Evidence | Critique | Status |
| --- | --- | --- | --- |
| ticket:c10lint24#ACC-001 | evidence:c10lint24-lint-diff-cli-validation; evidence:c10lint24-main-ci-success | critique:c10lint24-lint-diff-cli-review | accepted |
| ticket:c10lint24#ACC-002 | evidence:c10lint24-lint-diff-cli-validation; evidence:c10lint24-main-ci-success | critique:c10lint24-lint-diff-cli-review | accepted |
| ticket:c10lint24#ACC-003 | evidence:c10lint24-lint-diff-cli-validation; evidence:c10lint24-main-ci-success | critique:c10lint24-lint-diff-cli-review | accepted |
| ticket:c10lint24#ACC-004 | evidence:c10lint24-lint-diff-cli-validation; evidence:c10lint24-main-ci-success | critique:c10lint24-lint-diff-cli-review | accepted with red-evidence limitation |
| ticket:c10lint24#ACC-005 | evidence:c10lint24-lint-diff-cli-validation; evidence:c10lint24-main-ci-success | critique:c10lint24-lint-diff-cli-review | accepted |
| ticket:c10lint24#ACC-006 | evidence:c10lint24-lint-diff-cli-validation; evidence:c10lint24-main-ci-success | critique:c10lint24-lint-diff-cli-review | accepted |

# Execution Notes

Use existing `click.testing.CliRunner` patterns. If command signatures change, update docs in the same ticket.

# Blockers

None.

# Evidence

Existing evidence: evidence:oracle-backlog-scan. Validation evidence: evidence:c10lint24-lint-diff-cli-validation and evidence:c10lint24-main-ci-success. Missing evidence: none for this ticket's closure gate.

# Critique Disposition

Risk class: medium

Critique policy: recommended

Policy rationale: User-facing command semantics and selection behavior need review.

Required critique profiles: code-change, operator-clarity, test-coverage

Findings: critique:c10lint24-lint-diff-cli-review#FIND-001 is low severity and accepted as a non-blocking evidence-discipline limitation for `ticket:c10lint24#ACC-004`. No medium/high findings.

Disposition status: completed

Deferral / not-required rationale: No critique follow-up blocks acceptance; parent resolved the initial docs/test-strength review concerns before recording the final critique, and the remaining low red-evidence limitation is explicit in the claim matrix and acceptance basis.

# Retrospective / Promotion Disposition

Disposition status: completed

Promoted: CLI reference docs now record lint rule precedence and `lint model`/`lint project` selection defaults.

Deferred / not-required rationale: No separate wiki/research/spec promotion is needed; the durable user-facing explanation belongs in `docs/docs/reference/cli.md`, and the residual evidence limitation is recorded in evidence and critique.

# Wiki Disposition

Not required - CLI reference docs are the accepted explanation surface for this command-level behavior.

# Acceptance Decision

Accepted by: OpenCode
Accepted at: 2026-05-04T19:07:42Z.
Basis: `evidence:c10lint24-lint-diff-cli-validation` records child red evidence, parent green `113 passed` focused tests, Ruff/whitespace checks, changed-source basedpyright `errorCount: 0`, docs updates, and final critique `critique:c10lint24-lint-diff-cli-review` with no blocking findings. `evidence:c10lint24-main-ci-success` records successful main-branch lint, Tests, Labeler, and Release validation for commit `b9dc86279d5074397306663631b7d47f3e824be0` and closure of GitHub issue #375.
Residual risks: `ticket:c10lint24#ACC-004` has green behavior evidence but not distinct strict red evidence; schema diff tests use fixture/mocked comparison rather than a live warehouse diff; lint selection intentionally adopts project-owned, non-ephemeral, segment-FQN semantics instead of prior substring/external/ephemeral behavior.

# Dependencies

None.

# Journal

- 2026-05-03T21:10:43Z: Created from CLI/SQL/workbench and core architecture oracle findings.
- 2026-05-04T18:46:08Z: Activated ticket and compiled Ralph packet `packet:ralph-ticket-c10lint24-20260504T184608Z` for test-first lint/diff selector, disabled-rule, rule-precedence, output grouping, and lint model-selection fixes.
- 2026-05-04T19:07:42Z: Consumed Ralph output, addressed critique-driven docs and test-strength refinements, recorded validation evidence `evidence:c10lint24-lint-diff-cli-validation`, completed recommended critique `critique:c10lint24-lint-diff-cli-review`, accepted all scoped claims with an explicit low red-evidence limitation for ACC-004, and closed ticket.
- 2026-05-04T19:33:25Z: Pushed commit `b9dc86279d5074397306663631b7d47f3e824be0` to `origin/main`, observed successful lint `25338008487`, Tests `25338008545`, Labeler `25338008503`, and Release `25338519017` workflows, recorded `evidence:c10lint24-main-ci-success`, commented on GitHub issue #375, and closed the issue.
