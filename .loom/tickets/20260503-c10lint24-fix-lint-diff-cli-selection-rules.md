---
id: ticket:c10lint24
kind: ticket
status: ready
change_class: code-behavior
risk_class: medium
created_at: 2026-05-03T21:10:43Z
updated_at: 2026-05-03T21:10:43Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  initiative:
    - initiative:dbt-110-111-hardening
  evidence:
    - evidence:oracle-backlog-scan
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
| ticket:c10lint24#ACC-001 | evidence:oracle-backlog-scan | None | open |
| ticket:c10lint24#ACC-003 | evidence:oracle-backlog-scan | None | open |

# Execution Notes

Use existing `click.testing.CliRunner` patterns. If command signatures change, update docs in the same ticket.

# Blockers

None.

# Evidence

Existing evidence: evidence:oracle-backlog-scan. Missing evidence: CLI tests.

# Critique Disposition

Risk class: medium

Critique policy: recommended

Policy rationale: User-facing command semantics and selection behavior need review.

Required critique profiles: code-change, operator-clarity, test-coverage

Findings: None - no critique yet.

Disposition status: pending

Deferral / not-required rationale: None.

# Retrospective / Promotion Disposition

Disposition status: pending

Promoted: None - implementation not complete.

Deferred / not-required rationale: Docs update likely enough; wiki probably not needed.

# Wiki Disposition

N/A - no wiki promotion selected yet.

# Acceptance Decision

Accepted by: Not accepted yet.
Accepted at: N/A.
Basis: Pending tests and docs.
Residual risks: Shared node selection behavior may not be intended for lint.

# Dependencies

None.

# Journal

- 2026-05-03T21:10:43Z: Created from CLI/SQL/workbench and core architecture oracle findings.
