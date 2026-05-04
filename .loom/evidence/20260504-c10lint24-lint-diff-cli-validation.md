---
id: evidence:c10lint24-lint-diff-cli-validation
kind: evidence
status: recorded
created_at: 2026-05-04T19:05:24Z
updated_at: 2026-05-04T19:05:24Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  initiative:
    - initiative:dbt-110-111-hardening
  ticket:
    - ticket:c10lint24
  packets:
    - packet:ralph-ticket-c10lint24-20260504T184608Z
external_refs: {}
---

# Summary

Observed red/green and parent validation for the `ticket:c10lint24` lint/diff CLI behavior changes. The observations cover selector propagation, disabled lint rules, rule precedence, output grouping, lint model selection, docs wording, and local static checks.

# Procedure

Observed at: 2026-05-04T19:05:24Z

Source state: branch `loom/dbt-110-111-hardening`, baseline `b2e4e9825b5b0e6de372c8ae9cefd5b6e2d6e926`, with uncommitted `ticket:c10lint24` implementation/record diff.

Procedure: Ralph child added focused tests, observed red failures, implemented the bounded changes, and ran focused validation. Parent reviewed the diff, reran focused validation, ran critique, addressed low docs/test-strength findings, and reran validation.

Expected result when applicable: tests should fail before implementation for current ignored selectors, ignored disabled lint rules, and duplicated lint output; after implementation the focused CLI/lint/diff tests, Ruff, whitespace checks, and changed-source basedpyright should pass with zero errors.

Actual observed result: child reported the initial focused red command returned `5 failed, 3 passed, 105 deselected` for expected selector, disabled-rule forwarding, and duplicated-output failures. Parent post-critique validation returned green focused tests, Ruff, whitespace, and zero basedpyright errors on changed sources.

Procedure verdict / exit code: pass after implementation and parent refinement; partial red limitation for `ticket:c10lint24#ACC-004` because the child packet did not record a distinct failing precedence test before implementation.

# Artifacts

- Red command from Ralph packet: `uv run pytest tests/core/test_cli.py tests/core/test_sql_lint.py tests/core/test_diff.py -q -k "diff_schema or lint_file or lint_output or lint_project or lint_model or disable_rules or selector"` -> expected red, `5 failed, 3 passed, 105 deselected`.
- Parent focused test command after parent critique refinement: `uv run pytest tests/core/test_cli.py tests/core/test_sql_lint.py tests/core/test_diff.py -q` -> `113 passed in 21.92s`.
- Parent formatting check: `uv run ruff format --check src/dbt_osmosis/cli/main.py src/dbt_osmosis/core/sql_lint.py tests/core/test_cli.py tests/core/test_sql_lint.py tests/core/test_diff.py` -> `5 files already formatted` before critique refinement; `uv run ruff format tests/core/test_cli.py` after refinement -> `1 file left unchanged`.
- Parent lint/whitespace command after refinement: `uv run ruff check src/dbt_osmosis/cli/main.py src/dbt_osmosis/core/sql_lint.py tests/core/test_cli.py tests/core/test_sql_lint.py tests/core/test_diff.py && git diff --check` -> `All checks passed!`.
- Parent type check after refinement: `uv run basedpyright --outputjson src/dbt_osmosis/cli/main.py src/dbt_osmosis/core/sql_lint.py` -> JSON summary `errorCount: 0`, `warningCount: 386`.
- Diff summary before evidence creation: `6 files changed, 359 insertions(+), 44 deletions(-)` across ticket/docs/source/tests, plus the new Ralph packet record.
- Code observations: `src/dbt_osmosis/cli/main.py:1721` passes selector settings into `YamlRefactorSettings`; `src/dbt_osmosis/cli/main.py:2195` groups lint violations by `LintLevel`; `src/dbt_osmosis/core/sql_lint.py:512` applies disabled rules after enabled rules; `src/dbt_osmosis/core/sql_lint.py:606` reuses shared candidate-node filtering for lint model/project selection.
- Test observations: `tests/core/test_cli.py:151` and `tests/core/test_cli.py:191` now exercise real `SchemaDiff.compare_all()` selector flow; `tests/core/test_sql_lint.py:592`, `598`, `688`, and `734` cover disabled-rule precedence and lint model selection behavior.
- Documentation observation: `docs/docs/reference/cli.md:217-220` documents disabled-rule precedence and model/project lint selection defaults.

# Supports Claims

- `ticket:c10lint24#ACC-001` — parent green tests exercise `diff schema customers` through real `SchemaDiff.compare_all()` flow and observe only `customers` is compared.
- `ticket:c10lint24#ACC-002` — parent green tests exercise `diff schema missing_model` through real `SchemaDiff.compare_all()` flow and observe no nodes are compared, so the selector is not broadened to the whole project.
- `ticket:c10lint24#ACC-003` — child red evidence showed `lint file --disable-rules` was not forwarded before the fix; green tests verify disabled rules are forwarded and suppress `select-star`.
- `ticket:c10lint24#ACC-004` — green tests and docs verify disabled rules win after enabled-rule selection; support is behavioral green evidence, not full red evidence.
- `ticket:c10lint24#ACC-005` — child red evidence showed warnings duplicated under `Other`; green tests verify file/model/project output does not duplicate warning/info grouping.
- `ticket:c10lint24#ACC-006` — green tests verify lint model iteration excludes external and ephemeral models and uses shared segment FQN matching.
- `initiative:dbt-110-111-hardening#OBJ-006` — focused tests and docs support user-facing CLI selector/options behavior.

# Challenges Claims

- Challenges any claim that the Ralph iteration produced strict red-before-green evidence for every acceptance criterion: `ticket:c10lint24#ACC-004` has green behavior evidence and static pre-change rationale, but no distinct recorded red failure for the precedence test.

# Environment

Commit: baseline `b2e4e9825b5b0e6de372c8ae9cefd5b6e2d6e926` with uncommitted `ticket:c10lint24` diff

Branch: `loom/dbt-110-111-hardening`

Runtime: OpenCode tool session; `uv` project environment

OS: darwin

Relevant config: `pyproject.toml`, `tests/core/test_cli.py`, `tests/core/test_sql_lint.py`, `tests/core/test_diff.py`, `docs/docs/reference/cli.md`

External service / harness / data source when applicable: local filesystem and local test/type/lint commands only; no remote CI observed for this evidence

# Validity

Valid for: the uncommitted `ticket:c10lint24` implementation diff and local validation commands listed above.

Fresh enough for: ticket acceptance review and critique disposition for the named lint/diff CLI claims.

Recheck when: source, tests, docs, dependency versions, dbt manifest fixture behavior, selector helpers, or lint rule APIs change.

Invalidated by: a later implementation diff that changes the CLI selector flow, lint rule filtering, lint output grouping, or lint model selection without rerunning equivalent validation.

Supersedes / superseded by: none.

# Limitations

- The evidence is local-only and does not include GitHub Actions results for the eventual commit.
- The initial red evidence did not include a distinct recorded failing test for the `--rules`/`--disable-rules` overlap precedence claim.
- The CLI tests use mocks/fixture contexts for command flow and do not execute a live warehouse schema diff.
- Existing basedpyright warnings remain; this evidence only records that changed-source error count is zero.

# Result

The observed validation supports the implemented lint/diff CLI behavior and docs after parent refinement, with a limited red-evidence gap for the rule-precedence acceptance claim.

# Interpretation

The evidence is sufficient to support parent acceptance review if the ticket explicitly consumes the low red-evidence limitation. It does not by itself close the ticket, prove remote CI, or eliminate all selector edge cases outside the covered command flows.

# Related Records

- ticket:c10lint24
- packet:ralph-ticket-c10lint24-20260504T184608Z
- critique:c10lint24-lint-diff-cli-review
- initiative:dbt-110-111-hardening
