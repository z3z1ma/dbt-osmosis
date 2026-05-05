---
id: evidence:gh333meta-skip-inherited-meta-validation
kind: evidence
status: recorded
created_at: 2026-05-05T07:19:30Z
updated_at: 2026-05-05T07:19:30Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  ticket:
    - ticket:gh333meta
  packets:
    - packet:ralph-ticket-gh333meta-20260505T070046Z
external_refs:
  github_issue: https://github.com/z3z1ma/dbt-osmosis/issues/333
---

# Summary

Observed red/green and parent validation for the `skip-inheritance-for-meta-keys` inheritance option.

# Procedure

Observed at: 2026-05-05T07:19:30Z
Source state: uncommitted diff on `loom/dbt-110-111-hardening` based on `db73d9e4a6fb2c28939d31aab611ca8d1f2f7021`, with unrelated prior ticket diffs present outside this ticket scope.
Procedure: Ralph added failing inheritance/settings tests, implemented the setting/CLI option and inherited meta filtering, and parent reran focused tests, Ruff checks, format checks, CLI help, and whitespace checks.
Expected result when applicable: Configured meta keys should not inherit from ancestors, other meta keys should inherit, local child meta should survive, classic `meta` and `config.meta` should both work, and `skip-merge-meta` should remain the all-meta opt-out.
Actual observed result: Failing tests turned green; parent validation and Oracle critique passed.
Procedure verdict / exit code: Pass; all parent commands exited 0.

# Artifacts

Child red observation:

- New tests failed before implementation with 4 failures: missing setting errors plus assertions showing `expression`/`doc_blocks` still inherited.

Parent green commands:

```bash
uv run pytest tests/core/test_inheritance_behavior.py tests/core/test_settings.py -q
```

Observed result: `78 passed, 2 warnings`.

```bash
uv run ruff check src/dbt_osmosis/core/settings.py src/dbt_osmosis/core/inheritance.py src/dbt_osmosis/cli/main.py tests/core/test_inheritance_behavior.py tests/core/test_settings.py
```

Observed result: `All checks passed!`.

```bash
uv run ruff format --check src/dbt_osmosis/core/settings.py src/dbt_osmosis/core/inheritance.py src/dbt_osmosis/cli/main.py tests/core/test_inheritance_behavior.py tests/core/test_settings.py
```

Observed result: `5 files already formatted`.

```bash
uv run dbt-osmosis yaml refactor --help
```

Observed result: help output includes `--skip-inheritance-for-meta-keys TEXT`.

```bash
git diff --check
```

Observed result: passed with no output.

# Supports Claims

- ticket:gh333meta#ACC-001: tests show `expression` and `doc_blocks` do not inherit.
- ticket:gh333meta#ACC-002: tests show `gdpr` and `security_classification` still inherit.
- ticket:gh333meta#ACC-003: tests show local child meta survives.
- ticket:gh333meta#ACC-004: tests cover classic `meta` and nested `config.meta`.
- ticket:gh333meta#ACC-005: existing `skip-merge-meta` coverage remained green in the focused inheritance run.

# Challenges Claims

None.

# Environment

Commit: uncommitted diff based on `db73d9e4a6fb2c28939d31aab611ca8d1f2f7021`.
Branch: `loom/dbt-110-111-hardening`.
Runtime: `uv run`; external `VIRTUAL_ENV` warning observed and ignored by `uv`.
OS: macOS Darwin.
Relevant config: focused inheritance/settings unit tests and CLI help.
External service / harness / data source when applicable: GitHub issue #333.

# Validity

Valid for: local source state and focused inheritance/config behavior.
Fresh enough for: `ticket:gh333meta` parent acceptance review and critique.
Recheck when: inheritance graph merging, `resolve_setting()`, CLI option wiring, or `YamlRefactorSettings` changes.
Invalidated by: changes to metadata inheritance or settings precedence without rerunning the focused tests.
Supersedes / superseded by: N/A.

# Limitations

This evidence is focused and local. It does not include full-suite, remote CI, or a runtime CLI invocation beyond help output.

# Result

The observed implementation adds an opt-in inherited meta key skip list that preserves default behavior and other metadata inheritance.

# Interpretation

The evidence supports local acceptance of `ticket:gh333meta` after critique. It does not establish remote CI health.

# Related Records

- ticket:gh333meta
- critique:gh333meta-skip-inherited-meta-review
- packet:ralph-ticket-gh333meta-20260505T070046Z
- initiative:issue-pr-zero
