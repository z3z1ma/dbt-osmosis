---
id: evidence:gh326skip-description-inheritance-validation
kind: evidence
status: recorded
created_at: 2026-05-05T07:41:32Z
updated_at: 2026-05-05T07:41:32Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  ticket:
    - ticket:gh326skip
  packets:
    - packet:ralph-ticket-gh326skip-20260505T072053Z
external_refs:
  github_issue: https://github.com/z3z1ma/dbt-osmosis/issues/326
---

# Summary

Observed red/green and parent validation for the `skip-inherit-descriptions` inheritance option.

# Procedure

Observed at: 2026-05-05T07:41:32Z
Source state: uncommitted diff on `loom/dbt-110-111-hardening` based on `db73d9e4a6fb2c28939d31aab611ca8d1f2f7021`, with unrelated prior issue-ticket diffs present outside this ticket scope.
Procedure: Ralph added failing inheritance/settings tests, implemented the setting/CLI option and description inheritance skip, then parent formatted the touched test file and reran focused tests, Ruff checks, CLI help observations, and whitespace checks.
Expected result when applicable: Empty child descriptions should stay empty when skipping is enabled, existing child descriptions should remain local, tags and meta should still inherit, skip should win over force deterministically, and CLI/config resolution should expose the option.
Actual observed result: Failing tests turned green; parent validation and Oracle critique passed.
Procedure verdict / exit code: Pass; all parent validation commands exited 0.

# Artifacts

Child red observation:

- New skip-description tests failed before implementation with 5 failures, as expected.

Parent green commands:

```bash
uv run ruff format tests/core/test_settings.py
```

Observed result: `1 file reformatted`.

```bash
uv run pytest tests/core/test_inheritance_behavior.py tests/core/test_settings.py -q
```

Observed result: `83 passed, 2 warnings`.

```bash
uv run ruff check src/dbt_osmosis/core/settings.py src/dbt_osmosis/core/transforms.py src/dbt_osmosis/cli/main.py tests/core/test_inheritance_behavior.py tests/core/test_settings.py
```

Observed result: `All checks passed!`.

```bash
uv run ruff format --check src/dbt_osmosis/core/settings.py src/dbt_osmosis/core/transforms.py src/dbt_osmosis/cli/main.py tests/core/test_inheritance_behavior.py tests/core/test_settings.py
```

Observed result: `5 files already formatted`.

```bash
git diff --check
```

Observed result: passed with no output.

```bash
uv run dbt-osmosis yaml document --help
uv run dbt-osmosis yaml refactor --help
```

Observed result: both help outputs include `--skip-inherit-descriptions`.

# Supports Claims

- ticket:gh326skip#ACC-001: tests show empty child descriptions remain empty when skip is enabled.
- ticket:gh326skip#ACC-002: tests show existing child descriptions survive skip plus force.
- ticket:gh326skip#ACC-003: tests show tags and meta still inherit when description inheritance is skipped.
- ticket:gh326skip#ACC-004: tests show skip wins deterministically over force.
- ticket:gh326skip#ACC-005: settings tests and CLI help observations show the option works through resolver-backed settings and the YAML CLI surfaces.

# Challenges Claims

None.

# Environment

Commit: uncommitted diff based on `db73d9e4a6fb2c28939d31aab611ca8d1f2f7021`.
Branch: `loom/dbt-110-111-hardening`.
Runtime: `uv run`; external `VIRTUAL_ENV` warning observed and ignored by `uv`.
OS: macOS Darwin.
Relevant config: focused inheritance/settings unit tests and CLI help.
External service / harness / data source when applicable: GitHub issue #326.

# Validity

Valid for: local source state and focused inheritance/config behavior.
Fresh enough for: `ticket:gh326skip` parent acceptance review and critique.
Recheck when: inheritance transforms, `resolve_setting()`, CLI option wiring, or `YamlRefactorSettings` changes.
Invalidated by: changes to description inheritance or settings precedence without rerunning the focused tests.
Supersedes / superseded by: N/A.

# Limitations

This evidence is focused and local. It does not include full-suite, remote CI, or a runtime CLI invocation beyond help output.

# Result

The observed implementation adds an opt-in description inheritance skip that preserves default behavior and leaves tag/meta inheritance intact.

# Interpretation

The evidence supports local acceptance of `ticket:gh326skip` after critique. It does not establish remote CI health.

# Related Records

- ticket:gh326skip
- critique:gh326skip-description-inheritance-review
- packet:ralph-ticket-gh326skip-20260505T072053Z
- initiative:issue-pr-zero
