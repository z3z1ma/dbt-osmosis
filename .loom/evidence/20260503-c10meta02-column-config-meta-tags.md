---
id: evidence:c10meta02-column-config-meta-tags
kind: evidence
status: recorded
created_at: 2026-05-03T22:44:00Z
updated_at: 2026-05-03T22:44:00Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  ticket:
    - ticket:c10meta02
  critique:
    - critique:c10meta02-column-config-meta-tags
  packet:
    - packet:ralph-ticket-c10meta02-20260503T215906Z
    - packet:ralph-ticket-c10meta02-20260503T220442Z
    - packet:ralph-ticket-c10meta02-20260503T221219Z
external_refs:
  github_issue: https://github.com/z3z1ma/dbt-osmosis/issues/353
  implementation_commit: https://github.com/z3z1ma/dbt-osmosis/commit/68d72bb96d0abc8268a00936591a2b3862bac71e
---

# Summary

Observed red/green and post-commit validation for ticket:c10meta02. This evidence supports local behavior for effective column metadata and tags from legacy top-level fields plus dbt 1.10+ `config.meta` / `config.tags` in settings resolution, property access, and inheritance.

# Procedure

Observed at: 2026-05-03T22:44:00Z

Source state: branch `loom/dbt-110-111-hardening`, commit `68d72bb96d0abc8268a00936591a2b3862bac71e`.

Procedure:

- Ralph iteration 1 added focused settings/property/inheritance tests, observed `4 failed` before implementation, then implemented effective column meta/tag helpers and observed `4 passed`.
- Ralph iteration 2 added YAML-source property coverage; strict red state was not observed because iteration 1's uncommitted implementation already satisfied the new test.
- Ralph iteration 3 added a YAML fallback regression after critique found a real fallback risk, observed the focused regression fail before implementation, then pass after the fix.
- Parent reran `uv run pytest tests/core/test_settings_resolver.py tests/core/test_property_accessor.py tests/core/test_inheritance_behavior.py` after commit `68d72bb96d0abc8268a00936591a2b3862bac71e`.

Expected result when applicable: new tests should pass after the helpers and fallback guard are implemented; YAML-source column reads should fall back to manifest when YAML lacks the requested meta/tags fields.

Actual observed result:

- Post-commit parent validation passed with `50 passed, 3 skipped in 12.46s`.
- Prior parent validation before commit also observed `ruff check` passing and `pyright` reporting `0 errors` for changed core files.

Procedure verdict / exit code: final parent validation passed with exit code 0.

# Artifacts

- Command: `uv run pytest tests/core/test_settings_resolver.py tests/core/test_property_accessor.py tests/core/test_inheritance_behavior.py`
- Final observed output excerpt:

```text
collected 53 items

tests/core/test_settings_resolver.py ..............                      [ 26%]
tests/core/test_property_accessor.py ......................sss           [ 73%]
tests/core/test_inheritance_behavior.py ..............                   [100%]

======================== 50 passed, 3 skipped in 12.46s ========================
```

# Supports Claims

- ticket:c10meta02#ACC-001: column-level dbt-osmosis options under `config.meta` override node-level settings in focused tests.
- ticket:c10meta02#ACC-002: `PropertyAccessor` sees effective column meta from manifest and YAML-source reads.
- ticket:c10meta02#ACC-003: inheritance carries upstream `config.meta` / `config.tags` into downstream meta/tags behavior in fixture-backed tests.
- ticket:c10meta02#ACC-004: legacy top-level meta/tags remain supported with deterministic merge precedence.
- ticket:c10meta02#ACC-006: helper boundary and tests document the compatibility rule.

# Challenges Claims

None for the final observed local tests. This evidence is insufficient for ticket:c10meta02#ACC-005 under dbt 1.11.x.

# Environment

Commit: `68d72bb96d0abc8268a00936591a2b3862bac71e`

Branch: `loom/dbt-110-111-hardening`

Runtime: Python 3.13.9, pytest 8.3.5, local `uv` environment. Test output included a non-failing warning that `VIRTUAL_ENV=/Users/alexanderbutler/code_projects/personal/dbt-osmosis/.venv` did not match the sibling worktree `.venv` and was ignored.

OS: macOS / darwin.

Relevant config: focused unit tests plus existing fixture-backed inheritance tests; no dbt 1.11 matrix was run.

External service / harness / data source when applicable: none.

# Validity

Valid for: local behavior in commit `68d72bb96d0abc8268a00936591a2b3862bac71e` for effective column config metadata/tag reads and inheritance.

Fresh enough for: critique and ticket support for ticket:c10meta02#ACC-001 through ticket:c10meta02#ACC-004 and ticket:c10meta02#ACC-006.

Recheck when: settings resolution, property access, inheritance graph normalization, dbt `ColumnInfo.config`, or YAML-source fallback behavior changes.

Invalidated by: later changes that bypass the effective helper boundary, alter precedence, or remove the fallback regression coverage.

Supersedes / superseded by: not superseded.

# Limitations

This evidence does not establish dbt 1.11 adapter-backed parsed-fixture behavior, full CLI YAML refactor behavior, or all config-resolution precedence paths. Broader fixture and CI evidence remains with ticket:c10cfg12 and ticket:c10ci06.

# Result

The observed tests passed in committed state, supporting the local compatibility change while leaving dbt 1.11 runtime evidence as explicit follow-up.

# Interpretation

The evidence supports the ticket's local implementation claims but does not by itself justify closing the dbt 1.11 fixture acceptance gap.

# Related Records

- ticket:c10meta02
- critique:c10meta02-column-config-meta-tags
- packet:ralph-ticket-c10meta02-20260503T215906Z
- packet:ralph-ticket-c10meta02-20260503T220442Z
- packet:ralph-ticket-c10meta02-20260503T221219Z
- initiative:dbt-110-111-hardening
