---
id: evidence:c10loom04-dbt-loom-parser-validation
kind: evidence
status: recorded
created_at: 2026-05-04T21:14:22Z
updated_at: 2026-05-04T21:22:06Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  ticket:
    - ticket:c10loom04
  packet:
    - packet:ralph-ticket-c10loom04-20260504T210814Z
  research:
    - research:dbt-110-111-api-surfaces
external_refs: {}
---

# Summary

Observed red/green validation for replacing the dbt-loom `ModelParser.parse_from_dict(None, node)` parser hack with `ModelNode.from_dict(node)` and for isolating project manifest mutation behind `_set_project_manifest()`.

# Procedure

Observed at: 2026-05-04T21:14:22Z

Source state: commit `a33563f558b432a59f3d656fdfddb7ec365ac241` on branch `loom/dbt-110-111-hardening` with uncommitted changes in `src/dbt_osmosis/core/config.py`, `tests/core/test_config.py`, `ticket:c10loom04`, and `packet:ralph-ticket-c10loom04-20260504T210814Z`.

Procedure: Ralph child first added the regression test and ran the focused config test file before implementation. Parent then reviewed the source diff and reran focused validation after implementation. After `ruff-format` adjusted the regression test formatting during full pre-commit, parent reran full pre-commit and focused config tests.

Expected result when applicable: the pre-fix regression should fail because the old path calls `ModelParser.parse_from_dict(None, node)`. After implementation, eligible public/private dbt-loom model nodes should import through `ModelNode.from_dict(node)`, protected and non-model nodes should remain excluded, and changed files should pass Ruff and basedpyright without errors.

Actual observed result: child reported expected red with `1 failed, 20 passed`, proving the patched `ModelParser.parse_from_dict` was called and loom nodes were skipped. Parent observed green focused tests, green Ruff, and basedpyright with zero errors after implementation. Parent later observed full pre-commit pass and focused config tests still green after formatting.

Procedure verdict / exit code: mixed expected red then green. Parent-observed green commands exited successfully; basedpyright reported warnings but `0 errors`.

# Artifacts

Child-reported red command:

```bash
uv run pytest tests/core/test_config.py -q
```

Child-reported red result:

```text
1 failed, 20 passed
```

The failing regression patched `dbt.parser.models.ModelParser.parse_from_dict` to raise `AssertionError("ModelParser parser hack should not be called")`, causing the old implementation to skip loom nodes and leave `manifest.nodes` empty.

Parent-observed green commands:

```bash
uv run pytest tests/core/test_config.py -q
```

```text
21 passed in 5.92s
```

```bash
uv run ruff check src/dbt_osmosis/core/config.py tests/core/test_config.py
```

```text
All checks passed!
```

```bash
uv run basedpyright src/dbt_osmosis/core/config.py
```

```text
0 errors, 29 warnings, 0 notes
```

Parent-observed final local gate after formatting:

```bash
uv run pre-commit run --all-files
```

```text
check python ast.........................................................Passed
check json...............................................................Passed
check yaml...............................................................Passed
check toml...............................................................Passed
fix end of files.........................................................Passed
trim trailing whitespace.................................................Passed
detect private key.......................................................Passed
debug statements (python)................................................Passed
ruff-format..............................................................Passed
ruff.....................................................................Passed
basedpyright.............................................................Passed
Detect hardcoded secrets.................................................Passed
Lint GitHub Actions workflow files.......................................Passed
```

```bash
uv run pytest tests/core/test_config.py -q
```

```text
21 passed in 5.96s
```

Parent diff inspection observed:

- `src/dbt_osmosis/core/config.py` now imports `ModelNode` instead of `ModelParser`.
- `_add_cross_project_references()` now appends `ModelNode.from_dict(node)` for exposed model nodes.
- `_set_project_manifest()` isolates the private `_DbtProject__manifest` write and uses a public property setter if one becomes available.
- `create_dbt_project_context()` uses `_set_project_manifest()` after optional dbt-loom manifest augmentation.
- `tests/core/test_config.py` adds `test_add_cross_project_references_imports_exposed_models_without_parser_hack()`.

# Supports Claims

- ticket:c10loom04#ACC-001: focused test and diff show cross-project model nodes now construct through `ModelNode.from_dict(node)`.
- ticket:c10loom04#ACC-002: diff and guard test show `ModelParser.parse_from_dict(None, ...)` is no longer called.
- ticket:c10loom04#ACC-003: diff shows project manifest mutation is isolated behind `_set_project_manifest()` with compatibility documentation.
- ticket:c10loom04#ACC-004: regression test uses fake dbt-loom manifests with public/private/protected/non-model node cases and verifies eligible nodes are added.
- ticket:c10loom04#ACC-005: implementation still catches optional dbt-loom import/addition failures and logs node import failures by `unique_id`; focused tests do not challenge that behavior.
- initiative:dbt-110-111-hardening#OBJ-001: removes a dbt parser private-method call in a dbt 1.10/1.11 compatibility path.
- initiative:dbt-110-111-hardening#OBJ-007: records focused validation for the compatibility fix.

# Challenges Claims

None observed. The expected red failure challenged the pre-fix implementation, not the post-fix claims.

# Environment

Commit: `a33563f558b432a59f3d656fdfddb7ec365ac241` plus uncommitted c10loom04 implementation and Loom record changes.

Branch: `loom/dbt-110-111-hardening`

Runtime: `uv run python --version` -> `Python 3.10.15`

OS: macOS 15.7.5 build 24G624

Relevant config: base `uv` environment; `VIRTUAL_ENV` warning observed and ignored by `uv`.

External service / harness / data source when applicable: none.

# Validity

Valid for: local source state containing the post-format c10loom04 diff and the installed local dependency set.

Fresh enough for: recommended code-change/dbt-compatibility critique and ticket review state.

Recheck when: `src/dbt_osmosis/core/config.py`, `tests/core/test_config.py`, dbt-core/dbt-core-interface versions, or dbt-loom manifest shape changes.

Invalidated by: failing focused tests, Ruff failure, basedpyright errors, or evidence that real dbt-loom nodes differ materially from the fixture-derived node dictionaries.

Supersedes / superseded by: none.

# Limitations

This evidence does not establish full dbt-core matrix coverage, real dbt-loom package integration, remote CI success, or closure readiness. The regression derives node dictionaries from the demo project manifest to keep node shape realistic, but it does not exercise every possible dbt node field combination.

# Result

The observed post-fix source state passes focused local validation and no longer uses the invalid `ModelParser.parse_from_dict(None, node)` path in `_add_cross_project_references()`.

# Interpretation

The evidence supports moving `ticket:c10loom04` to review because implementation and local validation are complete. It does not by itself satisfy the recommended critique gate, remote CI expectations, or final acceptance decision.

# Related Records

- ticket:c10loom04
- packet:ralph-ticket-c10loom04-20260504T210814Z
- research:dbt-110-111-api-surfaces
- initiative:dbt-110-111-hardening
