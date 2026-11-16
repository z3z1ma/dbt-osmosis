Status: done
Created: 2026-11-16
Updated: 2026-11-16

# Cross-package patch ownership (#443)

## Scope
Investigate issue 443 against main 3cc00a57fa4883050fd5cffeba7adee9d6d0d4d3,
fix confirmed default-selection bug, and publish a tested draft PR.
The delegating user explicitly authorized implementation and draft publication.

## Non-goals
No merge, release, issue comment, external-package opt-in redesign, or changes to
`.github/workflows/tests.yml` and `tests/test_yaml_inheritance.py` (separate owner).

## Acceptance criteria
1. Reproduce package-owned patches using real dbt parsing, not just mocked nodes.
2. Default refactor must leave such patches alone, create no root duplicate or
   blank routed YAML, and leave the project parseable.
3. Verify dry-run with and without an existing same-relative-path root YAML.
4. Preserve local/unpatched candidates, read-only consumers including SQL lint,
   and explicit include-external selection.

## References
- Canonical report: https://github.com/z3z1ma/dbt-osmosis/issues/443 (open, read 2026-11-16).
- `.10x/reviews/2026-07-05-node-filters-complexipy.md`: default external exclusion.
- `src/dbt_osmosis/core/node_filters.py`: shared candidate-selection contract.
- Repository and core AGENTS.md; required 10x workflow.

## Assumptions
Source-backed: the shared selector governs restructuring and documentation transforms;
`include_external` is an explicit opt-in. A patch package is authoritative even if
the corresponding root-relative file exists or the package folder is renamed.
Use the existing external opt-in contract for patch ownership in bulk YAML
management. Read-only consumers must retain node-ownership selection.

## Journal
- Created isolated worktree/branch from freshly fetched main; original checkout untouched.
- Existing local uv-managed environment: Python 3.11.15, dbt-core 1.10.20.
- Added subprocess integration fixture with a disabled package seed and local override.
  Real manifest reports `seed.analytics.exceptions` patched by
  `patch_owner://seeds/seeds.yml`.
- Baseline command: `PYTHONPATH=src <existing-venv>/bin/python -m pytest tests/test_cross_package_patch.py -q`.
  Result: 2 failed, 2 passed. Normal refactor creates unexpected root YAML (with
  and without an existing empty seeds/seeds.yml). Both dry runs preserve all YAML bytes.
  Thus cross-package writes are confirmed; dry-run write allegation is not reproduced
  on current main under these conditions. Logging requires access to standard home log.
- Minimal fix extends default external exclusion to model/seed patch ownership.
  Does not reinterpret external patches as missing or resolve guessed package folders.

- Independently confirmed baseline generated an empty seeds/schema.yml and a populated
  seeds/seeds.yml; a subsequent full dbt parse fails with duplicate schema entries.
- Baseline dry-run on the populated duplicate exits 1 during dbt parsing with the
  duplicate-schema error; all YAML bytes and mtimes remain unchanged. This is an
  additional null result for the dry-run allegation, not proof across dbt versions.
- Review caught the initial shared-selector change also excluding local SQL models
  from lint. Revised to an explicit exclude_external_patches keyword enabled only
  by YAML-management callers; read-only consumers retain previous defaults.
- Initial post-fix adjacent run: 117 passed, 2 skipped.
- Expanded final-source run: 214 passed, 2 skipped, 4 failures in sync test doubles
  that accepted only the old selector signature. Updated all four doubles to accept
  the keyword and assert it is True; original behavioral assertions retained.
  Independent reviewer passed the test-only adjustment; full sync rerun passed: 34 passed, 2 dbt deprecation warnings in 34.01s.
  All 218 selected tests pass across these runs (2 fixture-related skips).

## Blockers
None currently.

## Evidence
- AC1: baseline real-parser regression run failed both normal cases as expected.
- AC2/AC3: all four final-source CLI integration cases passed in the expanded run,
  preserving YAML bytes/mtimes and full dbt parsing. Baseline populated-duplicate
  dry-run was separately checked and did not alter YAML.
- AC4: 24 selection matrix cases (model/seed, patch owner, explicit external opt-in,
  YAML-management/read-only mode) and both lint-selection cases passed.
- Expanded command (existing uv-managed Python environment, PYTHONPATH=src):
  `python -m pytest tests/test_cross_package_patch.py tests/core/test_node_filters.py tests/core/test_sql_lint.py tests/core/test_transforms.py tests/core/test_restructuring.py tests/core/test_path_management.py tests/core/test_schema.py tests/core/test_sync_operations.py -q`
- Ruff check/format checks passed on changed Python files; git diff --check passed.
- Two skipped tests require source nodes absent from demo_duckdb. Two warnings are
  dbt's Click parser deprecations. No full repository suite or dbt/adapter matrix claim.

## Review
Independent reviewer `/root/review_443` inspected the revised diff and actual
assertions without repeating tests. Verdict: pass; no blocking findings.
The initial significant SQL lint selection regression was fixed and covered.
Residual limits: explicit --include-external and direct per-node APIs bypassing
bulk selection retain existing behavior. No-action here: this ticket fixes default
bulk YAML workflows, not a general redesign of package path resolution.

## Retrospective
The patch owner and node owner differ for overrides. Existence checks on a
root-relative path cannot establish ownership. Dry-run allegations need independent
verification; baseline tests already pass for this case.

Review lesson: shared candidate selection also serves SQL lint, discovery, and
reporting. YAML write ownership restrictions must be explicit at mutation callers.
