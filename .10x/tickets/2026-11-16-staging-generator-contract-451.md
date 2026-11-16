Status: active
Created: 2026-11-16
Updated: 2026-11-16

# Complete staging generator contract compatibility (#451)

## Scope
Review and complete PR #451 against supported dbt-core-interface output; preserve contributor commit and existing legacy output compatibility.

## Non-goals
No release, dependency change, AI generator redesign or unrelated main changes.

## Acceptance Criteria
- Actual supported dependency output supplies model name and SQL/YAML consistently, including mixed-case table names where fallback differs.
- Legacy staging_name/sql/yaml and absent-name fallback remain compatible.
- Generator returns artifacts without writing; CLI writes matching SQL/YAML names; dry-run writes nothing.
- Focused tests, lint, independent review and exact-head CI pass before merge.

## References
- https://github.com/z3z1ma/dbt-osmosis/pull/451 (original 41c99b385609b74606550b53deed9bcd74fe99f6)
- pyproject.toml: dbt-core-interface>=1.1.7
- Installed dbt-core-interface 1.1.7 staging_generator.py: generate_from_source returns model_name/model_sql/schema_yml; _generate_model_name applies snake case.
- AGENTS.md and src/dbt_osmosis/core/AGENTS.md

## Assumptions
- User delegated review, fixes, pushes and merge decisions; no routine approval needed.
- Existing legacy key precedence and fallback are retained (source-backed).
- Dependency returned model name is authoritative for its schema model name (source-backed).

## Journal
- Isolated clone/worktree created; original checkout untouched.
- Current main verified 674de99a5ef26a098522e67e2ffd575671c214b5; Tests 43878946340 and lint 43878928289 succeeded.
- Original PR adds SQL/YAML aliases but ignores model_name; added fixture stg_orders coincides with fallback.
- Inspected real dependency 1.1.7: snake-case table normalization means CustomerOrders produces a different model name from raw fallback. Convenience generator returns content only, with no writes.
- Merged main into local PR branch without conflicts, preserving contributor commit.
- Added real-generator wrapper and CLI regression cases plus legacy key precedence/fallback coverage. Confirmed failures before editing production code, then accepted model_name as fallback alias after legacy staging_name; retained existing type/nonempty fallback and legacy SQL/YAML precedence.
- Full generator and CLI suites passed locally; no CLI production behavior or write safeguards changed.

## Blockers
None identified yet.

## Evidence
- Before the fix, `PYTHONPATH=src /Users/alexanderbut/code_projects/personal/dbt-osmosis/.venv/bin/python -m pytest tests/core/test_generators.py tests/core/test_cli_generate_group.py -k 'real_interface or output_key_compatibility' -q --no-cov`: **5 failed, 2 passed, 43 deselected**. The actual 1.1.7 generator returned a normalized schema name while the wrapper/CLI used `stg_CustomerOrders`; CLI created incorrectly named SQL/YAML files. Legacy precedence and absent-name fallback passed.
- After the alias fix, the same Python/PYTHONPATH command without `-k` ran both full focused suites: **50 passed**, 2 existing Click/dbt deprecation warnings, 19.83 seconds. This covers legacy and modern keys, fallback/precedence, actual generator SQL/YAML content and names, custom/default output paths, no wrapper writes, CLI real writes and dry-run, and existing overwrite/root-boundary guards.
- `ruff format --preview --check` and `ruff check` on the three changed Python files: passed.
- Local validation uses Python 3.11 and dbt-core-interface 1.1.7 (both declared lower bound and lock version); CI matrix and independent review remain separate gates. Normal dbt fixture/logging access required sandbox escalation after an initial logger PermissionError; the approved run passed. No shared environment/package changes were made.

## Review
Pending independent review.

## Retrospective
The contributor's content aliases are valid, but content alone does not satisfy the artifact contract: a generator-derived schema name must agree with filenames. A fixture whose model name equals the wrapper fallback masked the omission. Testing the real dependency with a mixed-case table caught the mismatch in both wrapper and CLI without mocking away the boundary. Existing write guards were retained unchanged.
