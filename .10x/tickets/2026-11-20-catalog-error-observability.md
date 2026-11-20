Status: active
Created: 2026-11-20
Updated: 2026-11-20
Parent: .10x/knowledge/maintenance-roadmap.md

# Surface supplied catalog generation failures

## Scope
Implement `.10x/specs/catalog-error-observability.md` with minimal loader warning, regression coverage and short artifact-reference documentation. Update maintenance roadmap accurately.
## Non-goals
No #468 actions, prerelease auditing, dependency upgrades, new validation rules, fallback changes, warehouse calls or release.
## Acceptance Criteria
1. Compatible partial artifacts warn once per load with path and error count, without raw error strings; nodes/sources remain available and input bytes unchanged.
2. Missing/null/empty errors stay quiet; context caching avoids duplicate warning on read.
3. Focused catalog/settings tests, Ruff and type check pass offline. Independent review and exact-head CI before merge.
## References
`.10x/specs/catalog-error-observability.md`; `.10x/specs/catalog-input-contract.md`; root/core AGENTS.md; `.10x/knowledge/maintenance-roadmap.md`.
## Assumptions
Record-backed: existing generated-catalog warnings establish incomplete catalog observability; official schema accepts partial catalog with errors. User-ratified: independently implement bounded roadmap improvement, no release, leave #468 untouched.
## Journal
- 2026-11-20: Isolated worktree `osmosis-next-quality` at current main 4843795. #473 merged certifi lock refresh; stable v1.6.0 published Nov 18. Existing source loads supplied errors silently. Primary docs/schema read. Original clone remains unchanged.
## Blockers
None for implementation; remote integration depends on permitted actions and exact-head checks.
## Evidence
Pending executor.
## Review
### Independent adversarial review (2026-11-20)
- **Verdict: pass**, subject to exact-head CI and integration gates. Reviewed the final diff against main `4843795`, governing specs, loader/deserializer boundary, context caching, actual test assertions, and executor evidence; did not repeat validation.
- **Finding (minor, resolved):** the original count assertion checked a bare digit that could occur in the temporary path. The executor now requires the complete `N generation error(s)` phrase and recorded six affected cases passing after correction. No remaining findings.
- The warning runs only after successful typed deserialization and passes only the supplied path and count to logging. Usable nodes/sources and producer errors remain on the returned artifact. Missing/null/empty errors remain quiet. Existing preflight caching avoids another load or warning on repeated context reads; no fallback, generation or validation behavior changes.
- Assertions cover one/multiple errors, no raw error content in logger calls, preserved artifact bytes, usable node/source metadata and actual context cache identity. Documentation accurately describes this bounded behavior. No dependency, prerelease, #468 or release changes appear in the diff.
- **Residual risk:** local evidence covers Core 1.11.17 / Python 3.13 / DuckDB only; the supported-version CI matrix remains required. This does not certify Fusion producers, catalog completeness, or external warehouse behavior. Existing generated-catalog warning behavior is unchanged.
## Retrospective
Pending executor.

### Executor journal
- Added six focused regressions before the implementation: one and two producer errors, missing/null/empty errors, and context preflight plus repeated reads. Initial sandbox import could not open the package's existing home log; approved offline rerun used the same environment. Baseline command `PYTHONPATH=src /tmp/osmosis-supported/bin/python -m pytest tests/test_catalog_contract.py -k 'generation_errors or partial_catalog' -q` produced **3 failed, 3 passed, 49 deselected**: warning cases failed because no warning existed. Log `/tmp/osmosis-catalog-warning-before.log`.
- Loader now warns only after successful deserialization and returns the same catalog object. Message contains supplied path, count and incomplete-metadata guidance; it does not format producer error strings. No generation, fallback, validation, cache or dependency changes.
- Tests check both nodes and sources remain usable, original artifact bytes remain unchanged, raw errors remain on returned object but absent from every logger call, and context preflight produces one warning with repeated reads remaining cached.
- Canonical artifact docs explain partial-catalog warning. Roadmap now reflects shipped documentation, bounded observability work and remaining producer-fixture evidence gap. Parent verified [#473](https://github.com/z3z1ma/dbt-osmosis/pull/473) merged at 4843795 and stable [v1.6.0](https://github.com/z3z1ma/dbt-osmosis/releases/tag/v1.6.0) published 2026-11-18; no release performed here.
- `ruff check --no-cache` and `ruff format --check --no-cache` on changed source/test files passed (initial nested test context manager lint was corrected). `basedpyright --pythonpath /tmp/osmosis-supported/bin/python src/dbt_osmosis/core/catalog_operations.py`: **0 errors, 47 warnings** (existing Any-style warnings). Log `/tmp/osmosis-catalog-warning-types.log`. `git diff --check` passed.
- Local environment: Python 3.13.14, Core 1.11.17, DuckDB 1.10.1. Existing environment used without dependency installation or lock changes.
- `PYTHONPATH=src /tmp/osmosis-supported/bin/python -m pytest tests/test_catalog_contract.py tests/core/test_settings.py -q`: **110 passed in 34.05s**, log `/tmp/osmosis-catalog-warning-after.log`. Includes existing real Core-generated catalog and non-dry-run CLI preservation regressions. Offline local DuckDB only, no Fusion/warehouse certification.
- Independent reviewer identified a weak count assertion that could match temporary-path digits. Tightened it to the complete `N generation error(s)` phrase. Re-ran only the six affected new cases: **6 passed, 49 deselected**, log `/tmp/osmosis-catalog-warning-review.log`.

### Acceptance evidence
1. One/two-error fixtures require exactly one warning, complete error-count phrase, path and actionable field guidance. Both node/source comments and error array survive loading; input bytes unchanged; all logger calls exclude error text.
2. Three no-error cases assert no warning. Real `YamlRefactorContext` preflight caches the loaded artifact and two reads retain object identity without another warning.
3. Focused suite 110 passed, tightened assertions six passed, lint/format clean, type check zero errors. Independent review and exact-head CI/integration remain parent-owned gates; not committed or published by executor.

### Executor retrospective
A catalog can be structurally compatible but incomplete after metadata-query failures. Surfacing this after deserialization preserves existing fallback behavior and avoids both misleading silence and an unnecessary new hard failure. Warning only on load naturally respects existing context caching. Raw producer errors remain in the artifact, avoiding additional log disclosure. Explicit count phrases produce stronger assertions than digit substrings in messages containing temporary paths.
