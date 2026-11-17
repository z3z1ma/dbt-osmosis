Status: active
Created: 2026-11-17
Updated: 2026-11-17
Parent: .10x/knowledge/maintenance-roadmap.md

# Complete catalog relation identity validation

## Scope
Close the missing consumed `metadata.database` shape check in existing catalog preflight, governed by `.10x/specs/catalog-input-contract.md`. Both node and source records must reject non-string, non-null database values before any YAML mutation. Missing, null and string values retain current compatibility.
## Non-goals
No engine certification, new CLI, stricter unconsumed-field validation, changed missing-path fallback, freshness policy, dependencies, release, or #468 changes.
## Acceptance Criteria
1. Reproduce current failure: the typed catalog deserializer accepts database=7 but CatalogTable.key() raises AttributeError; demonstrate the boundary currently accepts this invalid input.
2. Invalid number, boolean, array and object database values in nodes and sources fail with CatalogValidationError identifying the catalog path and metadata.database.
3. Missing, null and string database values remain accepted; existing generated Core artifact test stays valid.
4. Actual non-dry-run CLI rejects invalid database input before YAML/output changes, with actionable diagnostic and no traceback. Preserve input bytes.
5. Targeted contract/core tests, lint and independent adversarial review pass. Exact-head CI required before merge.
## References
`.10x/specs/catalog-input-contract.md`; `.10x/tickets/2026-11-17-catalog-input-contract.md`; root/core AGENTS.md; https://schemas.getdbt.com/dbt/catalog/v1.json (2026-11-17); https://docs.getdbt.com/reference/artifacts/catalog-json (2026-11-17).
## Assumptions
Record-backed: existing spec requires validation of consumed structural fields and failure before YAML mutation. `introspection._catalog_entry_matches` calls entry.key(); installed dbt CatalogTable.key uses lowercase(metadata.database). Published catalog-v1 schema defines optional database as string or null. This is completing existing contract, not designing new semantics. Explicit owner roadmap delegation authorizes implementation.
## Journal
- 2026-11-17: Inspected main a7632b5b in isolated worktree. Original checkout clean and unchanged at 3cc00a57. #468 remains untouched; no open issues from connector inventory.
- Read-only Python 3.13 probe with /tmp/osmosis-supported/bin/python: CatalogTable.from_dict accepted metadata.database=7, then key() raised AttributeError: int object has no attribute lower. Current validator checks type/schema/name but not consumed database.
- Current upstream docs still specify external catalog-v1 while v2 documentation reads information schema; manifest version does not certify Python API interchangeability. No speculative Fusion changes warranted.
## Blockers
None for bounded implementation; remote publication/merge subject to available authorized access and exact-head CI.
## Evidence
Pending executor.
## Review
Pending independent reviewer.
## Retrospective
Pending executor.

### Executor journal
- Added node/source regression cases for integer, float, boolean, array and object database values. Before implementation, `PYTHONPATH=src /tmp/osmosis-supported/bin/python -m pytest tests/test_catalog_contract.py -k malformed_database -q` failed all 10 cases because no exception was raised; log `/tmp/catalog-database-before.log`. Initial editing-script marker mismatch made no changes, then was corrected.
- Added nullable-string check adjacent to existing relation identity validation. Extended actual CLI checks to invalid database inputs for refactor, organize, document and migration plan; retained original manifest rejection and valid-input preservation assertions.
- `PYTHONPATH=src /tmp/osmosis-supported/bin/python -m pytest tests/test_catalog_contract.py tests/core/test_settings.py tests/core/test_cli.py -q`: **151 passed in 34.14s**, `/tmp/catalog-database-after.log`. This includes generated Core artifact, compatible optional database values on nodes/sources, and fresh subprocess non-dry-run refactor/organize/document/migration failure checks preserving YAML bytes/mtimes and existing migration output bytes/mtime, without traceback. Catalog input bytes remain unchanged.
- `ruff check --no-cache` and `ruff format --check --no-cache` on both changed Python files passed. `git diff --check` passed. First cached Ruff invocation was blocked from creating a cache temporary file by the read-only sandbox; retry with no-cache passed without elevation.
- `basedpyright --pythonpath /tmp/osmosis-supported/bin/python src/dbt_osmosis/core/catalog_operations.py`: **0 errors, 46 warnings**, existing Any annotations. No dependency edits.
- Local environment: Python 3.13.14, dbt-core 1.11.17, dbt-duckdb 1.10.1. Tests use temporary local DuckDB projects; no external warehouse execution. This is focused local verification, not all supported Core versions or Fusion certification. Parent reports separate existing main release-CI failure under Core1.12.6 is owned by another worker; this executor did not change that test or workflow.

### Acceptance evidence and remaining gates
1. Baseline 10 failures establish existing boundary accepted malformed database values; parent observed later key() AttributeError.
2. All 10 invalid node/source cases pass after the three-line validator guard and require path plus field diagnostic.
3. Six optional database cases and existing actual Core-generated artifact test pass.
4. Four new invalid-database subprocess cases pass alongside existing valid/manifest scenarios; no assertions weakened.
5. Focused tests, Ruff and type check pass. Independent review and exact-head CI remain parent-owned gates; not yet merged or published.

### Executor retrospective
Typed deserialization alone does not validate every consumed optional field: database values reach relation matching untouched, unlike an assumption of safe string coercion. The smallest fix extends the existing preflight rather than introducing schema libraries or catching late AttributeError after writes. Reusing the real CLI preservation matrix demonstrated the timing contract without duplicating project setup.

### Independent adversarial review (2026-11-17)
- Verdict: **pass**, subject to exact-head supported-version CI before merge. No critical, significant, minor or nit findings.
- Reviewed final source/test diff against `a7632b5b`, existing loader exception wrapping, constructor preflight order, actual CLI assertions and executor evidence; no verification commands were repeated.
- The three-line check covers both nodes and sources and rejects every non-null non-string database value before typed deserialization and relation matching. Missing and explicit null remain accepted; string values are not normalized or rewritten. Existing envelope checks, unknown producer handling, missing-path fallback and public exception conversion are unchanged.
- New invalid-shape cases require the field and supplied path in diagnostics. Positive cases cover missing/null/string values in both sections and call the relation key method. The CLI matrix adds invalid-database cases for refactor, organize, document and migration plan without removing prior valid-input or manifest cases; it asserts no traceback, unchanged YAML bytes/mtimes and catalog bytes, and unchanged existing migration output.
- Residual risk: local evidence is Python 3.13/Core 1.11.17/DuckDB 1.10.1 only. Exact-head CI must establish the supported matrix. This narrow guard does not certify Fusion, full JSON Schema compliance, catalog freshness or any external warehouse. The independently owned release-validation failure remains outside this patch.
