Status: active
Created: 2026-11-17
Updated: 2026-11-17
Parent: .10x/knowledge/maintenance-roadmap.md

# Validate supplied catalog artifacts

## Scope
Implement `.10x/specs/catalog-input-contract.md` at the existing catalog input boundary. Audit legacy behavior and validate early enough to avoid YAML mutation before invalid artifact detection. Use minimal concrete helpers and existing exceptions/CLI conventions.
## Non-goals
No new artifact-only execution mode, engine support claim, dependencies, release or broad exception refactor.
## Acceptance Criteria
1. Valid actual generated Core catalog still supplies columns/types/comments; compatible inputs with a schema-less metadata object still work.
2. Wrong artifact, malformed JSON/envelope/consumed shapes and unsupported declared schema fail with path and actionable cause; no silent warehouse fallback.
3. Actual CLI failure precedes YAML writes; original catalog and YAML remain byte-identical.
4. Missing path/default behavior retained; additive producer metadata and fixture limits documented.
5. Targeted tests, lint/type checks and independent review pass; exact-head CI required before merge.
## References
`.10x/specs/catalog-input-contract.md`; root/core AGENTS.md; prior catalog module split ticket `.10x/tickets/done/2026-07-05-break-core-codeql-cyclic-imports.md`.
## Assumptions
Record-backed existing `_load_catalog` calls json.loads then CatalogArtifact.from_dict without guard. No configured/missing path returns None. Explicit user roadmap implementation authority permits bounded validation; preserve supported input semantics.
## Journal
- Inspected code and official catalog v1 schema; created isolated branch feat/catalog-contract at main f1d8947f. Existing artifact reader has no schema/kind gate; tests mostly mock it. No existing active owner found.
## Blockers
None for execution; no actual Fusion-produced fixture available, so no Fusion certification claim.
## Evidence
See executor observations below.
## Review
Pending independent review.
## Retrospective
See executor observations below.

### Executor observations and evidence
- Existing Core 1.11.17 factory probe: absent `metadata` fails with mashumaro `MissingField`; `metadata: {}` succeeds; arbitrary `dbt_schema_version: garbage` also succeeds. Thus compatibility means preserving the schema-less metadata envelope, not newly accepting absent metadata. Specification wording corrected with parent agreement. Factory `InvalidFieldValue` and `MissingField` errors are now wrapped with the supplied path; no new dependencies.
- Added focused envelope/consumed-column validation in the existing catalog helper. Explicit input is loaded in `YamlRefactorContext.__post_init__` before thread-pool allocation and before any CLI source bootstrap/restructure. This calls only `_load_catalog`, never generation or live introspection. `yaml_opts` converts only `CatalogValidationError` to a Click diagnostic; other exception behavior is unchanged.
- `PYTHONPATH=src /tmp/osmosis-supported/bin/python -m pytest tests/test_catalog_contract.py -q`: initial **25 passed in 11.09s**. Expanded valid round-trip coverage then `PYTHONPATH=src /tmp/osmosis-supported/bin/python -m pytest tests/test_catalog_contract.py tests/core/test_settings.py -q`: **83 passed in 17.16s**. Log `/tmp/catalog-tests2.log`.
- New tests include real Core-produced catalog from the existing session fixture (not a hand-labelled Fusion fixture), schema-less metadata, synthetic future producer/additive metadata, malformed envelope/schema/consumed columns, no generation or pool allocation on failure, and fresh subprocess non-dry-run CLI `refactor`, `organize`, and `document`. Negative CLI cases preserve YAML bytes and mtimes before potential restructuring; valid CLI cases preserve unrelated nested semantic metadata and explicit descriptions. Input catalog bytes remain unchanged.
- Local environment: Python **3.13.14**, dbt-core **1.11.17**, dbt-duckdb **1.10.1**, mashumaro **3.14**. Local assertions do not certify every supported Core minor; exact-head matrix remains required. Tests use temporary local DuckDB projects only. Sandbox escalation was needed for the existing logger's home-directory output, not external warehouse access.
- Ruff check, Ruff format check (five changed Python files) and `git diff --check` passed. Basedpyright with `/tmp/osmosis-supported/bin/python` on the three changed core modules: **0 errors, 66 warnings**, existing permissive Any annotations. CLI typecheck recorded separately below.
- No actual Fusion producer artifact was available. The synthetic future producer test establishes only compatible catalog-v1 acceptance; no Fusion runtime or producer certification is claimed. No manifest-only mode was introduced; missing compiled SQL is outside this catalog boundary and unchanged.

### Executor retrospective
A deserializer accepting unknown schema URLs is not an artifact compatibility check. The existing CLI reordered YAML before catalog reads, so guarding only `_load_catalog` would not satisfy preservation: constructor preflight is essential. Keep valid legacy schema-less metadata, distinguish absent metadata (already invalid), and use real generated artifacts alongside explicitly synthetic future-producer fixtures. Published JSON Schema says columns are objects even though the prose reference calls them arrays; the schema and generated data are authoritative.

- CLI basedpyright invocation with the same explicit interpreter: **0 errors, 394 warnings**, no new error-level diagnostics (`/tmp/catalog-cli-types.log`).

### Independent adversarial review (2026-11-17)
- Target: immutable implementation `75050a1b84e9c5e80f7bbe104d14b6d414471203` against `f1d8947f`.
- Verdict: **concerns (minor)** pending the owner-selected migration CLI presentation fix and follow-up review, then exact-head supported Core/Python CI. No critical or significant findings. Reviewed source, call order, fixture provenance and assertions; did not repeat executor verification.
- Constructor preflight occurs before pool allocation and before `refactor`/`organize` source bootstrap, restructuring, or `document` transformation. It calls the loader only, never catalog generation. Invalid explicit input cannot silently trigger generation or warehouse fallback. Existing project parsing still precedes context construction; no claim of avoiding all dbt parse activity is made.
- Validator rejects explicit wrong kind/version, non-mapping envelope and malformed consumed column types; dbt typed deserialization retains responsibility for remaining required shapes. Compatible schema-less metadata objects remain accepted; entirely absent metadata already failed previously. Unknown producer metadata is accepted without runtime compatibility certification. Input files are read only, and unrelated YAML semantic metadata preservation is asserted in valid real CLI cases.
- New exception conversion is narrow for YAML CLI commands. **Minor presentation gap:** `migration_plan_command` independently exposes `--catalog-path` through `_create_cli_yaml_context` without `yaml_opts`, so an invalid catalog there retains a traceback rather than the YAML commands' Click diagnostic. The exception still includes the catalog path/cause and occurs before migration output writes; this does not weaken the YAML preservation gate. Owner can add a narrow helper catch/test or explicitly retain this pre-existing non-YAML presentation behavior in this slice.
- Residual risks: local producer fixture is Core 1.11.17/DuckDB, not Fusion; matrix must establish supported 1.8–1.12 compatibility, including mashumaro exception imports. This is focused consumed-field validation, not complete JSON Schema validation or a catalog/project freshness check. The contract intentionally preserves nonexistent-path fallback.

### Review follow-up: consistent CLI diagnostics
- Resolved migration-plan presentation gap with one narrow `_create_yaml_refactor_context` CLI factory. Existing CLI constructors now use that factory; the duplicate `yaml_opts` exception guard was removed. Core APIs continue raising `CatalogValidationError`; CLI consumers consistently convert only that exception to `ClickException`.
- Added fresh subprocess `migration plan --catalog-path <manifest> --output <existing-file>` regression: actionable path-specific diagnostic without traceback, existing output bytes/mtime unchanged, YAML bytes/mtime unchanged, and catalog bytes unchanged.
- `PYTHONPATH=src /tmp/osmosis-supported/bin/python -m pytest tests/test_catalog_contract.py tests/core/test_cli.py -q`: **76 passed in 28.68s**, `/tmp/catalog-migration-tests-final.log`. Initial editing-script syntax error made no source changes and is not counted as new-fix verification; the recorded final run followed the successful edit.
- Ruff check/format and diff check passed for the affected CLI and test files. Basedpyright on the CLI with the explicit same interpreter: **0 errors, 394 warnings**, `/tmp/catalog-cli-types-final.log`. No runtime loader, dependency, or catalog compatibility behavior changed in this follow-up. Awaiting independent follow-up review.

### Independent follow-up review (2026-11-17)
- Reviewed `75050a1..b281a5cff52f43c1b1e0c5898a0d4afb55ae610e`, including the new CLI factory and actual migration regression assertions. **Verdict: pass**, subject to exact-head matrix. The minor presentation finding is resolved.
- All CLI constructors now pass through the narrow factory; only `CatalogValidationError` is translated, with core API exception behavior unchanged. Migration planning validates at context creation before comparing schema or writing the output. The new test asserts no traceback and unchanged existing output bytes/mtime, YAML bytes/mtime and catalog bytes.
- No executor tests repeated. Prior residual compatibility/producer limits remain; no new blocking findings.
