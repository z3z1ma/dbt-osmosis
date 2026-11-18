Status: active
Created: 2026-11-18
Updated: 2026-11-18
Parent: .10x/knowledge/maintenance-roadmap.md

# Publish the existing artifact support boundary

## Scope
Document existing validated catalog behavior and Python engine dependency in canonical user docs; qualify misleading generalizations in historical FUSION_COMPAT.md without erasing its historical findings. Update roadmap completed state for #469/#470. This is documentation of source-backed behavior, not new product semantics.
## Non-goals
No runtime/dependency changes, manifest-only mode, Fusion certification, new release, legacy support removal, or changes to blocked PR #468.
## Acceptance Criteria
- User-facing docs distinguish catalog ingestion from parsing/compilation/warehouse independence.
- State accepted catalog v1/schema-less metadata envelopes, consumed-field validation, missing-path fallback and invalid-input early failure accurately from source/tests.
- Historical Fusion observations do not imply all current Fusion artifacts or installations are certified.
- Roadmap reflects merged #469 database-shape validation and #470 release test correction while retaining evidence limits and #468 blocker.
- Docs build and independent review pass; draft PR then exact-head CI before merge.
## References
.10x/specs/catalog-input-contract.md; src/dbt_osmosis/core/catalog_operations.py; src/dbt_osmosis/core/settings.py; tests/test_catalog_contract.py; FUSION_COMPAT.md; .10x/knowledge/maintenance-roadmap.md.
## Assumptions
Source-backed existing behavior only. Alex's standing main/roadmap ownership authorizes validated documentation publication. No additional runtime or warehouse claims.
## Journal
- Main 0cdf4a7 observed. Prior roadmap and dependency workers completed. Only open PR #468 is explicitly out of scope; zero open issues. Latest main Tests/lint/Release success. Catalog contract already implemented; public support explanation still relies on broad historical Fusion claims.
## Blockers
None for bounded documentation. Current actual Fusion producer fixtures unavailable; explicitly do not claim certification.
## Evidence
- Added canonical reference page and sidebar/index/settings links. Cross-checked accepted shapes against `_validate_catalog_input`, missing path behavior against `_load_catalog`, missing relation fallback against `get_columns`, and preflight timing against `YamlRefactorContext.__post_init__`. Read regression assertions in `tests/test_catalog_contract.py`; did not rerun Python tests for documentation-only edits.
- Qualified historical Fusion evidence at the top of FUSION_COMPAT.md, preserving the original experiment observations. Updated roadmap completed state from parent-provided verified #469/#470 evidence; retained current actual Fusion fixture gap and #468 action block.
- `npm --prefix docs ci --ignore-scripts --no-audit --no-fund`: success, 1,286 packages from unchanged lockfile. Node v26.7.0, npm 11.19.0. No dependency manifest/lockfile edits.
- `npm --prefix docs run build`: exit 0, client/server compilation succeeded and generated static files. Log `/private/tmp/dbt-artifact-docs-build.log`. Existing Node experimental localStorage warning; no broken-link/build error.
- `git diff --check`: passed. Generated `docs/build/docs/reference/artifacts/index.html` exists and contains the new page title. Initial path probe omitted Docusaurus's `/docs/` route prefix, then corrected; not a build failure.
- Independent review, remote CI, publication and merge remain pending. No runtime/source/test files changed.
## Review
Pending independent review.
## Retrospective
Source inspection exposed two existing settings descriptions that overstated offline behavior: a catalog miss may still query the warehouse, and disabling introspection does not prevent arbitrary macros/compilation queries. The new page and corrected settings distinguish these boundaries rather than expanding the runtime contract.
