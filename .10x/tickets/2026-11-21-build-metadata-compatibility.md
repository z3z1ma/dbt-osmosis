Status: active
Created: 2026-11-21
Updated: 2026-11-21

# Coherent build metadata validation upgrade
Scope: explicit follow-through authorization for #478. Independently reproduce baseline Hatchling1.31/Twine6.1 and proposed1.33 metadata failure. Compare smallest supported Twine+packaging upgrade based on primary release notes and declared Python support (workflow Python3.10). Align both explicit Hatchling workflow invocations intentionally if upgrading. Preserve Dependabot credit, Python support, version, publishing guards, permissions and runtime scope.
Acceptance: exact installed tool/metadata versions recorded; baseline and candidate wheel/sdist creation, twine check --strict, fresh install/CLI/module smoke offline after dependencies available. No upload/auth request/release dispatch/publication. Sound minimal candidate starts draft; independent source/evidence review and exact-head+integrated CI required before merge. Original#478 remains open until replacement validated. #468 and old denied actions untouched. Observe current automatic Release44951167180 read-only; wait if genuine interference risk before merge.
References: prior hatchling-nov21/.10x/tickets/2026-11-21-hatchling-478.md and reviews/2026-11-21-hatchling-478-review.md (outside repository under task-11), AGENTS.md, pyproject.toml, .github/workflows/release.yml, https://twine.readthedocs.io/en/stable/changelog.html (7.0 metadata2.5 support).
Journal: main fetched bbe8ad49; existing automatic Release currently validate in_progress. New authorization expands previous leave-open disposition to coherent validator investigation, not publication.
Blockers: none for isolated investigation; merge gates above.

## Evidence
- Base bbe8ad496ac6d8c18005d3007f7549ac258a0d32. Original Dependabot #478 commit37ad568 cherry-picked as3f837b4, author retained. Candidate aligns both explicit release Hatchling commands1.31.0 to1.33.0 and validator6.1.0 to7.0.0; package version/Python support/runtime dependencies/permissions/publishing guards unchanged.
- Primary sources: https://hatch.pypa.io/latest/history/hatchling/ documents metadata2.5 default from1.32; https://twine.readthedocs.io/en/stable/changelog.html documents metadata2.5 fix in7.0. Installed distribution METADATA: Hatchling1.31/1.33 Python>=3.10, packaging>=24.2; Twine6.1 Python>=3.8, packaging>=24.0; Twine7.0 Python>=3.10, packaging>=26.1. No separate packaging pin needed because7.0 declares sufficient minimum.
- Fresh Python3.10.20 baseline: `uvx --python 3.10 --from=hatchling==1.31.0 hatchling build -d /tmp/build-metadata-nov21-baseline` creates wheel/sdist with metadata2.4. Twine6.1.0 strict check passes both. Actual packaging26.3.
- Same source plus original bot backend update, Hatchling1.33 direct builds produce metadata2.5 wheel/sdist. Twine6.1.0 strict checks separately fail each: InvalidDistribution: Invalid distribution metadata: '2.5' is not a valid metadata version (exit1), despite packaging26.3. Thus the previous failure is independently reproduced, not assumed.
- Twine7.0.0/packaging26.3 strict checks pass both candidate artifacts. `uvx --python 3.10 --with=packaging==26.1 --from=twine==7.0.0 twine check --strict ...` also passes both, demonstrating declared minimum. All package-check commands are local check only, never upload/auth.
- Workflow parity: invoked uvx0.10.12 (workflow-pinned uv) on Python3.10, then exact Hatchling1.33 `hatchling build` and Twine7.0 `twine check --strict dist/*` commands: both formats pass. Runtime uv0.11.25 only bootstrapped pinned0.10.12; output records actual nested uvx version. No release dispatched.
- PEP517 `uv build --python3.10` independently builds sdist and wheel from sdist. Fresh Python3.10 wheel install with release smoke dbt-adapters>=1.16.3,<2.0 gives dbt-osmosis1.6.0/dbt-core1.12.7; pip check reports65 packages compatible. Installed console script and module --help pass with network connection blocked, Path.home redirected in-process to temporary log directory for sandbox isolation (no credential/HOME changes). Baseline wheel also gets separate fresh offline install and both entry-point checks.
- Compared all repeated core metadata fields and description: unchanged apart from Metadata-Version2.4→2.5. No lower-metadata workaround. Logs and build outputs retained locally under /tmp/build-metadata-*; no private paths required to reproduce commands.
- Existing automatic Release44951167180 observed read-only: validate still in_progress onbbe8ad4. Parent must account for active run before merge; no workflow dispatch or cancellation here.

## Recommendation and limits
Coordinated upgrade resolves the proven consumer mismatch without forcing old metadata. Candidate remains draft until independent review and exact-head/integrated CI. Local Mac/Python3.10 smoke is package/CLI startup evidence, not complete runtime coverage or a publication test. Original#478 stays open until replacement disposition.

## Retrospective
Artifact creation and installation can succeed while pinned validation rejects new metadata. Test exact consumer, Python floor, and declared dependency minimum; upgrade the actual builder and validator together rather than preserve an unestablished old metadata contract.
