Status: active
Created: 2026-11-16
Updated: 2026-11-16

# Refresh Pillow security lock (#436)
## Scope
Regenerate only Pillow 12.2.0 -> 12.3.0 on 7279fec, preserving runtime metadata and Python support. No release/version changes. Original contributor Dependabot retained.
## Acceptance and evidence
- `uv lock --upgrade-package pillow==12.3.0`: 178 packages resolved. Semantic package comparison proves only Pillow version changes; refreshed diff avoids old PR's unrelated marker normalization.
- `uv lock --check`: pass. `git diff --check`: pass.
- Python 3.13.14 with Pillow 12.3.0: in-memory RGB PNG save/read pixel roundtrip passes. Limited smoke, no full workbench coverage claimed.
- Upstream https://pillow.readthedocs.io/en/stable/releasenotes/12.3.0.html lists fixes for PDF decompression bomb, EPS infinite loop, image OOB access and Windows viewer injection. Lock update is useful security maintenance; direct requirement support remains unchanged.
## Blockers and review
Independent review and exact-head CI required before merge. Executor does not merge or close #436.
## Retrospective
Refresh targeted lock on current main rather than transplanting stale generated lock changes. Lock-only updates do not force already installed end-user environments to update.
