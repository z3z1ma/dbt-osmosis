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

## Independent review (dependency_review, 2026-11-16)
Target: d75649975b77e056fe7e4987fc9e186d09c94d2a against 7279fec9bebeee6bf2fb6b1fa6d4d0838883ee9f.
Verdict: pass for focused lock refresh, conditional on exact-head CI before merge.
Findings: no blocking defect. Diff is confined to Pillow package artifacts/version plus ticket; runtime/support metadata and dependency markers unchanged. Python 3.10–3.13 wheel entries remain. Dependabot coauthor retained. Recorded lock/image smoke evidence is appropriately limited and was not rerun.
Residual risk: 12.3.0 lock no longer lists manylinux2014/glibc 2.17 wheels (2.27/2.28 wheels remain), so older Linux installations may build from sdist; source artifact is present. This does not change declared Python support, and the project has no observed guarantee of binary-wheel availability on those platforms. Full workbench/image-format validation is not established by the PNG smoke. Lock-only update does not raise requirements.txt minimum or force existing pip installations to update.
