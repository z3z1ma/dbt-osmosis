Status: open
Created: 2026-11-18
Updated: 2026-11-18
Parent: .10x/tickets/2026-11-18-release-160.md

# Scope locked-dependency remediation and workbench support

## Scope
Own the existing security debt identified during 1.6.0 release assessment. This is a shaping/follow-up owner, not authority to change support semantics. Separate dependency fixes that preserve supported Python versions from the Python 3.10 optional-workbench decision before execution.
## Non-goals
No claim all GitHub alerts are exploitable or wheel runtime dependencies. No silent removal of Python 3.10 support, blanket alert dismissal, credential changes, or arbitrary dependency upgrades.
## Acceptance Criteria
- Re-read current alerts and dependency paths; distinguish wheel resolution, shipped uv.lock reproducibility, optional workbench/Azure paths, and docs tooling.
- Produce a bounded remediation ticket for compatible locked dependency updates, with dependency-tree/lock diff, representative supported-Python tests and exact-head CI gates.
- Establish an explicit workbench support decision for Python 3.10 before attempting Pillow 13 or changing Python requirements. Consider the maintained package base separately from optional workbench requirements.
- Document remaining exploitability limits and open advisories; only mark remediated versions fixed after verification.
## References
- https://github.com/advisories/GHSA-5x8w-8jj5-q52h — Pillow FPX JPEG heap overflow; alert #356 reported first patched 13.0.0.
- https://github.com/advisories/GHSA-6q48-6c7w-9gw2 — certifi compromised emSign root trust; alert #368 affected >=2019.03.09,<2026.10.18, advisory recommends 2026.10.19+.
- https://pypi.org/pypi/pillow/json — observed latest 13.0.0 requires Python >=3.11.
- https://pypi.org/pypi/streamlit/json — observed latest 1.67.0 requires Python >=3.11.
- Candidate uv.lock: Pillow 12.3.0 and certifi 2026.1.4. Project Python >=3.10,<3.14; workbench extra pulls Pillow through its dependency graph.
- .10x/tickets/2026-11-16-pillow-security-refresh.md documents the earlier narrow update, not current all-advisory remediation.
## Assumptions
Observed package metadata and advisory data are temporal and must be rechecked. Base wheel does not pin these transitive versions; sdist includes uv.lock. Optional-workbench compatibility and exploitability are not established by base CLI checks.
## Journal
- 2026-11-18: Independent release security scope review inspected authenticated alert examples and package metadata. GitHub banner reported 128 open alerts (9 critical/72 high/43 moderate/4 low), not 128 independently confirmed exploitable runtime vulnerabilities. No dependencies or alerts changed by that review.
## Blockers
Python 3.10 workbench policy requires an explicit decision; no known compatible Pillow 13 install on Python 3.10. Full transitive call-path exploitability audit not performed.
## Evidence
Review establishes existing vulnerable locked versions and incompatible patched Pillow Python floor, not comprehensive exploitability.
## Review
Pending scoped remediation design and execution review.
## Retrospective
A version-only release cannot be called security-clean because newer optional dependency fixes exist. Wheel requirements and repository lockfile exposure need separate statements.
