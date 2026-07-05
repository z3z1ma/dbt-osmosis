Status: done
Created: 2026-07-04
Updated: 2026-07-04
Parent: .10x/tickets/2026-07-04-quality-optimizer-hill-climb.md
Depends-On: None

# Remediate uv Audit Vulnerabilities

## Scope

Update the lockfile with the smallest dependency changes needed to clear current `uv audit --frozen` vulnerabilities.

In scope:

- Update `uv.lock` so vulnerable resolved packages are at fixed versions:
  - `gitpython >=3.1.50`
  - `msgpack >=1.2.1`
  - `pygments >=2.20.0`
- Prefer `uv lock --upgrade-package ...` with only the affected packages.
- Change `pyproject.toml` only if the resolver proves existing constraints prevent fixed versions.
- Verify the existing configured gates after the lock update.

Out of scope:

- Broad dependency modernization.
- Adding new dependencies or tools.
- Changing CI, Semgrep, Dependabot, or uv policy configuration.
- Workbench feature changes.
- Complexity refactoring.

## Acceptance Criteria

- ACC-001: `uv audit --frozen` exits 0 after the lockfile change.
- ACC-002: `uv lock --check` exits 0.
- ACC-003: `ruff check .` and `ruff format --check .` exit 0.
- ACC-004: `basedpyright` exits 0 errors in a dependency-complete temp or project environment.
- ACC-005: A focused test run covering dependency metadata and affected optional surfaces passes; full pytest is run unless a documented time/tooling constraint prevents it.
- ACC-006: The diff contains only the minimum required lockfile and directly necessary metadata changes.

## Evidence Expectations

- Capture before/after `uv audit --frozen` output.
- Capture resolver command used and package versions after resolution.
- Capture all verification command outcomes.

## Progress and Notes

- 2026-07-04: Baseline identified 7 vulnerabilities in `gitpython`, `msgpack`, and `pygments`. Dependency paths were traced with `uv tree --invert`.
- 2026-07-04: Started execution on branch `codex/quality-optimizer-hill-climb`.
- 2026-07-04: Ran `uv lock --upgrade-package gitpython --upgrade-package msgpack --upgrade-package pygments`. Resolver updated only the targeted vulnerable packages: `gitpython 3.1.46 -> 3.1.50`, `msgpack 1.1.2 -> 1.2.1`, and `pygments 2.19.2 -> 2.20.0`.
- 2026-07-04: `uv audit --frozen` now reports no known vulnerabilities. `uv lock --check` passes.
- 2026-07-04: Narrowed unrelated lockfile marker normalization back out of the diff, leaving only the required targeted package updates.
- 2026-07-04: Recorded verification evidence in `.10x/evidence/2026-07-04-uv-audit-remediation.md` and review in `.10x/reviews/2026-07-04-uv-audit-remediation-review.md`.

## Blockers

None.

## References

- `.10x/research/2026-07-04-quality-optimizer-baseline.md`
- `.10x/evidence/2026-07-04-quality-optimizer-baseline.md`
- `.10x/evidence/2026-07-04-uv-audit-remediation.md`
- `.10x/reviews/2026-07-04-uv-audit-remediation-review.md`
