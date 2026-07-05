Status: done
Created: 2026-07-05
Updated: 2026-07-05
Parent: None
Depends-On: None

# Remediate workbench requirements OSV vulnerabilities

## Scope

Update the tracked Streamlit deployment requirements file so OSV-Scanner no longer resolves vulnerable lower-bound transitive packages from `src/dbt_osmosis/workbench/requirements.txt`.

In scope:

- Keep `dbt-osmosis[workbench,duckdb]==1.4.0` as the deployment package entry.
- Add only the explicit fixed-version lower-bound constraints needed for packages OSV currently flags from this requirements file.
- Keep the existing `setuptools>=70,<81` compatibility bound unless verification proves it must change.
- Verify the requirements file resolves on Python 3.10 and no longer contributes OSV findings.
- Update package metadata tests only if they need to assert the deployment security constraints.

Out of scope:

- Changing `pyproject.toml` optional dependency contracts.
- Updating `uv.lock` or root dependency resolution.
- Changing workbench application behavior or Streamlit UI code.
- Fully clearing docs npm advisories, already owned by `.10x/tickets/2026-07-05-ratify-docs-node20-docusaurus-upgrade.md`.

## Acceptance Criteria

- ACC-001: OSV-Scanner no longer reports vulnerabilities sourced from `src/dbt_osmosis/workbench/requirements.txt`.
- ACC-002: The requirements file still references `dbt-osmosis[workbench,duckdb]==1.4.0`.
- ACC-003: `uv pip compile src/dbt_osmosis/workbench/requirements.txt --python-version 3.10` exits 0.
- ACC-004: Relevant package metadata tests pass.
- ACC-005: `uv audit --frozen` remains clean, or any result is proven unrelated to this requirements-only change.
- ACC-006: The diff is limited to the requirements deployment surface, tests for that surface, and 10x records.

## Evidence Expectations

- Capture before/after OSV summaries for this requirements file.
- Capture fixed lower-bound rationale from OSV fixed versions and resolver output.
- Capture resolver and package metadata test output.

## Progress and Notes

- 2026-07-05: OSV-Scanner reported vulnerable transitive package versions sourced from `src/dbt_osmosis/workbench/requirements.txt`: `gitpython 3.1.9`, `h11 0.9.0`, `idna 3.9.0`, `ipython 8.9.0`, `pillow 9.5.0`, and `pyarrow 9.0.0`.
- 2026-07-05: `uv pip compile src/dbt_osmosis/workbench/requirements.txt --python-version 3.10` resolves safe current versions without changing package metadata: `gitpython 3.1.50`, `h11 0.16.0`, `idna 3.18`, `ipython 8.39.0`, `pillow 12.3.0`, and `pyarrow 24.0.0`.
- 2026-07-05: Started execution on branch `codex/quality-optimizer-hill-climb`.
- 2026-07-05: Added fixed lower-bound constraints for the six vulnerable workbench deployment transitive packages and a package metadata regression test.
- 2026-07-05: After the change, OSV-Scanner still reports 5 docs package-lock records, but 0 records are sourced from `src/dbt_osmosis/workbench/requirements.txt`.
- 2026-07-05: `uv pip compile ... --python-version 3.10`, `uv run pytest tests/test_package_metadata.py`, `uv audit --frozen`, and `uvx ruff==0.15.17 check tests/test_package_metadata.py` exited 0. Evidence recorded in `.10x/evidence/2026-07-05-workbench-requirements-osv-remediation.md`.
- 2026-07-05: Closure review recorded in `.10x/reviews/2026-07-05-workbench-requirements-osv-remediation-review.md`.

## Blockers

None.

## References

- `/tmp/dbt-osmosis-ai-quality/continuation/osv-source.json`
- `.10x/evidence/2026-07-05-workbench-requirements-osv-remediation.md`
- `.10x/reviews/2026-07-05-workbench-requirements-osv-remediation-review.md`
- `src/dbt_osmosis/workbench/requirements.txt`
- `tests/test_package_metadata.py`
