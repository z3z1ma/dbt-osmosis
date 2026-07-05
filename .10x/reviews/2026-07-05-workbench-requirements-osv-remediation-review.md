Status: recorded
Created: 2026-07-05
Updated: 2026-07-05
Target: .10x/tickets/done/2026-07-05-remediate-workbench-requirements-osv-vulnerabilities.md
Verdict: pass

# Workbench requirements OSV remediation review

## Target

Review of the requirements-file hardening that adds explicit lower-bound constraints for vulnerable workbench deployment transitive dependencies.

## Findings

- No significant implementation defect found. The diff is limited to `src/dbt_osmosis/workbench/requirements.txt`, the package metadata test that guards it, and 10x records.
- The constraints use fixed lower bounds from OSV and preserve the existing deployment package entry `dbt-osmosis[workbench,duckdb]==1.4.0`.
- Residual OSV findings remain only in `docs/package-lock.json`; those are outside this ticket and are owned by `.10x/tickets/2026-07-05-ratify-docs-node20-docusaurus-upgrade.md`.

## Verdict

Pass. Acceptance criteria are supported by OSV before/after evidence, resolver output, package metadata tests, root uv audit, and Ruff on the touched Python test.

## Residual Risk

The requirements file is still not a lockfile, so future transitive advisories can reappear. The added constraints remove the known vulnerable lower-bound choices without changing the root Python dependency contract.
