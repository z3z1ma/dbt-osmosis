---
id: evidence:c10pkg10-main-ci-release-success
kind: evidence
status: recorded
created_at: 2026-05-04T04:58:20Z
updated_at: 2026-05-04T04:58:20Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  tickets:
    - ticket:c10pkg10
  evidence:
    - evidence:c10pkg10-package-metadata-smoke
  critique:
    - critique:c10pkg10-release-packaging-review
  wiki:
    - wiki:ci-compatibility-matrix
    - wiki:release-publishing-workflow
external_refs:
  github_actions:
    - https://github.com/z3z1ma/dbt-osmosis/actions/runs/25300857319
    - https://github.com/z3z1ma/dbt-osmosis/actions/runs/25300857392
    - https://github.com/z3z1ma/dbt-osmosis/actions/runs/25301413030
---

# Summary

Observed successful post-push CI and release validation for `ticket:c10pkg10` commit `f4e475314bbb412fea927577f99f8e78a7258f80`. This evidence supplements local package metadata smoke evidence with main-branch GitHub Actions results.

# Procedure

Observed at: 2026-05-04T04:58:20Z
Source state: commit `f4e475314bbb412fea927577f99f8e78a7258f80` on `main`.
Procedure: Queried GitHub Actions with `gh run watch`, `gh run view`, and `gh run list` for the pushed commit.
Expected result when applicable: `Tests`, `lint`, and Release validation should complete successfully for the committed package metadata changes.
Actual observed result: `Tests`, `lint`, and Release completed successfully for commit `f4e475314bbb412fea927577f99f8e78a7258f80`.
Procedure verdict / exit code: pass. `gh run watch 25300857319 --exit-status` and `gh run watch 25301413030 --exit-status` returned success; `gh run list --workflow lint` showed success.

# Artifacts

GitHub Actions observations:

- `Tests` run `25300857319` completed with conclusion `success` for head SHA `f4e475314bbb412fea927577f99f8e78a7258f80`.
- `Tests` job `Pip install smoke` (`74167846447`) completed with conclusion `success`; this is the CI copy of the independent base/OpenAI/Azure/workbench/DuckDB/proxy pip-smoke job.
- `Tests` job `Check uv lockfile` (`74167423381`) completed with conclusion `success`.
- `Tests` docs jobs completed with conclusion `success`: Node 18 job `74167846462` and Node 24 job `74167846477`.
- `Tests` latest compatibility jobs completed with conclusion `success`: dbt 1.10 job `74167846480` and dbt 1.11 job `74167846474`.
- `Tests` matrix pytest/integration jobs completed with conclusion `success` across Python/dbt rows, including Python 3.10/dbt 1.11 job `74167846549`, Python 3.11/dbt 1.11 job `74167846554`, Python 3.12/dbt 1.11 job `74167846571`, and Python 3.13/dbt 1.11 job `74167846574`.
- `lint` run `25300857392` completed with conclusion `success` for the same head SHA.
- Release run `25301413030` completed with conclusion `success` for the same head SHA.
- Release validation job `74168897100` completed with conclusion `success` and ran lock freshness, Ruff, basedpyright, demo parse, pytest, docs build, package build, package metadata validation, built-wheel install smoke, release-note source declaration, and artifact upload.
- Release publish job `74169981927` completed with conclusion `success`; PyPI publishing and release-note publishing were skipped on the no-new-version path.

Observed annotation:

- Release emitted a GitHub Actions Node.js 20 deprecation warning for several actions. This did not fail the run and does not challenge `ticket:c10pkg10` package metadata claims.

# Supports Claims

- ticket:c10pkg10#ACC-001: Main `Tests` and Release validated the committed base/package metadata state.
- ticket:c10pkg10#ACC-002: Main CI and Release package metadata/wheel smoke passed for the committed direct dependency and optional-extra routing.
- ticket:c10pkg10#ACC-003: Main CI and Release built and smoke-installed the committed workbench requirements/package metadata path.
- ticket:c10pkg10#ACC-004: Main `Tests`, `lint`, and Release validated the committed dev tooling and lockfile state.
- ticket:c10pkg10#ACC-005: Main CI validated the committed Taskfile/source state after `.python-version` source removal.
- ticket:c10pkg10#ACC-006: Main `Tests` pip-install-smoke job validated independent installs for base and supported optional extras.

# Challenges Claims

None - the observed post-push CI and release validation succeeded for the cited claims.

# Environment

Commit: `f4e475314bbb412fea927577f99f8e78a7258f80`
Branch: `main`
Runtime: GitHub-hosted Actions runners using the workflow-defined Python and Node versions.
OS: GitHub Actions runner environments.
Relevant config: `.github/workflows/tests.yml`, `.github/workflows/release.yml`, `.github/workflows/constraints.txt`, `pyproject.toml`, `uv.lock`, `Taskfile.yml`, and docs tooling.
External service / harness / data source when applicable: GitHub Actions via `gh` CLI.

# Validity

Valid for: post-push CI and release validation of commit `f4e475314bbb412fea927577f99f8e78a7258f80`.
Fresh enough for: `ticket:c10pkg10` acceptance and closure decision after retrospective disposition.
Recheck when: package metadata, optional extras, CI smoke logic, release validation, lockfile, or dependency constraints change.
Invalidated by: failed rerun for the same commit, source changes after this commit, or dependency/index drift that changes install resolution.
Supersedes / superseded by: Supplements evidence:c10pkg10-package-metadata-smoke with committed main-branch CI/release observations.

# Limitations

- This evidence does not launch the Streamlit workbench interactively.
- This evidence does not validate SQL proxy runtime semantics beyond dependency routing and package install behavior.
- This evidence does not address the GitHub Actions Node.js 20 deprecation warning except to note that it did not fail this run.
- This evidence does not prove future package-index resolution will remain identical.

# Result

Main CI and release validation passed for the committed `ticket:c10pkg10` package metadata changes. The `Tests` run validated matrix compatibility, docs builds, lock freshness, and independent pip smokes. The Release run validated package build, metadata, and built-wheel install smoke.

# Interpretation

The observations support closing `ticket:c10pkg10` once the ticket acceptance gate records critique disposition, retrospective / promotion disposition, and residual risks. They do not make the experimental proxy supported or prove interactive workbench runtime behavior.

# Related Records

- ticket:c10pkg10
- evidence:c10pkg10-package-metadata-smoke
- critique:c10pkg10-release-packaging-review
- wiki:ci-compatibility-matrix
- wiki:release-publishing-workflow
