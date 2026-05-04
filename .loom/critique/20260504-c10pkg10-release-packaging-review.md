---
id: critique:c10pkg10-release-packaging-review
kind: critique
status: final
created_at: 2026-05-04T04:18:06Z
updated_at: 2026-05-04T04:18:06Z
scope:
  kind: repository
  repositories:
    - repo:root
review_target: "ticket:c10pkg10 release-packaging diff from 83ee46a working tree"
links:
  tickets:
    - ticket:c10pkg10
  evidence:
    - evidence:c10pkg10-package-metadata-smoke
  packets:
    - packet:ralph-ticket-c10pkg10-20260504T033410Z
external_refs: {}
---

# Summary

Reviewed the `ticket:c10pkg10` release-packaging diff after Ralph implementation and follow-up fixes. The review focused on package metadata correctness, optional-extra isolation, install smoke strength, dev-tool consistency, and operator-facing install guidance.

# Review Target

Target: `ticket:c10pkg10` uncommitted working-tree diff on branch `loom/dbt-110-111-hardening`, based on commit `83ee46a57a7aaf9349edb9ac5acacb02f8dee236`.

Reviewed changed surfaces:

- `pyproject.toml`
- `uv.lock`
- `Taskfile.yml`
- `.github/workflows/tests.yml`
- `.pre-commit-config.yaml`
- `src/dbt_osmosis/workbench/requirements.txt`
- `README.md`
- `docs/docs/intro.md`
- `docs/docs/reference/cli.md`
- `docs/docs/tutorial-basics/installation.md`
- `docs/docs/tutorial-yaml/synthesize.md`
- `tests/test_package_metadata.py`
- `packet:ralph-ticket-c10pkg10-20260504T033410Z`

# Verdict

`pass_with_findings`.

The initial review found six packaging and operator-surface issues. The current diff addresses them with direct dependencies, independent per-extra pip smokes, stronger workbench import coverage, pinned dev tooling, and clearer proxy docs. No critique blocker remains before the ticket acceptance gate consumes the evidence and finding dispositions.

# Findings

## FIND-001: Python 3.10 metadata test path lacked `tomli`

Severity: medium
Confidence: high
State: open

Observation:

The package metadata regression test originally relied on `tomllib`, which exists only on Python 3.11+. That weakened `ticket:c10pkg10#ACC-004` for the supported Python 3.10 runtime.

Why it matters:

The ticket changes dev dependency sources and supports Python 3.10. A metadata test that cannot run on Python 3.10 can hide drift in the oldest supported runtime.

Follow-up:

Resolved in the reviewed diff by adding a `tomllib` / `tomli` compatibility import and `tomli>=2; python_version < "3.11"` to canonical dev dependency surfaces. The ticket should consume this as `resolved` with evidence from the Python 3.10 metadata test.

Challenges:

- ticket:c10pkg10#ACC-004

## FIND-002: Direct `yaml` import was not backed by a direct dependency

Severity: medium
Confidence: high
State: open

Observation:

`src/dbt_osmosis/cli/main.py` imports `yaml` directly, but the initial package metadata cleanup focused on `sqlglot` and did not add a direct `PyYAML` dependency.

Why it matters:

Direct imports should be backed by direct dependencies or explicit optional extras. Relying on transitive availability makes base installs brittle as dbt dependency graphs change.

Follow-up:

Resolved in the reviewed diff by adding `PyYAML>=6.0` to base dependencies and asserting `yaml` imports in the base pip smoke. The ticket should consume this as `resolved`.

Challenges:

- ticket:c10pkg10#ACC-002

## FIND-003: Workbench smoke was too weak to catch import-time dependency gaps

Severity: medium
Confidence: high
State: open

Observation:

Earlier workbench smoke only checked package presence. Strengthening the smoke to import representative workbench dependencies and components exposed a real `pkg_resources` failure from `ydata_profiling` with `setuptools-82.0.1`.

Why it matters:

The workbench extra is user-facing. A smoke that does not import the dependency/component boundary can pass even when installed workbench dependencies fail at import time.

Follow-up:

Resolved in the reviewed diff by importing representative workbench modules in Taskfile and CI smokes and by bounding workbench `setuptools` to `>=70,<81`. The ticket should consume this as `resolved`; interactive Streamlit launch remains a residual risk, not a closure blocker for this packaging ticket.

Challenges:

- ticket:c10pkg10#ACC-003
- ticket:c10pkg10#ACC-006

## FIND-004: DuckDB smoke masked the new `duckdb` extra

Severity: medium
Confidence: high
State: open

Observation:

An earlier smoke installed `dbt-duckdb` explicitly while also testing `.[duckdb]`, so it could pass even if the new `duckdb` extra did not provide the adapter.

Why it matters:

`ticket:c10pkg10#ACC-006` requires install smoke for each supported optional extra. The DuckDB check must prove the extra itself installs the adapter.

Follow-up:

Resolved in the reviewed diff by removing the explicit adapter argument from the independent `.[duckdb]` smoke and adding metadata tests that reject the masked form. The ticket should consume this as `resolved`.

Challenges:

- ticket:c10pkg10#ACC-006

## FIND-005: Taskfile pre-commit tool install was unpinned

Severity: low
Confidence: high
State: open

Observation:

The Taskfile initially used an unbounded `uv tool install pre-commit`, while the canonical dev dependency declared `pre-commit>3.0.0,<5`.

Why it matters:

Unpinned tooling can make local setup drift away from the dependency surface the ticket is trying to canonicalize.

Follow-up:

Resolved in the reviewed diff by changing Taskfile setup to `uv tool install 'pre-commit>3.0.0,<5'` and asserting that shape in metadata tests. The ticket should consume this as `resolved`.

Challenges:

- ticket:c10pkg10#ACC-004

## FIND-006: Proxy extra docs could imply product support

Severity: low
Confidence: high
State: open

Observation:

Adding `dbt-osmosis[proxy]` without clear wording could make the experimental SQL proxy look like a newly supported product surface.

Why it matters:

`ticket:c10pkg10` may route dependencies, but `ticket:c10proxy25` owns proxy support semantics. Operator-facing docs need to preserve that boundary.

Follow-up:

Resolved in the reviewed diff by documenting `dbt-osmosis[proxy]` as dependency-only and linking the support decision to `ticket:c10proxy25`. The ticket should consume this as `resolved`.

Challenges:

- ticket:c10pkg10#ACC-001
- ticket:c10pkg10#ACC-002

# Evidence Reviewed

Reviewed local diff, packet child output, package metadata tests, lock checks, Python 3.10 metadata test output, independent pip install smokes, `git diff --check`, and targeted pre-commit output.

Key evidence record:

- evidence:c10pkg10-package-metadata-smoke

# Residual Risks

- The workbench smoke imports representative dependencies and components but does not launch the full Streamlit app or exercise interactive dashboard flows.
- The proxy extra remains dependency-only. Runtime support/removal semantics stay owned by `ticket:c10proxy25`.
- Final acceptance should still consume post-commit/main-branch CI evidence because this critique reviewed a local working-tree diff.

# Required Follow-up

No critique-required implementation follow-up remains. Before closure, the ticket should record each open finding above with ticket-owned disposition `resolved`, preserve post-commit CI evidence if required by the workflow, and complete or explicitly disposition retrospective / promotion follow-through.

# Acceptance Recommendation

`no-critique-blockers`.
